"""Operations revision before formal training: allow 72h within the 300 CNY cap.
Scientific training/evaluation entrypoints and all seed groups stay unchanged.
The original 48h operation and its conservative-cost rejection are retained.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'paper-gpu-supplement-20261019-v1'
REPORT = ROOT/'reports'/STUDY
SEEDS = {'baseline': (43, 143, 243), 'combined': (42, 142, 242), 'deberta': (42, 142, 242)}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission-only', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    status_path = REPORT/'schedule-status-v2.json'
    if status_path.exists():
        status = read(status_path)
        if status['state'] == 'running':
            raise RuntimeError('Schedule already marked running; inspect its PID before retrying')
    else:
        status = dict(started_unix=time.time(), completed=[], rate_cny_per_hour=1.24,
                      total_budget_cny=300, maximum_schedule_hours=72, test_accessed=False)
    status.update(state='running', pid=os.getpid())
    write(status_path, status)
    deadline = status['started_unix']+72*3600
    def save(**fields):
        status.update(fields, updated_unix=time.time(), elapsed_hours=(time.time()-status['started_unix'])/3600)
        status['estimated_schedule_compute_cny'] = status['elapsed_hours']*1.24
        write(status_path, status)
    def execute(script, kind=None, seed=None, mode=None, pilot=False, limit=18000):
        python = ROOT/'environments'/('deberta-cu126' if kind in (None, 'deberta') else 'laya-cu126')/'.venv/bin/python'
        command = [str(python), '-u', str(ROOT/'scripts'/script), mode]
        if kind:
            command += ['--kind', kind, '--seed', str(seed)]
        if pilot:
            command += ['--pilot']
        label = '-'.join(str(x) for x in (script.removesuffix('.py'), mode, kind or '', seed or '', 'pilot' if pilot else '') if x)
        remaining = deadline-time.time()
        if remaining < 300:
            raise TimeoutError('Bounded schedule has insufficient time to start another task')
        save(phase=label, command=command)
        with (REPORT/(label+'.log')).open('a') as log:
            subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=min(limit, remaining))
        status['completed'].append(label)
        save()
    try:
        runner_manifest = read(ROOT/'runner-manifest-v2.json')
        for relative, expected in runner_manifest['source_sha256'].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Prepared runner source changed: '+relative)
        if not (REPORT/'source-contract-v2.json').exists():
            write(REPORT/'source-contract-v2.json', runner_manifest)
        if not read(ROOT/'upload-verification.json')['state'] == 'passed':
            raise ValueError('Upload did not pass verification')
        if read(REPORT/'environment-admission.json')['state'] != 'passed':
            raise ValueError('Locked environments/CUDA did not pass admission')
        if not (REPORT/'seed-admission.json').exists():
            execute('paper_gpu_supplement.py', mode='seed-audit', limit=1200)
        for kind, seed in [('deberta', 42), ('baseline', 43), ('combined', 42)]:
            path = ROOT/'reports'/f'{STUDY}-{kind}-s{seed}-admission'
            if not (path/'training-complete.json').exists():
                execute('paper_gpu_supplement.py', kind, seed, 'train', True, 1800)
            if not (path/'reload-verification.json').exists():
                execute('paper_gpu_supplement.py', kind, seed, 'verify', True, 600)
        pilots = {k: read(ROOT/'reports'/f'{STUDY}-{k}-s{SEEDS[k][0]}-admission/training-complete.json') for k in SEEDS}
        estimated = {k: v['training_seconds']/v['updates']*(1382 if k == 'deberta' else 5526)*3
                     for k, v in pilots.items()}
        # Deliberately conservative: four-step Laya pilots include startup costs;
        # additionally allow 70% training margin and two hours for I/O/evaluation.
        hours = (sum(estimated.values())*1.7+7200)/3600
        admission = dict(state='passed', pilots=pilots, estimated_training_seconds=estimated,
                         conservative_hours=hours, conservative_compute_cny=hours*1.24,
                         data_disk_free_bytes=shutil.disk_usage(ROOT).free, rate_cny_per_hour=1.24,
                         maximum_total_authorized_cny=300, official_test_accessed=False)
        write(REPORT/'resource-admission-v2.json', admission)
        if hours > 70 or hours*1.24 > 250:
            raise RuntimeError('Pilot projection exceeds bounded time/budget; do not start the nine-run schedule')
        if args.admission_only:
            save(state='admitted_waiting_for_launch', phase='admission-complete')
            return
        for repeat in range(3):
            for kind in ('deberta', 'baseline', 'combined'):
                seed = SEEDS[kind][repeat]
                run = ROOT/'reports'/f'{STUDY}-{kind}-s{seed}'
                # Maximal footprint for nine models and temporary writes fits in
                # 50GB; check before every new run, never delete unrelated files.
                if not (run/'training-complete.json').exists():
                    required_bytes = (12 if kind == 'deberta' else 5)*1024**3
                    if shutil.disk_usage(ROOT).free < required_bytes:
                        raise OSError('Need verified archive/cleanup before the next training run')
                    execute('paper_gpu_supplement.py', kind, seed, 'train')
                if not (run/'reload-verification.json').exists():
                    execute('paper_gpu_supplement.py', kind, seed, 'verify', limit=900)
                if not (run/'calibration.json').exists():
                    execute('paper_gpu_evaluate.py', kind, seed, 'calibration', limit=1800)
        if not (REPORT/'all-models-frozen.json').exists():
            execute('paper_gpu_evaluate.py', mode='freeze', limit=120)
        for repeat in range(3):
            for kind in ('baseline', 'combined', 'deberta'):
                seed = SEEDS[kind][repeat]
                run = ROOT/'reports'/f'{STUDY}-{kind}-s{seed}'
                if not (run/'development.json').exists():
                    execute('paper_gpu_evaluate.py', kind, seed, 'development', limit=3600)
        if not (REPORT/'summary.json').exists():
            execute('paper_gpu_evaluate.py', mode='aggregate', limit=120)
        save(state='completed', phase='nine-runs-and-development-completed')
    except BaseException as exc:
        save(state='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
