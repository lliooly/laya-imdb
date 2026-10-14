"""Bounded sequential study -> evidence gate -> strict window pair -> final study."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import ROOT, file_digest, write_json
from run_laya_decision_study import validate_study
from prepare_laya_window_evaluation import prepare_pair


def window_evidence(summary):
    """N4 tests training coverage, independently of whether N2 fusion succeeds."""
    if summary.get('state') != 'completed' or summary.get('test_accessed') is not False:
        raise ValueError('Require completed sealed-test fixed-weight study')
    models = summary.get('models', {})
    if set(models) != {'baseline-s43', 'combined-s42'}:
        raise ValueError('Require the two predeclared checkpoint identities')
    evidence = {}
    for name, model in models.items():
        metrics = model['single_and_fused']
        head, tail = metrics['head'], metrics['head_tail']
        if (head['all']['count'] != 2500 or tail['all']['count'] != 2500
                or head['truncated']['count'] != tail['truncated']['count']
                or head['truncated']['count'] < 1):
            raise ValueError('Require full paired development and truncation scope')
        evidence[name] = {
            'head_tail_delta_pp': 100*(tail['all']['accuracy']-head['all']['accuracy']),
            'truncated_delta_pp': 100*(tail['truncated']['accuracy']-head['truncated']['accuracy']),
            'n2_success': model['exp1']['success']}
    supported = all(e['head_tail_delta_pp'] > 0 and e['truncated_delta_pp'] > 0 for e in evidence.values())
    return {'supported': supported, 'models': evidence,
            'criterion': 'head-tail coverage improves both fixed checkpoints; no hyperparameter selection',
            'n2_is_independent': True}


def validate_pair_specs(prefix, random_view):
    normalized = []
    for spec, mode in ((prefix, 'prefix'), (random_view, 'random_head_tail')):
        if (spec.get('window_training') != {'mode': mode} or spec.get('runtime_profile') != 'strict'
                or spec['training']['seed'] != 42 or spec['training']['epochs'] != 2
                or spec['training']['amp'] is not False or spec['train_limit'] is not None
                or spec['validation_limit'] is not None):
            raise ValueError('Require full-source strict seed42 window training pair')
        clean = copy.deepcopy(spec)
        clean.pop('run_id')
        clean.pop('window_training')
        normalized.append(clean)
    if normalized[0] != normalized[1] or prefix['run_id'] == random_view['run_id']:
        raise ValueError('Window pair differs beyond window mode/run_id or reuses a run_id')


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-config', type=Path, required=True)
    p.add_argument('--prefix-config', type=Path, required=True)
    p.add_argument('--random-config', type=Path, required=True)
    p.add_argument('--run-id', default='laya-decision-execution-20261014-v1')
    args = p.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('Invalid pipeline run_id')
    study = validate_study(json.loads(args.study_config.read_text()))
    prefix = json.loads(args.prefix_config.read_text())
    random_view = json.loads(args.random_config.read_text())
    validate_pair_specs(prefix, random_view)
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    status = {'state': 'running', 'started_utc': now(), 'stages': [], 'test_accessed': False,
              'primary_checkpoint': 'final_epoch', 'early_stopping_on_development': False,
              'runner_sha256': file_digest(Path(__file__)),
              'config_sha256': {str(path): file_digest(path) for path in
                               (args.study_config, args.prefix_config, args.random_config)}}
    def save():
        write_json(out / 'status.json', status)
    def run(label, command, seconds):
        entry = {'stage': label, 'command': command, 'started_utc': now()}
        status['stages'].append(entry)
        save()
        with (out / (label+'.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            entry['pid'] = child.pid
            save()
            try:
                code = child.wait(timeout=seconds)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
        entry.update(returncode=code, finished_utc=now())
        save()
        if code:
            raise RuntimeError(label+' failed; see '+str(out / (label+'.log')))
    try:
        save()
        for name, path in (('prefix', args.prefix_config), ('random', args.random_config)):
            run(name+'-preflight', ['scripts/train_laya.py', '--config', str(path), '--check-only'], 180)
        run('fixed-weight-study', ['scripts/run_laya_decision_study.py', '--config', str(args.study_config)], 2400)
        summary_path = ROOT / 'reports' / study['run_id'] / 'summary.json'
        evidence = window_evidence(json.loads(summary_path.read_text()))
        write_json(out / 'window-evidence.json', dict(evidence, study_summary_sha256=file_digest(summary_path)))
        if not evidence['supported']:
            status.update(state='completed_stopped_by_evidence', finished_utc=now(),
                          reason='N4 coverage signal did not replicate across the two fixed checkpoints')
            save()
            return
        if shutil.disk_usage(ROOT).free < 8*2**30:
            raise RuntimeError('Require 8 GiB free for both training arms')
        for name, path, spec in (('prefix', args.prefix_config, prefix), ('random', args.random_config, random_view)):
            run(name+'-train', ['scripts/train_laya.py', '--config', str(path)], 3600)
            report = ROOT / 'reports' / spec['run_id']
            audit = json.loads((report / 'window_audit.json').read_text())
            optimization = json.loads((report / 'optimization_diagnostics.json').read_text())
            if (audit['state'] != 'completed' or [e['consumed'] for e in audit['epochs']] != [22100, 22100]
                    or optimization['state'] != 'completed' or optimization['completed_optimizer_updates'] != 5526):
                raise RuntimeError('Training consumption/update audit failed: '+name)
        final_config = prepare_pair(prefix['run_id'], random_view['run_id'], args.run_id+'-final-study')
        config_path = out / 'final-study-config.json'
        write_json(config_path, final_config)
        run('final-pair-study', ['scripts/run_laya_decision_study.py', '--config', str(config_path)], 2400)
        status.update(state='completed', finished_utc=now(), final_report='reports/'+final_config['run_id']+'/summary.json')
        save()
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), finished_utc=now())
        save()
        raise


if __name__ == '__main__':
    main()
