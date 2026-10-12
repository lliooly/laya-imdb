"""Bounded, fresh-process diagnostics; no full training or post-training eval."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, write_json
from laya_determinism import compare_workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT / 'reports'):
        raise ValueError('Output must be under reports')
    out.mkdir(parents=True, exist_ok=False)
    spec = json.loads((ROOT / 'configs/laya-rtx4090-ce-resampling-control-pool45-s43-epochs2-fp32-v2.json').read_text())
    status = {'state': 'running', 'completed': [], 'conditions': {}, 'test_accessed': False,
              'maximum_workers': 10, 'maximum_updates': 160}
    started = time.monotonic()
    write_json(out / 'status.json', status)

    def run_pair(profile, mode='original'):
        labels = []
        condition = profile + '-' + mode
        for repeat in (1, 2):
            label = f'{condition}-{repeat}'
            config = json.loads(json.dumps(spec))
            config['run_id'] = out.name + '-' + label
            if mode == 'original':
                config.pop('sampling')
            if profile == 'no-checkpoint':
                config['training']['gradient_checkpointing'] = False
            path = out / (label + '-config.json')
            write_json(path, config)
            status['active_worker'] = label
            write_json(out / 'status.json', status)
            with (out / (label + '.log')).open('x') as log:
                try:
                    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/run_laya_short_trace.py'),
                        '--worker', mode, '--config', str(path), '--out', str(out / label),
                        '--runtime-profile', profile, '--parameter-details'], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                except subprocess.TimeoutExpired:
                    status['conditions'][condition] = {'state': 'timeout', 'worker': label}
                    return None
            if proc.returncode:
                text = (out / (label + '.log')).read_text()
                if 'OutOfMemoryError' in text or 'CUDA out of memory' in text:
                    reason = 'cuda_oom'
                elif 'deterministic' in text.lower() and profile == 'strict' and 'RuntimeError' in text:
                    reason = 'deterministic_runtime_error'
                else:
                    raise RuntimeError('Unexpected worker failure: ' + label)
                status['conditions'][condition] = {'state': reason, 'worker': label, 'exit_code': proc.returncode}
                return None
            result = json.loads((out / label / 'result.json').read_text())
            assert result['state'] == 'diagnostic_completed' and result['completed_updates'] == 16
            assert result['full_schedule_updates'] == 5526
            labels.append(label)
            status['completed'].append(label)
            write_json(out / 'status.json', status)
        comparison = compare_workers(out / labels[0], out / labels[1])
        write_json(out / (condition + '-comparison.json'), comparison)
        status['conditions'][condition] = {'state': 'compared', **comparison}
        write_json(out / 'status.json', status)
        return comparison

    try:
        for profile in ('baseline', 'no-checkpoint', 'math-sdp', 'strict'):
            run_pair(profile)
            write_json(out / 'status.json', status)
        strict = status['conditions'].get('strict-original', {})
        if strict.get('identical'):
            control = run_pair('strict', 'control')
            if control is not None:
                write_json(out / 'strict-original-vs-control-comparison.json',
                           compare_workers(out / 'strict-original-1', out / 'strict-control-1'))
        status['state'] = 'completed'
    except BaseException as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        status.pop('active_worker', None)
        status['elapsed_seconds'] = time.monotonic() - started
        write_json(out / 'status.json', status)


if __name__ == '__main__':
    main()
