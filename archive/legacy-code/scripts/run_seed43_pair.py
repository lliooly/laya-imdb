"""Budget-bounded seed=43 replication; evaluates preserved epochs after training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import ROOT, file_digest, write_json
from comparison_metrics import paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--remaining-yuan', type=float, required=True)
    parser.add_argument('--hourly-price', type=float, default=1.24)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc).timestamp()
    configs = [ROOT / ('configs/laya-rtx4090-ce-' + name + '-s43-epochs2-fp32.json')
               for name in ('baseline', 'combined')]
    specs = [json.loads(p.read_text()) for p in configs]
    if specs[0]['training']['calib_seed'] != 42 or specs[1]['training']['calib_seed'] != 42:
        raise ValueError('Keep the frozen calibration split')
    for spec in specs:
        if spec['training']['seed'] != 43 or not spec['epoch_checkpoints']:
            raise ValueError('Seed and preservation protocol mismatch')
        smoke = ROOT / 'reports' / spec['run_id'].replace('ce-', 'smoke-ce-', 1) / 'numerical_audit.json'
        audit = json.loads(smoke.read_text())
        if not audit.get('passed'):
            raise ValueError('CUDA smoke audit did not pass')
        smoke_epochs = json.loads((smoke.parent / 'epoch_checkpoints.json').read_text())['epochs']
        if [e['epoch'] for e in smoke_epochs] != [1, 2] or len(audit['optimizer_steps']) != 28:
            raise ValueError('Smoke epoch/update accounting mismatch')
        for entry in smoke_epochs:
            if file_digest(Path(entry['checkpoint']) / 'model.safetensors') != entry['weights_sha256']:
                raise ValueError('Smoke epoch weights changed')
        if (ROOT / 'reports' / spec['run_id']).exists():
            raise FileExistsError(spec['run_id'])
        if (ROOT / 'checkpoints' / spec['run_id']).exists():
            raise FileExistsError(spec['run_id'])
    available = min(deadline - time.time(), args.remaining_yuan / args.hourly_price * 3600 - 300)
    if available < 5600:
        raise ValueError('Insufficient estimated balance/time for the paired experiment (5600s)')
    deadline = min(deadline, time.time() + available)
    output = ROOT / 'artifacts/ce-seed43-20261011'
    output.mkdir(parents=True, exist_ok=False)
    state = {'state': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'deadline_utc': datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
             'budget': vars(args), 'includes_idle': True, 'test_accessed': False,
             'runner_sha256': file_digest(Path(__file__)), 'stages': []}
    status = output / 'status.json'

    def run(stage, command, cap):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('Budget deadline reached')
        entry = {'stage': stage, 'started_utc': datetime.now(timezone.utc).isoformat(),
                 'command': command, 'timeout_seconds': min(cap, remaining)}
        state['stages'].append(entry)
        write_json(status, state)
        with (output / (stage + '.log')).open('w') as log:
            process = subprocess.Popen([sys.executable, *command], cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=entry['timeout_seconds'])
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        entry.update(returncode=code, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(status, state)
        if code:
            raise RuntimeError(stage + ' failed: ' + str(code))

    try:
        for name, config, spec in zip(('baseline', 'combined'), configs, specs):
            run(name + '-train', ['scripts/train_laya.py', '--config', str(config)], 3300)
            report = ROOT / 'reports' / spec['run_id']
            checkpoint = ROOT / 'checkpoints' / spec['run_id']
            epochs = json.loads((report / 'epoch_checkpoints.json').read_text())['epochs']
            if [e['epoch'] for e in epochs] != [1, 2]:
                raise ValueError('Both epoch checkpoints must exist')
            curve = []
            for epoch in epochs:
                path = checkpoint / ('epoch-%03d' % epoch['epoch'])
                if file_digest(path / 'model.safetensors') != epoch['weights_sha256']:
                    raise ValueError('Preserved epoch hash mismatch')
                export = report / ('cuda-saved-validation-epoch%d.json' % epoch['epoch'])
                run(name + '-epoch%d-validation' % epoch['epoch'],
                    ['scripts/export_cuda_laya_predictions.py', '--checkpoint', str(path), '--out', str(export)], 240)
                curve.append(dict(epoch, raw_metrics=json.loads(export.read_text())['raw_metrics']))
            export = report / 'cuda-saved-validation.json'
            run(name + '-final-validation', ['scripts/export_cuda_laya_predictions.py',
                '--checkpoint', str(checkpoint), '--out', str(export)], 240)
            write_json(report / 'epoch_validation_curve.json', {'epochs': curve,
                'primary_metric': 'final_saved_checkpoint_accuracy',
                'best_epoch': max(curve, key=lambda e: (e['raw_metrics']['accuracy'], e['epoch']))['epoch'],
                'epoch_temperature_is_pre_final_calibration': True, 'test_accessed': False})
        reports = [ROOT / 'reports' / s['run_id'] for s in specs]
        manifests = [json.loads((r / 'manifest.json').read_text()) for r in reports]
        for key in ('gradient_train_ids', 'calibration_ids', 'validation'):
            if manifests[0][key] != manifests[1][key]:
                raise ValueError('Paired data mismatch: ' + key)
        exports = [json.loads((r / 'cuda-saved-validation.json').read_text()) for r in reports]
        predictions = [e['validation_predictions'] for e in exports]
        if [(p['sample_id'], p['true_label'], p['input_sha256']) for p in predictions[0]] != [
                (p['sample_id'], p['true_label'], p['input_sha256']) for p in predictions[1]]:
            raise ValueError('Paired validation input mismatch')
        write_json(output / 'comparison.json', {'seed': 43, 'test_accessed': False,
            'input_hashes_all_match': True, 'metrics': {n: e['raw_metrics'] for n, e in zip(('baseline','combined'), exports)},
            'paired': paired([p['true_label'] for p in predictions[0]],
                             [p['logits'][1] - p['logits'][0] for p in predictions[0]],
                             [p['logits'][1] - p['logits'][0] for p in predictions[1]])})
        state['state'] = 'completed'
    except BaseException as exc:
        state.update(state='failed', error=repr(exc))
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(status, state)


if __name__ == '__main__':
    main()
