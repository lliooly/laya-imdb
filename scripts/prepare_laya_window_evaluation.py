"""Bind a completed EXP-3 pair to new fixed-weight study IDs and actual hashes."""
import argparse
import copy
import json
from pathlib import Path

from common import ROOT, file_digest, write_json
from run_laya_decision_study import validate_study


def prepare_pair(prefix_run, random_run, run_id):
    specs, checkpoints = [], []
    for run, mode, name in ((prefix_run, 'prefix', 'prefix-trained'),
                            (random_run, 'random_head_tail', 'random-window-trained')):
        if not isinstance(run, str) or not run or Path(run).name != run or run in ('.', '..'):
            raise ValueError('Training run must be a repository run ID')
        report = ROOT / 'reports' / run
        status = json.loads((report / 'status.json').read_text())
        audit = json.loads((report / 'window_audit.json').read_text())
        manifest_path = report / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        spec = manifest['spec']
        if (status['state'] != 'completed' or audit['state'] != 'completed'
                or spec['window_training'] != {'mode': mode} or spec['training']['seed'] != 42
                or spec['training']['epochs'] != 2 or spec['train_limit'] is not None
                or spec['validation_limit'] is not None or spec['runtime_profile'] != 'strict'
                or len(manifest['gradient_train_ids']) != 22100 or len(manifest['calibration_ids']) != 400
                or manifest.get('development_evaluation_scope') != 'separate_final_saved_checkpoint_only'
                or [e['consumed'] for e in audit['epochs']] != [22100, 22100]):
            raise ValueError('Require completed formal two-epoch strict window pair')
        normalized = copy.deepcopy(spec)
        normalized.pop('run_id')
        normalized.pop('window_training')
        specs.append((normalized, manifest['gradient_train_ids'], manifest['calibration_ids']))
        folder = ROOT / 'checkpoints' / run
        checkpoints.append({'name': name, 'checkpoint': str(folder.relative_to(ROOT)),
            'manifest': str(manifest_path.relative_to(ROOT)),
            'manifest_sha256': file_digest(manifest_path),
            'weight_sha256': file_digest(folder / 'model.safetensors')})
    if specs[0] != specs[1]:
        raise ValueError('EXP-3 arms differ beyond training window mode and run_id')
    return validate_study({'run_id': run_id, 'device': 'cuda', 'runtime_profile': 'strict',
                           'batch_size': 8, 'windows': [128, 256, 466], 'checkpoints': checkpoints})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix-run', required=True)
    parser.add_argument('--random-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a new evaluation config path')
    spec = prepare_pair(args.prefix_run, args.random_run, args.run_id)
    write_json(args.out, spec)


if __name__ == '__main__':
    main()
