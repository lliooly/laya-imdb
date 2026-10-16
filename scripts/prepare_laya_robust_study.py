"""Prepare a new immutable protocol; does not start CUDA or connect to a host."""
import argparse
import json
from pathlib import Path

from common import ROOT, file_digest
from laya_checkpoint_delivery import atomic_json
from run_laya_robust_loss import STUDY, PAIRS, contract_path


def prepare():
    from run_laya_head_control import TRAINING
    old = json.loads((ROOT/'configs/laya-head-control-cu126-20261015-v2.json').read_text())
    for rel, sha in old['source_sha256'].items():
        if file_digest(ROOT/rel) != sha:
            raise ValueError('Historical source changed: ' + rel)
    extra = [p for folder in ('scripts','tests') for p in (ROOT/folder).glob('*robust*.py')]
    extra += [ROOT/'scripts/laya_checkpoint_delivery.py', ROOT/'tests/test_laya_checkpoint_delivery.py']
    paths = set(old['source_sha256']) | {str(p.relative_to(ROOT)) for p in extra}
    d = {'study_id': STUDY, 'approved_design': 'docs/superpowers/specs/2026-10-15-laya-robust-loss-and-checkpoint-delivery-design.md',
         'training': TRAINING, 'pairs': [list(p) for p in PAIRS], 'loss': {'q':.7, 'gce_mixture_weight':.5},
         'runtime': {'device':'cuda:0','profile':'strict','dtype':'float32','amp':False,'tf32':False},
         'source_sha256': {p: file_digest(ROOT/p) for p in sorted(paths)},
         'local_delivery_required': True, 'minimum_free_bytes': 100*1024**3,
         'state': 'prepared_pending_cuda_and_transfer_admission', 'cuda_admission_passed': False,
         'test_accessed': False, 'old_l2_weight_recovery_required': False}
    path = contract_path()
    if path.exists():
        if json.loads(path.read_text()) != d:
            raise FileExistsError('Prepared contract changed; create a new version/run ID')
    else:
        atomic_json(path, d)
    return path, d


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    path, d = prepare()
    print(f'{path}: {len(d["source_sha256"])} sources bound; no CUDA admission or training performed')
