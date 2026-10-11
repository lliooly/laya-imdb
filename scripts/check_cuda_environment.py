"""Read-only checks of frozen local inputs and CUDA availability; never trains."""
import argparse
from importlib.metadata import version
import inspect
import json
from pathlib import Path

from common import ROOT, file_digest, write_json
import train_laya


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/local-preparation-v1/readiness.json')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Readiness record already exists; choose a new --out path')
    train_laya.block_network()
    import torch
    import laya.train as upstream

    lock = json.loads((ROOT / 'configs/laya_model_lock.json').read_text())
    split = json.loads((ROOT / 'reports/split_manifest.json').read_text())
    checks = {}
    for name, expected in lock['sha256'].items():
        path = ROOT / lock['local_dir'] / name
        checks[str(path.relative_to(ROOT))] = path.is_file() and file_digest(path) == expected
    for name in ('train', 'validation'):
        path = ROOT / f'data/processed/{name}.jsonl'
        checks[str(path.relative_to(ROOT))] = path.is_file() and file_digest(path) == split['jsonl_sha256'][name]
    if not all(checks.values()) or version('laya') != lock['laya_version']:
        raise ValueError('Frozen inputs or pinned Laya version failed readiness checks')
    configs = {}
    for path in sorted((ROOT / 'configs').glob('laya-cuda-smoke-ce-*.json')):
        cfg = json.loads(path.read_text())
        upstream.TrainConfig(**cfg['training']).validate()
        if cfg['device'] != 'cuda':
            raise ValueError('Smoke config must request CUDA')
        configs[path.name] = {'config_valid': True, 'run_id': cfg['run_id'],
                             'run_directory_available': not (ROOT / 'checkpoints' / cfg['run_id']).exists()
                                 and not (ROOT / 'reports' / cfg['run_id']).exists(),
                             'train_limit': cfg['train_limit'], 'validation_limit': cfg['validation_limit'],
                             'amp': cfg['training']['amp']}
    if len(configs) != 2 or not all(c['run_directory_available'] for c in configs.values()):
        raise ValueError('Expected two unused CUDA smoke configurations')
    cuda_available = torch.cuda.is_available()
    devices = [{'index': i, 'name': torch.cuda.get_device_name(i),
                'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
               for i in range(torch.cuda.device_count())] if cuda_available else []
    forward_source = inspect.getsource(upstream._forward)
    report = {'status': 'local_inputs_verified', 'test_accessed': False, 'training_performed': False,
              'weights_loaded': False, 'input_hash_checks': checks, 'configs': configs,
              'versions': {n: version(n) for n in ('laya', 'torch', 'transformers', 'tokenizers', 'numpy')},
              'torch_cuda_build': torch.version.cuda, 'cuda_available': cuda_available, 'devices': devices,
              'upstream_amp': {'forward_explicit_float16': 'dtype=torch.float16' in forward_source,
                               'note': 'Installed trainer explicitly requests FP16 autocast; checkpoint amp_dtype is not the training dtype selector.'},
              'cuda_functional_validation': 'pending real CUDA forward/backward/save/reload and Mac parity',
              'cuda_training_ready': False,
              'source_sha256': {'upstream_train': file_digest(Path(upstream.__file__)),
                                'train_adapter': file_digest(ROOT / 'scripts/train_laya.py'),
                                'uv_lock': file_digest(ROOT / 'uv.lock')}}
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
