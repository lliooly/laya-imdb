"""N8 fixed 50:50 parameter average; calibration and evaluation remain separate."""
import argparse
import json
from pathlib import Path
import shutil

from common import ROOT, file_digest, write_json


def average_tensor(torch, name, first, second):
    if first.shape != second.shape or first.dtype != second.dtype:
        raise ValueError('Checkpoint tensor shape/dtype mismatch: '+name)
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise FloatingPointError('Nonfinite checkpoint tensor: '+name)
    if not first.is_floating_point():
        if not torch.equal(first, second):
            raise ValueError('Non-floating buffers must match: '+name)
        return first.clone()
    # Temperature is calibration state, not a learned weight. Refit on 400 only.
    result = torch.ones_like(first) if name == 'temperature' else ((first.float()+second.float())/2).to(first.dtype)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite averaged tensor: '+name)
    return result.contiguous()


def normalized_config(folder):
    cfg = json.loads((folder / 'rl_agent_config.json').read_text())
    for key in ('temperature', 'temperature_by_options', 'abstention_thresholds'):
        cfg.pop(key, None)
    return cfg


def compare_assets(first, second):
    if normalized_config(first) != normalized_config(second):
        raise ValueError('Checkpoint architecture/configuration differs')
    for folder in ('tokenizer', 'encoder'):
        a = {p.relative_to(first / folder) for p in (first / folder).rglob('*') if p.is_file()}
        b = {p.relative_to(second / folder) for p in (second / folder).rglob('*') if p.is_file()}
        if a != b or not a:
            raise ValueError('Missing or different checkpoint assets')
        for path in a:
            pa, pb = first / folder / path, second / folder / path
            if folder == 'tokenizer' and str(path) == 'tokenizer.json':
                def semantics(p):
                    doc = json.loads(p.read_text())
                    doc.pop('padding', None)
                    doc.pop('truncation', None)
                    return doc
                equal = semantics(pa) == semantics(pb)
            else:
                equal = file_digest(pa) == file_digest(pb)
            if not equal:
                raise ValueError('Checkpoint assets differ: '+str(path))


def average_checkpoints(first, second, out, expected_first, expected_second):
    if out.exists():
        raise FileExistsError('Use a new checkpoint directory')
    paths = [p / 'model.safetensors' for p in (first, second)]
    expected = [expected_first, expected_second]
    if [file_digest(p) for p in paths] != expected:
        raise ValueError('Source checkpoint hash mismatch')
    compare_assets(first, second)
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    with safe_open(paths[0], framework='pt', device='cpu') as a, safe_open(paths[1], framework='pt', device='cpu') as b:
        if set(a.keys()) != set(b.keys()) or 'temperature' not in a.keys():
            raise ValueError('Checkpoint keys differ or temperature is missing')
        weights = {key: average_tensor(torch, key, a.get_tensor(key), b.get_tensor(key)) for key in a.keys()}
    if [file_digest(p) for p in paths] != expected:
        raise ValueError('Checkpoint changed during averaging')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('encoder', 'tokenizer'):
        shutil.copytree(second / name, out / name)
    cfg = normalized_config(second)
    cfg.update(temperature=[1., 1., 1.], temperature_by_options={})
    write_json(out / 'rl_agent_config.json', cfg)
    save_file(weights, out / 'model.safetensors')
    write_json(out / 'average_manifest.json', {'state': 'weights_created_uncalibrated',
        'inputs': [{'path': str(p.parent), 'weight_sha256': sha} for p, sha in zip(paths, expected)],
        'weights': [.5, .5], 'weight_sha256': file_digest(out / 'model.safetensors'),
        'temperature_reset': True, 'calibration_required': 'original_400_only',
        'training_performed': False, 'inference_performed': False, 'test_accessed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('first', 'second', 'out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--first-sha256', required=True)
    parser.add_argument('--second-sha256', required=True)
    args = parser.parse_args()
    for path in (args.first, args.second, args.out):
        if not path.resolve().is_relative_to(ROOT / 'checkpoints'):
            raise ValueError('N8 input/output must stay under repository checkpoints')
    average_checkpoints(args.first, args.second, args.out, args.first_sha256, args.second_sha256)


if __name__ == '__main__':
    main()
