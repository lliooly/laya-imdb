"""Offline real-model inference with the frozen two-model Laya fusion."""
import argparse
import json
from pathlib import Path

import train_laya
from common import ROOT, clean_text, file_digest
from laya_fusion_inference import FrozenFusionPredictor, MODEL_NAMES
from validate_laya_fusion_inference import relative_path, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', help='Movie review text')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    text = clean_text(args.text)
    if not text:
        parser.error('Review must not be empty')
    spec = json.loads(args.config.read_text())
    if spec.get('device') == 'mlx-fp32-precise':
        from validate_laya_mlx_precise import validate_config as validate_precise_config
        validate_precise_config(spec)
    elif spec.get('device') == 'mlx':
        from validate_laya_mlx_fusion import validate_config as validate_mlx_config
        validate_mlx_config(spec)
    else:
        validate_config(spec)
    frozen_path = relative_path(ROOT, spec['frozen_fit'])
    if file_digest(frozen_path) != spec['input_sha256'][spec['frozen_fit']]:
        raise ValueError('Frozen coefficient SHA256 mismatch')
    train_laya.block_network()
    if spec['device'] == 'mlx-fp32-precise':
        from laya_mlx_precise_inference import PreciseMLXFusionPredictor
        engine = PreciseMLXFusionPredictor(ROOT, spec['checkpoint_root'], spec['models'],
            json.loads(frozen_path.read_text()), spec['device'], spec['batch_size'])
    elif spec['device'] == 'mlx':
        from laya_mlx_fusion_inference import MLXFusionPredictor
        engine = MLXFusionPredictor(ROOT, spec['checkpoint_root'], spec['models'],
            json.loads(frozen_path.read_text()), 'mlx', spec['batch_size'])
    else:
        import torch
        train_laya.require_device(torch, spec['device'])
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        engine = FrozenFusionPredictor(ROOT, spec['checkpoint_root'], spec['models'],
            json.loads(frozen_path.read_text()), spec['device'], spec['batch_size'])
    for name in MODEL_NAMES:
        engine.load(name)
    prediction = engine.predict([text])
    p = prediction['primary'][0]
    print(json.dumps({'label': ('negative', 'positive')[int(p[1] > p[0])],
                      'probabilities': dict(zip(('negative', 'positive'), p)),
                      'backend': spec['device'], 'dtype': 'float16' if spec['device'] == 'mlx' else 'float32',
                      'truncation': prediction['stats'][0],
                      'model_forward_rows': prediction['forward_rows'],
                      'coefficient': engine.coefficient, 'parameters_refitted': False},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
