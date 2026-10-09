"""Offline Mac inference: Laya-CE / MLX FP16 by default; ModernBERT remains selectable."""
import argparse
import json
from pathlib import Path

import day2_runtime
from common import ROOT, clean_text, file_digest


LABELS = ('negative', 'positive')


def predict_laya(text, folder, dtype='float16', cache_gib=2):
    # Shared task definition and offline guard do not import PyTorch.
    import train_laya
    train_laya.block_network()
    import mlx.core as mx
    import numpy as np
    from laya_mlx import Agent
    from laya_mlx.agent import collate_items
    from laya_mlx.common import temp_bucket
    from comparison_metrics import sigmoid

    mx.set_cache_limit(cache_gib * 1024**3)
    agent = Agent(str(folder), device='gpu', dtype=dtype, compile=False, cache_prompts=False)
    items, _ = agent.prepare(text, {'sentiment': train_laya.QUESTION})
    batch = collate_items(items, agent.tok.pad_token_id)
    z = np.asarray(agent.forward(batch)[0], dtype=np.float64)[0]
    if not np.isfinite(z).all():
        raise FloatingPointError('Non-finite Laya logits; retry with --dtype float32')
    qt, k = items[0]['qtype'], len(items[0]['markers'])
    assert k == 2
    temperature = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
    margin = z[1] - z[0]
    positive = float(sigmoid(margin / temperature))
    raw_positive = float(sigmoid(margin))
    return {'model': 'Laya', 'backend': 'mlx', 'dtype': dtype,
            'checkpoint': str(folder.resolve()), 'label': LABELS[int(np.argmax(z))],
            'probabilities': {'negative': 1-positive, 'positive': positive},
            'probabilities_uncalibrated': {'negative': 1-raw_positive, 'positive': raw_positive},
            'temperature': float(temperature), 'calibration': 'saved checkpoint temperature; no refitting',
            'input_tokens': len(items[0]['ids']), 'max_tokens_including_question_and_options': agent.cfg['max_len'],
            'truncation': items[0]['state_stats'], 'mlx_cache_limit_bytes': cache_gib * 1024**3}


def predict_modernbert(text, folder):
    import mlx.core as mx
    from day2_data import encode
    from modernbert_mlx import Classifier

    run = json.loads((folder / 'run.json').read_text())
    if (folder / 'best.json').exists():
        best = json.loads((folder / 'best.json').read_text())
        weights = folder / best['weights']
        if file_digest(weights) != best['sha256']:
            raise ValueError('Selected checkpoint hash mismatch')
        length = run['inference_length']
        stage = f"Full baseline; validation-selected epoch {best['epoch']}; official test not evaluated"
    else:
        if 'pilot_length' not in run['config']:
            raise ValueError('Baseline has not produced a selected checkpoint yet')
        weights = folder / 'model.safetensors'
        length = run['config']['pilot_length']
        stage = 'Day2 short pilot; predictions may be unreliable'
    model = Classifier(json.loads((folder / 'config.json').read_text()))
    model.load_weights(str(weights), strict=True)
    model.eval()
    batch = encode([{'text': text, 'label': 0}], length)
    logits = model(mx.array(batch['ids']), mx.array(batch['mask']))
    probabilities = mx.softmax(logits, axis=-1)[0].tolist()
    return {'model': 'ModernBERT-base', 'backend': 'mlx', 'dtype': 'float32',
            'label': LABELS[int(probabilities[1] > probabilities[0])],
            'probabilities_uncalibrated': dict(zip(LABELS, probabilities)),
            'max_tokens_including_special': length, 'model_stage': stage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', help='Movie review text')
    parser.add_argument('--model', choices=('laya', 'modernbert'),
                        help='Default: Laya; explicit checkpoint type is inferred when omitted')
    parser.add_argument('--checkpoint', type=Path, help='Existing local checkpoint directory')
    parser.add_argument('--dtype', choices=('float16', 'float32'), default='float16', help='Laya inference precision')
    parser.add_argument('--cache-gib', type=int, default=2, help='Laya MLX cache limit in GiB')
    args = parser.parse_args()
    text = clean_text(args.text)
    if not text:
        parser.error('Review must not be empty')
    if args.cache_gib < 1:
        parser.error('--cache-gib must be at least 1')
    kind = args.model
    if kind is None:
        kind = 'modernbert' if args.checkpoint and (args.checkpoint / 'run.json').is_file() else 'laya'
    folder = args.checkpoint or ROOT / 'checkpoints' / ('laya-l512-s42-ce' if kind == 'laya' else 'modernbert-l256-s42')
    if not folder.is_dir():
        parser.error(f'Local checkpoint not found: {folder}')
    try:
        result = predict_laya(text, folder, args.dtype, args.cache_gib) if kind == 'laya' else predict_modernbert(text, folder)
    except ModuleNotFoundError as exc:
        parser.error(f'Missing dependency {exc.name}; install with uv sync --locked --extra laya-mlx')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
