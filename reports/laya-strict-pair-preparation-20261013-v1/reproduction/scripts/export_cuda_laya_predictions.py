"""Evaluate saved CUDA Laya weights on the frozen full validation split, offline."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import train_laya
from common import ROOT, file_digest, read_jsonl, write_json
from comparison_metrics import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose an unused export path')
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    train_laya.block_network()
    import numpy as np
    import torch
    import laya.train as t
    train_laya.require_device(torch, 'cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = ROOT / 'data/processed/validation.jsonl'
    split = json.loads((ROOT / 'reports/split_manifest.json').read_text())
    if file_digest(source) != split['jsonl_sha256']['validation']:
        raise ValueError('Frozen validation hash mismatch')
    rows = list(read_jsonl(source))
    if len(rows) != 2500:
        raise ValueError('Full validation must have 2500 samples')
    model, tok, cfg = t.load_checkpoint(str(args.checkpoint))
    model.eval().to('cuda')
    parallel = t.uses_parallel_layout(cfg)
    predictions = []
    began = time.monotonic()
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        typed = [{'state': row['text'], 'questions': {'sentiment': train_laya.QUESTION},
                  'expected': {'sentiment': ('negative', 'positive')[row['label']]}} for row in chunk]
        items, skipped = t.items_from_rows(tok, typed, cfg['max_len'], cfg['head_max_len'], label_smoothing=0.0)
        if skipped or len(items) != len(chunk):
            raise ValueError('Every frozen validation sample must be evaluated')
        records = t.calibration_records(model, tok, items, torch.device('cuda'),
            cfg['max_len'], cfg['head_max_len'], batch_size=args.batch_size, parallel=parallel)
        for row, item, record in zip(chunk, items, records):
            prepared = t.encode_item(tok, item, cfg['max_len'], cfg['head_max_len'], parallel=parallel)
            inputs = {key: prepared[key] for key in ('ids', 'markers', 'qtype')}
            encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
            logits = np.asarray(record[1], dtype=float)
            if logits.shape != (2,) or not np.isfinite(logits).all():
                raise FloatingPointError('Saved model produced invalid binary logits')
            probabilities = np.exp((logits - logits.max()) / cfg['temperature'][0])
            probabilities /= probabilities.sum()
            predictions.append({'sample_id': row['id'], 'true_label': row['label'],
                'input_sha256': hashlib.sha256(encoded).hexdigest(), 'logits': logits.tolist(),
                'probabilities': probabilities.tolist(), 'pred_label': int(logits.argmax())})
        if start % 400 == 0:
            print(min(start + args.batch_size, len(rows)), '/2500', flush=True)
    margins = [p['logits'][1] - p['logits'][0] for p in predictions]
    report = {'state': 'completed', 'passed': True, 'test_accessed': False,
        'checkpoint_sha256': file_digest(args.checkpoint / 'model.safetensors'),
        'validation_sha256': split['jsonl_sha256']['validation'],
        'temperature': cfg['temperature'][0], 'dtype': 'float32', 'tf32': False,
        'batch_size': args.batch_size, 'elapsed_seconds': time.monotonic() - began,
        'raw_metrics': metrics([r['label'] for r in rows], margins),
        'native_calibrated_metrics': metrics([r['label'] for r in rows], margins, cfg['temperature'][0]),
        'source_sha256': file_digest(Path(__file__)),
        'versions': {name: version(name) for name in ('torch', 'laya', 'transformers', 'numpy')},
        'validation_predictions': predictions}
    write_json(args.out, report)
    print(json.dumps(report['native_calibrated_metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
