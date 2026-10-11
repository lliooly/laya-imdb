"""Verify CUDA smoke checkpoint inputs and predictions on Mac MLX FP16."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import day2_runtime
import train_laya
from common import ROOT, file_digest, read_jsonl, write_json
from comparison_metrics import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audit', type=Path)
    group.add_argument('--cuda-export', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose an unused output path')
    reference = json.loads((args.audit or args.cuda_export).read_text())
    if not reference.get('passed'):
        raise ValueError('CUDA numerical audit must pass before Mac parity')
    reload = reference['checkpoint_reload'] if args.audit else reference
    digest = file_digest(args.checkpoint / 'model.safetensors')
    if digest != reload['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the audited CUDA artifact')
    frozen = ROOT / 'data/processed/validation.jsonl'
    split = json.loads((ROOT / 'reports/split_manifest.json').read_text())
    if file_digest(frozen) != split['jsonl_sha256']['validation']:
        raise ValueError('Frozen validation hash mismatch')
    rows = {row['id']: row for row in read_jsonl(frozen)}
    if args.cuda_export:
        if reference['validation_sha256'] != split['jsonl_sha256']['validation']:
            raise ValueError('CUDA export used different validation data')
        if [p['sample_id'] for p in reload['validation_predictions']] != list(rows):
            raise ValueError('Full export must include all 2500 IDs in frozen order')
    train_laya.block_network()
    import numpy as np
    import mlx.core as mx
    from laya_mlx import Agent
    from laya_mlx.agent import collate_items
    mx.set_cache_limit(2 * 1024**3)
    agent = Agent(str(args.checkpoint), dtype='float16', device='gpu',
                  compile=False, cache_prompts=False)
    cfg = json.loads((args.checkpoint / 'rl_agent_config.json').read_text())
    results = []
    for index, ref in enumerate(reload['validation_predictions']):
        row = rows[ref['sample_id']]
        item = agent.prepare(row['text'], {'sentiment': train_laya.QUESTION})[0][0]
        inputs = {key: item[key] for key in ('ids', 'markers', 'qtype')}
        encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
        input_digest = hashlib.sha256(encoded).hexdigest()
        if input_digest != ref['input_sha256']:
            raise ValueError('Prepared inputs differ on sample ' + row['id'])
        logits = np.asarray(agent.forward(collate_items([item], agent.tok.pad_token_id))[0])[0].astype(float)
        if not np.isfinite(logits).all():
            raise FloatingPointError('Nonfinite MLX logits')
        probabilities = np.exp((logits - logits.max()) / cfg['temperature'][0])
        probabilities /= probabilities.sum()
        results.append({'sample_id': row['id'], 'input_sha256': input_digest,
            'true_label': row['label'],
            'cuda_logits': ref['logits'], 'mlx_fp16_logits': logits.tolist(),
            'max_abs_logit_delta': float(np.abs(logits - ref['logits']).max()),
            'max_abs_probability_delta': float(np.abs(probabilities - ref['probabilities']).max()),
            'cuda_pred_label': ref['pred_label'], 'mlx_pred_label': int(logits.argmax())})
        if args.cuda_export and (index + 1) % 400 == 0:
            print('MLX parity', index + 1, '/2500', flush=True)
    labels_match = all(r['cuda_pred_label'] == r['mlx_pred_label'] for r in results)
    probability_delta = max(r['max_abs_probability_delta'] for r in results)
    # A declared absolute tolerance is an engineering smoke gate, not an accuracy claim.
    tolerance = 0.01
    passed = labels_match and probability_delta <= tolerance
    report = {'passed': passed, 'test_accessed': False, 'checkpoint_sha256': digest,
        'validation_sha256': split['jsonl_sha256']['validation'], 'dtype': 'float16',
        'inputs_match': len(results), 'labels_match': labels_match,
        'max_abs_probability_delta': probability_delta, 'probability_tolerance': tolerance,
        'scope': 'Full frozen validation' if args.cuda_export else 'Smoke validation items only; full validation parity remains a formal-run gate',
        'label_agreement': sum(r['cuda_pred_label'] == r['mlx_pred_label'] for r in results) / len(results),
        'mlx_raw_metrics': metrics([r['true_label'] for r in results],
                                  [r['mlx_fp16_logits'][1] - r['mlx_fp16_logits'][0] for r in results]),
        'mlx_native_calibrated_metrics': metrics([r['true_label'] for r in results],
             [r['mlx_fp16_logits'][1] - r['mlx_fp16_logits'][0] for r in results], cfg['temperature'][0]),
        'versions': {name: version(name) for name in ('mlx', 'laya-mlx', 'numpy')},
        'source_sha256': file_digest(Path(__file__)), 'samples': results}
    write_json(args.out, report)
    print(json.dumps({key: val for key, val in report.items() if key != 'samples'}, indent=2))
    if not passed:
        raise RuntimeError('Mac MLX FP16 smoke parity failed; inspect the report')


if __name__ == '__main__':
    main()
