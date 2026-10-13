"""Export fixed-weight head/head-tail predictions on calibration and development only."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time

import train_laya
from common import ROOT, file_digest, read_jsonl, write_json


def select_splits(manifest, split, train_rows, validation_rows):
    """Retain manifest calibration order; reject overlap, missing IDs and changed splits."""
    train_ids = [r['id'] for r in train_rows]
    dev_ids = [r['id'] for r in validation_rows]
    calib = manifest['calibration_ids']
    gradient = manifest['gradient_train_ids']
    if train_ids != split['ids']['train'] or dev_ids != split['ids']['validation']:
        raise ValueError('Frozen split IDs/order differ')
    if train_ids != manifest['train']['ids'] or dev_ids != manifest['validation']['ids']:
        raise ValueError('Formal manifest split IDs/order differ')
    if len(set(train_ids)) != len(train_ids) or len(set(dev_ids)) != len(dev_ids):
        raise ValueError('Duplicate source IDs')
    if len(calib) != 400 or len(set(calib)) != 400 or len(validation_rows) != 2500:
        raise ValueError('Expected unique calibration 400 and development 2500')
    if len(gradient) != len(set(gradient)) or set(calib) & set(gradient):
        raise ValueError('Calibration overlaps gradient training')
    if set(calib) | set(gradient) != set(train_ids) or set(train_ids) & set(dev_ids):
        raise ValueError('Invalid train/calibration/development partition')
    if any(r['label'] not in (0, 1) for r in train_rows + validation_rows):
        raise ValueError('Expected binary labels')
    by_id = {r['id']: r for r in train_rows}
    return {'calibration': [by_id[i] for i in calib], 'validation': validation_rows}


def prepare_rules(tok, rows, max_len=512, head_max_len=192):
    """Select original token IDs, never decode/re-encode the head-tail text."""
    import laya.train as upstream
    from laya.common import build_head
    typed = [{'state': r['text'], 'questions': {'sentiment': train_laya.QUESTION},
              'expected': {'sentiment': ('negative', 'positive')[r['label']]}} for r in rows]
    items, skipped = upstream.items_from_rows(tok, typed, max_len, head_max_len, label_smoothing=0.0)
    if skipped or len(items) != len(rows):
        raise ValueError('Every selected sample must produce one item')
    head, _, _ = build_head(tok, items[0]['q'], head_max_len)
    budget = max_len - len(head) - 1
    if (len(head), budget) != (45, 466) or max_len != 512 or head_max_len != 192:
        raise ValueError('Unexpected fixed question template/token budget')
    tok.backend_tokenizer.no_padding()
    tok.backend_tokenizer.no_truncation()
    enc = tok.backend_tokenizer.encode_batch(
        [r['text'].replace(tok.mask_token, ' ') for r in rows], add_special_tokens=False)
    variants, stats = [], []
    for item, encoding in zip(items, enc):
        full = encoding.ids
        if item['state_ids'] != full[:max_len]:
            raise ValueError('Original tokenization differs from training preprocessing')
        altered = dict(item)
        changed = len(full) > budget
        altered['state_ids'] = full[:budget // 2] + full[-(budget-budget // 2):] if changed else item['state_ids']
        original = upstream.encode_item(tok, item, max_len, head_max_len, parallel=False)
        modified = upstream.encode_item(tok, altered, max_len, head_max_len, parallel=False)
        if original['markers'] != modified['markers'] or original['qtype'] != modified['qtype']:
            raise ValueError('Head-tail changed question layout')
        if not changed and original != modified:
            raise ValueError('Untruncated inputs must be identical')
        variants.append(altered)
        stats.append({'truncated': changed, 'review_tokens': len(full), 'review_token_budget': budget})
    return items, variants, stats, len(head), budget


def input_digest(prepared):
    payload = {key: prepared[key] for key in ('ids', 'markers', 'qtype')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checked_logits(record):
    import numpy as np
    values = np.asarray(record[1], dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise FloatingPointError('Expected finite binary logits')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-weight-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'mps'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a new output run directory')
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    train_laya.block_network()
    manifest = json.loads(args.manifest.read_text())
    if manifest['question'] != train_laya.QUESTION:
        raise ValueError('Formal manifest question differs from fixed experiment template')
    split_path = ROOT / 'reports/split_manifest.json'
    split = json.loads(split_path.read_text())
    source_sha = {}
    source_rows = {}
    for name in ('train', 'validation'):
        path = ROOT / f'data/processed/{name}.jsonl'
        actual = file_digest(path)
        if actual != split['jsonl_sha256'][name] or actual != manifest[name]['source_sha256']:
            raise ValueError(f'Frozen {name} hash mismatch')
        source_sha[name] = actual
        source_rows[name] = list(read_jsonl(path))
    selected = select_splits(manifest, split, source_rows['train'], source_rows['validation'])
    actual_weight = file_digest(args.checkpoint / 'model.safetensors')
    if actual_weight != args.expected_weight_sha256:
        raise ValueError('Checkpoint hash differs from expected final weights')
    import numpy as np
    import torch
    import laya.train as upstream
    train_laya.require_device(torch, args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tok, cfg = upstream.load_checkpoint(str(args.checkpoint))
    if upstream.uses_parallel_layout(cfg):
        raise ValueError('This experiment requires the current non-parallel model layout')
    training_cfg = manifest['spec']['training']
    if cfg['max_len'] != training_cfg['max_len'] or cfg['head_max_len'] != training_cfg['head_max_len']:
        raise ValueError('Checkpoint context settings differ from training manifest')
    # Saved tokenizers may persist temporary padding/truncation; only semantic changes are disallowed.
    original_tok = ROOT / manifest['model_lock']['local_dir'] / 'tokenizer/tokenizer.json'
    if file_digest(original_tok) != manifest['model_lock']['sha256']['tokenizer/tokenizer.json']:
        raise ValueError('Original tokenizer hash mismatch')
    def semantics(path):
        value = json.loads(path.read_text())
        value.pop('truncation', None)
        value.pop('padding', None)
        return value
    if semantics(args.checkpoint / 'tokenizer/tokenizer.json') != semantics(original_tok):
        raise ValueError('Checkpoint tokenizer semantics differ')
    temperature = float(cfg['temperature'][0])
    if not math.isfinite(temperature) or temperature <= 0:
        raise FloatingPointError('Invalid native temperature')
    model = model.float().eval().to(args.device)
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise FloatingPointError('Checkpoint has non-finite parameters')
    args.out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    write_json(args.out / 'status.json', {'state': 'running', 'test_accessed': False})
    metadata = {'schema_version': 1, 'run_id': args.out.name, 'state': 'running',
        'training_performed': False, 'test_accessed': False, 'checkpoint_sha256': actual_weight,
        'manifest_sha256': file_digest(args.manifest), 'source_sha256': source_sha,
        'split_manifest_sha256': file_digest(split_path), 'native_temperature': temperature,
        'rules': ['head', 'head_tail'], 'split_counts': {k: len(v) for k,v in selected.items()},
        'device': args.device, 'dtype': 'float32', 'amp': False, 'tf32': False,
        'parallel_layout': False, 'batch_size': args.batch_size,
        'rule_definition': {'head': 'first 466 original tokens', 'head_tail': 'first 233 + last 233 original tokens if truncated; otherwise identical head input'},
        'source_script_sha256': file_digest(Path(__file__)),
        'versions': {n: version(n) for n in ('torch', 'laya', 'transformers', 'numpy')}}
    try:
        with torch.no_grad():
            for split_name, rows in selected.items():
                head, tail, stats, head_length, budget = prepare_rules(tok, rows, cfg['max_len'], cfg['head_max_len'])
                metadata.update(head_token_length=head_length, review_token_budget=budget)
                changed = [i for i,s in enumerate(stats) if s['truncated']]
                metadata.setdefault('forward_counts', {})[split_name] = {'head': len(rows), 'head_tail': len(changed)}
                write_json(args.out / 'metadata.json', metadata)
                outputs = {}
                for rule, items, indices in [('head', head, list(range(len(rows)))), ('head_tail', tail, changed)]:
                    result = [dict(r, reused_head_prediction=True) for r in outputs['head']] if rule == 'head_tail' else [None]*len(rows)
                    for start in range(0, len(indices), args.batch_size):
                        ii = indices[start:start+args.batch_size]
                        chunk = [items[i] for i in ii]
                        records = upstream.calibration_records(model, tok, chunk, torch.device(args.device), cfg['max_len'], cfg['head_max_len'], batch_size=args.batch_size, parallel=False)
                        if len(records) != len(ii):
                            raise ValueError('Forward result count mismatch')
                        for i, item, record in zip(ii, chunk, records):
                            logits = checked_logits(record)
                            probs = np.exp((logits-logits.max())/temperature)
                            probs /= probs.sum()
                            if not np.isfinite(probs).all():
                                raise FloatingPointError('Non-finite probabilities')
                            prepared = upstream.encode_item(tok, item, cfg['max_len'], cfg['head_max_len'], parallel=False)
                            result[i] = {'sample_id': rows[i]['id'], 'true_label': rows[i]['label'],
                                'input_sha256': input_digest(prepared), 'logits': logits.tolist(),
                                'probabilities': probs.tolist(), 'pred_label': int(logits.argmax()),
                                **stats[i], 'reused_head_prediction': False}
                        print(f'{split_name}/{rule}: {min(start+args.batch_size,len(indices))}/{len(indices)}', flush=True)
                    outputs[rule] = result
                    write_json(args.out / f'{split_name}-{rule}.json', {'schema_version': 1, 'split': split_name, 'rule': rule, 'rows': result, 'test_accessed': False})
        if file_digest(args.checkpoint / 'model.safetensors') != actual_weight:
            raise ValueError('Checkpoint changed during export')
        metadata.update(state='completed', elapsed_seconds=time.monotonic()-began)
        write_json(args.out / 'metadata.json', metadata)
        write_json(args.out / 'status.json', {'state': 'completed', 'test_accessed': False, 'elapsed_seconds': metadata['elapsed_seconds']})
    except Exception as exc:
        write_json(args.out / 'status.json', {'state': 'failed', 'test_accessed': False, 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()
