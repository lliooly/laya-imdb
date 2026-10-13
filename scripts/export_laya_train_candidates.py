"""Offline fixed-weight inference on frozen gradient-training IDs only."""
import argparse
import json
from pathlib import Path
import time

import train_laya
from common import ROOT, file_digest, read_jsonl, write_json
from export_laya_posthoc import checked_logits, input_digest


def select_gradient(manifest, split, rows):
    ids = [r['id'] for r in rows]
    gradient = manifest['gradient_train_ids']
    calibration = manifest['calibration_ids']
    validation = manifest['validation']['ids']
    groups = [set(gradient), set(calibration), set(validation)]
    if ids != split['ids']['train'] or ids != manifest['train']['ids']:
        raise ValueError('Frozen source IDs/order differ')
    if validation != split['ids']['validation']:
        raise ValueError('Frozen validation IDs differ')
    if (len(ids), len(gradient), len(calibration), len(validation)) != (22500, 22100, 400, 2500):
        raise ValueError('Expected original frozen partition sizes')
    if any(len(g) != len(v) for g, v in zip(groups, (gradient, calibration, validation))):
        raise ValueError('Duplicate partition IDs')
    if len(set(ids)) != len(ids) or groups[0] | groups[1] != set(ids):
        raise ValueError('Invalid training partition coverage')
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Data partition overlap')
    if any(not r['id'].startswith('train/') or type(r['label']) is not int or r['label'] not in (0, 1) for r in rows):
        raise ValueError('Expected original training IDs and binary labels')
    by_id = {r['id']: r for r in rows}
    return [by_id[i] for i in gradient]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose an unused export path')
    train_laya.block_network()
    source = ROOT / 'data/processed/train.jsonl'
    split = json.loads((ROOT / 'reports/split_manifest.json').read_text())
    manifest = json.loads(args.manifest.read_text())
    if file_digest(source) != split['jsonl_sha256']['train'] or file_digest(source) != manifest['train']['source_sha256']:
        raise ValueError('Frozen training SHA mismatch')
    if manifest['question'] != train_laya.QUESTION:
        raise ValueError('Unexpected question template')
    rows = select_gradient(manifest, split, list(read_jsonl(source)))
    reference = json.loads((args.manifest.parent / 'cuda-saved-validation.json').read_text())
    weight_sha = file_digest(args.checkpoint / 'model.safetensors')
    if weight_sha != reference['checkpoint_sha256']:
        raise ValueError('Weight SHA mismatch')
    import torch
    import laya.train as upstream
    train_laya.require_device(torch, 'cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tok, cfg = upstream.load_checkpoint(str(args.checkpoint))
    if upstream.uses_parallel_layout(cfg) or (cfg['max_len'], cfg['head_max_len']) != (512, 192):
        raise ValueError('Unexpected model layout/context')
    original = ROOT / manifest['model_lock']['local_dir'] / 'tokenizer/tokenizer.json'
    if file_digest(original) != manifest['model_lock']['sha256']['tokenizer/tokenizer.json']:
        raise ValueError('Original tokenizer SHA mismatch')
    def semantics(p):
        j = json.loads(p.read_text())
        j.pop('padding', None); j.pop('truncation', None)
        return j
    if semantics(original) != semantics(args.checkpoint / 'tokenizer/tokenizer.json'):
        raise ValueError('Tokenizer semantics mismatch')
    model = model.float().eval().to('cuda')
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise FloatingPointError('Nonfinite model parameters')
    began = time.monotonic()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(rows), 8):
            chunk = rows[start:start + 8]
            typed = [{'state': r['text'], 'questions': {'sentiment': train_laya.QUESTION},
                      'expected': {'sentiment': ('negative', 'positive')[r['label']]}} for r in chunk]
            items, skipped = upstream.items_from_rows(tok, typed, 512, 192, label_smoothing=0.0)
            if skipped or len(items) != len(chunk):
                raise ValueError('Every gradient sample must produce one item')
            records = upstream.calibration_records(model, tok, items, torch.device('cuda'),
                512, 192, batch_size=8, parallel=False)
            if len(records) != len(chunk):
                raise ValueError('Forward count mismatch')
            for r, item, record in zip(chunk, items, records):
                logits = checked_logits(record)
                prepared = upstream.encode_item(tok, item, 512, 192, parallel=False)
                predictions.append({'sample_id': r['id'], 'true_label': r['label'],
                    'input_sha256': input_digest(prepared), 'logits': logits.tolist(),
                    'pred_label': int(logits.argmax())})
            if start % 800 == 0:
                print(f'{min(start + 8, len(rows))}/{len(rows)}', flush=True)
    if file_digest(args.checkpoint / 'model.safetensors') != weight_sha:
        raise ValueError('Weights changed during inference')
    write_json(args.out, {'state': 'completed', 'split': 'gradient_train', 'count': len(rows),
        'training_performed': False, 'test_accessed': False, 'validation_body_accessed': False,
        'calibration_body_used_for_inference': False, 'seen_training_samples': True,
        'checkpoint_sha256': weight_sha, 'manifest_sha256': file_digest(args.manifest),
        'train_source_sha256': file_digest(source), 'script_sha256': file_digest(Path(__file__)),
        'dtype': 'float32', 'tf32': False, 'batch_size': 8, 'input_rule': 'original_head',
        'elapsed_seconds': time.monotonic() - began, 'rows': predictions})


if __name__ == '__main__':
    main()
