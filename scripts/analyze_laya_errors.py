"""Audit saved Laya-CE validation errors on CPU, without loading model weights."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from common import ROOT, file_digest, read_jsonl, write_json
from comparison_metrics import metrics, sigmoid
import train_laya


def checked_rows(path, expected):
    if file_digest(path) != expected:
        raise ValueError(f"Frozen source hash mismatch: {path}")
    return list(read_jsonl(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/laya-ce-error-analysis-v1')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Analysis already exists; choose a new --out directory')
    train_laya.block_network()
    from transformers import AutoTokenizer
    import laya.train as upstream
    from laya.common import build_head, build_sequence

    checkpoint = ROOT / 'checkpoints/laya-l512-s42-ce'
    source = ROOT / 'reports/model-comparison/laya_ce_full'
    split = json.loads((ROOT / 'reports/split_manifest.json').read_text())
    export = json.loads((source / 'export.json').read_text())
    rows = checked_rows(ROOT / 'data/processed/validation.jsonl', split['jsonl_sha256']['validation'])
    raw = checked_rows(source / 'raw.jsonl', export['predictions_sha256'])
    if len(rows) != 2500 or len(raw) != len(rows):
        raise ValueError('Expected full 2500-row validation')
    if [r['id'] for r in rows] != split['ids']['validation']:
        raise ValueError('Frozen validation IDs differ')
    if [(r['id'], r['label']) for r in rows] != [(r['sample_id'], r['true_label']) for r in raw]:
        raise ValueError('Prediction IDs/labels do not align')
    cfg = json.loads((checkpoint / 'rl_agent_config.json').read_text())
    if file_digest(checkpoint / 'model.safetensors') != export['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from prediction source')
    original = json.loads((ROOT / 'reports/laya-l512-s42-ce/manifest.json').read_text())
    tokenizer_file = checkpoint / 'tokenizer/tokenizer.json'
    original_tokenizer = ROOT / original['model_lock']['local_dir'] / 'tokenizer/tokenizer.json'
    if file_digest(original_tokenizer) != original['model_lock']['sha256']['tokenizer/tokenizer.json']:
        raise ValueError('Original tokenizer differs from training source')
    # save_pretrained can persist the last temporary option truncation (48 tokens).
    # Compare tokenizer semantics after removing runtime padding/truncation settings.
    def tokenizer_definition(path):
        value = json.loads(path.read_text())
        value.pop('truncation', None)
        value.pop('padding', None)
        return value
    if tokenizer_definition(tokenizer_file) != tokenizer_definition(original_tokenizer):
        raise ValueError('Saved tokenizer semantics differ from original model')
    tok = AutoTokenizer.from_pretrained(str(checkpoint / 'tokenizer'), local_files_only=True)
    question = upstream.to_internal('sentiment', train_laya.QUESTION)
    head, _, _ = build_head(tok, question, cfg['head_max_len'])
    budget = cfg['max_len'] - len(head) - 1
    if (len(head), budget) != (export['head_token_length'], export['review_token_budget']):
        raise ValueError('Question template budget differs from exported model')
    tok.backend_tokenizer.no_truncation()
    tok.backend_tokenizer.no_padding()
    encodings = tok.backend_tokenizer.encode_batch(
        [r['text'].replace(tok.mask_token, ' ') for r in rows], add_special_tokens=False)
    labels = np.asarray([r['label'] for r in rows])
    z = np.asarray([r['logits'] for r in raw], dtype=np.float64)
    margin = z[:, 1] - z[:, 0]
    pred = (margin > 0).astype(int)
    correct = pred == labels
    temperature = float(cfg['temperature'][0])
    confidence = np.maximum(sigmoid(margin / temperature), sigmoid(-margin / temperature))
    lengths = np.asarray([len(e.ids) for e in encodings])
    truncated = lengths > budget
    saved = list(read_jsonl(ROOT / 'reports/laya-l512-s42-ce/validation-saved.jsonl'))
    if [(r['id'], r['label'], r['prediction']) for r in saved] != [
        (r['id'], r['label'], int(p)) for r, p in zip(rows, pred)]:
        raise ValueError('Saved validation and exported predictions differ')
    baseline = np.asarray([r['baseline_prediction'] for r in saved])
    # Verify full-token length reconstruction uses the same sequence as training,
    # which first caps state tokens at max_len before applying the question budget.
    for row, encoding in zip(rows, encodings):
        text = row['text'].replace(tok.mask_token, ' ')
        full = build_sequence(tok, text, question, cfg['max_len'], cfg['head_max_len'])
        capped = build_sequence(tok, text, question, cfg['max_len'], cfg['head_max_len'],
                                state_ids=encoding.ids[:cfg['max_len']])
        if full != capped or len(full[0]) != len(head) + min(len(encoding.ids), budget) + 1:
            raise ValueError('Reconstructed input differs from upstream training input')

    def group(keep):
        m = metrics(labels[keep], margin[keep], temperature)
        return {k: m[k] for k in ('count', 'accuracy', 'correct', 'macro_f1', 'nll',
                'brier_multiclass', 'ece_15_equal_width_top_label', 'error_mean_confidence',
                'errors_with_confidence_ge_095')} | {'errors': int((~correct & keep).sum())}

    groups = {'all': group(np.ones(len(rows), dtype=bool)),
              'untruncated': group(~truncated), 'truncated': group(truncated)}
    for low, high in ((0, 128), (128, 256), (256, budget), (budget, 1024), (1024, None)):
        keep = (lengths > low) & ((lengths <= high) if high else True)
        if keep.any():
            groups[f'review_tokens_{low + 1}_{high or "plus"}'] = group(keep)
    errors = []
    for i in np.flatnonzero(~correct):
        r, e = rows[i], encodings[i]
        cut = e.offsets[budget][0] if len(e.ids) > budget else len(r['text'].replace(tok.mask_token, ' '))
        text = r['text'].replace(tok.mask_token, ' ')
        errors.append({'sample_id': r['id'], 'true_label': int(labels[i]), 'pred_label': int(pred[i]),
                       'confidence_native_calibrated': float(confidence[i]),
                       'margin_uncalibrated': float(margin[i]), 'review_tokens': int(lengths[i]),
                       'review_budget': budget, 'truncated': bool(truncated[i]),
                       'dropped_tokens': max(0, int(lengths[i]) - budget),
                       'baseline_correct': bool(baseline[i] == labels[i]),
                       'text_sha256': r['text_sha256'], 'text': r['text'],
                       'omitted_text': text[cut:] if truncated[i] else '',
                       'annotation_status': 'unreviewed', 'semantic_tags': [], 'review_note': ''})
    errors.sort(key=lambda r: (-r['confidence_native_calibrated'], r['sample_id']))
    if len(errors) != 106 or groups['all']['correct'] != 2394:
        raise ValueError('Historical accuracy recount failed')
    truncated_errors = int((~correct & truncated).sum())
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'development validation; descriptive analysis, not causal attribution',
               'test_accessed': False, 'training_performed': False, 'weights_loaded': False,
               'template_tokens': len(head), 'review_budget': budget, 'temperature': temperature,
               'groups': groups,
               'truncated_fraction': float(truncated.mean()),
               'errors_truncated_fraction': truncated_errors / len(errors),
               'baseline_overlap': {'both_wrong': int(((baseline != labels) & ~correct).sum()),
                                    'baseline_correct_laya_wrong': int(((baseline == labels) & ~correct).sum())},
               'target': {'accuracy': .98, 'allowed_errors': 50, 'net_errors_to_fix': len(errors) - 50,
                          'idealized_truncation_only_accuracy': (2394 + truncated_errors) / 2500,
                          'idealized_bound_note': 'Assumes correcting every currently truncated error, preserving every other prediction; not a predicted gain or an upper bound on all possible training effects.'},
               'source_sha256': {'validation': split['jsonl_sha256']['validation'],
                                 'predictions': export['predictions_sha256'],
                                 'checkpoint': export['checkpoint_sha256'],
                                 'tokenizer': file_digest(tokenizer_file),
                                 'script': file_digest(Path(__file__)),
                                 'upstream_train': file_digest(Path(upstream.__file__))}}
    args.out.mkdir(parents=True)
    data = args.out / 'data'
    data.mkdir()
    with (data / 'errors.jsonl').open('w', encoding='utf-8') as f:
        for row in errors:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    fields = [k for k in errors[0] if k not in ('text', 'omitted_text', 'semantic_tags', 'review_note')]
    with (args.out / 'error_index.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in errors)
    write_json(args.out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
