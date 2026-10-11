"""Compare the completed CUDA reference to frozen historical CE predictions."""
import argparse
import json
from pathlib import Path

import train_laya
from common import ROOT, file_digest, read_jsonl, write_json
from comparison_metrics import metrics, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/laya-cuda-reference-ce-l512-s42-fp32-v1')
    args = parser.parse_args()
    out = args.report / 'comparison_summary.json'
    if out.exists():
        raise FileExistsError('Summary already exists')
    export_path = args.report / 'cuda-saved-validation.json'
    exported = json.loads(export_path.read_text())
    if not exported['passed'] or exported['state'] != 'completed':
        raise ValueError('Saved CUDA evaluation must complete first')
    old_dir = ROOT / 'reports/model-comparison/laya_ce_full'
    old_meta = json.loads((old_dir / 'export.json').read_text())
    if file_digest(old_dir / 'raw.jsonl') != old_meta['predictions_sha256']:
        raise ValueError('Historical predictions hash mismatch')
    old = list(read_jsonl(old_dir / 'raw.jsonl'))
    new = exported['validation_predictions']
    split = json.loads((ROOT / 'reports/split_manifest.json').read_text())
    source = ROOT / 'data/processed/validation.jsonl'
    if file_digest(source) != split['jsonl_sha256']['validation'] or exported['validation_sha256'] != split['jsonl_sha256']['validation']:
        raise ValueError('Frozen validation hash mismatch')
    rows = list(read_jsonl(source))
    ids = [r['id'] for r in rows]
    labels = [r['label'] for r in rows]
    if len(rows) != 2500 or [p['sample_id'] for p in new] != ids or [p['sample_id'] for p in old] != ids:
        raise ValueError('Predictions must use the frozen full validation order')
    if [p['true_label'] for p in new] != labels or [p['true_label'] for p in old] != labels:
        raise ValueError('Prediction labels differ from frozen source')
    a = [p['logits'][1] - p['logits'][0] for p in old]
    b = [p['logits'][1] - p['logits'][0] for p in new]
    old_metrics = metrics(labels, a, old_meta['native_temperature'])
    new_metrics = metrics(labels, b, exported['temperature'])
    train_laya.block_network()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / '.cache/laya/english/tokenizer'), local_files_only=True)
    tok.backend_tokenizer.no_truncation()
    tok.backend_tokenizer.no_padding()
    lengths = [len(tok.encode(r['text'].replace(tok.mask_token, ' '), add_special_tokens=False)) for r in rows]
    budget = old_meta['review_token_budget']
    strata = {}
    for name, keep in [('not_truncated', [i for i, n in enumerate(lengths) if n <= budget]),
                       ('truncated', [i for i, n in enumerate(lengths) if n > budget])]:
        strata[name] = {'count': len(keep),
            'historical_errors': sum((a[i] > 0) != bool(labels[i]) for i in keep),
            'cuda_reference_errors': sum((b[i] > 0) != bool(labels[i]) for i in keep)}
    old_train = json.loads((ROOT / 'reports/laya-l512-s42-ce/manifest.json').read_text())
    new_train = json.loads((args.report / 'manifest.json').read_text())
    matching = {key: old_train[key] == new_train[key] for key in ('gradient_train_ids', 'calibration_ids')}
    if not all(matching.values()):
        raise ValueError('Training/calibration IDs differ; comparison protocol needs review')
    summary = {'test_accessed': False, 'validation_count': 2500,
        'historical_mps_native_metrics': old_metrics, 'cuda_saved_native_metrics': new_metrics,
        'accuracy_difference_percentage_points': (new_metrics['accuracy'] - old_metrics['accuracy']) * 100,
        'paired': paired(labels, a, b), 'length_strata': strata, 'review_token_budget': budget,
        'target_98_allowed_errors': 50, 'remaining_net_corrections_to_98': max(0, 2500 - new_metrics['correct'] - 50),
        'train_calibration_ids_match_historical': matching,
        'scope': 'Historical MPS vs CUDA reference, with platform and micro-batch regrouping changed; not an isolated method-gain experiment. Single seed.',
        'input_sha256': {'cuda_export': file_digest(export_path), 'historical_raw': file_digest(old_dir / 'raw.jsonl'),
                          'validation': file_digest(source)}, 'source_sha256': file_digest(Path(__file__))}
    write_json(out, summary)
    print(json.dumps({k:v for k,v in summary.items() if 'metrics' not in k}, indent=2))


if __name__ == '__main__':
    main()
