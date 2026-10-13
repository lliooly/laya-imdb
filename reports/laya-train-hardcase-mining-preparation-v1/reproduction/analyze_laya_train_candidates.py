"""Join training-only model diagnostics with frozen keyword candidates; prepare review."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from common import ROOT, file_digest, read_jsonl, write_json
from export_laya_train_candidates import select_gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose new analysis directory')
    prep = ROOT / 'reports/laya-train-hardcase-mining-preparation-v1/review-protocol.json'
    protocol = json.loads(prep.read_text())
    manifest_path = ROOT / 'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = ROOT / 'data/processed/train.jsonl'
    split = json.loads((ROOT / 'reports/split_manifest.json').read_text())
    if file_digest(source) != manifest['train']['source_sha256'] or file_digest(source) != split['jsonl_sha256']['train']:
        raise ValueError('Changed frozen training source')
    rows = select_gradient(manifest, split, list(read_jsonl(source)))
    by_id = {r['id']: r for r in rows}
    predictions = {}
    hashes = {str(p.relative_to(ROOT)): file_digest(p) for p in (prep, manifest_path, source)}
    for model in ('baseline', 'combined'):
        path = args.exports / f'{model}-gradient-predictions.json'
        doc = json.loads(path.read_text())
        model_folder = ROOT / f'reports/laya-rtx4090-ce-{model}-s43-epochs2-fp32-v1'
        model_manifest = json.loads((model_folder / 'manifest.json').read_text())
        for key in ('gradient_train_ids', 'calibration_ids', 'validation'):
            if model_manifest[key] != manifest[key]:
                raise ValueError('Paired training data differ')
        if doc['manifest_sha256'] != file_digest(model_folder / 'manifest.json'):
            raise ValueError('Export manifest SHA mismatch')
        if doc['checkpoint_sha256'] != json.loads((model_folder / 'cuda-saved-validation.json').read_text())['checkpoint_sha256']:
            raise ValueError('Export weight SHA mismatch')
        if doc['state'] != 'completed' or doc['split'] != 'gradient_train' or doc['train_source_sha256'] != file_digest(source):
            raise ValueError('Invalid training export')
        if doc['training_performed'] or doc['test_accessed'] or doc['validation_body_accessed']:
            raise ValueError('Invalid export data scope')
        pp = doc['rows']
        if [(p['sample_id'], p['true_label']) for p in pp] != [(r['id'], r['label']) for r in rows]:
            raise ValueError('Expected full ordered original gradient IDs')
        logits = np.asarray([p['logits'] for p in pp], dtype=float)
        if logits.shape != (22100, 2) or not np.isfinite(logits).all():
            raise ValueError('Invalid logits')
        if any(p['pred_label'] != int(np.argmax(p['logits'])) for p in pp):
            raise ValueError('Invalid predictions')
        predictions[model] = pp
        hashes[str(path.relative_to(ROOT))] = file_digest(path)
    if [r['input_sha256'] for r in predictions['baseline']] != [r['input_sha256'] for r in predictions['combined']]:
        raise ValueError('Paired model input hashes differ')
    candidate_path = ROOT / 'reports/laya-train-hardcase-audit-v1/candidates.csv'
    with candidate_path.open() as f:
        candidates = list(csv.DictReader(f))
    if len(candidates) != len({c['sample_id'] for c in candidates}):
        raise ValueError('Duplicate rule candidates')
    categories = {}
    for c in candidates:
        r = by_id[c['sample_id']]
        if int(c['label']) != r['label'] or hashlib.sha256(r['text'].encode()).hexdigest() != c['text_sha256']:
            raise ValueError('Candidate label/text SHA mismatch')
        categories[c['sample_id']] = c['categories'].split('|')
    hashes[str(candidate_path.relative_to(ROOT))] = file_digest(candidate_path)
    ranking = []
    for i, r in enumerate(rows):
        cs = categories.get(r['id'], [])
        category = next((c for c in protocol['exclusive_category_priority'] if c in cs), 'unflagged')
        scores = {}
        for model in ('baseline', 'combined'):
            p = predictions[model][i]
            margin = p['logits'][1] - p['logits'][0]
            loss = float(np.logaddexp(0., -(2 * r['label'] - 1) * margin))
            probability = float(np.exp(-np.logaddexp(0., -margin)))
            entropy = float(np.logaddexp(0., margin) - margin * probability)
            scores[model] = {'pred_label': p['pred_label'], 'raw_nll': loss, 'entropy': entropy}
        both_wrong = all(v['pred_label'] != r['label'] for v in scores.values())
        disagreement = scores['baseline']['pred_label'] != scores['combined']['pred_label']
        ranking.append({'sample_id': r['id'], 'label': r['label'], 'exclusive_category': category,
            'rule_categories': cs, 'both_wrong': both_wrong, 'disagreement': disagreement,
            'mean_raw_nll': sum(v['raw_nll'] for v in scores.values()) / 2,
            'mean_entropy': sum(v['entropy'] for v in scores.values()) / 2,
            'model_scores': scores, 'text_sha256': hashlib.sha256(r['text'].encode()).hexdigest()})
    ranking.sort(key=lambda x: (-x['both_wrong'], -x['disagreement'], -x['mean_raw_nll'], x['sample_id']))
    selected = []
    rng = random.Random(protocol['seed'])
    cells = []
    for label in protocol['labels']:
        for category in protocol['exclusive_category_priority']:
            cell = [r for r in ranking if r['label'] == label and r['exclusive_category'] == category]
            hard = cell[:protocol['per_label_category']['model_priority']]
            remaining = sorted(cell[len(hard):], key=lambda x: x['sample_id'])
            randoms = rng.sample(remaining, min(len(remaining), protocol['per_label_category']['random_remaining']))
            cells.append({'label': label, 'category': category, 'population': len(cell), 'model_priority': len(hard), 'random_remaining': len(randoms)})
            for selection, items in (('model_priority', hard), ('random_remaining', randoms)):
                for r in items:
                    selected.append(dict(r, selection=selection, text=by_id[r['sample_id']]['text']))
    from transformers import AutoTokenizer
    tokenizer_dir = ROOT / manifest['model_lock']['local_dir'] / 'tokenizer'
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    tok.backend_tokenizer.no_padding(); tok.backend_tokenizer.no_truncation()
    for r in selected:
        sanitized = r['text'].replace(tok.mask_token, ' ')
        encoded = tok.backend_tokenizer.encode(sanitized, add_special_tokens=False)
        r['review_tokens'] = len(encoded.ids)
        r['truncated'] = len(encoded.ids) > 466
        r['visible_head_text'] = sanitized[:encoded.offsets[465][1]] if r['truncated'] else sanitized
        r['review_token_budget'] = 466
    if len(selected) > protocol['review_max_count'] or len({r['sample_id'] for r in selected}) != len(selected):
        raise ValueError('Invalid review selection')
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'ranking.json', {'rows': ranking, 'scope': 'seen_gradient_training_only'})
    write_json(args.out / 'review-samples.json', {'protocol': protocol, 'rows': selected})
    summary = {'scope': 'training_candidate_ranking_not_out_of_fold', 'count': len(rows),
        'candidate_count': len(candidates), 'both_models_wrong': sum(r['both_wrong'] for r in ranking),
        'models_disagree': sum(r['disagreement'] for r in ranking),
        'model_errors': {m: sum(r['model_scores'][m]['pred_label'] != r['label'] for r in ranking) for m in predictions},
        'review_count': len(selected), 'review_cells': cells, 'source_sha256': hashes,
        'analysis_script_sha256': file_digest(Path(__file__)),
        'training_performed': False, 'test_accessed': False, 'validation_body_accessed': False,
        'weights_assigned': False, 'review_status': 'pending'}
    write_json(args.out / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'review_cells')}, indent=2))


if __name__ == '__main__': main()
