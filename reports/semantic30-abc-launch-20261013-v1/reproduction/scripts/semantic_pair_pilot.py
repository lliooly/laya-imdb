"""Build auditable, review-gated semantic triplets from frozen gradient sources.

Raw text stays in reports/**/data (ignored by Git). No model training or test reads.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from common import ROOT, file_digest, read_jsonl, write_json

VARIANTS = ('original', 'ordinary', 'semantic')
CATEGORIES = ('irony', 'complex_relation', 'evaluation_target')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def source_partition(rows, split, manifest):
    ids = [r['id'] for r in rows]
    gradient, calibration = manifest['gradient_train_ids'], manifest['calibration_ids']
    validation = split['ids']['validation']
    if ids != split['ids']['train'] or ids != manifest['train']['ids']:
        raise ValueError('Frozen source IDs/order mismatch')
    groups = [set(x) for x in (ids, gradient, calibration, validation)]
    if any(len(s) != len(v) for s, v in zip(groups, (ids, gradient, calibration, validation))):
        raise ValueError('Duplicate partition IDs')
    if groups[1] & groups[2] or groups[1] | groups[2] != groups[0] or groups[0] & groups[3]:
        raise ValueError('Invalid partition isolation')
    if validation != manifest['validation']['ids']:
        raise ValueError('Validation ID manifest mismatch')
    if any(type(r['label']) is not int or r['label'] not in (0, 1) for r in rows):
        raise ValueError('Invalid source labels')
    return {r['id']: r for r in rows}, gradient


def load_source(manifest_path):
    split = json.loads((ROOT / 'reports/split_manifest.json').read_text())
    manifest = json.loads(manifest_path.read_text())
    path = ROOT / 'data/processed/train.jsonl'
    sha = file_digest(path)
    if sha != split['jsonl_sha256']['train'] or sha != manifest['train']['source_sha256']:
        raise ValueError('Frozen training source hash mismatch')
    rows = list(read_jsonl(path))
    by_id, gradient = source_partition(rows, split, manifest)
    if (len(rows), len(gradient), len(manifest['calibration_ids']), len(split['ids']['validation'])) != (22500, 22100, 400, 2500):
        raise ValueError('Unexpected frozen partition sizes')
    return by_id, gradient, manifest, sha


def edit_text(text, edits, variant):
    """Apply nonoverlapping exact spans against the original, never chained replacements."""
    spans = []
    for edit in edits:
        before, after = edit['before'], edit[variant]
        if not before or text.count(before) != 1:
            raise ValueError('Edit must match exactly once')
        if not after.strip() or before == after:
            raise ValueError('Each variant must change its span')
        start = text.index(before)
        spans.append((start, start + len(before), after))
    spans.sort()
    if not spans or any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError('Empty or overlapping edits')
    result, cursor, changed = '', 0, []
    for start, end, after in spans:
        result += text[cursor:start]
        changed.append((len(result), len(result) + len(after)))
        result += after
        cursor = end
    return result + text[cursor:], changed


def spans_visible(text, spans, tokenizer, budget=466):
    encoding = tokenizer.encode(text, add_special_tokens=False)
    visible = encoding.offsets[:budget]
    return bool(visible) and all(start >= visible[0][0] and end <= visible[-1][1] for start, end in spans)


def build_triplets(recipes, by_id, gradient, tokenizer):
    output, used, hashes = [], set(), set()
    for recipe in recipes:
        parent = recipe['parent_id']
        if parent not in set(gradient) or parent in used:
            raise ValueError('Source must be unique and in gradient partition')
        used.add(parent)
        source = by_id[parent]
        text = source['text']
        if recipe['source_sha256'] != digest(text) or recipe['label'] != source['label']:
            raise ValueError('Source hash or label mismatch')
        if recipe['category'] not in CATEGORIES:
            raise ValueError('Unknown semantic category')
        evidence = recipe['evidence']
        if not evidence or text.count(evidence) != 1:
            raise ValueError('Evidence must match exactly once')
        evidence_span = (text.index(evidence), text.index(evidence) + len(evidence))
        original_spans = [(text.index(e['before']), text.index(e['before']) + len(e['before']))
                          for e in recipe['edits'] if e['before'] in text]
        variants = {'original': text}
        if not spans_visible(text, [evidence_span] + original_spans, tokenizer):
            raise ValueError('Original evidence/edit outside visible prefix: ' + parent)
        for name in VARIANTS[1:]:
            altered, spans = edit_text(text, recipe['edits'], name)
            if not spans_visible(altered, spans, tokenizer):
                raise ValueError('Edited evidence outside visible prefix: ' + parent)
            variants[name] = altered
        if len(set(variants.values())) != 3:
            raise ValueError('Triplet variants must differ')
        texts = {name: {'text': value, 'sha256': digest(value),
                       'tokens': len(tokenizer.encode(value, add_special_tokens=False).ids)}
                 for name, value in variants.items()}
        if hashes & {v['sha256'] for v in texts.values()}:
            raise ValueError('Exact duplicate within pilot')
        hashes.update(v['sha256'] for v in texts.values())
        output.append({'parent_id': parent, 'label': source['label'], 'category': recipe['category'],
                       'evidence': evidence, 'rationale_zh': recipe['rationale_zh'],
                       'review_status': 'ai_draft_pending_human', 'variants': texts})
    return output


def require_human_review(triplets, review):
    if review.get('reviewer_kind') != 'human' or not review.get('reviewer', '').strip():
        raise ValueError('Independent human review is required for training export')
    decisions = review.get('decisions', [])
    if len(decisions) != len(triplets) or len({r['parent_id'] for r in decisions}) != len(decisions):
        raise ValueError('Review must cover each parent exactly once')
    by_id = {r['parent_id']: r for r in decisions}
    for item in triplets:
        row = by_id.get(item['parent_id'], {})
        if row.get('decision') != 'accept' or row.get('label') != item['label']:
            raise ValueError('All triplets need accepted, label-matched human decisions')
        expected = {k: v['sha256'] for k, v in item['variants'].items()}
        if row.get('variant_sha256') != expected:
            raise ValueError('Stale review: variant hashes differ')
        if not all(row.get(k) is True for k in ('label_preserved', 'natural_language', 'evidence_sufficient')):
            raise ValueError('Human semantic quality checks incomplete')


def make_schedule(gradient, labels, parents, seed=42, fraction=.05):
    """Two epochs: full original first epoch; fixed same-label replacements in second."""
    if len(gradient) != len(set(gradient)) or set(labels) != set(gradient):
        raise ValueError('Gradient IDs or labels invalid')
    if any(type(v) is not int or v not in (0, 1) for v in labels.values()):
        raise ValueError('Invalid binary labels')
    if not parents or len(parents) != len(set(parents)) or not set(parents) <= set(gradient):
        raise ValueError('Invalid unique parent IDs')
    if not 0 < fraction < .5:
        raise ValueError('Replacement fraction must be between 0 and .5')
    count = round(2 * len(gradient) * fraction)
    if count < 1:
        raise ValueError('Replacement budget is empty')
    candidates = {label: sorted(p for p in parents if labels[p] == label) for label in (0, 1)}
    if any(not v for v in candidates.values()):
        raise ValueError('Both labels must be represented among parents')
    displaced = [p for p in gradient if p not in set(parents)]
    random.Random(seed + 100003).shuffle(displaced)
    if count > len(displaced):
        raise ValueError('Insufficient unique replaceable positions')
    counter = Counter()
    replacements = {}
    for old in displaced[:count]:
        label = labels[old]
        pool = candidates[label]
        replacements[old] = pool[counter[label] % len(pool)]
        counter[label] += 1
    result = []
    for epoch in range(2):
        order = list(gradient)
        random.Random(seed + epoch).shuffle(order)
        result.append([{'original_id': old,
                        'parent_id': replacements.get(old, old) if epoch == 1 else old,
                        'augmented': epoch == 1 and old in replacements,
                        'label': labels[old]} for old in order])
    return result


def materialize(schedule, by_id, triplets, arm):
    if arm not in ('A', 'B', 'C'):
        raise ValueError('Arm must be A, B or C')
    variant = {'A': 'original', 'B': 'ordinary', 'C': 'semantic'}[arm]
    pairs = {row['parent_id']: row for row in triplets}
    epochs = []
    for epoch in schedule:
        records = []
        for slot in epoch:
            parent = slot['parent_id']
            original = by_id[parent]
            if original['label'] != slot['label']:
                raise ValueError('Schedule changed source label')
            value = pairs[parent]['variants'][variant]['text'] if slot['augmented'] else original['text']
            records.append({**slot, 'variant': variant if slot['augmented'] else 'original',
                            'text': value, 'text_sha256': digest(value)})
        epochs.append(records)
    return epochs


def review_markdown(triplets):
    lines = ['# 30 组语义改写试制：人工审核材料', '',
             'AI 起草，尚未获得人工标签审核。以下材料不是准确率证据。', '',
             '请先阅读各版本自行判断，再填写 review-decisions.json。任何歧义可选择 reject。', '']
    for i, row in enumerate(triplets, 1):
        lines.extend([f'## 样本 {i:02d} · {row["category"]}', '',
                      '先独立判断三段文字的总体态度是否一致。', ''])
        for name, title in zip(VARIANTS, ('原文', '普通改写 B', '针对性改写 C')):
            lines.extend([f'### {title}', '', row['variants'][name]['text'], ''])
        lines.extend(['<details><summary>起草依据与原标签（判断后再展开）</summary>', '',
                      f'来源：{row["parent_id"]}；原标签：{row["label"]}；{row["rationale_zh"]}', '',
                      f'原文证据：{row["evidence"]}', '', '</details>', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'reports/laya-l512-s42-ce/manifest.json')
    parser.add_argument('--review', type=Path, help='Required only for full training data export')
    parser.add_argument('--export-training', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use an unused output directory')
    by_id, gradient, manifest, source_sha = load_source(args.manifest)
    from tokenizers import Tokenizer
    lock = json.loads((ROOT / 'configs/laya_model_lock.json').read_text())
    tok_path = ROOT / lock['local_dir'] / 'tokenizer/tokenizer.json'
    if file_digest(tok_path) != lock['sha256']['tokenizer/tokenizer.json']:
        raise ValueError('Tokenizer hash mismatch')
    tok = Tokenizer.from_file(str(tok_path))
    tok.no_padding(); tok.no_truncation()
    recipes = json.loads(args.recipes.read_text())['recipes']
    triplets = build_triplets(recipes, by_id, gradient, tok)
    if args.export_training:
        if not args.review:
            raise ValueError('Human review file required for training export')
        require_human_review(triplets, json.loads(args.review.read_text()))
    schedule = make_schedule(gradient, {p: by_id[p]['label'] for p in gradient},
                             [r['parent_id'] for r in triplets])
    data = args.out / 'data'
    data.mkdir(parents=True)
    pack = {'state': ('human_review_accepted' if args.export_training else 'ai_draft_pending_human'), 'source_sha256': source_sha,
            'manifest_sha256': file_digest(args.manifest), 'tokenizer_sha256': file_digest(tok_path),
            'question': manifest['question'], 'rows': triplets}
    write_json(data / 'triplets.json', pack)
    (data / 'REVIEW.md').write_text(review_markdown(triplets), encoding='utf-8')
    write_json(data / 'review-decisions.json', {'reviewer_kind': None, 'reviewer': '', 'decisions': [
        {'parent_id': r['parent_id'], 'decision': 'pending', 'label': None,
         'variant_sha256': {k: v['sha256'] for k, v in r['variants'].items()},
         'label_preserved': None, 'natural_language': None, 'evidence_sufficient': None,
         'notes': ''} for r in triplets]})
    write_json(data / 'schedule-preview.json', schedule)
    if args.export_training:
        for arm in ('A', 'B', 'C'):
            for epoch, records in enumerate(materialize(schedule, by_id, triplets, arm), 1):
                with (data / f'{arm}-epoch-{epoch}.jsonl').open('w') as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    summary = {'state': ('human_review_accepted_training_data_exported' if args.export_training
                         else 'mechanically_validated_pending_human_review'), 'training_performed': False,
               'training_data_exported': args.export_training, 'test_accessed': False,
               'validation_body_accessed': False, 'calibration_used_for_augmentation': False,
               'count': len(triplets), 'versions': 3 * len(triplets),
               'category_counts': dict(Counter(r['category'] for r in triplets)),
               'label_counts': dict(Counter(r['label'] for r in triplets)),
               'source_sha256': source_sha, 'manifest_sha256': file_digest(args.manifest),
               'recipes_sha256': file_digest(args.recipes), 'pack_sha256': file_digest(data / 'triplets.json'),
               'script_sha256': file_digest(Path(__file__)),
               'accepted_review_sha256': file_digest(args.review) if args.export_training else None,
               'total_exposures_per_arm': sum(map(len, schedule)),
               'augmented_slots_per_arm': sum(s['augmented'] for e in schedule for s in e),
               'epoch_label_counts': [dict(Counter(s['label'] for s in e)) for e in schedule],
               'lengths': [{'parent_id': r['parent_id'],
                           **{k: v['tokens'] for k, v in r['variants'].items()}} for r in triplets],
               'limitations': [('AI authored; human review accepted, not a guarantee of label correctness'
                                if args.export_training else 'AI authored; human semantic review pending'),
                               'Selected seen-training candidates; no independent accuracy estimate',
                               'All modified spans visible; this does not prove all decisive evidence is visible',
                               'No syntactic parser or gold long-sentence annotations']}
    write_json(args.out / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('state', 'count', 'category_counts', 'label_counts',
                                             'total_exposures_per_arm', 'augmented_slots_per_arm')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
