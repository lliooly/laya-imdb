"""Independently verify saved MLX inputs and full A/B/C data budgets on CPU."""
import argparse
from collections import Counter
import json
from pathlib import Path

from common import ROOT, file_digest, write_json
from semantic_pair_pilot import VARIANTS, digest, load_source, make_schedule, materialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'reports/laya-l512-s42-ce/manifest.json')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new verification output')
    pack = json.loads(args.pack.read_text())
    raw_predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    predictions = {r['parent_id']: r for r in raw_predictions}
    if len(predictions) != len(raw_predictions) or set(predictions) != {r['parent_id'] for r in pack['rows']}:
        raise ValueError('Probe IDs mismatch or duplicate')
    source, gradient, manifest, source_sha = load_source(args.manifest)
    if source_sha != pack['source_sha256'] or file_digest(args.manifest) != pack['manifest_sha256']:
        raise ValueError('Source provenance mismatch')
    if pack['question'] != manifest['question']:
        raise ValueError('Question mismatch')
    from transformers import AutoTokenizer
    import laya.train as upstream
    tok = AutoTokenizer.from_pretrained(str(ROOT / '.cache/laya/english/tokenizer'), local_files_only=True)
    verified = 0
    for row in pack['rows']:
        parent = row['parent_id']
        if parent not in set(gradient) or source[parent]['text'] != row['variants']['original']['text']:
            raise ValueError('Source text or membership mismatch')
        if source[parent]['label'] != row['label'] or predictions[parent]['label'] != row['label']:
            raise ValueError('Source label mismatch')
        for name in VARIANTS:
            value = row['variants'][name]
            if digest(value['text']) != value['sha256']:
                raise ValueError('Variant text hash mismatch')
            typed = {'state': value['text'], 'questions': {'sentiment': pack['question']},
                     'expected': {'sentiment': ('negative', 'positive')[row['label']]}}
            items, skipped = upstream.items_from_rows(tok, [typed], 512, 192, label_smoothing=0)
            if skipped or len(items) != 1:
                raise ValueError('Unexpected upstream item count')
            encoded = upstream.encode_item(tok, items[0], 512, 192, parallel=False)
            sha = digest(json.dumps({k: encoded[k] for k in ('ids', 'markers', 'qtype')}, sort_keys=True))
            saved = predictions[parent]['predictions'][name]
            if sha != saved['input_sha256'] or saved['text_sha256'] != value['sha256']:
                raise ValueError('MLX/PyTorch actual input differs')
            verified += 1
    schedule = make_schedule(gradient, {i: source[i]['label'] for i in gradient},
                             [r['parent_id'] for r in pack['rows']])
    arms = {arm: materialize(schedule, source, pack['rows'], arm) for arm in ('A', 'B', 'C')}
    unchanged, changed = 0, 0
    classes = Counter(source[i]['label'] for i in gradient)
    for epoch in range(2):
        for arm in arms.values():
            if len(arm[epoch]) != len(gradient) or Counter(r['label'] for r in arm[epoch]) != classes:
                raise ValueError('Exposure or class budget mismatch')
        for a, b, c in zip(*(arms[arm][epoch] for arm in ('A', 'B', 'C'))):
            if not a['parent_id'] == b['parent_id'] == c['parent_id'] or not a['label'] == b['label'] == c['label']:
                raise ValueError('Arm alignment failed')
            if a['augmented']:
                if len({r['text_sha256'] for r in (a, b, c)}) != 3:
                    raise ValueError('Augmented versions do not differ')
                changed += 1
            else:
                if a != b or b != c:
                    raise ValueError('Nonaugmented versions differ')
                unchanged += 1
    original_hashes = {digest(r['text']) for r in source.values()}
    if any(r['variants'][name]['sha256'] in original_hashes for r in pack['rows'] for name in VARIANTS[1:]):
        raise ValueError('New variant exactly duplicates original training text')
    lengths = [r['variants']['semantic']['tokens'] - r['variants']['ordinary']['tokens'] for r in pack['rows']]
    write_json(args.out, {'state': 'verified', 'verified_prepared_inputs': verified,
        'mlx_pytorch_input_hashes_identical': True, 'training_performed': False,
        'test_accessed': False, 'validation_body_accessed': False,
        'nonaugmented_positions_identical': unchanged, 'augmented_positions_label_aligned': changed,
        'each_arm_exposures': 2 * len(gradient),
        'original_coverage_first_epoch': len({r['parent_id'] for r in schedule[0]}),
        'new_variant_exact_collision_with_original_training_texts': False,
        'near_duplicate_audit_completed': False,
        'semantic_minus_ordinary_tokens': {'min': min(lengths), 'max': max(lengths), 'mean': sum(lengths)/len(lengths)},
        'pack_sha256': file_digest(args.pack), 'predictions_sha256': file_digest(args.predictions),
        'source_manifest_sha256': file_digest(args.manifest), 'script_sha256': file_digest(Path(__file__))})
    print(f'Verified {verified} prepared inputs; {unchanged} unchanged and {changed} augmented positions per arm.')


if __name__ == '__main__':
    main()
