import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from semantic_pair_pilot import (build_triplets, digest, edit_text, make_schedule,
                                  materialize, require_human_review, source_partition, review_markdown)
from probe_semantic_pairs import summarize


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        class Encoding:
            ids = list(range(len(text)))
            offsets = [(i, i + 1) for i in range(len(text))]
        return Encoding()


class SemanticPairTests(unittest.TestCase):
    def setUp(self):
        self.source = {'p': {'id': 'p', 'text': 'It is not bad.', 'label': 1}}
        self.recipe = {'parent_id': 'p', 'label': 1, 'source_sha256': digest('It is not bad.'),
                       'category': 'complex_relation', 'evidence': 'not bad', 'rationale_zh': '否定范围',
                       'edits': [{'before': 'not bad', 'ordinary': 'not poor', 'semantic': 'quite acceptable'}]}

    def pack(self):
        return build_triplets([self.recipe], self.source, ['p'], CharacterTokenizer())

    def test_local_edit_preserves_surroundings(self):
        row = self.pack()[0]
        self.assertEqual(row['variants']['semantic']['text'], 'It is quite acceptable.')
        self.assertEqual(row['review_status'], 'ai_draft_pending_human')

    def test_partition_overlap_and_order_rejected(self):
        rows = [{'id': 'p', 'label': 1}, {'id': 'c', 'label': 0}]
        split = {'ids': {'train': ['p', 'c'], 'validation': ['v']}}
        manifest = {'gradient_train_ids': ['p'], 'calibration_ids': ['c'],
                    'train': {'ids': ['p', 'c']}, 'validation': {'ids': ['v']}}
        source_partition(rows, split, manifest)
        bad = copy.deepcopy(manifest); bad['calibration_ids'] = ['p']
        with self.assertRaisesRegex(ValueError, 'isolation'):
            source_partition(rows, split, bad)
        with self.assertRaisesRegex(ValueError, 'order'):
            source_partition(list(reversed(rows)), split, manifest)

    def test_excluded_source_and_mismatched_label_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, 'gradient'):
            build_triplets([self.recipe], self.source, ['other'], CharacterTokenizer())
        for field, value in [('label', 0), ('source_sha256', '0'*64)]:
            bad = copy.deepcopy(self.recipe); bad[field] = value
            with self.assertRaisesRegex(ValueError, 'hash or label'):
                build_triplets([bad], self.source, ['p'], CharacterTokenizer())

    def test_repeated_and_overlapping_edits_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            edit_text('bad bad', [{'before': 'bad', 'ordinary': 'poor'}], 'ordinary')
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            edit_text('not bad', [{'before': 'not bad', 'ordinary': 'fine'},
                                  {'before': 'bad', 'ordinary': 'poor'}], 'ordinary')

    def test_invisible_original_or_replacement_rejected(self):
        text = 'x'*470 + 'It is not bad.'
        bad = copy.deepcopy(self.recipe); bad['source_sha256'] = digest(text)
        with self.assertRaisesRegex(ValueError, 'outside visible'):
            build_triplets([bad], {'p': {'text': text, 'label': 1}}, ['p'], CharacterTokenizer())
        self.recipe['edits'][0]['semantic'] = 'x'*470
        with self.assertRaisesRegex(ValueError, 'outside visible'):
            self.pack()

    def test_review_is_human_and_hash_bound(self):
        pack = self.pack()
        review = {'reviewer_kind': 'human', 'reviewer': 'test-only reviewer', 'decisions': [{
            'parent_id': 'p', 'label': 1, 'decision': 'accept',
            'variant_sha256': {k: v['sha256'] for k, v in pack[0]['variants'].items()},
            'label_preserved': True, 'natural_language': True, 'evidence_sufficient': True}]}
        require_human_review(pack, review)
        bad = copy.deepcopy(review); bad['reviewer_kind'] = 'AI'
        with self.assertRaisesRegex(ValueError, 'human'):
            require_human_review(pack, bad)
        bad = copy.deepcopy(review); bad['decisions'][0]['variant_sha256']['semantic'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Stale'):
            require_human_review(pack, bad)
        bad = copy.deepcopy(review); bad['decisions'][0]['decision'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'accepted'):
            require_human_review(pack, bad)

    def test_review_source_filename_hidden_until_details(self):
        pack = self.pack()
        pack[0]['parent_id'] = 'train/pos/rating_10.txt'
        markdown = review_markdown(pack)
        self.assertNotIn('rating_10.txt', markdown.split('<details>')[0])
        self.assertIn('rating_10.txt', markdown.split('<details>')[1])

    def test_schedule_and_arms_preserve_budgets_and_coverage(self):
        from collections import Counter
        ids = [str(i) for i in range(100)]
        labels = {i: int(i) % 2 for i in ids}
        sources = {i: {'id': i, 'label': labels[i], 'text': 'original-' + i} for i in ids}
        parents = ['0', '1', '2', '3']
        triplets = [{'parent_id': p, 'variants': {k: {'text': sources[p]['text'] if k == 'original' else k+p}
                                               for k in ('original', 'ordinary', 'semantic')}} for p in parents]
        schedule = make_schedule(ids, labels, parents)
        self.assertEqual(schedule, make_schedule(ids, labels, parents))
        self.assertEqual(sum(r['augmented'] for e in schedule for r in e), 10)
        self.assertEqual({r['parent_id'] for r in schedule[0]}, set(ids))
        for e in schedule:
            self.assertEqual(Counter(r['label'] for r in e), Counter(labels.values()))
        arms = {arm: materialize(schedule, sources, triplets, arm) for arm in ('A', 'B', 'C')}
        for epoch in range(2):
            for a, b, c in zip(*(arms[arm][epoch] for arm in ('A', 'B', 'C'))):
                self.assertEqual((a['parent_id'], a['label']), (c['parent_id'], c['label']))
                if a['augmented']:
                    self.assertEqual(len({a['text'], b['text'], c['text']}), 3)
                else:
                    self.assertEqual(a, b); self.assertEqual(b, c)

    def test_probe_counts_regressions_and_double_errors(self):
        rows = [{'parent_id': 'fixed', 'label': 1,
                 'predictions': {k: {'label': v} for k, v in zip(('original', 'ordinary', 'semantic'), (0, 0, 1))}},
                {'parent_id': 'lost', 'label': 0,
                 'predictions': {k: {'label': v} for k, v in zip(('original', 'ordinary', 'semantic'), (0, 0, 1))}}]
        result = summarize(rows)
        self.assertEqual(result['transitions']['semantic']['fixed'], ['fixed'])
        self.assertEqual(result['transitions']['semantic']['regressed'], ['lost'])
        self.assertEqual(result['transitions']['semantic']['pair_both_correct'], 0)


if __name__ == '__main__':
    unittest.main()
