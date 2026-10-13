import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import ROOT, read_jsonl
from export_laya_posthoc import prepare_rules, select_splits, checked_logits, input_digest


class PosthocInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(str(ROOT / '.cache/laya/english/tokenizer'), local_files_only=True)

    def test_short_input_identical_and_long_input_uses_real_tail(self):
        import laya.train as upstream
        rows = [{'id': 'short', 'label': 1, 'text': 'A genuinely enjoyable film.'},
                {'id': 'long', 'label': 0, 'text': 'The story unfolds slowly. '*220 + 'But the ending is an absolute disappointment.'}]
        head, tail, stats, head_len, budget = prepare_rules(self.tok, rows)
        self.assertEqual((head_len, budget), (45, 466))
        original = [upstream.encode_item(self.tok, i, 512, 192, parallel=False) for i in head]
        altered = [upstream.encode_item(self.tok, i, 512, 192, parallel=False) for i in tail]
        self.assertEqual(original[0], altered[0])
        self.tok.backend_tokenizer.no_truncation()
        full = self.tok.backend_tokenizer.encode(rows[1]['text'], add_special_tokens=False).ids
        self.assertEqual(tail[1]['state_ids'], full[:233]+full[-233:])
        self.assertNotEqual(tail[1]['state_ids'][-233:], full[233:466])
        self.assertEqual(altered[1]['ids'][45:-1], full[:233]+full[-233:])
        self.assertEqual(len(altered[1]['ids']), 512)
        self.assertEqual(original[1]['markers'], altered[1]['markers'])
        self.assertFalse(stats[0]['truncated'])
        self.assertTrue(stats[1]['truncated'])

    def test_calibration_order_and_isolation_reject_overlap(self):
        train = [{'id': f'train/{i}', 'label': i%2} for i in range(401)]
        dev = [{'id': f'dev/{i}', 'label': i%2} for i in range(2500)]
        ids = [r['id'] for r in train]
        vi = [r['id'] for r in dev]
        manifest = {'calibration_ids': list(reversed(ids[:400])), 'gradient_train_ids': ids[400:],
                    'train': {'ids': ids}, 'validation': {'ids': vi}}
        split = {'ids': {'train': ids, 'validation': vi}}
        selected = select_splits(manifest, split, train, dev)
        self.assertEqual([r['id'] for r in selected['calibration']], manifest['calibration_ids'])
        bad = copy.deepcopy(manifest)
        bad['gradient_train_ids'].append(ids[0])
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            select_splits(bad, split, train, dev)
        with self.assertRaisesRegex(ValueError, 'order'):
            select_splits(manifest, split, train, list(reversed(dev)))

    def test_head_matches_saved_real_validation_inputs_and_budget_rejected(self):
        import laya.train as upstream
        predictions = json.loads((ROOT / 'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/cuda-saved-validation.json').read_text())['validation_predictions']
        by_id = {r['id']: r for r in read_jsonl(ROOT / 'data/processed/validation.jsonl')}
        ids = [predictions[0]['sample_id'], 'train/neg/838_2.txt']
        original, _, _, _, _ = prepare_rules(self.tok, [by_id[i] for i in ids])
        expected = {p['sample_id']: p['input_sha256'] for p in predictions}
        for sample_id, item in zip(ids, original):
            self.assertEqual(input_digest(upstream.encode_item(self.tok, item, 512, 192, parallel=False)), expected[sample_id])
        with self.assertRaisesRegex(ValueError, 'budget'):
            prepare_rules(self.tok, [by_id[ids[0]]], max_len=1024)

    def test_nonfinite_and_wrong_shape_logits_rejected(self):
        for logits in ([float('nan'), 0], [float('inf'), 0], [0], [[0, 1]]):
            with self.assertRaises(FloatingPointError):
                checked_logits((0, logits, [1, 0], 2))
        self.assertEqual(checked_logits((0, [1, -1], [1, 0], 2)).tolist(), [1, -1])


if __name__ == '__main__':
    unittest.main()
