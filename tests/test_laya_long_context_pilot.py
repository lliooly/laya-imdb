"""Check token retention beyond the historical 512-token pre-clamp."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT
from laya_long_context_pilot import prepare


class LongContextInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(str(ROOT/'.cache/laya/english/tokenizer'), local_files_only=True)

    def test_middle_and_tail_survive_2048_when_full_review_fits(self):
        from laya_decision_views import full_tokens
        text = 'Wonderful movie. ' * 210 + ' However the ending is awful. '
        row = {'id': 'synthetic-only', 'text': text, 'label': 0}
        full = full_tokens(self.tok, [text])[0]
        self.assertGreater(len(full), 512)
        self.assertLess(len(full), 2002)
        short, _ = prepare(self.tok, [row], 512)
        long, sizes = prepare(self.tok, [row], 2048)
        self.assertEqual(long[0]['ids'][45:-1], full)
        self.assertEqual(sizes, [len(full)])
        self.assertEqual(len(short[0]['ids']), 512)
        self.assertEqual(short[0]['ids'][45:-1], full[:233]+full[-233:])
        self.assertEqual(short[0]['target'], long[0]['target'])
        self.assertEqual(short[0]['markers'], long[0]['markers'])

    def test_short_inputs_identical_and_overflow_has_both_ends(self):
        from laya_decision_views import full_tokens
        rows = [{'id':'short', 'text':'A good movie.', 'label':1},
                {'id':'long', 'text':'Good movie. '*1000+'Bad ending.', 'label':0}]
        a, _ = prepare(self.tok, rows, 512)
        b, sizes = prepare(self.tok, rows, 2048)
        self.assertEqual(a[0], b[0])
        full = full_tokens(self.tok, [rows[1]['text']])[0]
        self.assertGreater(sizes[1], 2002)
        self.assertEqual(b[1]['ids'][45:-1], full[:1001]+full[-1001:])
        self.assertEqual(len(b[1]['ids']), 2048)


if __name__ == '__main__':
    unittest.main()
