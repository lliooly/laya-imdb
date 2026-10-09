"""Guard against split leakage and accidental preprocessing damage."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import clean_text, digest
from prepare_data import partition


def record(identity, text, label):
    return {"id": identity, "text": text, "label": label, "text_sha256": digest(text)}


class DataTests(unittest.TestCase):
    def test_preserve_negation_case_and_punctuation(self):
        self.assertEqual(clean_text("  I do NOT like it!<br /><br>But &quot;nice&quot; acting. "),
                         'I do NOT like it! But "nice" acting.')

    def test_grouping_conflicts_balance_and_reproducibility(self):
        rows = [record(f"{y}-{i}", f"label{y} review{i}", y) for y in (0, 1) for i in range(20)]
        rows += [record("duplicate", "label0 review0", 0),
                 record("conflict0", "mixed", 0), record("conflict1", "mixed", 1)]
        train, val = partition(rows, 42, 5)
        self.assertEqual((train, val), partition(rows, 42, 5))
        self.assertEqual(len(train) + len(val), len(rows))
        self.assertEqual([sum(r["label"] == y for r in val) for y in (0, 1)], [5, 5])
        self.assertFalse({r["text_sha256"] for r in train} & {r["text_sha256"] for r in val})
        self.assertTrue({"conflict0", "conflict1"} <= {r["id"] for r in train})


if __name__ == "__main__":
    unittest.main()
