import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from training_utils import classification_metrics, epoch_order, is_better, learning_rate


class TrainingTests(unittest.TestCase):
    def test_macro_f1_handles_missing_predictions(self):
        result = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0], 4)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 1/3)

    def test_warmup_decay_boundary(self):
        rates = [learning_rate(s, 10, 1, 0.2) for s in range(10)]
        self.assertEqual(rates[:2], [0.5, 1])
        self.assertEqual(rates[-1], 0)
        self.assertTrue(all(a >= b for a, b in zip(rates[1:], rates[2:])))

    def test_ranking_uses_loss_only_on_tie(self):
        best = {"macro_f1": 0.8, "loss": 0.4}
        self.assertTrue(is_better({"macro_f1": 0.8, "loss": 0.3}, best))
        self.assertFalse(is_better({"macro_f1": 0.79, "loss": 0.1}, best))

    def test_epoch_order_covers_remainder_and_is_stable(self):
        order = epoch_order(19, 42, 0)
        self.assertEqual(sorted(order.tolist()), list(range(19)))
        self.assertEqual(order.tolist(), epoch_order(19, 42, 0).tolist())
        self.assertNotEqual(order.tolist(), epoch_order(19, 42, 1).tolist())
