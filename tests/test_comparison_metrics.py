import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unittest
import numpy as np
from comparison_metrics import metrics, fit_temperature, paired


class ComparisonMetricsTests(unittest.TestCase):
    def test_binary_metrics_and_brier_convention(self):
        m = metrics([0, 1], np.log([1/3, 3]))
        self.assertAlmostEqual(m["nll"], -np.log(.75))
        self.assertEqual(m["accuracy"], 1.)
        self.assertEqual(m["auroc"], 1.)
        self.assertAlmostEqual(m["brier_multiclass"], .125)
        self.assertAlmostEqual(m["brier_positive"], .0625)
        self.assertAlmostEqual(m["ece_15_equal_width_top_label"], .25)

    def test_auc_ties_and_argmax_tie(self):
        m = metrics([0, 1], [0., 0.])
        self.assertEqual(m["auroc"], .5)
        self.assertEqual(m["correct"], 1)

    def test_temperature_optimum_and_preserved_predictions(self):
        # Three correct and one wrong equal-confidence decisions: optimal confidence=.75.
        y, z = np.array([1, 1, 1, 0]), np.array([2., 2., 2., 2.])
        t = fit_temperature(z, y)
        self.assertAlmostEqual(t, 2/np.log(3), places=9)
        self.assertEqual(metrics(y, z)["accuracy"], metrics(y, z, t)["accuracy"])
        self.assertLess(metrics(y, z, t)["nll"], metrics(y, z)["nll"])

    def test_exact_mcnemar_known_case(self):
        r = paired(np.ones(6), np.ones(6), -np.ones(6))
        self.assertEqual(r["a_only_correct"], 6)
        self.assertAlmostEqual(r["mcnemar_exact_two_sided_p"], .03125)


if __name__ == "__main__":
    unittest.main()
