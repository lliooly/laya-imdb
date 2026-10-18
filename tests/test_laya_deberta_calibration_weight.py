import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_laya_deberta_calibration_weight import blend, fit_weight


class CalibrationWeightTests(unittest.TestCase):
    def test_identical_models_keep_equal_weights(self):
        p = np.array([.1, .9])
        self.assertEqual(fit_weight(p, p, [0, 1])['alpha_laya'], .5)
        np.testing.assert_array_equal(blend(p, p, .5), p)

    def test_strictly_better_model_can_receive_full_weight(self):
        self.assertEqual(fit_weight([.99, .01], [.6, .4], [1, 0])['alpha_laya'], 1.)
        self.assertEqual(fit_weight([.6, .4], [.99, .01], [1, 0])['alpha_laya'], 0.)

    def test_complementary_models_have_interior_optimum(self):
        fit = fit_weight([.9, .8], [.2, .1], [1, 0])
        self.assertAlmostEqual(fit['alpha_laya'], .5)
        self.assertAlmostEqual(fit['derivative_at_solution'], 0.)

    def test_invalid_or_misaligned_inputs_fail(self):
        for a, b, y in [([.1], [.9, .8], [0]), ([np.nan], [.8], [1]), ([.1], [.9], [2])]:
            with self.assertRaises(ValueError):
                fit_weight(a, b, y)


if __name__ == '__main__':
    unittest.main()
