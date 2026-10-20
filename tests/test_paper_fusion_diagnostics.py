import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from paper_fusion_diagnostics import align, changes, fit_weight


class FusionDiagnosticsTests(unittest.TestCase):
    def test_known_calibration_optima(self):
        y = np.array([1, 0])
        self.assertEqual(fit_weight([.9, .1], [.4, .6], y), 1.)
        self.assertEqual(fit_weight([.4, .6], [.9, .1], y), 0.)
        self.assertAlmostEqual(fit_weight([.99, .9], [.1, .01], y), .5, places=10)
        self.assertEqual(fit_weight([.8, .2], [.8, .2], y), 1.)

    def test_misaligned_ids_are_rejected(self):
        a = dict(sample_ids=['a', 'b'], true_labels=[0, 1], positive_probabilities=[.1, .9], test_accessed=False)
        b = dict(a, sample_ids=['b', 'a'])
        with self.assertRaises(ValueError):
            align({'a': a, 'b': b}, 2)

    def test_repair_harm_direction_and_tie(self):
        r = changes(np.array([1, 0, 1]), np.array([.4, .4, .6]), np.array([.6, .6, .5]))
        self.assertEqual((r['repaired'], r['harmed'], r['net_correct']), (1, 2, -1))


if __name__ == '__main__':
    unittest.main()
