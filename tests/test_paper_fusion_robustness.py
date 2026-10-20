import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from paper_fusion_robustness import grouped_folds, fit_pipeline, cluster_interval


class FusionRobustnessTests(unittest.TestCase):
    def test_duplicate_texts_do_not_cross_fold(self):
        groups = np.array(['a', 'a', 'b', 'c', 'd', 'e', 'f', 'g'])
        folds = grouped_folds(groups, 23, k=3)
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(8)))
        for test in folds:
            train = np.setdiff1d(np.arange(8), test)
            self.assertFalse(set(groups[test]) & set(groups[train]))

    def test_heldout_label_poisoning_does_not_change_fit(self):
        data = {k: dict(native=np.array([-1., 2., -3., 4., 5., -6.])*scale,
                        head=np.array([-1., 2., -3., 4., 5., -6.])*scale,
                        tail=np.array([-1., 2., -3., 4., 5., -6.])*scale,
                        truncated=np.zeros(6, bool)) for k, scale in [('baseline', .7), ('combined', 1.2), ('deberta', .9)]}
        y = np.array([0, 1, 0, 1, 0, 1])
        poisoned = y.copy(); poisoned[4:] = 1-poisoned[4:]
        first = fit_pipeline(data, y, np.arange(4))[0]
        second = fit_pipeline(data, poisoned, np.arange(4))[0]
        self.assertEqual(first, second)

    def test_group_bootstrap_point_keeps_row_weights_and_pairing(self):
        groups = np.array(['a', 'b', 'b'])
        delta = np.array([[1., 0., 0.], [1., 0., 0.]])
        result, samples = cluster_interval(delta, groups, samples=200, seed=7)
        self.assertAlmostEqual(result['mean_delta_pp'], 100/3)
        self.assertEqual(len(samples), 200)
        self.assertTrue(all(any(abs(x-v)<1e-10 for v in (0., 100/3, 100.)) for x in samples))


if __name__ == '__main__':
    unittest.main()
