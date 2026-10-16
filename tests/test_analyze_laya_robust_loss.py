import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_laya_robust_loss import continuation, paired_summary


class PairedAnalysisTests(unittest.TestCase):
    def test_same_id_bootstrap_preserves_perfect_seed_dependence(self):
        y = np.array([0,0,1,1])
        ce = np.ones((4,3), bool)
        mixed = ce.copy();mixed[0] = False
        result = paired_summary(y, ce, mixed, draws=300)
        self.assertEqual(result['mean_mixed_minus_ce_pp'], -25.)
        for seed in result['per_seed']:
            self.assertEqual(seed['repairs'], 0)
            self.assertEqual(seed['regressions'], 1)
            self.assertEqual(seed['conditional_bootstrap95_pp'], result['conditional_shared_id_bootstrap95_pp'])
        self.assertEqual(result['seed_sign_flip_nominal_two_sided_p'], .25)

    def test_identical_predictions_have_zero_effect_and_interval(self):
        y = np.array([0,1,0,1]);a = np.ones((4,3), bool)
        result = paired_summary(y, a, a, draws=100)
        self.assertEqual(result['conditional_shared_id_bootstrap95_pp'], [0.,0.])
        self.assertEqual(result['seed_sign_flip_nominal_two_sided_p'], 1.)
        self.assertFalse(continuation(result)['passed'])

    def test_continuation_is_conjunctive_and_boundary_inclusive(self):
        group = {'mean_mixed_minus_ce_pp':.2, 'per_seed':[{'delta_pp':x} for x in [-.2,.4,.4]]}
        self.assertTrue(continuation(group)['passed'])
        group['per_seed'][0]['delta_pp'] = -.2001
        self.assertFalse(continuation(group)['passed'])


if __name__ == '__main__': unittest.main()
