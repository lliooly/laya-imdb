"""Budget regression guards for a narrowly scoped inference-only completion."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from laya_evaluation_completion import COMPLETION, JOBS, reserve


class EvaluationBudgetTests(unittest.TestCase):
    def base(self):
        return {'jobs': [{'run_id': 'prior', 'state': 'stopped_budget_confirmed'}],
                'maximum_compute_usd': 24., 'upper_bound_used_usd': 24.}

    def test_single_dollar_reservation_and_duplicate_block(self):
        ledger = self.base()
        record = reserve(ledger)
        self.assertEqual(record['reserved_usd'], 1.)
        self.assertEqual(ledger['upper_bound_used_usd'], 25.)
        with self.assertRaises(ValueError): reserve(ledger)

    def test_unresolved_prior_call_is_not_ignored(self):
        ledger = self.base(); ledger['jobs'][0]['state'] = 'reserved'
        before = copy.deepcopy(ledger)
        with self.assertRaises(ValueError): reserve(ledger)
        self.assertEqual(ledger, before)

    def test_budget_change_and_relaunch_fail_closed(self):
        for amount in (23., 24.01, 25.):
            ledger = self.base(); ledger['upper_bound_used_usd'] = amount
            with self.assertRaises(ValueError): reserve(ledger)
        ledger = self.base(); ledger['jobs'].append({'run_id': COMPLETION, 'state': 'completed'})
        with self.assertRaises(ValueError): reserve(ledger)

    def test_exact_five_inference_jobs(self):
        self.assertEqual(len(JOBS), 5)
        self.assertTrue(all(j['evaluate'] is True for j in JOBS))
        self.assertNotIn({'seed':42,'arm':'ce','evaluate':True}, JOBS)
        self.assertEqual(len({(j['seed'],j['arm']) for j in JOBS}), 5)


if __name__ == '__main__': unittest.main()
