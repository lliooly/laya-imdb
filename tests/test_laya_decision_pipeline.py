import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import ROOT
from run_laya_decision_pipeline import window_evidence, validate_pair_specs


class DecisionPipelineTests(unittest.TestCase):
    def fixture(self):
        row = {'single_and_fused': {
            'head': {'all': {'count': 2500, 'accuracy': .965}, 'truncated': {'count': 407, 'accuracy': .95}},
            'head_tail': {'all': {'count': 2500, 'accuracy': .967}, 'truncated': {'count': 407, 'accuracy': .962}}},
            'exp1': {'success': False}}
        return {'state': 'completed', 'test_accessed': False,
                'models': {'baseline-s43': copy.deepcopy(row), 'combined-s42': copy.deepcopy(row)}}

    def test_n4_coverage_gate_is_independent_of_fusion_success(self):
        self.assertTrue(window_evidence(self.fixture())['supported'])
        doc = self.fixture()
        doc['models']['combined-s42']['single_and_fused']['head_tail']['truncated']['accuracy'] = .949
        self.assertFalse(window_evidence(doc)['supported'])

    def test_partial_or_changed_checkpoint_study_is_rejected(self):
        for key, value in (('state', 'running'), ('test_accessed', True), ('models', {})):
            doc = self.fixture()
            doc[key] = value
            with self.assertRaises(ValueError):
                window_evidence(doc)

    def test_training_pair_keeps_single_factor_and_new_run_ids(self):
        a, b = [json.loads((ROOT / f'configs/laya-decision-window-{mode}-s42-strict-20261013-v1.json').read_text())
                for mode in ('prefix', 'random_head_tail')]
        validate_pair_specs(a, b)
        b['training']['encoder_lr'] *= 2
        with self.assertRaises(ValueError):
            validate_pair_specs(a, b)


if __name__ == '__main__':
    unittest.main()
