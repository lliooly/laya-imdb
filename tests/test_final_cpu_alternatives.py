import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_final_cpu_alternatives import bounded_minimum, holm, make_head, numpy_head_state, install_guard, ROOT


class FinalCPUAlternativeTests(unittest.TestCase):
    def test_weak_extra_branch_can_be_rejected_at_zero(self):
        self.assertEqual(bounded_minimum(lambda w: 1+w, .5), 0.)
        self.assertEqual(bounded_minimum(lambda w: w-1, .5), .5)
        self.assertAlmostEqual(bounded_minimum(lambda w: w-.13, .5), .13)

    def test_holm_adjustment_preserves_family_error_control(self):
        self.assertEqual(holm(dict(a=.01, b=.04, c=.03)), dict(a=.03, c=.06, b=.06))

    def test_head_eval_has_one_deterministic_path_and_safe_state_roundtrip(self):
        config = dict(heads=dict(hidden=64, dropout=.2))
        torch.manual_seed(42)
        model = make_head(torch, config).eval()
        x = torch.randn(4, 1024)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(x), model(x)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'head.safetensors'
            save_file(dict(model.state_dict()), path)
            state = numpy_head_state(path)
            for name, tensor in model.state_dict().items():
                np.testing.assert_array_equal(state[name], tensor.detach().numpy())

    def test_z_fit_guard_blocks_development_and_official_test_before_open(self):
        install_guard('fit')
        for path in ['data/processed/test.jsonl', 'artifacts/synthetic/validation.npz', 'reports/synthetic/2048-development.json']:
            with self.assertRaises(PermissionError):
                (ROOT/path).read_bytes()


if __name__ == '__main__':
    unittest.main()
