import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from laya_frozen_probe import (MarkerCapture, standardize, fit_readout, logits,
                               score, select_candidate, paired_accuracy)


class FrozenProbeTests(unittest.TestCase):
    def test_raw_encoder_capture_is_independent_of_decision_head(self):
        import torch
        from transformers import BertConfig, BertModel
        from laya.common import DecisionModel
        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = DecisionModel(BertModel(BertConfig(vocab_size=40, hidden_size=16,
            num_hidden_layers=1, num_attention_heads=1, intermediate_size=32,
            max_position_embeddings=32)), head_layers=1).eval().requires_grad_(False)
        tokens = torch.tensor([[2, 4, 8, 9, 8, 6], [2, 8, 10, 11, 8, 6]])
        mask = torch.ones_like(tokens)
        pos = torch.tensor([[2, 4], [1, 4]])
        marker_mask = torch.ones_like(pos, dtype=torch.bool)
        qtype = torch.zeros(2, dtype=torch.long)
        captures = []
        with torch.inference_mode():
            for changed in (False, True):
                if changed:
                    # A large head intervention must leave the raw encoder probe unchanged.
                    model.head.layers[0].linear2.weight.mul_(0)
                    model.head.layers[0].linear1.weight.add_(1)
                with MarkerCapture(model, torch) as capture:
                    capture.begin(pos)
                    model(tokens, mask, pos, marker_mask, qtype)
                    captures.append(capture.consume())
            raw = model.encoder(input_ids=tokens, attention_mask=mask).last_hidden_state
            expected = raw[torch.arange(2), pos[:, 1]]-raw[torch.arange(2), pos[:, 0]]
        np.testing.assert_array_equal(captures[0]['encoder'], expected.numpy())
        np.testing.assert_array_equal(captures[0]['encoder'], captures[1]['encoder'])
        self.assertFalse(np.allclose(captures[0]['post_head'], captures[1]['post_head']))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        self.assertFalse(model.encoder._forward_hooks)
        self.assertFalse(model.scorer._forward_pre_hooks)

    def test_convex_probe_converges_and_scaling_uses_training_only(self):
        import torch
        torch.set_num_threads(1)
        x = np.column_stack((np.linspace(-3, 3, 100), np.ones(100)))
        y = (x[:, 0] > 0).astype(int)
        mean, scale = standardize(x)
        self.assertEqual(scale[1], 1.)
        preserved = mean.copy()
        result = fit_readout(torch, x, y, .01, mean, scale)
        self.assertTrue(result['converged'])
        self.assertEqual(score(logits(x, result, mean, scale), y)['correct'], 100)
        logits(x+1000, result, mean, scale)
        np.testing.assert_array_equal(mean, preserved)

    def test_calibration_selection_order_and_paired_identity(self):
        candidates = [{'lambda': l, 'calibration': {'correct': c, 'nll': n}}
                      for l, c, n in [(1., 390, .1), (.1, 391, .2), (.01, 391, .2), (.001, 390, .01)]]
        self.assertEqual(select_candidate(copy.deepcopy(candidates))['lambda'], .1)
        y = np.array([0, 1]*10)
        result = paired_accuracy(y, y, y, repetitions=200, seed=3)
        self.assertEqual(result['delta_accuracy_pp'], 0.)
        self.assertEqual(result['paired_stratified_bootstrap95_pp'], [0., 0.])
        self.assertEqual(result['mcnemar_exact_nominal_two_sided_p'], 1.)

    def test_invalid_markers_and_nonfinite_features_fail(self):
        import torch
        from types import SimpleNamespace
        capture = MarkerCapture(SimpleNamespace(), torch)
        for pos in (torch.tensor([[1, -1]]), torch.tensor([[1, 2, 3]])):
            with self.assertRaises(ValueError):
                capture.begin(pos)
        with self.assertRaises(ValueError):
            standardize(np.array([[float('nan')]]))


if __name__ == '__main__':
    unittest.main()
