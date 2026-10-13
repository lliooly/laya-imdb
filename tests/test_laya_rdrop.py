"""Numerical and isolation checks for the training-only consistency loss."""
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from laya_rdrop import rdrop_training, symmetric_kl, validate_alpha
from laya_optimization import optimize_training, validate_optimization


class RDropTests(unittest.TestCase):
    def test_alpha_admission(self):
        for alpha in [True, -1, float('nan'), float('inf'), '1']:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                validate_optimization({'rdrop_alpha': alpha})
        self.assertEqual(validate_alpha(1), 1.0)

    def test_known_distributions_padding_and_both_branch_gradients(self):
        import torch
        a = torch.tensor([[math.log(.8), math.log(.2), 1e20]], requires_grad=True)
        b = torch.tensor([[math.log(.6), math.log(.4), -1e20]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        value = symmetric_kl(torch, a, b, mask)
        expected = .5 * (.8 * math.log(.8/.6) + .2 * math.log(.2/.4)
                         + .6 * math.log(.6/.8) + .4 * math.log(.4/.2))
        self.assertAlmostEqual(float(value.detach()), expected, places=6)
        self.assertAlmostEqual(float(symmetric_kl(torch, b, a, mask).detach()), expected, places=6)
        value.backward()
        for gradient in (a.grad, b.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient[:, :2].abs().sum()), 0)
            self.assertEqual(float(gradient[0, 2]), 0)
        self.assertEqual(float(symmetric_kl(torch, a, a, mask).detach()), 0)

    def test_kl_averages_examples_and_rejects_empty_options(self):
        import torch
        a, b = torch.tensor([[2., 0.]]), torch.tensor([[0., 2.]])
        mask = torch.ones_like(a, dtype=torch.bool)
        single = symmetric_kl(torch, a, b, mask)
        doubled = symmetric_kl(torch, a.repeat(2, 1), b.repeat(2, 1), mask.repeat(2, 1))
        self.assertEqual(float(single), float(doubled))
        with self.assertRaises(ValueError):
            symmetric_kl(torch, a, b, torch.zeros_like(mask))

    def test_eval_and_no_grad_do_not_pair_or_change_original_loss(self):
        import torch
        calls = []
        original_loss = lambda logits, target, mask: logits.sum()
        upstream = SimpleNamespace(_forward=lambda *args: calls.append(1) or torch.ones((1, 2)),
                                   soft_ce_loss=original_loss)
        original_forward = upstream._forward
        with tempfile.TemporaryDirectory() as directory:
            with rdrop_training(torch, upstream, 1, Path(directory)):
                upstream._forward(SimpleNamespace(training=False), {}, 'cpu', False, False)
                with torch.no_grad():
                    upstream._forward(SimpleNamespace(training=True), {}, 'cpu', False, False)
                self.assertEqual(float(upstream.soft_ce_loss(torch.ones((1, 2)), None, None)), 2)
            self.assertEqual(len(calls), 2)
            self.assertIs(upstream._forward, original_forward)
            self.assertIs(upstream.soft_ce_loss, original_loss)

    def test_unconsumed_pair_raises_and_restores(self):
        import torch
        original_forward = lambda *args: torch.ones((1, 2), requires_grad=True)
        original_loss = lambda *args: None
        upstream = SimpleNamespace(_forward=original_forward, soft_ce_loss=original_loss)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'unconsumed'):
                with rdrop_training(torch, upstream, 1, Path(directory)):
                    upstream._forward(SimpleNamespace(training=True), {}, 'cpu', False, False)
            self.assertIs(upstream._forward, original_forward)
            self.assertIs(upstream.soft_ce_loss, original_loss)

    def test_rdrop_rejects_rlcd_before_forward(self):
        import torch
        upstream = SimpleNamespace(train_model=lambda *a, **k: None, items_from_rows=lambda *a, **k: None)
        with tempfile.TemporaryDirectory() as directory:
            with optimize_training(torch, upstream, {'rdrop_alpha': 1}, Path(directory)):
                with self.assertRaisesRegex(ValueError, 'soft-ce'):
                    upstream.train_model(None, None, [], SimpleNamespace(label_smoothing=0, loss='rlcd'),
                                         'cpu', 512, 192)


if __name__ == '__main__':
    unittest.main()
