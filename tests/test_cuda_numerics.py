"""Verify nonfinite training cannot advance optimizer/scheduler, and hooks are restored."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cuda_numerics import guard_training


class NumericalGuardTests(unittest.TestCase):
    def test_nonfinite_gradient_aborts_before_update_and_restores_hooks(self):
        import torch
        p = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([p], lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        p.grad = torch.tensor([float('inf')])
        upstream = SimpleNamespace(soft_ce_loss=lambda value: value, rlcd_loss=lambda value: value)
        original_clip = torch.nn.utils.clip_grad_norm_
        original_loss = upstream.soft_ce_loss
        with self.assertRaises(RuntimeError):
            with guard_training(torch, upstream):
                torch.nn.utils.clip_grad_norm_([p], 1.0)
                optimizer.step()
                scheduler.step()
        self.assertEqual(p.item(), 1.0)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertFalse(optimizer.state)
        self.assertIs(torch.nn.utils.clip_grad_norm_, original_clip)
        self.assertIs(upstream.soft_ce_loss, original_loss)

    def test_nonfinite_loss_is_rejected(self):
        import torch
        upstream = SimpleNamespace(soft_ce_loss=lambda value: value, rlcd_loss=lambda value: value)
        with guard_training(torch, upstream):
            for name in ('soft_ce_loss', 'rlcd_loss'):
                with self.assertRaises(FloatingPointError):
                    getattr(upstream, name)(torch.tensor(float('nan')))

    def test_finite_update_remains_equal_to_unguarded_update(self):
        import torch
        values = []
        upstream = SimpleNamespace(soft_ce_loss=lambda value: value, rlcd_loss=lambda value: value)
        from contextlib import nullcontext
        for guarded in (False, True):
            p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
            optimizer = torch.optim.AdamW([p], lr=0.1)
            with guard_training(torch, upstream) if guarded else nullcontext():
                upstream.soft_ce_loss(p.square().sum()).backward()
                torch.nn.utils.clip_grad_norm_([p], 1.0)
                optimizer.step()
            values.append(p.detach().clone())
        self.assertTrue(torch.equal(*values))


if __name__ == '__main__':
    unittest.main()
