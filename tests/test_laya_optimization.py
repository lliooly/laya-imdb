"""CPU regressions for the pinned training loop and opt-in experiment adapter."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from laya_optimization import make_scheduler, optimize_training, validate_optimization
from cuda_numerics import guard_training


class OptimizationTests(unittest.TestCase):
    def test_invalid_options_are_rejected(self):
        for options in [[], {'warmup_ratio': True}, {'warmup_ratio': float('nan')},
                        {'warmup_ratio': float('inf')}, {'warmup_ratio': -0.1},
                        {'warmup_ratio': 1}, {'diagnostics_every': 0},
                        {'diagnostics_every': True}, {'warmup_steps': 4}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_optimization(options)

    def test_actual_update_rates_reach_peak_and_floor_in_both_groups(self):
        import torch
        p, q = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
        optimizer = torch.optim.AdamW([{'params': [p], 'lr': 2.5e-5},
                                      {'params': [q], 'lr': 1e-4}])
        scheduler = make_scheduler(torch, optimizer, 5526, 1e-6, 0.05)
        rates = []
        for _ in range(5526):
            rates.append([g['lr'] for g in optimizer.param_groups])
            optimizer.step()
            scheduler.step()
        self.assertEqual(rates[0], [2.5e-5 / 277, 1e-4 / 277])
        self.assertEqual(rates[276], [2.5e-5, 1e-4])
        self.assertEqual(rates[-1], [1e-6, 1e-6])
        for index in range(2):
            self.assertTrue(all(a[index] <= b[index] for a, b in zip(rates[:276], rates[1:277])))
            self.assertTrue(all(a[index] >= b[index] for a, b in zip(rates[276:-1], rates[277:])))
        optimizer.step()
        scheduler.step()
        self.assertEqual([g['lr'] for g in optimizer.param_groups], rates[-1])

    def test_warmup_rejects_budget_without_decay(self):
        import torch
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)
        with self.assertRaises(ValueError):
            make_scheduler(torch, optimizer, 1, 0.01, 0.05)

    def run_toy_training(self, options, smoothing=0.0, fail=False, dropout=0.0):
        import torch
        import laya.train as upstream

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Embedding(12, 4)
                self.head = torch.nn.Linear(4, 1)
                self.dropout = torch.nn.Dropout(dropout)
                self.calls = 0

            def forward(self, input_ids, attention_mask, marker_pos, marker_mask,
                        qtype, detach_encoder=False):
                self.calls += 1
                if fail is True and self.calls == 3:
                    raise RuntimeError('synthetic interrupted forward')
                hidden = self.dropout(self.encoder(input_ids))
                logits = self.head(hidden).squeeze(-1).gather(1, marker_pos)
                if ((fail == 'nonfinite' and self.calls == 3)
                        or (fail == 'second_nonfinite' and self.calls == 2)):
                    logits = logits * float('nan')
                return logits, None

        torch.manual_seed(42)
        model = ToyModel()
        # Five micro-batches per epoch, including a partial final accumulation
        # window: this exercises actual upstream optimizer update accounting.
        items = [{'ids': [i + 1, i + 2], 'markers': [0, 1], 'qtype': 0,
                  'q': {'t': 'choice'}, 'k': 2,
                  'target': [1.0, 0.0] if i % 2 else [0.0, 1.0]} for i in range(9)]
        original_items = copy.deepcopy(items)
        observed_targets = []

        def encode(tok, item, *args, **kwargs):
            observed_targets.append(list(item['target']))
            return item

        config = upstream.TrainConfig(epochs=2, micro_batch=2, grad_accum=2,
                                      encoder_lr=0.02, head_lr=0.04, min_lr=0.001,
                                      loss='soft-ce', amp=False, gradient_checkpointing=False,
                                      log_every=0, label_smoothing=smoothing)
        originals = (upstream.train_model, upstream.items_from_rows, upstream._forward, upstream.soft_ce_loss,
                     torch.optim.lr_scheduler.CosineAnnealingLR,
                     torch.nn.utils.clip_grad_norm_, torch.optim.AdamW.step)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.object(upstream, 'encode_item', encode):
                try:
                    with guard_training(torch, upstream), optimize_training(torch, upstream, options, path):
                        upstream.train_model(model, SimpleNamespace(pad_token_id=0), items, config,
                                             torch.device('cpu'), 512, 192)
                except (RuntimeError, FloatingPointError):
                    if not fail:
                        raise
                else:
                    self.assertFalse(fail, 'Synthetic failure must propagate')
            self.assertEqual(items, original_items)
            self.assertEqual(originals, (upstream.train_model, upstream.items_from_rows, upstream._forward, upstream.soft_ce_loss,
                             torch.optim.lr_scheduler.CosineAnnealingLR,
                             torch.nn.utils.clip_grad_norm_, torch.optim.AdamW.step))
            audit = json.loads((path / 'optimization_diagnostics.json').read_text()) if options else None
            if audit is not None:
                audit['model_forward_calls'] = model.calls
                if (path / 'rdrop_diagnostics.json').exists():
                    audit['rdrop'] = json.loads((path / 'rdrop_diagnostics.json').read_text())
            updates = [json.loads(s) for s in (path / 'optimizer_updates.jsonl').read_text().splitlines()] if options else []
        return model.state_dict(), audit, updates, observed_targets

    def test_diagnostics_with_zero_warmup_preserve_exact_parameters(self):
        import torch
        plain, _, _, _ = self.run_toy_training(None)
        observed, audit, updates, _ = self.run_toy_training({'warmup_ratio': 0.0})
        self.assertTrue(all(torch.equal(value, observed[key]) for key, value in plain.items()))
        self.assertEqual(audit['completed_optimizer_updates'], 6)
        self.assertEqual(audit['gradient_windows'], 6)
        self.assertEqual(audit['scheduler'], 'upstream_cosine')
        self.assertEqual(len(updates), 6)

    def test_warmup_follows_optimizer_updates_with_partial_accumulation(self):
        _, audit, updates, _ = self.run_toy_training({'warmup_ratio': 0.3})
        self.assertEqual(audit['warmup_updates'], 2)
        self.assertEqual(audit['completed_optimizer_updates'], 6)
        self.assertEqual(updates[0]['learning_rates'], [0.01, 0.02])
        self.assertEqual(updates[1]['learning_rates'], [0.02, 0.04])
        self.assertEqual(updates[-1]['learning_rates'], [0.001, 0.001])
        self.assertTrue(all(math.isfinite(x['gradient_norm_before_clipping']) for x in updates))

    def test_label_smoothing_only_changes_training_targets(self):
        _, audit, _, targets = self.run_toy_training({'warmup_ratio': 0.0}, smoothing=0.05)
        self.assertEqual(audit['label_smoothing_scope'], 'gradient_training_only')
        self.assertTrue(all(sorted(t) == [0.025, 0.975] for t in targets))
        import torch
        calls = []
        upstream = SimpleNamespace(train_model=lambda *args, **kwargs: None,
            items_from_rows=lambda *args, **kwargs: calls.append(kwargs['label_smoothing']))
        original = upstream.items_from_rows
        with tempfile.TemporaryDirectory() as directory:
            with optimize_training(torch, upstream, {}, Path(directory)):
                upstream.items_from_rows(None, [], 512, 192, label_smoothing=0.05)
                upstream.items_from_rows(None, [], 512, 192, label_smoothing=0.0)
        self.assertEqual(calls, [0.0, 0.0])
        self.assertIs(upstream.items_from_rows, original)

    def test_failure_keeps_diagnostics_and_restores_hooks(self):
        _, audit, updates, _ = self.run_toy_training({'warmup_ratio': 0.3}, fail=True)
        self.assertEqual(audit['state'], 'failed')
        self.assertEqual(audit['completed_optimizer_updates'], 1)
        self.assertEqual(len(updates), 1)

    def test_nonfinite_loss_with_adapter_stops_before_next_update(self):
        _, audit, updates, _ = self.run_toy_training({'warmup_ratio': 0.3}, fail='nonfinite')
        self.assertEqual(audit['state'], 'failed')
        self.assertIn('Nonfinite', audit['error'])
        self.assertEqual(audit['completed_optimizer_updates'], 1)
        self.assertEqual(audit['gradient_windows'], 1)
        self.assertEqual(len(updates), 1)

    def test_zero_rdrop_preserves_dropout_rng_and_exact_parameters(self):
        import torch
        plain, _, _, _ = self.run_toy_training({'warmup_ratio': 0.3}, dropout=0.3)
        observed, audit, _, _ = self.run_toy_training(
            {'warmup_ratio': 0.3, 'rdrop_alpha': 0}, dropout=0.3)
        self.assertTrue(all(torch.equal(value, observed[key]) for key, value in plain.items()))
        self.assertNotIn('rdrop', audit)
        self.assertEqual(audit['model_forward_calls'], 10)

    def test_rdrop_two_forwards_per_batch_keep_optimizer_budget(self):
        _, audit, _, _ = self.run_toy_training(
            {'warmup_ratio': 0.3, 'rdrop_alpha': 1.0}, smoothing=0.05, dropout=0.3)
        self.assertEqual(audit['completed_optimizer_updates'], 6)
        self.assertEqual(audit['model_forward_calls'], 20)
        self.assertEqual(audit['rdrop']['paired_batches'], 10)
        self.assertEqual(audit['rdrop']['training_forward_calls'], 20)
        self.assertGreater(audit['rdrop']['mean_symmetric_kl'], 0)
        self.assertEqual(audit['rdrop']['state'], 'completed')

    def test_rdrop_second_forward_nonfinite_stops_without_optimizer_update(self):
        _, audit, updates, _ = self.run_toy_training(
            {'warmup_ratio': 0.3, 'rdrop_alpha': 1.0}, fail='second_nonfinite', dropout=0.3)
        self.assertEqual(audit['state'], 'failed')
        self.assertEqual(audit['completed_optimizer_updates'], 0)
        self.assertEqual(updates, [])
        self.assertEqual(audit['rdrop']['state'], 'failed')

    def test_rdrop_forward_exception_restores_hooks(self):
        _, audit, _, _ = self.run_toy_training(
            {'warmup_ratio': 0.3, 'rdrop_alpha': 1.0}, fail=True, dropout=0.3)
        self.assertEqual(audit['state'], 'failed')
        self.assertEqual(audit['rdrop']['state'], 'failed')

    def test_configs_keep_independent_factors_and_smoke_has_real_warmup(self):
        import laya.train as upstream
        root = Path(__file__).resolve().parents[1]
        base = json.loads((root / 'configs/laya-rtx4090-ce-epochs2-fp32.json').read_text())
        for name, ratio, smoothing in [('ls005', 0.0, 0.05), ('warmup005', 0.05, 0.0)]:
            spec = json.loads((root / f'configs/laya-rtx4090-ce-epochs2-{name}-fp32.json').read_text())
            expected = dict(base['training'], label_smoothing=smoothing)
            self.assertEqual(spec['training'], expected)
            self.assertEqual(spec['optimization']['warmup_ratio'], ratio)
            for key in ['device', 'train_limit', 'validation_limit', 'eval_batch_size']:
                self.assertEqual(spec[key], base[key])
            smoke = json.loads((root / f'configs/laya-rtx4090-smoke-ce-{name}-fp32.json').read_text())
            upstream.TrainConfig(**spec['training']).validate()
            upstream.TrainConfig(**smoke['training']).validate()
            config = smoke['training']
            total = upstream.optimizer_updates(112, config['micro_batch'], config['grad_accum'], config['epochs'])
            if ratio:
                self.assertEqual(math.ceil(total * ratio), 3)


if __name__ == '__main__':
    unittest.main()
