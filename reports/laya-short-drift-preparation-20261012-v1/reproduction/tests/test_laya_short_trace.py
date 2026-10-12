"""Diagnostic instrumentation must preserve parameters and RNG, and restore hooks."""
from pathlib import Path
import sys
import json
from contextlib import ExitStack
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from laya_short_trace import DiagnosticStop, short_trace, tensor_hash


class TraceTests(unittest.TestCase):
    def test_real_upstream_guards_scheduler_and_control_clone(self):
        import torch
        import laya.train as upstream
        from common import ROOT
        from laya_optimization import optimize_training
        from cuda_numerics import guard_training
        from laya_resampling import resample_training, CANDIDATE_SHA256
        manifest = json.loads((ROOT / 'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json').read_text())
        candidate_file = 'reports/laya-train-hardcase-review-v2/frozen-candidates.json'
        hard = json.loads((ROOT / candidate_file).read_text())['rows']
        source = {row['id']: row['label'] for row in
                  map(json.loads, (ROOT / 'data/processed/train.jsonl').read_text().splitlines())}
        selected = {row['sample_id'] for row in hard}
        for label in (0, 1):
            needed = 64 - sum(source[sample_id] == label for sample_id in selected)
            selected.update([sample_id for sample_id in manifest['gradient_train_ids']
                             if sample_id not in selected and source[sample_id] == label][:needed])
        ids = [sample_id for sample_id in manifest['gradient_train_ids'] if sample_id in selected]
        labels = {sample_id: source[sample_id] for sample_id in ids}
        items = [{'sample_id': sample_id, 'ids': [1, 2 + labels[sample_id]], 'markers': [0, 1],
                  'qtype': 0, 'q': {'t': 'choice'}, 'k': 2,
                  'target': [1. - labels[sample_id], float(labels[sample_id])]} for sample_id in ids]
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Embedding(8, 4)
                self.head = torch.nn.Linear(4, 1)
                self.dropout = torch.nn.Dropout(.2)
            def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
                return self.head(self.dropout(self.encoder(input_ids))).squeeze(-1).gather(1, marker_pos), None
        config = upstream.TrainConfig(epochs=2, micro_batch=8, grad_accum=1, encoder_lr=.02,
                                      head_lr=.04, min_lr=.001, loss='soft-ce', amp=False,
                                      gradient_checkpointing=False, log_every=0, label_smoothing=.05, seed=43)
        traces = []
        for mode in (None, 'control'):
            torch.manual_seed(17)
            model = Toy()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                with self.assertRaises(DiagnosticStop):
                    with patch.object(upstream, 'encode_item', lambda tok, item, *a, **kw: item), ExitStack() as stack:
                        stack.enter_context(short_trace(torch, upstream, path / 'trace'))
                        stack.enter_context(guard_training(torch, upstream))
                        if mode:
                            stack.enter_context(resample_training(upstream, {'mode': mode,
                                'candidate_file': candidate_file, 'candidate_sha256': CANDIDATE_SHA256}, ids, labels, path))
                        stack.enter_context(optimize_training(torch, upstream, {'warmup_ratio': .05}, path))
                        upstream.train_model(model, SimpleNamespace(pad_token_id=0), items, config,
                                             torch.device('cpu'), 512, 192)
                diagnostics = json.loads((path / 'optimization_diagnostics.json').read_text())
                self.assertEqual(diagnostics['completed_optimizer_updates'], 16)
                traces.append((path / 'trace/updates.jsonl').read_text())
        self.assertEqual(traces[0], traces[1])

    def test_trace_preserves_updates_and_rng_and_stops_before_next_forward(self):
        import torch
        torch.set_num_threads(1)
        def forward(model, batch, device, *args):
            return model(batch)
        def loss(logits, target, mask=None):
            return (logits - target).square().mean()
        upstream = SimpleNamespace(_forward=forward, soft_ce_loss=loss)
        outputs = []
        for traced in (False, True, True):
            torch.manual_seed(47)
            model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.3))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            batch = torch.ones(2, 3)
            target = torch.zeros(2, 2)
            with tempfile.TemporaryDirectory() as directory:
                def run():
                    for _ in range(4 if traced else 3):
                        logits = upstream._forward(model, batch, 'cpu')
                        value = upstream.soft_ce_loss(logits, target)
                        value.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                if traced:
                    with self.assertRaises(DiagnosticStop):
                        with short_trace(torch, upstream, Path(directory) / 'trace', limit=3) as state:
                            run()
                    self.assertEqual(state['state'], 'diagnostic_completed')
                    self.assertEqual(state['completed_updates'], 3)
                    self.assertEqual(state['stopped_before_forward_update'], 4)
                else:
                    run()
                outputs.append(([tensor_hash(value) for value in model.state_dict().values()],
                                tensor_hash(torch.get_rng_state())))
                self.assertIs(upstream._forward, forward)
                self.assertIs(upstream.soft_ce_loss, loss)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_early_failure_restores_hooks_and_is_not_completed(self):
        import torch
        forward = lambda *args: None
        loss = lambda *args: None
        upstream = SimpleNamespace(_forward=forward, soft_ce_loss=loss)
        original_step = torch.optim.AdamW.step
        original_clip = torch.nn.utils.clip_grad_norm_
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'before reaching'):
                with short_trace(torch, upstream, Path(directory) / 'trace', limit=3) as state:
                    pass
            self.assertEqual(state['state'], 'failed')
        self.assertIs(upstream._forward, forward)
        self.assertIs(upstream.soft_ce_loss, loss)
        self.assertIs(torch.optim.AdamW.step, original_step)
        self.assertIs(torch.nn.utils.clip_grad_norm_, original_clip)


if __name__ == '__main__':
    unittest.main()
