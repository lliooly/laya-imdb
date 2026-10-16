"""Input and update journals with verified epoch-prefix recovery."""
import hashlib
import json
from pathlib import Path

import torch

from laya_checkpoint_delivery import canonical, atomic_json
from laya_head_control import expected_inputs, state_digest


class TrainingAudit:
    def __init__(self, upstream, model, tok, items, ids, config, folder, admission=False, prefix=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.plan = iter(expected_inputs(upstream, tok, items, ids, config, upstream.encode_item))
        self.identity = {id(item): sid for item, sid in zip(items, ids, strict=True)}
        self.count, self.updates, self.hash = 0, 0, hashlib.sha256()
        self.expected_updates = upstream.optimizer_updates(len(items), config.micro_batch, 1, config.epochs)
        self.admission, self.latest = admission, {}
        if prefix is not None:
            prefix = Path(prefix)
            with (prefix/'actual_inputs.jsonl').open('rb') as src, (self.folder/'actual_inputs.jsonl').open('xb') as dst:
                for line in src:
                    if json.loads(line) != next(self.plan):
                        raise ValueError('Saved input prefix differs from deterministic plan')
                    dst.write(line); self.hash.update(line); self.count += 1
            with (prefix/'updates.jsonl').open('rb') as src, (self.folder/'updates.jsonl').open('xb') as dst:
                for line in src:
                    entry = json.loads(line)
                    self.updates += 1
                    if entry['update'] != self.updates:
                        raise ValueError('Saved optimizer update prefix is not consecutive')
                    dst.write(line)
            if self.count != len(items) or self.updates != self.expected_updates // 2:
                raise ValueError('Incomplete first-epoch recovery journal')
        self.inputs = (self.folder/'actual_inputs.jsonl').open('ab')
        self.traces = (self.folder/'updates.jsonl').open('ab')

    def input(self, epoch, item, order, encoded):
        actual = {'epoch': epoch + 1, 'sample_id': self.identity[id(item)], 'option_order': order,
                  **{k: encoded[k] for k in ('ids', 'markers', 'qtype', 'target')}}
        if actual != next(self.plan):
            raise ValueError('Actual input differs from fixed paired schedule')
        line = canonical(actual) + b'\n'
        self.inputs.write(line); self.hash.update(line); self.count += 1

    def before_step(self, model, index):
        self.latest = {}
        if index == 1 or index == self.expected_updates:
            for name, group in [('encoder', model.encoder.parameters()),
                                ('head', (p for n, p in model.named_parameters() if not n.startswith('encoder.')))]:
                squared = [p.grad.detach().float().square().sum() for p in group if p.grad is not None]
                norm = float(torch.stack(squared).sum().sqrt()) if squared else 0.
                if not 0 < norm < float('inf'):
                    raise FloatingPointError('Missing finite nonzero ' + name + ' gradient')
                self.latest[name + '_gradient_norm_after_clipping'] = norm
        if self.admission:
            self.latest['gradient_sha256'] = state_digest({n: p.grad for n, p in model.named_parameters() if p.grad is not None})

    def update(self, model, index, rates, loss, ce, norm):
        if index != self.updates + 1:
            raise ValueError('Nonconsecutive optimizer updates')
        entry = {'update': index, 'learning_rates': rates, 'loss': loss, 'unsmoothed_ce': ce,
                 'gradient_norm_before_clipping': norm, **self.latest}
        if self.admission or index == self.expected_updates or index % 512 == 0:
            entry['model_sha256'] = state_digest(model.state_dict())
        self.traces.write(canonical(entry) + b'\n'); self.traces.flush()
        self.updates = index

    def flush(self):
        self.inputs.flush(); self.traces.flush()

    def finish(self):
        self.flush()
        if self.updates != self.expected_updates or next(self.plan, None) is not None:
            raise ValueError('Incomplete training input/update budget')
        value = {'state': 'completed', 'updates': self.updates, 'gradient_exposures': self.count,
                 'input_stream_sha256': self.hash.hexdigest(),
                 'test_accessed': False, 'development_body_read': False}
        atomic_json(self.folder/'training_audit.json', value)
        return value

    def close(self):
        self.inputs.close(); self.traces.close()
