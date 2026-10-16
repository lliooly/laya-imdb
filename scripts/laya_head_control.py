"""CE-only Laya heads, explicit initialization and lossless FP32 checkpoints."""
from contextlib import contextmanager
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from common import file_digest, write_json


FORMAT = 'laya-ce-head-control-v1'
KINDS = ('original', 'linear')
EXPECTED_PARAMETERS = {'original': 421029889, 'linear': 394782720}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(canonical([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class HeadControl(nn.Module):
    def __init__(self, encoder, kind, seed):
        super().__init__()
        if kind not in KINDS or type(seed) is not int:
            raise ValueError('Unknown head or seed')
        self.encoder, self.head_kind, self.training_seed = encoder, kind, seed
        self.head_checkpointing = False
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1000003 + seed)
            if kind == 'original':
                from laya.common import DecisionModel
                reference = DecisionModel(encoder, head_layers=2, n_act=2, dropout=.1)
                self.head, self.type_emb, self.scorer = reference.head, reference.type_emb, reference.scorer
            else:
                self.head = None
                self.scorer = nn.Linear(encoder.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
                detach_encoder=False, position_ids=None, option_ids=None):
        if position_ids is not None or option_ids is not None:
            raise ValueError('Only the fixed sequential layout is allowed')
        if marker_pos.ndim != 2 or marker_pos.shape[1] != 2 or not marker_mask.all():
            raise ValueError('Require exactly two actual candidate markers')
        if (marker_pos < 0).any() or (marker_pos >= input_ids.shape[1]).any():
            raise ValueError('Invalid candidate positions')
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        if self.head_kind == 'original':
            h = h + self.type_emb(qtype)[:, None, :]
            for layer in self.head.layers:
                if self.head_checkpointing and self.training and torch.is_grad_enabled():
                    h = checkpoint(layer, h, src_key_padding_mask=~attention_mask.bool(), use_reentrant=False)
                else:
                    h = layer(h, src_key_padding_mask=~attention_mask.bool())
        positions = marker_pos[:, :, None].expand(-1, -1, h.shape[-1])
        logits = self.scorer(torch.gather(h, 1, positions)).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4), None


def encoder_from_directory(directory):
    from transformers import AutoConfig, AutoModel
    from laya.common import _apply_rope_config, _no_init_weights
    cfg = AutoConfig.from_pretrained(str(directory), local_files_only=True)
    _apply_rope_config(cfg)
    if hasattr(cfg, 'reference_compile'):
        cfg.reference_compile = False
    with torch.random.fork_rng(devices=[]), _no_init_weights():
        encoder = AutoModel.from_config(cfg, attn_implementation='sdpa')
    return encoder.float()


def from_source(directory, kind, seed):
    from safetensors import safe_open
    directory = Path(directory)
    encoder = encoder_from_directory(directory/'encoder')
    with safe_open(directory/'model.safetensors', framework='pt', device='cpu') as f:
        state = {k.removeprefix('encoder.'): f.get_tensor(k)
                 for k in f.keys() if k.startswith('encoder.')}
    encoder.load_state_dict(state, strict=True)
    del state
    dropout = {k: v for k, v in encoder.config.to_dict().items() if 'dropout' in k}
    if not dropout or any(v != 0 for v in dropout.values()):
        raise ValueError('Source encoder dropout differs from the registered protocol')
    if any(isinstance(m, nn.Dropout) and m.p != 0 for m in encoder.modules()):
        raise ValueError('An actual encoder dropout module is nonzero')
    model = HeadControl(encoder, kind, seed).float()
    if sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMETERS[kind]:
        raise ValueError('Unexpected trainable parameter count')
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError('All CE parameters must be trainable')
    return model


def save_fp32(model, directory):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
           for v in state.values()):
        raise FloatingPointError('Only finite FP32 checkpoint tensors are allowed')
    state_sha = state_digest(state)
    save_file(state, str(directory/'model.safetensors'))
    model.encoder.config.save_pretrained(directory/'encoder')
    metadata = {'format': FORMAT, 'head_kind': model.head_kind,
                'training_seed': model.training_seed, 'state_sha256': state_sha,
                'encoder_config_sha256': file_digest(directory/'encoder/config.json'),
                'weights_sha256': file_digest(directory/'model.safetensors'),
                'dtype': 'float32', 'parameter_count': sum(p.numel() for p in model.parameters())}
    write_json(directory/'head_control.json', metadata)
    return metadata


def load_fp32(directory):
    from safetensors.torch import load_file
    directory = Path(directory)
    metadata = json.loads((directory/'head_control.json').read_text())
    if metadata['format'] != FORMAT or metadata['dtype'] != 'float32':
        raise ValueError('Unknown checkpoint format or precision')
    if file_digest(directory/'model.safetensors') != metadata['weights_sha256']:
        raise ValueError('Checkpoint bytes changed')
    if file_digest(directory/'encoder/config.json') != metadata['encoder_config_sha256']:
        raise ValueError('Saved encoder configuration changed')
    state = load_file(str(directory/'model.safetensors'), device='cpu')
    if any(v.is_floating_point() and v.dtype != torch.float32 for v in state.values()):
        raise ValueError('Checkpoint contains quantized parameters')
    if state_digest(state) != metadata['state_sha256']:
        raise ValueError('Checkpoint tensor state changed')
    model = HeadControl(encoder_from_directory(directory/'encoder'),
                        metadata['head_kind'], metadata['training_seed']).float()
    model.load_state_dict(state, strict=True)
    return model, metadata


def fit_temperature(margins, labels):
    m, y = np.asarray(margins, dtype=np.float64), np.asarray(labels)
    if m.ndim != 1 or not len(m) or y.shape != m.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(m).all():
        raise ValueError('Expected finite aligned binary calibration data')
    def derivative(a):
        return float(np.mean(m*(.5*(1+np.tanh(a*m/2))-y)))
    lo, hi = .2, 2.
    if np.all(m == 0):
        alpha = 1.
    elif derivative(lo) >= 0:
        alpha = lo
    elif derivative(hi) <= 0:
        alpha = hi
    else:
        for _ in range(80):
            mid = (lo+hi)/2
            if derivative(mid) > 0:
                hi = mid
            else:
                lo = mid
        alpha = (lo+hi)/2
    return {'temperature': 1/alpha, 'inverse_temperature': alpha,
            'bounds': [.5, 5.], 'fit_count': len(m),
            'raw_nll': float(np.mean(np.logaddexp(0, m)-y*m)),
            'calibrated_nll': float(np.mean(np.logaddexp(0, alpha*m)-y*alpha*m)),
            'boundary_hit': alpha in (.2, 2.)}


def logit_metrics(labels, logits, truncated, temperature=1.):
    """Compute NLL from original logits, without probability clipping caps."""
    from comparison_metrics import metrics
    y, z, mask = np.asarray(labels), np.asarray(logits, dtype=np.float64), np.asarray(truncated, dtype=bool)
    if z.shape != (len(y), 2) or mask.shape != y.shape or not len(y) or not np.isfinite(z).all() or not np.isin(y, [0, 1]).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Expected aligned finite binary prediction data')
    margin = z[:, 1]-z[:, 0]
    return {name: metrics(y[keep], margin[keep], temperature) if keep.any() else {'count': 0}
            for name, keep in [('all', np.ones(len(y), dtype=bool)), ('truncated', mask), ('untruncated', ~mask)]}


def expected_inputs(upstream, tok, items, ids, config, encode):
    identity = {id(item): sid for item, sid in zip(items, ids, strict=True)}
    if len(identity) != len(items) or len(set(ids)) != len(ids):
        raise ValueError('Duplicate gradient items or IDs')
    rng = random.Random(config.seed)
    for epoch in range(config.epochs):
        order = list(items)
        random.Random(config.seed+epoch).shuffle(order)
        for item in order:
            option = upstream.draw_option_order(item, rng, config.shuffle_options)
            prepared = encode(tok, item, 512, 192, option, False)
            yield {'epoch': epoch+1, 'sample_id': identity[id(item)],
                   'option_order': option, **{k: prepared[k] for k in ('ids', 'markers', 'qtype', 'target')}}


@contextmanager
def audit_training(upstream, model, tok, items, ids, config, report, admission):
    """Observe the pinned loop; never replace its updates, loss or RNG streams."""
    from laya_resampling import TRAIN_FUNCTION_SHA256
    if hashlib.sha256(inspect.getsource(upstream.train_model).encode()).hexdigest() != TRAIN_FUNCTION_SHA256:
        raise ValueError('Upstream training loop differs from the pinned source')
    report = Path(report)
    encode, loss_fn = upstream.encode_item, upstream.soft_ce_loss
    clip_fn, step_fn = torch.nn.utils.clip_grad_norm_, torch.optim.AdamW.step
    plan = iter(expected_inputs(upstream, tok, items, ids, config, encode))
    counter, update = 0, 0
    input_hash = hashlib.sha256()
    latest = {}
    identity = {id(item): sid for item, sid in zip(items, ids, strict=True)}
    expected_updates = upstream.optimizer_updates(len(items), config.micro_batch, config.grad_accum, config.epochs)
    with gzip.open(report/'actual_inputs.jsonl.gz', 'xt', compresslevel=1) as inputs, (report/'updates.jsonl').open('x') as traces:
        def encode_checked(item_tok, item, max_len, head_max_len, option_order=None, parallel=False):
            nonlocal counter
            if (max_len, head_max_len, parallel) != (512, 192, False):
                raise ValueError('Input settings differ')
            value = encode(item_tok, item, max_len, head_max_len, option_order, parallel)
            expected = next(plan)
            actual = {'epoch': counter//len(items)+1, 'sample_id': identity[id(item)],
                      'option_order': option_order, **{k: value[k] for k in ('ids', 'markers', 'qtype', 'target')}}
            if actual != expected:
                raise ValueError('Actual gradient input differs from the fixed paired schedule')
            line = canonical(actual)+'\n'
            inputs.write(line); input_hash.update(line.encode()); counter += 1
            return value

        def loss_checked(*args, **kwargs):
            value = loss_fn(*args, **kwargs)
            latest['loss'] = float(value.detach())
            return value

        def clip_checked(parameters, max_norm, *args, **kwargs):
            parameters = list(parameters)
            for name in ('encoder', 'head'):
                latest.pop(name+'_gradient_norm_after_clipping', None)
            norm = clip_fn(parameters, max_norm, *args, **kwargs)
            latest['gradient_norm_before_clipping'] = float(norm.detach())
            if not math.isfinite(latest['gradient_norm_before_clipping']):
                raise FloatingPointError('Nonfinite gradient norm')
            if update == 0 or update+1 == expected_updates:
                for name, group in [('encoder', model.encoder.parameters()),
                                    ('head', (p for n, p in model.named_parameters() if not n.startswith('encoder.')))]:
                    squared = [p.grad.detach().float().square().sum() for p in group if p.grad is not None]
                    value = float(torch.stack(squared).sum().sqrt()) if squared else 0.
                    if not math.isfinite(value) or value <= 0:
                        raise FloatingPointError(f'{name} has no finite nonzero gradient')
                    latest[name+'_gradient_norm_after_clipping'] = value
            if admission:
                latest['gradient_sha256'] = state_digest({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
            return norm

        def step_checked(optimizer, *args, **kwargs):
            nonlocal update
            rates = [g['lr'] for g in optimizer.param_groups]
            value = step_fn(optimizer, *args, **kwargs)
            update += 1
            entry = {'update': update, 'learning_rates': rates, **latest}
            if admission or update == expected_updates or update % 512 == 0:
                entry['model_sha256'] = state_digest(model.state_dict())
            if admission:
                entry['torch_rng_sha256'] = state_digest({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()})
            traces.write(canonical(entry)+'\n');traces.flush()
            return value

        upstream.encode_item, upstream.soft_ce_loss = encode_checked, loss_checked
        torch.nn.utils.clip_grad_norm_, torch.optim.AdamW.step = clip_checked, step_checked
        try:
            yield
            if update != expected_updates or counter != len(items)*config.epochs or next(plan, None) is not None:
                raise ValueError('Incomplete gradient exposure or update budget')
            write_json(report/'training_audit.json', {'state': 'completed', 'updates': update,
                       'gradient_exposures': counter, 'input_stream_sha256': input_hash.hexdigest(),
                       'test_accessed': False, 'development_body_read': False})
        finally:
            upstream.encode_item, upstream.soft_ce_loss = encode, loss_fn
            torch.nn.utils.clip_grad_norm_, torch.optim.AdamW.step = clip_fn, step_fn
