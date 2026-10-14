"""Real label-free FP32 inference with a previously frozen probability mixture."""
import json
from pathlib import Path
import time

import numpy as np

from common import file_digest
from export_laya_posthoc import input_digest, prepare_rules
from laya_probability_fusion import apply_gated_mixture

MODEL_NAMES = ('baseline-s43', 'combined-s42')


def synchronize(torch, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def empty_cache(torch, device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def tokenizer_semantics(path):
    value = json.loads(Path(path).read_text())
    value.pop('padding', None)
    value.pop('truncation', None)
    return value


class FrozenFusionPredictor:
    """Two real checkpoints, fixed native temperatures and one persisted coefficient.

    CPU is supported for small-model unit tests. Production CLI only admits CUDA/MPS.
    Labels are absent from this API; placeholder targets never enter model.forward.
    """
    def __init__(self, root, checkpoint_root, assets, frozen_fit, device, batch_size=8):
        import torch
        import laya.train as upstream
        if tuple(assets) != MODEL_NAMES or batch_size < 1:
            raise ValueError('Require the fixed ordered model pair and positive batch size')
        coefficient = frozen_fit['mixture']['coefficient']
        if (frozen_fit.get('primary_models') != list(MODEL_NAMES)
                or frozen_fit.get('validation_used_for_fitting') is not False
                or not np.isfinite(coefficient) or not 0 <= coefficient <= 1):
            raise ValueError('Invalid frozen coefficient or model identity')
        self.torch, self.upstream = torch, upstream
        self.root, self.checkpoint_root = Path(root), Path(checkpoint_root)
        self.assets, self.coefficient = assets, coefficient
        self.device, self.batch_size = torch.device(device), batch_size
        self.models, self.tokenizers, self.configs, self.load_seconds = {}, {}, {}, {}
        self.verify_assets()
        original = None
        for name in MODEL_NAMES:
            folder = self.folder(name)
            semantics = tokenizer_semantics(folder/'tokenizer/tokenizer.json')
            if original is not None and semantics != original:
                raise ValueError('Tokenizers differ between frozen models')
            original = semantics

    def folder(self, name):
        value = Path(self.assets[name]['directory'])
        path = self.root/self.checkpoint_root/value
        if (value.is_absolute() or len(value.parts) != 1 or value.name in ('.', '..')
                or not path.resolve().is_relative_to(self.root.resolve())):
            raise ValueError('Checkpoint must stay inside the repository')
        return path

    def verify_assets(self):
        for name in MODEL_NAMES:
            hashes = self.assets[name]['sha256']
            if not {'model.safetensors', 'rl_agent_config.json', 'encoder/config.json',
                    'tokenizer/tokenizer.json'} <= set(hashes):
                raise ValueError('Bind complete model assets')
            for rel, expected in hashes.items():
                p = Path(rel)
                if p.is_absolute() or '..' in p.parts or file_digest(self.folder(name)/p) != expected:
                    raise ValueError(f'Model asset SHA256 mismatch: {name}/{rel}')

    def load(self, name):
        if name in self.models:
            return
        started = time.perf_counter()
        model, tok, cfg = self.upstream.load_checkpoint(str(self.folder(name)))
        if ((cfg['max_len'], cfg['head_max_len']) != (512, 192)
                or self.upstream.uses_parallel_layout(cfg) or cfg.get('temperature_by_options')
                or not np.isfinite(cfg['temperature'][0]) or cfg['temperature'][0] <= 0):
            raise ValueError('Require fixed sequential context and native scalar temperature')
        model.float().eval().to(self.device)
        if any(not self.torch.isfinite(p).all().item() for p in model.parameters()):
            raise FloatingPointError('Nonfinite model parameters')
        if any(p.device.type != self.device.type or p.dtype != self.torch.float32
               or (self.device.index is not None and p.device.index != self.device.index)
               for p in model.parameters()):
            raise RuntimeError('Model device/dtype fallback is forbidden')
        synchronize(self.torch, self.device)
        self.models[name], self.tokenizers[name], self.configs[name] = model, tok, cfg
        self.load_seconds[name] = time.perf_counter()-started

    def _infer(self, name, items, indices, stats, ids, reuse=None):
        torch, upstream = self.torch, self.upstream
        tok, cfg, model = self.tokenizers[name], self.configs[name], self.models[name]
        result = [dict(r, reused_head_prediction=True) for r in reuse] if reuse is not None else [None]*len(items)
        with torch.inference_mode():
            for start in range(0, len(indices), self.batch_size):
                ii = indices[start:start+self.batch_size]
                prepared = [upstream.encode_item(tok, items[i], 512, 192, [0, 1], parallel=False) for i in ii]
                batch = upstream.collate_items([prepared], tok.pad_token_id)
                z = upstream._forward(model, batch, self.device, False, False).cpu().numpy()
                if z.shape != (len(ii), 2) or not np.isfinite(z).all():
                    raise FloatingPointError('Invalid real model logits')
                for j, i in enumerate(ii):
                    logits = z[j].astype(np.float64)
                    p = np.exp((logits-logits.max())/float(cfg['temperature'][0]))
                    p /= p.sum()
                    if not np.isfinite(p).all():
                        raise FloatingPointError('Invalid native probabilities')
                    result[i] = {'sample_id': ids[i], 'input_sha256': input_digest(prepared[j]),
                                 'logits': logits.tolist(), 'probabilities': p.tolist(),
                                 'pred_label': int(logits.argmax()), **stats[i],
                                 'reused_head_prediction': False}
        return result

    def predict(self, texts, sample_ids=None, mode='fusion'):
        if (mode not in ('fusion', 'baseline_head_tail') or not isinstance(texts, list)
                or not texts or any(not isinstance(t, str) or not t.strip() for t in texts)):
            raise ValueError('Require nonempty review texts and a fixed inference mode')
        ids = sample_ids if sample_ids is not None else [f'request/{i}' for i in range(len(texts))]
        if len(ids) != len(texts) or len(set(ids)) != len(ids):
            raise ValueError('Require unique aligned request IDs')
        names = MODEL_NAMES if mode == 'fusion' else MODEL_NAMES[:1]
        for name in names:
            if name not in self.models:
                raise RuntimeError('Explicitly load the required models before predicting')
        # The upstream helper attaches targets, but forward consumes only encoded inputs.
        placeholders = [{'id': sid, 'text': text, 'label': 0} for sid, text in zip(ids, texts)]
        head, tail, stats, _, _ = prepare_rules(self.tokenizers[MODEL_NAMES[0]], placeholders)
        all_indices = list(range(len(texts)))
        changed = [i for i, row in enumerate(stats) if row['truncated']]
        mask = np.asarray([s['truncated'] for s in stats], dtype=bool)
        traces, forwards = {}, {}
        if mode == 'baseline_head_tail':
            rows = self._infer(names[0], tail, all_indices, stats, ids)
            return {'sample_ids': ids, 'stats': stats, 'models': {names[0]: {'head_tail': rows}},
                    'primary': [r['probabilities'] for r in rows], 'forward_rows': len(texts)}
        for name in names:
            h = self._infer(name, head, all_indices, stats, ids)
            t = self._infer(name, tail, changed, stats, ids, reuse=h)
            traces[name] = {'head': h, 'head_tail': t}
            forwards[name] = {'head': len(texts), 'head_tail': len(changed)}
        args = [np.asarray([r['probabilities'] for r in traces[n][v]])
                for n in names for v in ('head', 'head_tail')]
        primary = apply_gated_mixture(*args, mask, self.coefficient)
        equal = apply_gated_mixture(*args, mask, .5)
        return {'sample_ids': ids, 'stats': stats, 'models': traces,
                'primary': primary.tolist(), 'equal': equal.tolist(),
                'baseline_head_tail': args[1].tolist(),
                'forward_rows': sum(sum(v.values()) for v in forwards.values()),
                'forward_rows_by_model': forwards}
