"""Fixed CE/GCE experiment; explicit epoch-boundary optimizer and RNG recovery."""
from dataclasses import dataclass
import hashlib
import inspect
import math
import random

import numpy as np
import torch

from laya_head_control import state_digest

ARMS = ('ce', 'mixed')
Q = 0.7
MIXTURE = 0.5
RESUME_FORMAT = 'laya-robust-epoch-resume-v1'


def loss_values(logits, target, mask, arm):
    """Return objective and common unsmoothed CE; only two one-hot slots allowed."""
    if arm not in ARMS:
        raise ValueError('Unknown loss arm')
    if (logits.ndim != 2 or logits.shape[1] != 2 or not logits.shape[0]
            or target.shape != logits.shape or mask.shape != logits.shape
            or mask.dtype != torch.bool or not bool(mask.all())):
        raise ValueError('Require two actual candidate slots per nonempty row')
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(target).all()):
        raise FloatingPointError('Nonfinite logits or labels')
    if not bool(((target == 0) | (target == 1)).all()) or not bool((target.sum(-1) == 1).all()):
        raise ValueError('Require unsmoothed one-hot labels in permuted slot order')
    # Keep the CE path operation-for-operation equal to upstream.soft_ce_loss.
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
    ce = -(target * logp * mask).sum(-1)
    if arm == 'ce':
        loss = ce.mean()
    else:
        log_py = (target * logp).sum(-1)
        gce = -torch.expm1(Q * log_py) / Q
        loss = ((1 - MIXTURE) * ce + MIXTURE * gce).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite training objective')
    return loss, ce.mean()


def capture_rng(device):
    state = np.random.get_state()
    return {'torch_cpu': torch.get_rng_state().clone(),
            'torch_cuda': [x.cpu().clone() for x in torch.cuda.get_rng_state_all()] if device.type == 'cuda' else [],
            'python': random.getstate(),
            'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def restore_rng(state, device):
    torch.set_rng_state(state['torch_cpu'].cpu())
    if device.type == 'cuda':
        if len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device topology changed')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])
    elif state['torch_cuda']:
        raise ValueError('Cannot resume CUDA training on CPU')
    random.setstate(state['python'])
    alg, keys, pos, has_gauss, cached = state['numpy']
    np.random.set_state((alg, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


@dataclass
class EpochContext:
    model: object
    optimizer: object
    scheduler: object
    order_rng: object
    device: object
    identity: dict
    epoch: int
    updates: int
    exposures: int
    history: list

    def snapshot(self):
        return {'format': RESUME_FORMAT, 'identity': self.identity,
                'model_state_sha256': state_digest(self.model.state_dict()),
                'next_epoch': self.epoch + 1, 'updates': self.updates,
                'exposures': self.exposures, 'history': list(self.history),
                'optimizer': cpu_tree(self.optimizer.state_dict()),
                'scheduler': self.scheduler.state_dict(),
                'option_order_rng': self.order_rng.getstate(), 'rng': capture_rng(self.device)}


def read_resume(path, identity, model_sha):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state['format'] != RESUME_FORMAT or state['identity'] != identity
            or state['model_state_sha256'] != model_sha or state['next_epoch'] != 1
            or len(state['history']) != 1 or state['updates'] < 1 or state['exposures'] < 1):
        raise ValueError('Resume identity, model or epoch differs')
    if not finite_tree(state):
        raise FloatingPointError('Nonfinite optimizer or recovery state')
    return state


def train_model(model, tok, items, config, device, arm, identity,
                on_epoch_end, observer=None, resume=None):
    """Pinned Laya FP32 loop with explicit recovery; callback must gate each epoch."""
    import laya.train as upstream
    from laya_resampling import TRAIN_FUNCTION_SHA256
    if hashlib.sha256(inspect.getsource(upstream.train_model).encode()).hexdigest() != TRAIN_FUNCTION_SHA256:
        raise ValueError('Pinned upstream training loop changed')
    config.validate()
    if (not items or config.epochs != 2 or config.grad_accum != 1
            or config.loss != 'soft-ce' or config.amp is not False
            or config.freeze_encoder or config.label_smoothing != 0 or arm not in ARMS
            or not callable(on_epoch_end)):
        raise ValueError('Require approved two-epoch FP32 experiment and an epoch gate')
    checkpointing = bool(config.gradient_checkpointing)
    if checkpointing and hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.head_checkpointing = checkpointing
    model.to(device).train()
    groups = [{'params': [p for n, p in model.named_parameters() if n.startswith('encoder.') and p.requires_grad],
               'lr': config.encoder_lr},
              {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad],
               'lr': config.head_lr}]
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=upstream.optimizer_updates(len(items), config.micro_batch, 1, config.epochs), eta_min=config.min_lr)
    torch.manual_seed(config.seed)
    order_rng = random.Random(config.seed)
    params = [p for g in groups for p in g['params']]
    history, first_epoch, updates, exposures = [], 0, 0, 0
    if resume is not None:
        if resume['identity'] != identity or resume['model_state_sha256'] != state_digest(model.state_dict()):
            raise ValueError('Recovery state is bound to a different model or experiment')
        first_epoch = resume['next_epoch']
        expected = math.ceil(len(items) / config.micro_batch)
        if first_epoch != 1 or resume['updates'] != expected or resume['exposures'] != len(items):
            raise ValueError('Recovery must start at the verified first-epoch boundary')
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        order_rng.setstate(resume['option_order_rng'])
        history, updates, exposures = list(resume['history']), resume['updates'], resume['exposures']
        restore_rng(resume['rng'], device)
    for epoch in range(first_epoch, config.epochs):
        epoch_items = list(items)
        random.Random(config.seed + epoch).shuffle(epoch_items)
        total, n_steps = 0., 0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(epoch_items), config.micro_batch):
            chunk = []
            for item in epoch_items[start:start + config.micro_batch]:
                order = upstream.draw_option_order(item, order_rng, config.shuffle_options)
                encoded = upstream.encode_item(tok, item, 512, 192, order, False)
                if observer is not None:
                    observer.input(epoch, item, order, encoded)
                chunk.append(encoded)
                exposures += 1
            batch = upstream.collate_items([chunk], tok.pad_token_id)
            logits = upstream._forward(model, batch, device, False, False)
            mask, target = batch['marker_mask'].to(device), batch['target'].to(device)
            loss, ce = loss_values(logits, target, mask, arm)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, config.grad_clip, error_if_nonfinite=True)
            rates = [g['lr'] for g in optimizer.param_groups]
            if observer is not None:
                observer.before_step(model, updates + 1)
            optimizer.step()
            if not bool(torch.stack([torch.isfinite(p).all() for p in params]).all()):
                raise FloatingPointError('Nonfinite model parameters after optimizer update')
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if observer is not None:
                observer.update(model, updates, rates, float(loss.detach()), float(ce.detach()), float(norm))
            n_steps += 1
            total += loss.item()
            if config.log_every and n_steps % config.log_every == 0:
                print(f'epoch {epoch + 1}/{config.epochs} step {n_steps} loss {loss.item():.4f}', flush=True)
        mean = total / max(1, n_steps)
        history.append(mean)
        print(f'epoch {epoch + 1}/{config.epochs} mean loss {mean:.4f}', flush=True)
        context = EpochContext(model, optimizer, scheduler, order_rng, device, identity,
                               epoch, updates, exposures, history)
        rng = capture_rng(device)
        on_epoch_end(epoch, mean, context)
        # A receiver must run its own subprocess; archive work cannot perturb dropout.
        after = capture_rng(device)
        if (not torch.equal(rng['torch_cpu'], after['torch_cpu'])
                or any(not torch.equal(a, b) for a, b in zip(rng['torch_cuda'], after['torch_cuda'], strict=True))
                or rng['python'] != after['python'] or rng['numpy'] != after['numpy']):
            raise RuntimeError('Epoch delivery changed the training RNG')
    model.eval()
    return history
