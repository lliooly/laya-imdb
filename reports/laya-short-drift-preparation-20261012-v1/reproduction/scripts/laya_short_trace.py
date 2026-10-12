"""Diagnostic hooks for existing FP32 training; never changes the loss or RNG."""
from contextlib import contextmanager
from functools import wraps
import hashlib
import json


class DiagnosticStop(BaseException):
    """Expected boundary after the requested completed optimizer updates."""


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    metadata = json.dumps([str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
    return hashlib.sha256(metadata + value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def tree_hash(value):
    import torch
    if isinstance(value, torch.Tensor):
        return tensor_hash(value)
    if isinstance(value, dict):
        value = {str(key): tree_hash(item) for key, item in sorted(value.items())}
    elif isinstance(value, (list, tuple)):
        value = [tree_hash(item) for item in value]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def model_hash(model, gradients=False):
    values = {}
    source = model.named_parameters() if gradients else model.state_dict().items()
    for name, tensor in source:
        value = tensor.grad if gradients else tensor
        values[name] = None if value is None else tensor_hash(value)
    return tree_hash(values)


def rng_hash(torch, device):
    value = {'cpu': tensor_hash(torch.get_rng_state())}
    if str(device).startswith('cuda'):
        value['cuda_all'] = [tensor_hash(state) for state in torch.cuda.get_rng_state_all()]
    return value


@contextmanager
def short_trace(torch, upstream, directory, limit=16):
    """Install before existing guard/optimizer contexts; restore even on stop."""
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError('Diagnostic limit must be between 1 and 16')
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    active = {}
    original_forward = upstream._forward
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = torch.optim.AdamW.step
    original_loss = upstream.soft_ce_loss
    state = {'state': 'running', 'limit': limit, 'completed_updates': 0,
             'training_performed': True, 'test_accessed': False,
             'instrumentation_changes_timing_and_cuda_synchronization': True}

    def save():
        (directory / 'status.json').write_text(json.dumps(state, indent=2) + '\n')

    @wraps(original_forward)
    def forward(model, batch, device, *args, **kwargs):
        if not model.training:
            return original_forward(model, batch, device, *args, **kwargs)
        if len(rows) == limit:
            raise DiagnosticStop()
        if active.get('pending'):
            raise ValueError('Only one forward per update and grad_accum=1 are supported')
        active.update(model=model, device=device, pending=True)
        record = {'update': len(rows) + 1, 'batch_sha256': tree_hash(batch),
                  'rng_before_forward': rng_hash(torch, device)}
        if not rows:
            record['initial_model_sha256'] = model_hash(model)
        logits = original_forward(model, batch, device, *args, **kwargs)
        record.update(logits_sha256=tensor_hash(logits), rng_after_forward=rng_hash(torch, device))
        active['record'] = record
        return logits

    @wraps(original_loss)
    def loss(*args, **kwargs):
        result = original_loss(*args, **kwargs)
        if active.get('pending'):
            active['record'].update(loss=float(result.detach()), loss_sha256=tensor_hash(result))
        return result

    @wraps(original_clip)
    def clip(parameters, *args, **kwargs):
        if active.get('pending'):
            active['record']['gradient_before_clip_sha256'] = model_hash(active['model'], True)
        result = original_clip(parameters, *args, **kwargs)
        if active.get('pending'):
            active['record']['gradient_norm_before_clipping'] = float(result.detach())
        return result

    @wraps(original_step)
    def step(optimizer, *args, **kwargs):
        if not active.get('pending'):
            raise ValueError('Optimizer update without recorded training forward')
        record = active['record']
        record['gradient_after_clip_sha256'] = model_hash(active['model'], True)
        record['learning_rates'] = [group['lr'] for group in optimizer.param_groups]
        result = original_step(optimizer, *args, **kwargs)
        record.update(model_after_update_sha256=model_hash(active['model']),
                      rng_after_update=rng_hash(torch, active['device']))
        rows.append(record)
        with (directory / 'updates.jsonl').open('a') as log:
            log.write(json.dumps(record, sort_keys=True) + '\n')
        active['pending'] = False
        state['completed_updates'] = len(rows)
        save()
        return result

    upstream._forward, upstream.soft_ce_loss = forward, loss
    torch.nn.utils.clip_grad_norm_, torch.optim.AdamW.step = clip, step
    save()
    try:
        yield state
        raise ValueError('Training ended before reaching the diagnostic stop boundary')
    except DiagnosticStop:
        state.update(state='diagnostic_completed', stopped_before_forward_update=limit + 1)
        save()
        raise
    except BaseException as error:
        state.update(state='failed', error=repr(error))
        save()
        raise
    finally:
        upstream._forward, upstream.soft_ce_loss = original_forward, original_loss
        torch.nn.utils.clip_grad_norm_, torch.optim.AdamW.step = original_clip, original_step
