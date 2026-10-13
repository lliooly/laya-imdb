"""Fail-fast CUDA training guards; restore all upstream functions on exit."""
from contextlib import contextmanager
from functools import wraps


@contextmanager
def guard_training(torch, upstream):
    originals = {name: getattr(upstream, name) for name in ('soft_ce_loss', 'rlcd_loss')}
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = torch.optim.AdamW.step

    def checked_loss(function):
        @wraps(function)
        def evaluate(*args, **kwargs):
            value = function(*args, **kwargs)
            if not bool(torch.isfinite(value.detach()).all()):
                raise FloatingPointError('Nonfinite CUDA training loss; stop this run')
            return value
        return evaluate

    @wraps(original_clip)
    def clip(parameters, *args, **kwargs):
        # Stop before optimizer/scheduler advancement, including AMP overflow.
        kwargs['error_if_nonfinite'] = True
        return original_clip(parameters, *args, **kwargs)

    @wraps(original_step)
    def step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        params = [p for group in optimizer.param_groups for p in group['params']]
        if not bool(torch.stack([torch.isfinite(p).all() for p in params]).all()):
            raise FloatingPointError('Nonfinite CUDA parameters after optimizer update; stop this run')
        return result

    for name, function in originals.items():
        setattr(upstream, name, checked_loss(function))
    torch.nn.utils.clip_grad_norm_ = clip
    torch.optim.AdamW.step = step
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(upstream, name, function)
        torch.nn.utils.clip_grad_norm_ = original_clip
        torch.optim.AdamW.step = original_step
