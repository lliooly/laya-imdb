"""Training-only, paired dropout forwards for the pinned single-process loop."""
from contextlib import contextmanager
from functools import wraps
import math

from common import write_json


def validate_alpha(alpha):
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or alpha < 0):
        raise ValueError('rdrop_alpha must be a finite nonnegative number')
    return float(alpha)


def symmetric_kl(torch, first, second, mask):
    """Sum over valid options, average over examples; both branches keep gradients."""
    if first.shape != second.shape or first.shape != mask.shape:
        raise ValueError('R-Drop logits and mask shapes must match')
    if mask.dtype != torch.bool or not bool(mask.any(-1).all()):
        raise ValueError('R-Drop requires a boolean mask with valid options in every row')
    a = torch.log_softmax(first.float().masked_fill(~mask, -torch.inf), -1).masked_fill(~mask, 0)
    b = torch.log_softmax(second.float().masked_fill(~mask, -torch.inf), -1).masked_fill(~mask, 0)
    return (0.5 * (a.exp() - b.exp()) * (a - b) * mask).sum(-1).mean()


@contextmanager
def rdrop_training(torch, upstream, alpha, report_dir):
    alpha = validate_alpha(alpha)
    if alpha == 0:
        yield
        return
    original_forward, original_loss = upstream._forward, upstream.soft_ce_loss
    pending = None
    audit = {'state': 'running', 'test_accessed': False, 'alpha': alpha,
             'paired_batches': 0, 'training_forward_calls': 0,
             'ce_sum': 0.0, 'symmetric_kl_sum': 0.0, 'max_symmetric_kl': 0.0,
             'loss_definition': 'mean(two CE) + alpha * mean(two directional KL)',
             'inference_changes': False}

    @wraps(original_forward)
    def forward(model, batch, device, amp, detach_encoder):
        nonlocal pending
        if not model.training or not torch.is_grad_enabled():
            return original_forward(model, batch, device, amp, detach_encoder)
        if pending is not None:
            raise RuntimeError('R-Drop previous paired forward was not consumed by its loss')
        first = original_forward(model, batch, device, amp, detach_encoder)
        second = original_forward(model, batch, device, amp, detach_encoder)
        pending = (first, second)
        audit['training_forward_calls'] += 2
        return first

    @wraps(original_loss)
    def loss(logits, target, mask):
        nonlocal pending
        if pending is None:
            return original_loss(logits, target, mask)
        first, second = pending
        if logits is not first:
            raise RuntimeError('R-Drop loss does not match its paired forward')
        pending = None
        ce = (original_loss(first, target, mask) + original_loss(second, target, mask)) / 2
        kl = symmetric_kl(torch, first, second, mask)
        result = ce + alpha * kl
        if not bool(torch.isfinite(result.detach()).all()):
            raise FloatingPointError('Nonfinite R-Drop training loss; stop this run')
        ce_value, kl_value = float(ce.detach()), float(kl.detach())
        audit['paired_batches'] += 1
        audit['ce_sum'] += ce_value
        audit['symmetric_kl_sum'] += kl_value
        audit['max_symmetric_kl'] = max(audit['max_symmetric_kl'], kl_value)
        return result

    upstream._forward, upstream.soft_ce_loss = forward, loss
    try:
        yield
        if pending is not None:
            raise RuntimeError('R-Drop training ended with an unconsumed paired forward')
        audit['state'] = 'completed'
    except BaseException as exc:
        audit.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     error=repr(exc))
        raise
    finally:
        pending = None
        upstream._forward, upstream.soft_ce_loss = original_forward, original_loss
        count = audit['paired_batches']
        audit['mean_ce'] = audit.pop('ce_sum') / count if count else None
        audit['mean_symmetric_kl'] = audit.pop('symmetric_kl_sum') / count if count else None
        write_json(report_dir / 'rdrop_diagnostics.json', audit)
