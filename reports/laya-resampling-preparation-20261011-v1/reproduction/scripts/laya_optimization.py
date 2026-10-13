"""Opt-in warmup and update diagnostics for the pinned, single-process trainer."""
from contextlib import contextmanager
from functools import wraps
import json
import math

from common import write_json
from laya_rdrop import rdrop_training, validate_alpha


def validate_optimization(options):
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError('optimization must be an object')
    unknown = set(options) - {'warmup_ratio', 'diagnostics_every', 'rdrop_alpha'}
    if unknown:
        raise ValueError('Unknown optimization fields: ' + ', '.join(sorted(unknown)))
    ratio = options.get('warmup_ratio', 0.0)
    every = options.get('diagnostics_every', 16)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio < 1:
        raise ValueError('warmup_ratio must be a finite number in [0, 1)')
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError('diagnostics_every must be a positive integer')
    result = {'warmup_ratio': float(ratio), 'diagnostics_every': every}
    if 'rdrop_alpha' in options:
        result['rdrop_alpha'] = validate_alpha(options['rdrop_alpha'])
    return result


def make_scheduler(torch, optimizer, total_updates, min_lr, warmup_ratio,
                   original_cosine=None):
    """Rates are for actual optimizer updates, not micro-batches.

    With W warmup updates and U total updates, update j (one-based) uses
    peak*j/W for j<=W, then cosine((j-W)/(U-W)), ending at min_lr on j=U.
    Zero warmup delegates to the untouched upstream scheduler exactly.
    """
    ratio = validate_optimization({'warmup_ratio': warmup_ratio})['warmup_ratio']
    if isinstance(total_updates, bool) or not isinstance(total_updates, int) or total_updates < 1:
        raise ValueError('total_updates must be a positive integer')
    cosine = original_cosine or torch.optim.lr_scheduler.CosineAnnealingLR
    if ratio == 0:
        return cosine(optimizer, T_max=total_updates, eta_min=min_lr)
    warmup_updates = math.ceil(total_updates * ratio)
    if warmup_updates >= total_updates:
        raise ValueError('Warmup must leave at least one cosine update; increase the run budget')
    if not math.isfinite(min_lr) or min_lr < 0 or any(
            not math.isfinite(group['lr']) or group['lr'] < min_lr
            for group in optimizer.param_groups):
        raise ValueError('Warmup requires finite peak learning rates >= min_lr >= 0')

    class WarmupCosine(torch.optim.lr_scheduler.LRScheduler):
        def get_lr(self):
            update = min(self.last_epoch + 1, total_updates)
            if update <= warmup_updates:
                return [peak * update / warmup_updates for peak in self.base_lrs]
            progress = (update - warmup_updates) / (total_updates - warmup_updates)
            factor = (1 + math.cos(math.pi * progress)) / 2
            return [min_lr + (peak - min_lr) * factor for peak in self.base_lrs]

    return WarmupCosine(optimizer)


@contextmanager
def optimize_training(torch, upstream, options, report_dir):
    """Scope temporary hooks to train_model, excluding calibration/evaluation.

    This adapter does not replace the training loop, loss, clipping, optimizer,
    shuffling or checkpointing. Hooks are restored even on interrupted runs.
    """
    options = validate_optimization(options)
    if options is None:
        yield
        return
    original_train = upstream.train_model
    original_items = upstream.items_from_rows

    @wraps(original_items)
    def hard_items(tok, rows, max_len, head_max_len, label_smoothing=0.0):
        # Upstream smooths before splitting calibration. Preserve hard targets
        # here and smooth only the gradient-training subset in train_model.
        return original_items(tok, rows, max_len, head_max_len, label_smoothing=0.0)

    @wraps(original_train)
    def train(model, tok, items, config, device, max_len, head_max_len,
              on_epoch_end=None, parallel=False):
        smoothing = config.label_smoothing
        alpha = options.get('rdrop_alpha', 0.0)
        if alpha and config.loss != 'soft-ce':
            raise ValueError('R-Drop only supports soft-ce training')
        if smoothing:
            items = [dict(item, target=[(1 - smoothing) * value + smoothing / item['k']
                                       for value in item['target']]) for item in items]
        original_cosine = torch.optim.lr_scheduler.CosineAnnealingLR
        original_clip = torch.nn.utils.clip_grad_norm_
        original_step = torch.optim.AdamW.step
        audit = {'state': 'running', 'test_accessed': False, 'optimization': options,
                 'label_smoothing': smoothing, 'label_smoothing_scope': 'gradient_training_only',
                 'completed_optimizer_updates': 0, 'gradient_windows': 0,
                 'clipped_windows': 0, 'max_gradient_norm': 0.0,
                 'gradient_norm_sum': 0.0,
                 'lr_scope': 'Learning rates used by each sampled optimizer update',
                 'sampling': 'First 16, warmup boundary +/-1, every diagnostics_every, and final update'}
        last_gradient = None

        def schedule(optimizer, T_max, eta_min=0.0, **values):
            if values:
                raise ValueError('Unsupported upstream scheduler arguments: ' + str(values))
            audit.update(total_optimizer_updates=T_max,
                         warmup_updates=math.ceil(T_max * options['warmup_ratio']),
                         peak_learning_rates=[group['lr'] for group in optimizer.param_groups],
                         min_lr=eta_min,
                         scheduler=('upstream_cosine' if options['warmup_ratio'] == 0
                                    else 'linear_warmup_cosine'))
            return make_scheduler(torch, optimizer, T_max, eta_min,
                                  options['warmup_ratio'], original_cosine)

        @wraps(original_clip)
        def clip(parameters, max_norm, *values, **clip_kwargs):
            nonlocal last_gradient
            norm = original_clip(parameters, max_norm, *values, **clip_kwargs)
            value = float(norm.detach())
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite gradient norm; stop optimization run')
            last_gradient = {'gradient_norm_before_clipping': value,
                             'clipped': value > max_norm}
            audit['gradient_windows'] += 1
            audit['clipped_windows'] += int(value > max_norm)
            audit['max_gradient_norm'] = max(audit['max_gradient_norm'], value)
            audit['gradient_norm_sum'] += value
            return norm

        with (report_dir / 'optimizer_updates.jsonl').open('x', encoding='utf-8') as log:
            @wraps(original_step)
            def step(optimizer, *values, **step_kwargs):
                rates = [group['lr'] for group in optimizer.param_groups]
                result = original_step(optimizer, *values, **step_kwargs)
                audit['completed_optimizer_updates'] += 1
                update = audit['completed_optimizer_updates']
                warmup = audit.get('warmup_updates', 0)
                if (update <= 16 or update % options['diagnostics_every'] == 0
                        or update in (warmup - 1, warmup, warmup + 1)
                        or update == audit.get('total_optimizer_updates')):
                    log.write(json.dumps({'update': update, 'learning_rates': rates,
                                          **(last_gradient or {})}) + '\n')
                    log.flush()
                return result

            torch.optim.lr_scheduler.CosineAnnealingLR = schedule
            torch.nn.utils.clip_grad_norm_ = clip
            torch.optim.AdamW.step = step
            try:
                with rdrop_training(torch, upstream, alpha, report_dir):
                    result = original_train(model, tok, items, config, device, max_len, head_max_len,
                                            on_epoch_end=on_epoch_end, parallel=parallel)
                if audit['completed_optimizer_updates'] != audit.get('total_optimizer_updates'):
                    raise RuntimeError('Observed optimizer updates differ from scheduler budget')
                audit['state'] = 'completed'
                return result
            except BaseException as exc:
                audit.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                             error=repr(exc))
                raise
            finally:
                torch.optim.lr_scheduler.CosineAnnealingLR = original_cosine
                torch.nn.utils.clip_grad_norm_ = original_clip
                torch.optim.AdamW.step = original_step
                count = audit['gradient_windows']
                audit['mean_gradient_norm'] = audit.pop('gradient_norm_sum') / count if count else None
                audit['clipping_fraction'] = audit['clipped_windows'] / count if count else None
                write_json(report_dir / 'optimization_diagnostics.json', audit)

    upstream.train_model = train
    upstream.items_from_rows = hard_items
    try:
        yield
    finally:
        upstream.train_model = original_train
        upstream.items_from_rows = original_items
