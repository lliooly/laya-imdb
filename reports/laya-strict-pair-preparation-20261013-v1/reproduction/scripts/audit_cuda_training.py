"""Observe a CUDA smoke run without changing its loss, optimizer or scheduler."""
import argparse
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from common import ROOT, file_digest, write_json
import train_laya


def sample_indices(torch, numel, device):
    # Integer arithmetic avoids FP32 rounding an endpoint up to numel for large tensors.
    count = min(16, numel)
    return torch.arange(count, device=device, dtype=torch.int64) * (numel - 1) // max(1, count - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / args.config).read_text())
    effective_limit = (spec.get('sampling') or {}).get('smoke_items', spec['train_limit'])
    if spec['device'] != 'cuda' or effective_limit is None or effective_limit > 128:
        raise ValueError('Audit is limited to small CUDA smoke runs')
    report_dir = ROOT / 'reports' / spec['run_id']
    output_dir = ROOT / 'checkpoints' / spec['run_id']
    if report_dir.exists() or output_dir.exists():
        raise FileExistsError('Use a new run_id; audit never overwrites a run')
    train_laya.prepare_runtime(spec)
    import torch
    import laya.train as upstream
    train_laya.require_device(torch, 'cuda')
    torch.cuda.reset_peak_memory_stats()
    audit = {'state': 'running', 'test_accessed': False, 'optimizer_steps': [],
             'gradient_windows': [], 'losses': [], 'scheduler_steps': 0,
             'checkpoint_reload': None, 'instrumentation_changes_training_math': False,
             'source_sha256': file_digest(Path(__file__))}
    original_step = torch.optim.AdamW.step
    original_clip = torch.nn.utils.clip_grad_norm_
    original_schedule = torch.optim.lr_scheduler.LRScheduler.step
    original_loss = upstream.soft_ce_loss
    original_save = upstream.save_checkpoint

    @wraps(original_loss)
    def loss(*values, **kwargs):
        result = original_loss(*values, **kwargs)
        scalar = float(result.detach())
        if not math.isfinite(scalar):
            raise FloatingPointError('Nonfinite training loss')
        audit['losses'].append(scalar)
        return result

    @wraps(original_clip)
    def clip(parameters, *values, **kwargs):
        parameters = list(parameters)
        norm = original_clip(parameters, *values, **kwargs)
        scalar = float(norm.detach())
        audit['gradient_windows'].append({'norm_before_clipping': scalar if math.isfinite(scalar) else None,
                                          'finite': math.isfinite(scalar)})
        return norm

    @wraps(original_step)
    def step(optimizer, *values, **kwargs):
        # Inspect a deterministic sample in each parameter group; no RNG draws.
        samples = [[(p, sample_indices(torch, p.numel(), p.device)) for p in group['params']]
                   for group in optimizer.param_groups]
        snapshots = [[p.detach().flatten()[indices].clone() for p, indices in group]
                     for group in samples]
        result = original_step(optimizer, *values, **kwargs)
        finite = bool(torch.stack([torch.isfinite(p).all() for group in optimizer.param_groups
                                  for p in group['params']]).all())
        changes = [float(torch.stack([(p.detach().flatten()[indices] - before).abs().max()
                                      for (p, indices), before in zip(group, snapshots[i])]).max())
                   for i, group in enumerate(samples)]
        audit['optimizer_steps'].append({'parameters_finite': finite,
                                         'sample_max_abs_update_by_group': changes})
        if not finite:
            raise FloatingPointError('Nonfinite parameters after optimizer update')
        return result

    @wraps(original_schedule)
    def schedule(scheduler, *values, **kwargs):
        # The scheduler also calls step during initialization.
        initialized = getattr(scheduler, '_step_count', 0) > 0
        result = original_schedule(scheduler, *values, **kwargs)
        if initialized:
            audit['scheduler_steps'] += 1
        return result

    @wraps(original_save)
    def save(model, tokenizer, config, path):
        with torch.no_grad():
            if not bool(torch.stack([torch.isfinite(p.to(torch.float16)).all()
                                    for p in model.state_dict().values()]).all()):
                raise FloatingPointError('Checkpoint would contain nonfinite FP16 weights')
        original_save(model, tokenizer, config, path)
        if Path(path).resolve() != output_dir.resolve():
            return
        # Compare in-memory FP32 inference to actual saved FP16 weights reloaded as FP32.
        rows = upstream.read_data(str(report_dir / 'data/validation.jsonl'))
        items, skipped = upstream.items_from_rows(tokenizer, rows, config['max_len'],
                                                 config['head_max_len'], label_smoothing=0.0)
        if skipped or len(items) != spec['validation_limit']:
            raise ValueError('Reload verification must use every smoke validation item')
        dev = next(model.parameters()).device
        parallel = upstream.uses_parallel_layout(config)
        before = upstream.calibration_records(model, tokenizer, items, dev,
                                               config['max_len'], config['head_max_len'], parallel=parallel)
        loaded, loaded_tokenizer, loaded_config = upstream.load_checkpoint(str(path))
        exact = all(torch.equal(p.detach().half().cpu(), loaded.state_dict()[name].half())
                    for name, p in model.state_dict().items())
        if not exact:
            raise ValueError('Reloaded weights differ from saved FP16 representation')
        loaded.to(dev)
        after_items, after_skipped = upstream.items_from_rows(loaded_tokenizer, rows,
            config['max_len'], config['head_max_len'], label_smoothing=0.0)
        if after_skipped or len(after_items) != len(items):
            raise ValueError('Reload changed usable validation inputs')
        after = upstream.calibration_records(loaded, loaded_tokenizer, after_items, dev,
                                              config['max_len'], config['head_max_len'], parallel=parallel)
        import numpy as np
        delta = max(float(np.abs(a[1] - b[1]).max()) for a, b in zip(before, after))
        same = all(int(a[1].argmax()) == int(b[1].argmax()) for a, b in zip(before, after))
        audit['checkpoint_reload'] = {'weights_match_fp16_representation': exact,
            'validation_items': len(items), 'predictions_match': same,
            'max_abs_logit_delta_fp32_vs_saved_fp16': delta,
            'checkpoint_sha256': file_digest(Path(path) / 'model.safetensors'),
            'config_max_len': loaded_config['max_len']}
        manifest = json.loads((report_dir / 'manifest.json').read_text())
        audit['checkpoint_reload']['validation_predictions'] = []
        for sample_id, item, record in zip(manifest['validation']['ids'], after_items, after):
            prepared = upstream.encode_item(loaded_tokenizer, item, config['max_len'],
                                            config['head_max_len'], parallel=parallel)
            inputs = {key: prepared[key] for key in ('ids', 'markers', 'qtype')}
            encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
            logits = np.asarray(record[1], dtype=float)
            probabilities = np.exp((logits - logits.max()) / loaded_config['temperature'][0])
            probabilities /= probabilities.sum()
            audit['checkpoint_reload']['validation_predictions'].append({
                'sample_id': sample_id, 'input_sha256': hashlib.sha256(encoded).hexdigest(),
                'logits': logits.tolist(), 'probabilities': probabilities.tolist(),
                'pred_label': int(logits.argmax())})
        del loaded

    torch.optim.AdamW.step = step
    torch.nn.utils.clip_grad_norm_ = clip
    torch.optim.lr_scheduler.LRScheduler.step = schedule
    upstream.soft_ce_loss = loss
    upstream.save_checkpoint = save
    started = time.monotonic()
    original_argv = sys.argv
    try:
        sys.argv = ['train_laya.py', '--config', args.config]
        train_laya.main()
        run_manifest = json.loads((report_dir / 'manifest.json').read_text())
        actual_count = len(run_manifest['gradient_train_ids'])
        if spec.get('sampling'):
            sampling_audit = json.loads((report_dir / 'sampling_audit.json').read_text())
            if sampling_audit['state'] != 'completed':
                raise ValueError('Sampling smoke audit did not complete')
            actual_count = sampling_audit['actual_train_count']
        expected = upstream.optimizer_updates(actual_count, spec['training']['micro_batch'],
                                               spec['training']['grad_accum'], spec['training']['epochs'])
        audit['expected_optimizer_steps'] = expected
        audit['passed'] = (len(audit['optimizer_steps']) == expected
            and audit['scheduler_steps'] == expected
            and len(audit['gradient_windows']) == expected
            and all(window['finite'] for window in audit['gradient_windows'])
            and all(all(change > 0 for change in step['sample_max_abs_update_by_group'])
                    for step in audit['optimizer_steps'])
            and audit['checkpoint_reload'] is not None
            and audit['checkpoint_reload']['predictions_match'])
        audit['state'] = 'completed' if audit['passed'] else 'failed_validation'
        if not audit['passed']:
            raise RuntimeError('CUDA audit failed; inspect numerical_audit.json before further training')
    except BaseException as exc:
        audit.update(state='failed', passed=False, error=repr(exc))
        raise
    finally:
        sys.argv = original_argv
        torch.optim.AdamW.step = original_step
        torch.nn.utils.clip_grad_norm_ = original_clip
        torch.optim.lr_scheduler.LRScheduler.step = original_schedule
        upstream.soft_ce_loss = original_loss
        upstream.save_checkpoint = original_save
        audit['elapsed_seconds_with_audit'] = time.monotonic() - started
        try:
            audit['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            audit['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        except RuntimeError as exc:
            audit['memory_query_error'] = repr(exc)
        if report_dir.exists():
            write_json(report_dir / 'numerical_audit.json', audit)


if __name__ == '__main__':
    main()
