"""Length-aware CE epochs with exact optimizer/RNG continuation and input journals."""
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from laya_head_control import state_digest
from laya_robust_training import capture_rng, restore_rng, cpu_tree, finite_tree

FORMAT = 'laya-context-epoch-resume-v1'


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def setup(model, config, count, device):
    if (config.epochs != 2 or config.grad_accum != 1 or config.amp or config.loss != 'soft-ce'
            or config.freeze_encoder or config.label_smoothing != 0 or count < 1):
        raise ValueError('Require the approved two-epoch, unsmoothed FP32 CE protocol')
    config.validate()
    if config.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.head_checkpointing = bool(config.gradient_checkpointing)
    model.to(device).train()
    optimizer = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if n.startswith('encoder.')], 'lr': config.encoder_lr},
        {'params': [p for n,p in model.named_parameters() if not n.startswith('encoder.')], 'lr': config.head_lr}],
        weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=math.ceil(count/config.micro_batch)*2, eta_min=config.min_lr)
    torch.manual_seed(config.seed)
    return optimizer, scheduler, random.Random(config.seed)


def snapshot(model, optimizer, scheduler, order_rng, device, identity, epoch, count, history, micro_batch):
    return {'format': FORMAT, 'identity': identity, 'next_epoch': epoch+1,
            'updates': math.ceil(count/micro_batch)*(epoch+1), 'exposures': count*(epoch+1),
            'model_state_sha256': state_digest(model.state_dict()), 'history': list(history),
            'optimizer': cpu_tree(optimizer.state_dict()), 'scheduler': scheduler.state_dict(),
            'option_order_rng': order_rng.getstate(), 'rng': capture_rng(device)}


def restore(value, model, optimizer, scheduler, order_rng, device, identity, count, micro_batch, expected_epoch):
    if (value['format'] != FORMAT or value['identity'] != identity
            or value['next_epoch'] != expected_epoch
            or value['updates'] != math.ceil(count/micro_batch)*expected_epoch
            or value['exposures'] != count*expected_epoch or len(value['history']) != expected_epoch
            or value['model_state_sha256'] != state_digest(model.state_dict())):
        raise ValueError('Checkpoint identity, state or complete epoch differs')
    if not finite_tree(value):
        raise FloatingPointError('Nonfinite recovery state')
    optimizer.load_state_dict(value['optimizer']); scheduler.load_state_dict(value['scheduler'])
    order_rng.setstate(value['option_order_rng']); restore_rng(value['rng'], device)
    return list(value['history'])


@dataclass
class Journal:
    folder: Path

    def __post_init__(self):
        self.folder.mkdir(parents=True, exist_ok=False)
        self.inputs = gzip.open(self.folder/'inputs.jsonl.gz', 'xt', compresslevel=1)
        self.steps = (self.folder/'updates.jsonl').open('x')
        self.input_hash, self.schedule_hash = hashlib.sha256(), hashlib.sha256()

    def input(self, epoch, sid, order, value):
        schedule = {'epoch': epoch+1, 'sample_id': sid, 'option_order': order}
        self.schedule_hash.update((canonical(schedule)+'\n').encode())
        line = canonical(dict(schedule, **value))+'\n'
        self.inputs.write(line); self.input_hash.update(line.encode())

    def update(self, value):
        self.steps.write(canonical(value)+'\n'); self.steps.flush()

    def close(self):
        self.inputs.close(); self.steps.close()


def run_epoch(model, tok, items, ids, config, device, cap, epoch, optimizer, scheduler,
              order_rng, journal=None, deadline=float('inf')):
    import laya.train as upstream
    if cap not in (512, 2048) or epoch not in (0, 1) or len(items) != len(ids) or not items:
        raise ValueError('Invalid epoch or input identity')
    identity = {id(item): sid for item,sid in zip(items,ids,strict=True)}
    if len(identity) != len(items) or len(set(ids)) != len(ids):
        raise ValueError('Repeated gradient item or sample ID')
    ordered = list(items); random.Random(config.seed+epoch).shuffle(ordered)
    params = [p for group in optimizer.param_groups for p in group['params']]
    model.train(); optimizer.zero_grad(set_to_none=True)
    total, steps = 0., 0
    started = time.monotonic()
    for start in range(0, len(ordered), config.micro_batch):
        if time.time() >= deadline-120:
            raise TimeoutError('Absolute study compute deadline reached')
        chunk = []
        for item in ordered[start:start+config.micro_batch]:
            order = upstream.draw_option_order(item, order_rng, config.shuffle_options)
            value = upstream.encode_item(tok, item, cap, 192, order, False)
            if len(value['ids']) != len(item['state_ids'])+46:
                raise ValueError('Input length changed after review budget selection')
            if journal:
                journal.input(epoch, identity[id(item)], order, value)
            chunk.append(value)
        batch = upstream.collate_items([chunk], tok.pad_token_id)
        logits = upstream._forward(model, batch, device, False, False)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite logits')
        loss = upstream.soft_ce_loss(logits, batch['target'].to(device), batch['marker_mask'].to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, config.grad_clip, error_if_nonfinite=True)
        rates = [g['lr'] for g in optimizer.param_groups]
        if steps == 0:
            for group in optimizer.param_groups:
                gradients = [p.grad.square().sum() for p in group['params'] if p.grad is not None]
                if not gradients or not 0 < float(torch.stack(gradients).sum()) < float('inf'):
                    raise FloatingPointError('Encoder/head gradient missing or nonfinite')
        optimizer.step()
        if not torch.stack([torch.isfinite(p).all() for p in params]).all():
            raise FloatingPointError('Nonfinite parameters after AdamW')
        scheduler.step(); optimizer.zero_grad(set_to_none=True)
        steps += 1; total += loss.item()
        if journal:
            journal.update({'epoch': epoch+1, 'update': epoch*math.ceil(len(items)/config.micro_batch)+steps,
                            'learning_rates': rates, 'loss': loss.item(), 'gradient_norm': float(norm)})
        if config.log_every and steps % config.log_every == 0:
            print(f'length={cap} epoch={epoch+1} step={steps}/{math.ceil(len(items)/config.micro_batch)} '
                  f'loss={loss.item():.6f} elapsed={time.monotonic()-started:.1f}s', flush=True)
    return total/steps, {'updates': steps, 'exposures': len(items), 'seconds': time.monotonic()-started,
                        'input_sha256': journal.input_hash.hexdigest() if journal else None,
                        'schedule_sha256': journal.schedule_hash.hexdigest() if journal else None}
