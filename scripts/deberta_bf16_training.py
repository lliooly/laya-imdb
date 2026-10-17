"""BF16 forward only, with the frozen FP32 optimizer, master weights and RNG."""
from contextlib import nullcontext
import time

import torch
import deberta_training as fp32

from deberta_training import (seed_all,collate,optimizer_for,groups_for,load_initial,load_saved)

RECIPE=dict(fp32.RECIPE,precision='bfloat16')


def autocast_context(device,reference=False):
    if device.type!='cuda' or reference:return nullcontext()
    return torch.autocast('cuda',dtype=torch.bfloat16)


def setup_model(model,device):
    if device.type=='cuda' and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native BF16 unavailable; no precision fallback is permitted')
    return fp32.setup_model(model,device)


def update(model,items,pad,opt,sched,device,deadline=float('inf'),micro=8):
    if not 0<len(items)<=32:raise ValueError('One update requires1..32 examples')
    model.train();opt.zero_grad(set_to_none=True)
    total=0.;started=time.monotonic()
    if device.type=='cuda':torch.cuda.synchronize()
    for start in range(0,len(items),micro):
        if time.time()>deadline-120:raise TimeoutError('Insufficient time for update and safe shutdown')
        part=items[start:start+micro];batch=collate(part,pad,device)
        with autocast_context(device):
            output=model(**batch).logits
        logits=output.float()
        if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite BF16 logits')
        y=torch.tensor([x['label'] for x in part],device=device)
        loss=torch.nn.functional.cross_entropy(logits,y,reduction='sum')/len(items)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
        loss.backward();total+=loss.item()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    head=[p.grad.square().sum() for n,p in model.named_parameters()
          if n.startswith('classifier.') and p.grad is not None]
    if head and not 0<float(torch.stack(head).sum())<float('inf'):
        raise FloatingPointError('Missing classifier gradient')
    rate=opt.param_groups[0]['lr'];opt.step();sched.step();opt.zero_grad(set_to_none=True)
    if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
        raise FloatingPointError('Nonfinite parameters after update')
    if device.type=='cuda':torch.cuda.synchronize()
    return dict(loss=total,gradient_norm=float(norm),learning_rate=rate,
                samples=len(items),seconds=time.monotonic()-started)


def predict(model,items,pad,device,deadline=float('inf'),batch_size=8):
    model.eval();values=[];started=time.monotonic()
    reference=getattr(model,'_fp32_storage_reference',False)
    with torch.inference_mode():
        for i in range(0,len(items),batch_size):
            if time.time()>deadline-60:raise TimeoutError('Inference deadline')
            with autocast_context(device,reference):
                output=model(**collate(items[i:i+batch_size],pad,device)).logits
            logits=output.float()
            if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite inference logits')
            values.extend(logits.cpu().tolist())
    if device.type=='cuda':torch.cuda.synchronize()
    return values,time.monotonic()-started


def snapshot(model,opt,sched,device,identity,progress):
    if any(p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('All persisted master parameters must remain FP32')
    state=fp32.snapshot(model,opt,sched,device,identity,progress)
    state['compute_precision']='bfloat16';state['master_precision']='float32'
    return state


def restore(state,model,opt,sched,device,identity):
    if state.get('compute_precision')!='bfloat16' or state.get('master_precision')!='float32':
        raise ValueError('Resume precision contract differs')
    return fp32.restore(state,model,opt,sched,device,identity)
