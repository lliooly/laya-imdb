"""FP32 sequence classification with sample-weighted accumulation and exact resume."""
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from common import file_digest
from laya_head_control import state_digest
from laya_robust_training import capture_rng, restore_rng, cpu_tree, finite_tree
from deberta_inputs import encode,RECIPE


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def collate(items, pad, device):
    width=max(len(x['input_ids']) for x in items)
    ids=torch.full((len(items),width),pad,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids)
    for i,x in enumerate(items):
        n=len(x['input_ids']); ids[i,:n]=torch.tensor(x['input_ids'],device=device); mask[i,:n]=1
    return dict(input_ids=ids,attention_mask=mask)


def optimizer_for(model, recipe=RECIPE):
    excluded=set()
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.LayerNorm):
            excluded.update(f'{name}.{n}' for n,_ in module.named_parameters(recurse=False))
    groups=[[],[]]
    for name,p in model.named_parameters():
        if p.requires_grad: groups[int(name.endswith('.bias') or name in excluded)].append(p)
    opt=torch.optim.AdamW([{'params':groups[0],'weight_decay':recipe['weight_decay']},
                           {'params':groups[1],'weight_decay':0.}],
                          lr=recipe['learning_rate'],betas=tuple(recipe['betas']),
                          eps=recipe['adam_epsilon'],foreach=False)
    def factor(step):
        warm,total=recipe['warmup_steps'],recipe['total_steps']
        return min((step+1)/warm,max(0.,(total-step)/(total-warm)))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,factor)


def groups_for(count, epoch, seed=42):
    indices=list(range(count)); random.Random(seed+epoch).shuffle(indices)
    return [indices[i:i+32] for i in range(0,count,32)]


def update(model, items, pad, opt, sched, device, deadline=float('inf'), micro=8):
    if not 0<len(items)<=32: raise ValueError('One update requires 1..32 examples')
    model.train(); opt.zero_grad(set_to_none=True)
    total=0.; started=time.monotonic()
    if device.type=='cuda': torch.cuda.synchronize()
    for start in range(0,len(items),micro):
        if time.time()>deadline-120: raise TimeoutError('Insufficient time for update and safe shutdown')
        part=items[start:start+micro];batch=collate(part,pad,device)
        logits=model(**batch).logits.float()
        if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite logits')
        y=torch.tensor([x['label'] for x in part],device=device)
        loss=torch.nn.functional.cross_entropy(logits,y,reduction='sum')/len(items)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
        loss.backward();total+=loss.item()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    # Check the new classification head really receives a finite nonzero gradient.
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


def load_initial(directory, loading_path=None):
    from transformers import AutoModelForSequenceClassification
    seed_all()
    model,info=AutoModelForSequenceClassification.from_pretrained(
        str(directory),local_files_only=True,num_labels=2,
        id2label={0:'neg',1:'pos'},label2id={'neg':0,'pos':1},
        dtype=torch.float32,output_loading_info=True,attn_implementation='eager')
    allowed=('classifier.','pooler.')
    if any(not k.startswith(allowed) for k in info['missing_keys']) or info['mismatched_keys'] or info['error_msgs']:
        raise ValueError('Unexpected missing or mismatched encoder weights: '+str(info))
    # Pretraining-only prediction heads are allowed to be discarded, never encoder weights.
    for k in info['unexpected_keys']:
        unused_position=(k=='deberta.embeddings.position_embeddings.weight'
                         and not model.config.position_biased_input)
        if not unused_position and not k.startswith(('lm_predictions.','mask_predictions.')):
            raise ValueError('Unexpected pretrained parameters: '+k)
    if loading_path:
        from common import write_json
        write_json(loading_path,info)
    return model.float()


def load_saved(folder):
    from transformers import AutoConfig,AutoModelForSequenceClassification
    from safetensors.torch import load_file
    cfg=AutoConfig.from_pretrained(str(folder),local_files_only=True)
    model=AutoModelForSequenceClassification.from_config(cfg,attn_implementation='eager').float()
    model.load_state_dict(load_file(str(folder/'model.safetensors')),strict=True)
    return model


def setup_model(model,device):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model.to(device)


def snapshot(model,opt,sched,device,identity,progress):
    return dict(format='deberta-fp32-resume-v1',identity=identity,progress=progress,
                model_state_sha256=state_digest(model.state_dict()),
                optimizer=cpu_tree(opt.state_dict()),scheduler=sched.state_dict(),rng=capture_rng(device))


def restore(state,model,opt,sched,device,identity):
    if (state['format']!='deberta-fp32-resume-v1' or state['identity']!=identity
            or state['model_state_sha256']!=state_digest(model.state_dict()) or not finite_tree(state)):
        raise ValueError('Recovery state differs from sealed model/contract')
    opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
    restore_rng(state['rng'],device)
    return state['progress']


def predict(model,items,pad,device,deadline=float('inf'),batch_size=8):
    model.eval(); values=[];started=time.monotonic()
    with torch.inference_mode():
        for i in range(0,len(items),batch_size):
            if time.time()>deadline-60:raise TimeoutError('Inference deadline')
            logits=model(**collate(items[i:i+batch_size],pad,device)).logits.float()
            if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Nonfinite inference logits')
            values.extend(logits.cpu().tolist())
    if device.type=='cuda':torch.cuda.synchronize()
    return values,time.monotonic()-started
