"""Frozen single DeBERTa study, full epoch bundles, admission and evaluation."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

from common import file_digest,read_jsonl,write_json
from deberta_inputs import ROOT,STUDY,VOLUME,BASE,read,training_rows,encode

SECONDS=5400
CAP_USD=5


def config(root=ROOT,evaluation=False):
    from deberta_inputs import RECIPE
    c=read(root/'configs'/f'{STUDY}.json')
    if (c['study_id']!=STUDY or c['recipe']!=RECIPE or c['maximum_gpu_seconds']!=SECONDS
            or c['reserve_usd']!=CAP_USD or c['fusion_new_weight']!=.5):
        raise ValueError('Approved scientific scope changed')
    for rel,sha in c['source_sha256'].items():
        if not evaluation and rel in c['evaluation_only_files']:continue
        if file_digest(root/rel)!=sha:raise ValueError('Bound source changed: '+rel)
    return c


def identity():
    return dict(study_id=STUDY,run_id='deberta-v3-large-s42',seed=42,max_len=512,
                contract_sha256=file_digest(ROOT/'configs'/f'{STUDY}.json'))


def training_inputs():
    config();rows,split=training_rows(ROOT)
    folder=ROOT/'artifacts'/STUDY;stats=read(folder/'input-summary.json')
    items=list(read_jsonl(folder/'train-tokens.jsonl'))
    if ([x['id'] for x in items]!=split['gradient_train_ids'] or len(items)!=22100
            or any(x['label']!=rows[x['id']]['label'] or not 1<=len(x['input_ids'])<=512 for x in items)
            or stats['train_tokens_sha256']!=file_digest(folder/'train-tokens.jsonl')):
        raise ValueError('Gradient cache differs from source members/labels')
    return items,stats


def bundle(epoch):return ROOT/'checkpoints'/STUDY/f'epoch-{epoch:03d}'


def seal(folder,stage):
    files=[dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=file_digest(p))
           for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='bundle.json']
    m=dict(format='deberta-bundle-v1',identity=identity(),stage=stage,
           bundle=str(folder.relative_to(ROOT)),files=files)
    write_json(folder/'bundle.json',m);return m


def check_bundle(folder):
    from laya_checkpoint_delivery import within
    m=read(folder/'bundle.json')
    if m['format']!='deberta-bundle-v1' or m['identity']['study_id']!=STUDY:
        raise ValueError('Unexpected bundle study')
    if len({x['path'] for x in m['files']})!=len(m['files']):raise ValueError('Repeated bundle file')
    for f in m['files']:
        p=within(folder,f['path'])
        if p.stat().st_size!=f['bytes'] or file_digest(p)!=f['sha256']:raise ValueError('Bundle corrupt: '+f['path'])
    return m


def tree_digest(value):
    import torch
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            x=x.detach().cpu().contiguous();h.update(str((str(x.dtype),list(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(str(type(x)).encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def save(model,opt,sched,device,progress,items,pad,folder,stage):
    import torch
    from deberta_training import snapshot,predict
    from laya_robust_training import finite_tree
    if folder.exists():raise FileExistsError('Sealed bundle already exists')
    temp=folder.with_name(folder.name+'.writing');temp.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(temp,safe_serialization=True)
    state=snapshot(model,opt,sched,device,identity(),progress)
    if not finite_tree(state):raise FloatingPointError('Nonfinite saved optimizer/RNG')
    torch.save(state,temp/'optimizer_rng.pt');del state
    write_json(temp/'training.json',dict(identity=identity(),progress=progress,test_accessed=False))
    # Reference rows always come from gradient training, never development/calibration.
    reference=items[:2];logits,_=predict(model,reference,pad,device)
    write_json(temp/'reference.json',dict(items=reference,pad_token_id=pad,logits=logits))
    assets=temp/'assets';assets.mkdir()
    shutil.copytree(ROOT/'artifacts'/STUDY/'tokenizer',assets/'tokenizer')
    shutil.copyfile(ROOT/'artifacts'/STUDY/'input-summary.json',assets/'input-summary.json')
    if stage.startswith('epoch-'):
        shutil.copyfile(ROOT/'reports'/STUDY/'audit'/(stage+'.jsonl'),assets/'updates.jsonl')
    c=config()
    for rel in c['source_sha256']:
        if rel.startswith(('scripts/','environments/','configs/')):
            target=assets/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    shutil.copyfile(ROOT/'configs'/f'{STUDY}.json',assets/'contract.json')
    from importlib.metadata import version
    write_json(assets/'runtime.json',dict(packages={p:version(p) for p in
        ['torch','transformers','numpy','safetensors','tokenizers','sentencepiece','protobuf']},
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        amp=False,tf32=torch.backends.cuda.matmul.allow_tf32,deterministic=torch.are_deterministic_algorithms_enabled()))
    m=seal(temp,stage);m['bundle']=str(folder.relative_to(ROOT));write_json(temp/'bundle.json',m)
    temp.rename(folder);return m


def verify(folder,output):
    import torch
    from deberta_training import load_saved,predict
    from laya_head_control import state_digest
    from laya_robust_training import finite_tree
    if folder.resolve() in output.resolve().parents:raise ValueError('Receipt cannot modify immutable bundle')
    torch.set_num_threads(4);m=check_bundle(folder)
    model=load_saved(folder);sha=state_digest(model.state_dict())
    state=torch.load(folder/'optimizer_rng.pt',map_location='cpu',weights_only=True)
    if (state['model_state_sha256']!=sha or state['identity']!=m['identity']
            or state['progress']!=read(folder/'training.json')['progress'] or not finite_tree(state)):
        raise ValueError('Saved model and training state differ')
    del state;gc.collect()
    ref=read(folder/'reference.json');actual,_=predict(model,ref['items'],ref['pad_token_id'],torch.device('cpu'))
    actual=torch.tensor(actual);expected=torch.tensor(ref['logits'])
    torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)
    result=dict(strict_load=True,optimizer_rng_finite=True,state_sha256=sha,
                max_abs_logit_difference=float((actual-expected).abs().max()),atol=1e-4,rtol=1e-4,
                device='cpu',bundle_sha256=file_digest(folder/'bundle.json'))
    write_json(output,result);print(json.dumps(result),flush=True)


def pilot(deadline):
    import torch
    from deberta_training import load_initial,setup_model,optimizer_for,update,predict,snapshot,seed_all
    from laya_head_control import state_digest
    items,stats=training_inputs();device=torch.device('cuda');seed_all()
    report=ROOT/'reports'/STUDY;report.mkdir(parents=True,exist_ok=True)
    model=setup_model(load_initial(ROOT/config()['model_dir'],report/'loading-info.json'),device)
    opt,sched=optimizer_for(model);seed_all()
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    selected=stats['pilot_representative'];groups=[[items[i] for i in selected[j:j+32]] for j in range(0,128,32)]
    first=update(model,groups[0],stats['pad_token_id'],opt,sched,device,deadline)
    initial_progress=dict(next_epoch=0,updates=1,exposures=32,history=[],initial_state_sha256=None)
    before=time.monotonic()
    save(model,opt,sched,device,initial_progress,items,stats['pad_token_id'],
         ROOT/'checkpoints'/STUDY/'pilot-boundary','pilot-boundary')
    save_seconds=time.monotonic()-before
    second=update(model,groups[1],stats['pad_token_id'],opt,sched,device,deadline)
    expected=snapshot(model,opt,sched,device,identity(),dict(updates=2))
    expected_sha=tree_digest(expected);del expected;gc.collect()
    timing=[second]+[update(model,g,stats['pad_token_id'],opt,sched,device,deadline) for g in groups[2:]]
    stress=update(model,[items[i] for i in stats['pilot_stress']],stats['pad_token_id'],opt,sched,device,deadline)
    _,inference=predict(model,[items[i] for i in selected],stats['pad_token_id'],device,deadline)
    result=dict(study_id=STUDY,state='pilot_continuous_complete',warmup=first,
        representative=timing,stress=stress,reference_group_indices=selected[32:64],
        continuous_after_second_update_sha256=expected_sha,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),save_seconds=save_seconds,
        inference_rows=128,inference_seconds=inference,pilot_gpu_seconds=time.monotonic()-start,
        test_accessed=False,development_accessed=False)
    write_json(report/'pilot-continuous.json',result);print(json.dumps(result),flush=True)


def pilot_resume(deadline):
    import torch
    from deberta_training import load_saved,setup_model,optimizer_for,restore,update,snapshot,seed_all
    items,stats=training_inputs();device=torch.device('cuda');seed_all()
    folder=ROOT/'checkpoints'/STUDY/'pilot-boundary';check_bundle(folder)
    model=setup_model(load_saved(folder),device);opt,sched=optimizer_for(model)
    state=torch.load(folder/'optimizer_rng.pt',map_location='cpu',weights_only=True)
    restore(state,model,opt,sched,device,identity());del state;gc.collect()
    result=read(ROOT/'reports'/STUDY/'pilot-continuous.json')
    update(model,[items[i] for i in result['reference_group_indices']],stats['pad_token_id'],opt,sched,device,deadline)
    actual=tree_digest(snapshot(model,opt,sched,device,identity(),dict(updates=2)))
    if actual!=result['continuous_after_second_update_sha256']:raise ValueError('CUDA resume not bitwise equal')
    result.update(state='pilot_resume_passed',exact_model_optimizer_scheduler_rng_resume=True,
                  resumed_after_second_update_sha256=actual)
    write_json(ROOT/'reports'/STUDY/'pilot-result.json',result)
    print('CUDA_RESUME_EXACT',actual,flush=True)


def train(epoch,deadline):
    import torch
    from deberta_training import (seed_all,load_initial,load_saved,setup_model,optimizer_for,
                                  restore,groups_for,update)
    from laya_head_control import state_digest
    seed_all();items,stats=training_inputs();device=torch.device('cuda');report=ROOT/'reports'/STUDY
    if epoch==0:
        model=load_initial(ROOT/config()['model_dir']);history=[];initial=state_digest(model.state_dict())
    elif epoch==1:
        previous=bundle(1);check_bundle(previous)
        receipt=read(report/'cloud-delivery/epoch-001.json')
        if not receipt['verification']['strict_load'] or receipt['bundle_sha256']!=file_digest(previous/'bundle.json'):
            raise ValueError('Epoch1 not independently accepted')
        model=load_saved(previous);progress=read(previous/'training.json')['progress']
        if progress['next_epoch']!=1 or progress['updates']!=691 or progress['exposures']!=22100:
            raise ValueError('Epoch1 counts differ')
        initial=progress['initial_state_sha256'];history=progress['history']
    else:raise ValueError('Only epochs0,1')
    model=setup_model(model,device);opt,sched=optimizer_for(model)
    if epoch:
        state=torch.load(bundle(1)/'optimizer_rng.pt',map_location='cpu',weights_only=True)
        restore(state,model,opt,sched,device,identity());del state;gc.collect()
    else:seed_all()
    journal=report/'audit'/f'epoch-{epoch+1:03d}.jsonl';journal.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();groups=groups_for(len(items),epoch);total=0.
    with journal.open('x') as out:
        for step,indices in enumerate(groups):
            value=update(model,[items[i] for i in indices],stats['pad_token_id'],opt,sched,device,deadline)
            total+=value['loss']*len(indices)
            out.write(json.dumps(dict(epoch=epoch+1,update=epoch*691+step+1,
                                      ids=[items[i]['id'] for i in indices],**value))+'\n');out.flush()
            if (step+1)%16==0:print(f'epoch={epoch+1} update={step+1}/691 loss={value["loss"]:.6f} elapsed={time.monotonic()-start:.1f}s',flush=True)
    history.append(dict(epoch=epoch+1,updates=len(groups),exposures=len(items),
                        mean_loss=total/len(items),seconds=time.monotonic()-start,journal_sha256=file_digest(journal)))
    progress=dict(next_epoch=epoch+1,updates=691*(epoch+1),exposures=22100*(epoch+1),
                  initial_state_sha256=initial,history=history)
    save(model,opt,sched,device,progress,items,stats['pad_token_id'],bundle(epoch+1),f'epoch-{epoch+1:03d}')


def evaluate(deadline):
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    from deberta_training import load_saved,predict,seed_all
    from comparison_metrics import metrics,sigmoid,fit_temperature
    from analyze_laya_length_router import comparison,wilson
    config(evaluation=True);report=ROOT/'reports'/STUDY;folder=bundle(2)
    m=check_bundle(folder);r=read(report/'cloud-delivery/epoch-002.json')
    progress=read(folder/'training.json')['progress']
    if (progress['next_epoch']!=2 or progress['updates']!=1382 or progress['exposures']!=44200
            or not r['verification']['strict_load'] or r['bundle_sha256']!=file_digest(folder/'bundle.json')):
        raise ValueError('Fixed epoch2 not frozen/accepted')
    write_json(report/'final-frozen.json',dict(identity=identity(),bundle_sha256=file_digest(folder/'bundle.json'),test_accessed=False))
    tok=AutoTokenizer.from_pretrained(str(folder/'assets/tokenizer'),local_files_only=True,use_fast=False)
    rows,split=training_rows(ROOT)
    dev=list(read_jsonl(ROOT/'data/processed/validation.jsonl'))
    if [x['id'] for x in dev]!=split['validation']['ids']:raise ValueError('Development identity changed')
    model=load_saved(folder).to('cuda');seed_all();out=report/'evaluation'
    if out.exists():raise FileExistsError('Evaluation already exists')
    out.mkdir();data={};runtime={}
    for name,source in [('calibration',[rows[sid] for sid in split['calibration_ids']]),('development',dev)]:
        items=[dict(id=x['id'],label=x['label'],**encode(tok,x['text'])) for x in source]
        z,seconds=predict(model,items,tok.pad_token_id,torch.device('cuda'),deadline)
        data[name]=dict(sample_ids=[x['id'] for x in source],true_labels=[x['label'] for x in source],
                        review_token_lengths=[x['review_tokens'] for x in items],
                        truncated=[x['truncated'] for x in items],logits=z,test_accessed=False)
        runtime[name]=seconds;write_json(out/f'{name}.json',data[name])
    calibration=data['calibration'];d=data['development'];z=np.asarray(d['logits']);margin=z[:,1]-z[:,0]
    cz=np.asarray(calibration['logits']);temperature=fit_temperature(cz[:,1]-cz[:,0],calibration['true_labels'],bounds=(.5,5))
    old=read(ROOT/'reports/laya-long-fusion-bc-20261017-v1/predictions.json')
    if d['sample_ids']!=old['sample_ids'] or d['true_labels']!=old['true_labels']:raise ValueError('Frozen main predictions misaligned')
    y=np.asarray(d['true_labels']);p=np.asarray(old['positive_probabilities']['old_primary']);q=sigmoid(margin/temperature)
    full2048=np.asarray(old['positive_probabilities']['new2048']);blend=(p+q)/2
    logit=lambda v:np.log(np.clip(v,1e-15,1-1e-15))-np.log1p(-np.clip(v,1e-15,1-1e-15))
    margins=dict(frozen_main=logit(p),deberta=margin/temperature,fusion_50_50=logit(blend))
    if int(((p>.5)==y).sum())!=2424:raise ValueError('96.96% frozen main failed')
    long=np.asarray(old['long_mask']);common=((p>.5)!=y)&((full2048>.5)!=y)
    if int(common.sum())!=63:raise ValueError('63 common-error set failed')
    groups={}
    for name,mask in [('all',np.ones(len(y),dtype=bool)),('laya_truncated',long),('laya_complete',~long)]:
        groups[name]={}
        for label,a in margins.items():
            value=metrics(y[mask],a[mask]);value['accuracy_wilson95']=wilson(value['correct'],value['count'])
            groups[name][label]=value
        groups[name]['deberta_vs_main']=comparison(y[mask],margins['frozen_main'][mask],margins['deberta'][mask])
        groups[name]['fusion_vs_main']=comparison(y[mask],margins['frozen_main'][mask],margins['fusion_50_50'][mask])
    result=dict(study_id=STUDY,state='evaluated',temperature=temperature,groups=groups,
        deberta_uncalibrated=metrics(y,margin),old_common_errors=63,
        deberta_fixes_on_old_common_errors=int((common&((q>.5)==y)).sum()),
        deberta_truncated_count=sum(d['truncated']),laya_truncated_count=int(long.sum()),
        inference_seconds=runtime,test_accessed=False,selection='fixed_epoch2_seed42',
        limitations=['single_seed','reused_development','heterogeneous_pipeline_not_encoder_only_ablation',
                     'conditional_bootstrap_not_training_variance'])
    write_json(out/'summary.json',result)
    write_json(out/'predictions.json',dict(sample_ids=d['sample_ids'],true_labels=d['true_labels'],
        positive_probabilities=dict(frozen_main=p.tolist(),deberta=q.tolist(),fusion_50_50=blend.tolist()),test_accessed=False))
    lines=['# DeBERTa 异构互补：固定两轮结果','','原始22100训练、400校准、2500开发；未访问官方测试。','',
           '| 方案 | 正确 / 2500 | Accuracy | Macro-F1 | AUROC | NLL | ECE |','|---|---:|---:|---:|---:|---:|---:|']
    for name in margins:
        g=groups['all'][name];lines.append(f'| {name} | {g["correct"]} | {g["accuracy"]*100:.2f}% | {g["macro_f1"]:.6f} | {g["auroc"]:.6f} | {g["nll"]:.6f} | {g["ece_15_equal_width_top_label"]:.6f} |')
    for label in ['deberta_vs_main','fusion_vs_main']:
        v=groups['all'][label];ci=[x*100 for x in v['delta_accuracy_conditional_bootstrap95']]
        lines+=['',f'{label}：修复{v["b_only_correct"]}，退化{v["a_only_correct"]}；Δ={v["delta_accuracy"]*100:+.2f} pp，条件95%区间[{ci[0]:+.2f}, {ci[1]:+.2f}] pp，精确McNemar p={v["mcnemar_exact_two_sided_p"]:.6g}。']
    lines+=['','单种子且开发集被多轮研究使用；条件区间不含训练方差。完整分组、共错修复和成本见 summary.json。不搜索权重，不自动替换部署。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');seal(out,'evaluated')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','pilot-resume','train','evaluate','verify'])
    p.add_argument('--epoch',type=int);p.add_argument('--deadline',type=float,default=float('inf'))
    p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='verify':verify(a.bundle,a.output)
    elif a.mode=='pilot':pilot(a.deadline)
    elif a.mode=='pilot-resume':pilot_resume(a.deadline)
    elif a.mode=='train':train(a.epoch,a.deadline)
    else:evaluate(a.deadline)


if __name__=='__main__':main()
