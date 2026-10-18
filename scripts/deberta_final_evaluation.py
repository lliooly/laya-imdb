"""Frozen local CPU test evaluation. No training, cloud compute, fitting or test selection."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from common import ROOT,file_digest,read_jsonl,write_json
from laya_checkpoint_delivery import atomic_json
from deberta_inputs import encode,BASE
from comparison_metrics import metrics,sigmoid,paired
from run_laya_risk_coverage import load_inputs,wilson

TRAIN='deberta-heterogeneous-bf16-budget7-s42-20261017-v1'
RUN='deberta-laya-final-cpu-evaluation-20261018-v1'
OUT=ROOT/'reports'/RUN
CONTRACT=ROOT/'configs'/f'{RUN}.json'
OLD=ROOT/'reports/laya-frozen-official-test-20261014-v1'
MODEL=ROOT/'checkpoints'/TRAIN/'epoch-002'


def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def relative(p):return str(Path(p).relative_to(ROOT))


def check():
    c=read(CONTRACT)
    if c['fusion_new_weight']!=.5 or c['temperature']!=1.1374971945707235 or c['device']!='cpu' or c['dtype']!='float32':
        raise ValueError('Frozen scientific scope differs')
    for rel,sha in c['sources_sha256'].items():
        if file_digest(ROOT/rel)!=sha:raise ValueError('Frozen source changed: '+rel)
    return c


def freeze():
    if CONTRACT.exists() or OUT.exists():raise FileExistsError('New run only; explicit resume uses existing contract')
    local=read(ROOT/'reports'/TRAIN/'local-delivery.json')
    if not local['local_delivery_verified'] or not local['bundles']['epoch-002']['local_delivery_verified']:
        raise ValueError('Final model not completely delivered')
    progress=read(MODEL/'training.json')['progress']
    if progress['updates']!=1382 or progress['exposures']!=44200 or progress['next_epoch']!=2:raise ValueError('Wrong final checkpoint')
    split=read(ROOT/'reports/split_manifest.json')
    paths=[Path(__file__),ROOT/'scripts/deberta_final_analysis.py',ROOT/'scripts/deberta_training.py',
        ROOT/'scripts/deberta_inputs.py',ROOT/'scripts/comparison_metrics.py',ROOT/'scripts/run_laya_risk_coverage.py',
        ROOT/'scripts/laya_checkpoint_delivery.py',ROOT/'scripts/common.py',ROOT/'scripts/laya_head_control.py',
        ROOT/'scripts/laya_robust_training.py',ROOT/'scripts/laya_probability_fusion.py',ROOT/'scripts/laya_decision_views.py',
        ROOT/'docs/superpowers/specs/2026-10-18-deberta-final-evaluation-design.md',
        ROOT/'tests/test_deberta_final_evaluation.py',ROOT/'uv.lock',ROOT/'environments/deberta-prep/uv.lock',
        ROOT/'environments/deberta-prep/pyproject.toml',ROOT/'reports'/BASE.removeprefix('reports/'),
        ROOT/'reports/split_manifest.json',ROOT/'reports'/TRAIN/'evaluation/summary.json',
        ROOT/'reports'/TRAIN/'evaluation/calibration.json',ROOT/'reports'/TRAIN/'evaluation/development.json',
        ROOT/'reports/laya-probability-fusion-s43-s42-20261014-v1/frozen-fit.json',
        OLD/'test-actual.json',OLD/'summary.json',ROOT/'data/processed/train.jsonl',ROOT/'data/processed/validation.jsonl',
        MODEL/'config.json',MODEL/'model.safetensors',MODEL/'bundle.json',MODEL/'reference.json']
    paths+=list((MODEL/'assets/tokenizer').glob('*'))
    c=dict(run_id=RUN,frozen_utc=now(),authorization='User requested frozen50:50 full official test and offline47-case analysis; noGPU',
        training_study_id=TRAIN,checkpoint='epoch-002',fusion_new_weight=.5,temperature=1.1374971945707235,
        device='cpu',dtype='float32',threads=12,batch_size=8,chunk_rows=64,
        ordering='stable increasing token length; restore official source order',new_fitting=False,
        primary='fusion_50_50',primary_reference='original_laya_main',
        engineering_gate={'calibration_count':400,'development_source_prefix':128,'max_probability_difference':.1,
                          'mean_probability_difference':.005,'max_class_disagreement_fraction':.01,'storage_atol':1e-4,'storage_rtol':1e-4},
        test=dict(path='data/processed/test.jsonl',sha256=split['jsonl_sha256']['test'],ids=split['ids']['test'],count=25000),
        sources_sha256={relative(p):file_digest(p) for p in paths if p.is_file()},
        backend_limitation='Development main report uses CUDA BF16 DeBERTa cache; formal test DeBERTa uses fixed localCPU FP32',
        paired_bootstrap_replicates=20000,paired_bootstrap_seed=20261018,
        case_labels_modified=False,training_performed=False,default_deployment_changed=False,
        test_previously_used_in_project=True,official_train_test_overlap_unique_texts=123)
    write_json(CONTRACT,c);OUT.mkdir();atomic_json(OUT/'status.json',dict(state='frozen_before_new_test',test_accessed=False,contract_sha256=file_digest(CONTRACT),updated_utc=now()))
    print('FROZEN',file_digest(CONTRACT),flush=True)


def prepare(split):
    c=check()
    if split=='test':
        if not read(OUT/'admission.json')['passed']:raise ValueError('Engineering admission required')
        gate=read(OUT/'test-gate.json')
        for rel,sha in gate['source_sha256'].items():
            if file_digest(ROOT/rel)!=sha:raise ValueError('Pre-test gate changed')
        # Mark access before any new official-test body read. Historical cached baseline was already evaluated.
        atomic_json(OUT/'status.json',dict(state='test_tokenizing',test_accessed=True,test_opened_utc=now(),updated_utc=now(),contract_sha256=file_digest(CONTRACT)))
        if file_digest(ROOT/c['test']['path'])!=c['test']['sha256']:raise ValueError('Official test changed')
        rows=list(read_jsonl(ROOT/c['test']['path']));ids=c['test']['ids']
    else:
        manifest=read(ROOT/BASE)
        if split=='calibration':
            all_rows={r['id']:r for r in read_jsonl(ROOT/'data/processed/train.jsonl')};ids=manifest['calibration_ids'];rows=[all_rows[i] for i in ids]
        elif split=='admission-development':
            rows=list(read_jsonl(ROOT/'data/processed/validation.jsonl'))[:128];ids=manifest['validation']['ids'][:128]
        else:raise ValueError('Unknown split')
    if [r['id'] for r in rows]!=ids or len(set(ids))!=len(ids):raise ValueError('Input IDs differ')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(MODEL/'assets/tokenizer'),local_files_only=True,use_fast=False)
    records=[]
    for i,row in enumerate(rows):
        if hashlib.sha256(row['text'].encode()).hexdigest()!=row['text_sha256']:raise ValueError('Text SHA differs')
        item=encode(tok,row['text']);item.update(sample_id=row['id'],source_index=i,text_sha256=row['text_sha256'])
        item['input_sha256']=hashlib.sha256(json.dumps(item['input_ids'],separators=(',',':')).encode()).hexdigest();records.append(item)
    path=OUT/'tokens'/(split+'.json')
    if path.exists():raise FileExistsError('Token cache already exists')
    write_json(path,dict(split=split,items=records,pad_token_id=tok.pad_token_id,contract_sha256=file_digest(CONTRACT)))
    print('PREPARED',split,len(records),flush=True)


def chunk_identity(items,contract_sha):
    return hashlib.sha256(json.dumps([contract_sha,[(x['sample_id'],x['input_sha256']) for x in items]],separators=(',',':')).encode()).hexdigest()


def infer(model,split,c):
    from deberta_training import predict
    import torch
    cache=read(OUT/'tokens'/(split+'.json'));items=cache['items'];sha=file_digest(CONTRACT)
    if cache['contract_sha256']!=sha:raise ValueError('Token contract differs')
    order=sorted(range(len(items)),key=lambda i:len(items[i]['input_ids']));records={};started=time.monotonic()
    for start in range(0,len(items),64):
        indices=order[start:start+64];part=[items[i] for i in indices];identity=chunk_identity(part,sha)
        path=OUT/'chunks'/split/f'{start:06d}.json'
        if path.exists():
            result=read(path)
            if (result['identity']!=identity or [r['sample_id'] for r in result['rows']]!=[x['sample_id'] for x in part]
                    or result['rows_sha256']!=hashlib.sha256(json.dumps(result['rows'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
                    or [r['input_sha256'] for r in result['rows']]!=[x['input_sha256'] for x in part]
                    or [r['source_index'] for r in result['rows']]!=indices
                    or np.asarray([r['logits'] for r in result['rows']]).shape!=(len(part),2)
                    or not np.isfinite([r['logits'] for r in result['rows']]).all()):raise ValueError('Saved chunk identity/content differs')
        else:
            z,seconds=predict(model,part,cache['pad_token_id'],torch.device('cpu'),batch_size=8)
            if not np.isfinite(z).all():raise FloatingPointError('Nonfinite CPU output')
            result=dict(identity=identity,device='cpu',dtype='float32',seconds=seconds,created_utc=now(),
                rows=[dict(sample_id=x['sample_id'],source_index=x['source_index'],input_sha256=x['input_sha256'],logits=value) for x,value in zip(part,z,strict=True)])
            result['rows_sha256']=hashlib.sha256(json.dumps(result['rows'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            atomic_json(path,result)
        for row in result['rows']:records[row['source_index']]=row
        atomic_json(OUT/'inference-progress.json',dict(split=split,completed=len(records),total=len(items),elapsed_this_process_seconds=time.monotonic()-started,updated_utc=now(),test_accessed=split=='test'))
        if start%512==0:print('CPU_PROGRESS',split,len(records),'/',len(items),flush=True)
    rows=[records[i] for i in range(len(items))]
    if [r['sample_id'] for r in rows]!=[x['sample_id'] for x in items]:raise ValueError('Inference restore order differs')
    write_json(OUT/(split+'-predictions.json'),dict(split=split,rows=rows,device='cpu',dtype='float32',contract_sha256=sha,test_accessed=split=='test'))
    return rows


def admission(model,c,cal,dev):
    import torch
    from deberta_training import predict
    ref=read(MODEL/'reference.json');z,_=predict(model,ref['items'],ref['pad_token_id'],torch.device('cpu'))
    torch.testing.assert_close(torch.tensor(z),torch.tensor(ref['logits']),atol=1e-4,rtol=1e-4)
    checks={}
    for split,actual,source in [('calibration',cal,'calibration'),('admission-development',dev,'development')]:
        expected=read(ROOT/'reports'/TRAIN/'evaluation'/(source+'.json'));n=len(actual)
        if n!=(400 if split=='calibration' else 128):raise ValueError('Fixed engineering sample count differs')
        if [r['sample_id'] for r in actual]!=expected['sample_ids'][:n]:raise ValueError('Admission IDs differ')
        a=np.asarray([r['logits'] for r in actual]);b=np.asarray(expected['logits'][:n]);p=sigmoid((a[:,1]-a[:,0])/c['temperature']);q=sigmoid((b[:,1]-b[:,0])/c['temperature'])
        limits=c['engineering_gate'];value=dict(count=n,max_abs_probability_difference=float(np.max(abs(p-q))),
            mean_abs_probability_difference=float(np.mean(abs(p-q))),class_disagreement_count=int(((p>.5)!=(q>.5)).sum()),max_abs_logit_difference=float(abs(a-b).max()))
        value['passed']=(value['max_abs_probability_difference']<=limits['max_probability_difference'] and value['mean_abs_probability_difference']<=limits['mean_probability_difference'] and value['class_disagreement_count']/n<=limits['max_class_disagreement_fraction'])
        checks[split]=value
    passed=all(v['passed'] for v in checks.values());write_json(OUT/'admission.json',dict(passed=passed,checks=checks,storage_reference_passed=True,official_test_opened=False,backend_exact_bf16_equivalence_claimed=False))
    if not passed:raise ValueError('CPU engineering gate failed; no new official test access')
    inputs,_,_=load_inputs();original=inputs['calibration'];a=np.asarray([r['logits'] for r in cal]);deberta=sigmoid((a[:,1]-a[:,0])/c['temperature'])
    if [r['sample_id'] for r in cal]!=original['ids']:raise ValueError('Calibration baseline IDs differ')
    p={'original_laya_main':original['fusion'][:,1],'deberta_cpu':deberta,'fusion_50_50':(original['fusion'][:,1]+deberta)/2};thresholds={}
    for name,prob in p.items():
        confidence=np.maximum(prob,1-prob)
        thresholds[name]={str(coverage):(.5 if coverage==1 else float(np.sort(confidence)[-round(coverage*400)])) for coverage in [1.,.9,.8,.7,.5]}
    write_json(OUT/'calibration-thresholds.json',thresholds)
    write_json(OUT/'test-gate.json',dict(sealed_utc=now(),contract_sha256=file_digest(CONTRACT),new_test_accessed=False,
        source_sha256={relative(OUT/name):file_digest(OUT/name) for name in ['admission.json','calibration-thresholds.json','calibration-predictions.json','admission-development-predictions.json']}))
    print('CPU_ADMISSION_PASSED',json.dumps(checks),flush=True)


def paired_delta_interval(y,a,b,reps=20000,seed=20261018):
    rng=np.random.default_rng(seed);draws=np.zeros(reps,dtype=int);delta=b.astype(int)-a.astype(int)
    for label in [0,1]:
        values=delta[y==label];counts=np.array([(values==v).sum() for v in [-1,0,1]])
        sample=rng.multinomial(len(values),counts/counts.sum(),size=reps);draws+=sample[:,2]-sample[:,0]
    return (np.quantile(draws/len(y),[.025,.975])).tolist()


def summarize(c):
    check();test=list(read_jsonl(ROOT/c['test']['path']));ids=[r['id'] for r in test];y=np.asarray([r['label'] for r in test])
    pred=read(OUT/'test-predictions.json');rows=pred['rows'];old=read(OLD/'test-actual.json')
    if ids!=c['test']['ids'] or ids!=old['sample_ids'] or ids!=[r['sample_id'] for r in rows] or len(ids)!=25000:raise ValueError('Test ID mismatch')
    if int((y==0).sum())!=12500 or int((y==1).sum())!=12500:raise ValueError('Test balance differs')
    z=np.asarray([r['logits'] for r in rows]);base=np.asarray(old['primary'])[:,1];q=sigmoid((z[:,1]-z[:,0])/c['temperature'])
    probs={'original_laya_main':base,'deberta_cpu':q,'fusion_50_50':(base+q)/2}
    pmetrics=lambda yy,p:metrics(yy,np.log(np.clip(p,1e-15,1-1e-15))-np.log1p(-np.clip(p,1e-15,1-1e-15)))
    masks={'all':np.ones(25000,dtype=bool),'laya_truncated':np.asarray([r['truncated'] for r in old['stats']]),'laya_complete':~np.asarray([r['truncated'] for r in old['stats']])}
    group={name:{k:pmetrics(y[keep],p[keep]) for k,p in probs.items()} for name,keep in masks.items()}
    if group['all']['original_laya_main']['correct']!=24086:raise ValueError('96.344% baseline did not reproduce')
    comparisons={}
    for name in ['deberta_cpu','fusion_50_50']:
        pair=paired(y,probs['original_laya_main']-.5,probs[name]-.5)
        pair['delta_accuracy']=group['all'][name]['accuracy']-group['all']['original_laya_main']['accuracy']
        pair['delta_accuracy_conditional_stratified_bootstrap95']=paired_delta_interval(y,(base>.5)==y,(probs[name]>.5)==y)
        comparisons[name+'_vs_original_main']=pair
    thresholds=read(OUT/'calibration-thresholds.json');selective={}
    for name,p in probs.items():
        conf=np.maximum(p,1-p);correct=(p>.5)==y;values=[]
        for target,threshold in thresholds[name].items():
            keep=conf>=threshold;n=int(keep.sum());hit=int(correct[keep].sum())
            values.append(dict(calibration_target_coverage=float(target),frozen_threshold=threshold,retained=n,actual_test_coverage=n/25000,correct=hit,errors=n-hit,accuracy=hit/n if n else None,wilson95=wilson(hit,n)))
        selective[name]=values
    result=dict(run_id=RUN,state='completed',split='official_test',count=25000,backend='local_cpu_fp32',
        groups=group,primary_comparison=comparisons['fusion_50_50_vs_original_main'],comparisons=comparisons,
        primary_accuracy_wilson95=wilson(group['all']['fusion_50_50']['correct'],25000),
        selective_at_calibration_only_thresholds=selective,contract_sha256=file_digest(CONTRACT),
        admission=read(OUT/'admission.json'),test_accessed=True,test_used_for_selection=False,new_fitting=False,
        training_performed=False,cloud_compute_started=False,default_deployment_changed=False,
        limitations=['historically_reused_test_in_project','single_deberta_seed','test_backend_differs_from_cached_bf16_development','conditional_intervals_exclude_training_calibration_selection','123_unique_original_train_test_text_overlaps'])
    write_json(OUT/'summary.json',result)
    write_json(OUT/'test-fused-predictions.json',dict(sample_ids=ids,true_labels=y.tolist(),positive_probabilities={k:p.tolist() for k,p in probs.items()},contract_sha256=file_digest(CONTRACT),test_accessed=True))
    lines=['# 固定异构融合：完整官方测试结果','',
        '候选、温度、50:50 规则和 CPU 推理协议已在本轮新增测试前冻结；测试已在历史研究中使用，不称作全项目首次独立留出评估。没有训练、新拟合、融合搜索或云端计算。','',
        '| 方案 | 正确 / 25000 | Accuracy | Macro-F1 | AUROC | NLL | ECE |','|---|---:|---:|---:|---:|---:|---:|']
    for name,g in group['all'].items():lines.append(f'| {name} | {g["correct"]} | {g["accuracy"]*100:.3f}% | {g["macro_f1"]:.6f} | {g["auroc"]:.8f} | {g["nll"]:.6f} | {g["ece_15_equal_width_top_label"]*100:.3f}% |')
    pc=result['primary_comparison'];ci=[100*v for v in pc['delta_accuracy_conditional_stratified_bootstrap95']]
    lines+=['',f'固定融合相对原主融合：修复 {pc["b_only_correct"]} 条，新增错误 {pc["a_only_correct"]} 条，净变化 {pc["b_only_correct"]-pc["a_only_correct"]:+d} 条，准确率差 {100*pc["delta_accuracy"]:+.3f} pp；精确 McNemar p={pc["mcnemar_exact_two_sided_p"]:.8g}，条件配对类别分层 bootstrap 95% 区间 [{ci[0]:+.3f}, {ci[1]:+.3f}] pp。',
        '',f'冻结契约 SHA256：`{file_digest(CONTRACT)}`。最终 DeBERTa 使用本地 CPU FP32，原 Laya 概率来自冻结 CUDA FP32 正式测试缓存；原完整开发集 97.28% 使用 DeBERTa CUDA BF16。因此不把测试差异仅归因于模型结构，也不声称 CPU/BF16 逐位等价。工程核对数值见 admission.json。',
        '','完整原始测试每类 12,500 条，保留旧审计中 123 个原训练/测试唯一文本重叠。单 seed 与历史数据使用、参数选择、校准估计和重复文本依赖不在条件区间内；结果不用于改权重、重选阈值、改标签或自动切换部署。',
        '','拒答阈值全部来自原 400 条校准集，并在本轮测试前封存。实际测试覆盖率与准确率见 summary.json，不能将固定覆盖排序曲线当成部署阈值，也不附加总体风险保证。',
        '','原测试基线 96.344% 已逐样本复现；新模型块状预测与输入摘要持久化，完整 ID/标签对齐后才统一评分。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(OUT/'status.json',dict(state='completed',test_accessed=True,completed_utc=now(),primary_accuracy=group['all']['fusion_50_50']['accuracy'],contract_sha256=file_digest(CONTRACT)))
    print('FORMAL_TEST_COMPLETED',json.dumps({'primary':group['all']['fusion_50_50']['accuracy'],'comparison':pc}),flush=True)


def run():
    c=check()
    if (OUT/'summary.json').exists():raise FileExistsError('Completed test is never rerun')
    prior=read(OUT/'status.json');opened=prior.get('test_accessed',False)
    atomic_json(OUT/'status.json',dict(state='cpu_admission_running' if not opened else 'resuming_official_test',updated_utc=now(),test_accessed=opened,contract_sha256=file_digest(CONTRACT)))
    import torch
    from deberta_training import seed_all,load_saved
    torch.set_num_threads(c['threads']);seed_all()
    prep=ROOT/'environments/deberta-prep/.venv/bin/python'
    model=load_saved(MODEL)
    if not (OUT/'test-gate.json').exists():
        for split in ['calibration','admission-development']:
            if not (OUT/'tokens'/(split+'.json')).exists():subprocess.run([str(prep),str(Path(__file__)),'prepare','--split',split],check=True,cwd=ROOT)
        cal=infer(model,'calibration',c);dev=infer(model,'admission-development',c);admission(model,c,cal,dev)
    else:
        gate=read(OUT/'test-gate.json')
        if gate['contract_sha256']!=file_digest(CONTRACT) or not read(OUT/'admission.json')['passed']:raise ValueError('Frozen engineering gate differs')
        for rel,sha in gate['source_sha256'].items():
            if file_digest(ROOT/rel)!=sha:raise ValueError('Frozen pre-test artifact changed')
    if not (OUT/'tokens/test.json').exists():subprocess.run([str(prep),str(Path(__file__)),'prepare','--split','test'],check=True,cwd=ROOT)
    seal=OUT/'test-inputs.json'
    token_sha=file_digest(OUT/'tokens/test.json')
    if seal.exists():
        value=read(seal)
        if value['token_sha256']!=token_sha or value['contract_sha256']!=file_digest(CONTRACT):raise ValueError('Frozen test token input differs')
    else:atomic_json(seal,dict(token_sha256=token_sha,contract_sha256=file_digest(CONTRACT),gate_sha256=file_digest(OUT/'test-gate.json'),sealed_utc=now()))
    atomic_json(OUT/'status.json',dict(state='official_test_cpu_inference',updated_utc=now(),test_accessed=True,contract_sha256=file_digest(CONTRACT)))
    infer(model,'test',c);del model
    if file_digest(OUT/'tokens/test.json')!=read(seal)['token_sha256']:raise ValueError('Frozen token input changed during inference')
    summarize(c)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','prepare','run']);p.add_argument('--split');a=p.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='prepare':prepare(a.split)
    else:
        try:run()
        except BaseException as e:
            old=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
            atomic_json(OUT/'status.json',dict(old,state='failed',error=repr(e),updated_utc=now()));raise


if __name__=='__main__':main()
