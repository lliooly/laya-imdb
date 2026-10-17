"""Isolated adapter: reuse frozen CE epochs, change only the reviewed dataset."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

from common import file_digest, read_jsonl, write_json

ROOT=Path(__file__).resolve().parents[1]
STUDY='laya-reviewed-data-s42-modal-20261017-v1'
CONTROL='laya-context-pair-s42-modal-20261016-v1'
DATA='laya-94-user-label-revision-20261017-v2'
VOLUME='end-imdb-laya-robust-deferred-20261015-v2'
SECONDS=3900
CAP_USD=4


def read(p):return json.loads(Path(p).read_text())


def config(root=ROOT):
    c=read(root/'configs'/f'{STUDY}.json')
    if (c['study_id']!=STUDY or c['additional_reserve_usd']!=4 or c['maximum_seconds']!=SECONDS
            or c['lengths']!=[512] or c['seed']!=42 or c['dataset_revision']!=DATA):
        raise ValueError('Approved one-model data-study scope differs')
    for rel,sha in c['source_sha256'].items():
        if file_digest(root/rel)!=sha:raise ValueError('Bound study input changed: '+rel)
    evidence=read(root/'reports'/CONTROL/'admission.json')
    if evidence['state']!='passed' or not evidence['exact_resume_weights_rng_losses']:
        raise ValueError('Existing full CUDA epoch-recovery admission is absent')
    from run_laya_head_control import TRAINING
    if c['training']!=dict(TRAINING,seed=42):raise ValueError('Training recipe differs from the control')
    return c


def data_rows(root=ROOT):
    revised=read(root/'data/label-revisions'/DATA/'manifest.json')
    original=read(root/'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json')
    if file_digest(root/revised['train']['path'])!=revised['train']['sha256']:
        raise ValueError('Reviewed data SHA differs')
    rows=list(read_jsonl(root/revised['train']['path']))
    baseline=list(read_jsonl(root/'data/processed/train.jsonl'))
    old={r['id']:r for r in baseline}
    patch=read(root/'data/label-revisions'/DATA/'cumulative-label-patch.json')['rows']
    changes={r['sample_id']:r for r in patch}
    dropped={r['sample_id'] for r in patch if r['action']=='delete'}
    if len(patch)!=94 or len(changes)!=94 or len(dropped)!=3:
        raise ValueError('User decision inventory differs')
    for r in rows:
        d=changes.get(r['id'])
        expected=dict(old[r['id']],label=d['new_label']) if d else old[r['id']]
        if r!=expected or r['id'] in dropped:raise ValueError('Unrequested training record differs')
    if ([r['id'] for r in baseline]!=original['train']['ids']
            or [r['id'] for r in rows]!=[i for i in original['train']['ids'] if i not in dropped]
            or [r['id'] for r in rows]!=revised['train']['ids']
            or len(rows)!=22497 or len({r['id'] for r in rows})!=22497
            or revised['gradient_train_ids']!=[i for i in original['gradient_train_ids'] if i not in dropped]
            or len(revised['gradient_train_ids'])!=22097
            or revised['calibration_ids']!=original['calibration_ids']):
        raise ValueError('Reviewed membership, row order or fixed partitions differ')
    by={r['id']:r for r in rows}
    if any(by[i]!=old[i] for i in revised['calibration_ids']):raise ValueError('Calibration changed')
    if sum(r['label']!=old[r['id']]['label'] for r in rows)!=61:raise ValueError('Expected exactly 61 flips')
    split=dict(revised,validation=original['validation'])
    return by,split


def install(root=ROOT):
    """Load frozen code into a fresh namespace; never mutate its imported module."""
    root=Path(root)
    spec=importlib.util.spec_from_file_location('reviewed_data_frozen_ce',root/'scripts/laya_context_pair.py')
    cp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cp)
    cp.ROOT=root;cp.STUDY=STUDY;cp.CAP_USD=CAP_USD;cp.SECONDS=SECONDS;cp.config=lambda:config(root)
    original_training_config=cp.training_config
    def training_config(cap,micro_batch=8):
        if cap!=512:raise ValueError('This study admits only 512')
        return original_training_config(cap,micro_batch)
    cp.training_config=training_config
    def tokenizer_and_rows():
        from transformers import AutoTokenizer
        by,split=data_rows(root)
        lock=read(root/'configs/laya_model_lock.json')
        for rel,sha in lock['sha256'].items():
            if file_digest(root/lock['local_dir']/rel)!=sha:raise ValueError('Published model changed')
        tok=AutoTokenizer.from_pretrained(str(root/lock['local_dir']/'tokenizer'),local_files_only=True)
        return tok,by,split,lock
    cp.tokenizer_and_rows=tokenizer_and_rows
    original_assets=cp.assets
    def assets(folder):
        import shutil
        original_assets(folder)
        dest=folder/'assets/data-revision';dest.mkdir()
        for name in ('manifest.json','cumulative-label-patch.json','original-labels-94.json'):
            shutil.copyfile(root/'data/label-revisions'/DATA/name,dest/name)
    cp.assets=assets
    cp.freeze=lambda:freeze(cp,root)
    return cp


def freeze(cp,root=ROOT):
    for epoch in (0,1):cp.accepted(512,epoch,root)
    final=read(cp.bundle(512,1,root)/'training.json')
    control=read(root/'checkpoints'/CONTROL/'context-512-s42/epoch-002/training.json')
    if (final['initial_state_sha256']!=control['initial_state_sha256'] or final['updates']!=5526
            or final['exposures']!=44194 or len(final['history'])!=2
            or any(h['updates']!=2763 or h['exposures']!=22097 for h in final['history'])):
        raise ValueError('Initialization or complete two-epoch update/exposure audit differs')
    gate={'state':'fixed_epoch_2_frozen','study_id':STUDY,'test_accessed':False,
          'initial_state_sha256':final['initial_state_sha256'],
          'final_bundle_sha256':file_digest(cp.bundle(512,1,root)/'bundle.json'),
          'baseline_bundle_sha256':file_digest(root/'checkpoints'/CONTROL/'context-512-s42/epoch-002/bundle.json')}
    path=root/'reports'/STUDY/'final-frozen.json'
    if path.exists() and read(path)!=gate:raise ValueError('Frozen model changed')
    write_json(path,gate);return gate


def evaluate(cp,deadline):
    import numpy as np
    import torch
    import laya.train as upstream
    from laya_head_control import load_fp32
    from comparison_metrics import metrics,fit_temperature
    from analyze_laya_length_router import comparison,wilson
    gate=cp.freeze();root=cp.ROOT;report=root/'reports'/STUDY
    out=report/'evaluation.writing';out.mkdir(exist_ok=False)
    tok,by,split,_=cp.tokenizer_and_rows()
    control=read(root/'reports'/CONTROL/'evaluation/512-development.json')
    control_summary=read(root/'reports'/CONTROL/'evaluation/summary.json')
    dev=list(read_jsonl(root/'data/processed/validation.jsonl'))
    if ([r['id'] for r in dev]!=control['sample_ids'] or control['sample_ids']!=split['validation']['ids']
            or [r['label'] for r in dev]!=control['true_labels']):raise ValueError('Development/control rows differ')
    model,_=load_fp32(cp.bundle(512,1));model.to('cuda').eval()
    values={};lengths=None
    for part,rows in [('calibration',[by[i] for i in split['calibration_ids']]),('development',dev)]:
        items,sizes=cp.items_for(tok,rows,512);blocks=[]
        with torch.inference_mode():
            for start in range(0,len(items),8):
                if time.time()>=deadline-120:raise TimeoutError('Deadline during frozen evaluation')
                prepared=[upstream.encode_item(tok,item,512,192,[0,1],False) for item in items[start:start+8]]
                batch=upstream.collate_items([prepared],tok.pad_token_id)
                logits=upstream._forward(model,batch,torch.device('cuda'),False,False).cpu().numpy()
                if logits.shape!=(len(prepared),2) or not np.isfinite(logits).all():raise FloatingPointError('Invalid prediction')
                blocks.extend(logits.tolist())
        values[part]=np.asarray(blocks)
        write_json(out/f'512-{part}.json',{'sample_ids':[r['id'] for r in rows],
                   'true_labels':[r['label'] for r in rows],'review_token_lengths':sizes,'logits':blocks,'test_accessed':False})
        if part=='development':lengths=np.asarray(sizes)
    if lengths.tolist()!=control['review_token_lengths']:raise ValueError('Input length semantics changed')
    temperature=fit_temperature(np.diff(values['calibration'],axis=1).ravel(),
        [by[i]['label'] for i in split['calibration_ids']],bounds=(.5,5.))
    y=np.asarray(control['true_labels']);a=np.diff(np.asarray(control['logits']),axis=1).ravel()
    b=np.diff(values['development'],axis=1).ravel();baseline_temp=control_summary['models']['512']['temperature']
    result={'study_id':STUDY,'state':'evaluated','control_study_id':CONTROL,'dataset_revision':DATA,
            'test_accessed':False,'temperature':temperature,'control_temperature':baseline_temp,'groups':{},
            'selection':'fixed_epoch_2_seed42','frozen_gate':gate,
            'limitations':['single_seed','reused_development','combined_relabel_and_delete_effect','sample_order_changes_after_deletion']}
    for name,keep in [('all',np.ones(len(y),dtype=bool)),('truncated_at_512',lengths>466),('complete_at_512',lengths<=466)]:
        groups={}
        for label,margin,t in [('original',a,baseline_temp),('revised',b,temperature)]:
            raw=metrics(y[keep],margin[keep]);groups[label]={'raw':raw,'calibrated':metrics(y[keep],margin[keep],t),
                'accuracy_wilson95':wilson(raw['correct'],raw['count'])}
        groups['paired']=comparison(y[keep],a[keep],b[keep]);result['groups'][name]=groups
    if result['groups']['all']['original']['raw']['correct']!=2408:raise ValueError('96.32% control failed to reproduce')
    write_json(out/'summary.json',result)
    lines=['# Laya-512 用户修订数据对照','', '同一初始权重，seed42、固定两轮，仅更换训练数据；开发2500条，官方测试未读取。','',
           '| 数据 | 正确 / 2500 | Accuracy | Macro-F1 | AUROC | 校准NLL | 校准ECE |','|---|---:|---:|---:|---:|---:|---:|']
    for label in ('original','revised'):
        g=result['groups']['all'][label];r,k=g['raw'],g['calibrated']
        lines.append(f"| {label} | {r['correct']} | {r['accuracy']*100:.2f}% | {r['macro_f1']:.6f} | {r['auroc']:.6f} | {k['nll']:.6f} | {k['ece_15_equal_width_top_label']:.6f} |")
    lines+=['','## 配对证据','']
    for name,g in result['groups'].items():
        p=g['paired'];ci=[v*100 for v in p['delta_accuracy_conditional_bootstrap95']]
        lines.append(f"- {name}：修复{p['b_only_correct']}、退化{p['a_only_correct']}；Δ={100*p['delta_accuracy']:+.2f}pp，条件配对95%区间[{ci[0]:+.2f},{ci[1]:+.2f}]pp，精确McNemar p={p['mcnemar_exact_two_sided_p']:.6g}。")
    lines+=['','累计61条用户裁定翻标和3条删除，梯度22100→22097，两轮暴露44200→44194；更新同为5526。删除改变实际洗牌和选项序列，本结果不单独归因于翻标。校准400不变，固定466阈值分组。单seed及重复使用开发集的边界不因条件样本区间而消除。未按结果追加epoch、种子、融合或训练。','']
    (out/'REPORT.md').write_text('\n'.join(lines));cp.assets(out)
    cp.seal(out,{'study_id':STUDY},'evaluated')
    m=read(out/'bundle.json');m['bundle']=str((report/'evaluation').relative_to(root));write_json(out/'bundle.json',m)
    out.rename(report/'evaluation')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['train','evaluate','verify'])
    p.add_argument('--epoch',type=int,choices=[0,1]);p.add_argument('--deadline',type=float,default=0)
    p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path);p.add_argument('--cap',type=int,choices=[512],default=512)
    args=p.parse_args();cp=install()
    if args.mode=='verify':cp.verify(args.bundle,args.output);return
    config()
    from laya_long_context_pilot import strict_runtime
    strict_runtime()
    from train_laya import block_network
    block_network()
    if args.mode=='train':cp.train_epoch(512,args.epoch,args.deadline)
    else:evaluate(cp,args.deadline)


if __name__=='__main__':main()
