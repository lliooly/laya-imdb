"""Offline development risk/coverage and exact three-pipeline common-error export."""
import json
from pathlib import Path
import numpy as np
from common import ROOT,file_digest,read_jsonl,write_json
from comparison_metrics import metrics,sigmoid
from run_laya_risk_coverage import load_inputs,wilson

TRAIN='deberta-heterogeneous-bf16-budget7-s42-20261017-v1'
RUN='deberta-laya-final-analysis-20261018-v1'
POINTS=(1.,.9,.8,.7,.5)


def probability_metrics(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-15,1-1e-15)
    return metrics(y,np.log(p)-np.log1p(-p))


def selective(y,p):
    p=np.asarray(p);confidence=np.maximum(p,1-p);order=np.argsort(-confidence,kind='stable')
    correct=(p>.5)==np.asarray(y);risk=np.cumsum(~correct[order])/np.arange(1,len(p)+1)
    return dict(confidence=confidence,order=order,correct=correct,risk=risk,aurc=float(risk.mean()))


def bootstrap(y,prepared,replicates=20000,seed=20261018):
    names=list(prepared);n=len(y);grid=np.arange(1,101)/100;k=np.rint(grid*n).astype(int)
    curves=np.empty((replicates,len(names),100));areas=np.empty((replicates,len(names)))
    rng=np.random.default_rng(seed);groups=[np.flatnonzero(np.asarray(y)==v) for v in (0,1)]
    for b in range(replicates):
        ii=np.concatenate([g[rng.integers(len(g),size=len(g))] for g in groups])
        count=np.bincount(ii,minlength=n)
        for j,name in enumerate(names):
            m=prepared[name];error=np.repeat(~m['correct'][m['order']],count[m['order']])
            risk=np.cumsum(error)/np.arange(1,n+1);curves[b,j]=1-risk[k-1];areas[b,j]=risk.mean()
    q=lambda x:np.quantile(x,[.025,.975],axis=0).T.tolist()
    return dict(replicates=replicates,seed=seed,paired=True,class_stratified=True,pointwise_not_simultaneous=True,
        grid=grid.tolist(),accuracy_ci95={name:q(curves[:,j]) for j,name in enumerate(names)},
        aurc_ci95={name:q(areas[:,j]) for j,name in enumerate(names)},
        delta_fusion_minus_main_ci95=q(curves[:,names.index('fusion')]-curves[:,names.index('main')]),
        delta_aurc_fusion_minus_main_ci95=q(areas[:,names.index('fusion')]-areas[:,names.index('main')]))


def main():
    out=ROOT/'reports'/RUN
    if out.exists():raise FileExistsError('Preserve analysis run')
    out.mkdir();source=ROOT/'reports'/TRAIN/'evaluation';s=json.loads((source/'summary.json').read_text())
    pred=json.loads((source/'predictions.json').read_text());old=json.loads((ROOT/'reports/laya-long-fusion-bc-20261017-v1/predictions.json').read_text())
    assert pred['sample_ids']==old['sample_ids'] and pred['true_labels']==old['true_labels']
    y=np.asarray(pred['true_labels']);pp=pred['positive_probabilities']
    probs={'main':np.asarray(pp['frozen_main']),'deberta':np.asarray(pp['deberta']),'fusion':np.asarray(pp['fusion_50_50'])}
    np.testing.assert_array_equal(probs['fusion'],(probs['main']+probs['deberta'])/2)
    inputs,_,_=load_inputs();cal=inputs['calibration'];c=json.loads((source/'calibration.json').read_text())
    assert c['sample_ids']==cal['ids'] and c['true_labels']==cal['labels'].tolist()
    cz=np.asarray(c['logits']);cp=sigmoid((cz[:,1]-cz[:,0])/s['temperature'])
    calp={'main':cal['fusion'][:,1],'deberta':cp,'fusion':(cal['fusion'][:,1]+cp)/2}
    sources=[source/'predictions.json',source/'calibration.json',source/'summary.json',ROOT/'reports/laya-long-fusion-bc-20261017-v1/predictions.json',ROOT/'data/processed/validation.jsonl',Path(__file__)]
    write_json(out/'protocol.json',dict(run_id=RUN,source_sha256={str(p.relative_to(ROOT)):file_digest(p) for p in sources},
        methods=list(probs),replicates=20000,seed=20261018,tie_break='stable source order',
        confidence='max(p,1-p)',temperature=s['temperature'],new_fitting=False,
        official_test_accessed=False,development_exploratory=True,thresholds='calibration-only inclusive kth order statistic'))
    prepared={k:selective(y,p) for k,p in probs.items()};b=bootstrap(y,prepared);write_json(out/'bootstrap.json',b)
    points=[];thresholds=[]
    for coverage in POINTS:
        k=round(coverage*len(y));row={'coverage':coverage,'retained':k,'methods':{}}
        for name,m in prepared.items():
            hit=int(m['correct'][m['order'][:k]].sum());i=round(coverage*100)-1
            row['methods'][name]={'correct':hit,'errors':k-hit,'accuracy':hit/k,'wilson95':wilson(hit,k),'paired_bootstrap95':b['accuracy_ci95'][name][i]}
            confidence=np.maximum(calp[name],1-calp[name]);tau=.5 if coverage==1 else float(np.sort(confidence)[-round(coverage*400)])
            keep=m['confidence']>=tau;n=int(keep.sum());h=int(m['correct'][keep].sum())
            thresholds.append(dict(method=name,target_calibration_coverage=coverage,threshold=tau,calibration_retained=int((confidence>=tau).sum()),
                development_retained=n,development_coverage=n/2500,development_accuracy=h/n if n else None,development_wilson95=wilson(h,n)))
        row['delta_fusion_minus_main_pp']=100*(row['methods']['fusion']['accuracy']-row['methods']['main']['accuracy'])
        row['delta_ci95_pp']=[100*x for x in b['delta_fusion_minus_main_ci95'][round(coverage*100)-1]];points.append(row)
    all_old={name:probability_metrics(y,p) for name,p in old['positive_probabilities'].items()}
    all_new={name:probability_metrics(y,p) for name,p in probs.items()}
    pool=dict(all_old,**{'heterogeneous_'+name:v for name,v in all_new.items()})
    best={key:{'best_method':min(pool,key=lambda n:pool[n][key]) if key!='auroc' else max(pool,key=lambda n:pool[n][key]),
                'fusion_best_in_this_pool': all(all_new['fusion'][key]<=v[key]+1e-14 for v in pool.values()) if key!='auroc' else all(all_new['fusion'][key]>=v[key]-1e-14 for v in pool.values())}
          for key in ['nll','ece_15_equal_width_top_label','auroc']}
    summary=dict(run_id=RUN,metrics=all_new,comparison_pool_metrics=all_old,best_in_explicit_pool=best,coverage_points=points,
        aurc={name:{'estimate':m['aurc'],'paired_stratified_bootstrap95':b['aurc_ci95'][name]} for name,m in prepared.items()},
        delta_aurc_fusion_minus_main=float(prepared['fusion']['aurc']-prepared['main']['aurc']),
        delta_aurc_ci95=b['delta_aurc_fusion_minus_main_ci95'],
        fusion_worse_prefixes_than_main=int((prepared['fusion']['risk']>prepared['main']['risk']+1e-15).sum()),
        calibration_only_thresholds=thresholds,test_accessed=False,
        limitations=['reused_development','single_deberta_seed','conditional_on_frozen_temperatures_weights','pointwise_intervals','no_guaranteed_risk_control'])
    write_json(out/'summary.json',summary);write_json(out/'calibration-thresholds.json',thresholds)
    write_json(out/'full-curves.json',{name:{'risk':m['risk'].tolist(),'retained_sample_ids':[pred['sample_ids'][i] for i in m['order']]} for name,m in prepared.items()})
    full2048=np.asarray(old['positive_probabilities']['new2048'])
    common=((probs['main']>.5)!=y)&((probs['deberta']>.5)!=y)&((full2048>.5)!=y)
    assert int(common.sum())==47
    rows={v['id']:v for v in read_jsonl(ROOT/'data/processed/validation.jsonl')}
    lengths=json.loads((source/'development.json').read_text())['review_token_lengths'];cases=[]
    for i in np.flatnonzero(common):
        raw=rows[pred['sample_ids'][i]];assert raw['label']==int(y[i])
        case=dict(case_id=f'H{len(cases)+1:03d}',sample_id=raw['id'],original_label=int(y[i]),original_label_name='pos' if y[i] else 'neg',
            original_label_not_adjudicated=True,p_positive={name:float(p[i]) for name,p in probs.items()},p_positive_2048=float(full2048[i]),
            laya_truncated=bool(old['long_mask'][i]),deberta_review_tokens=lengths[i],deberta_truncated=lengths[i]>510,
            text_sha256=raw['text_sha256'],text=raw['text'])
        cases.append(case)
    write_json(out/'common-47.json',{'scope':'development exact three-pipeline intersection','count':47,'cases':cases,'labels_modified':False})
    lines=['# 47 条跨流水线共错样本','',
        '这是当前主融合、2048 视角和 DeBERTa 同时预测错误的开发样本集合。模型一致错误不等于标签错误；原标签不修改，不把尚未人工复核的反讽、语义混合或错标假设写成定论。','']
    for case in cases:
        lines += [f'## {case["case_id"]} · {case["sample_id"]}','',f'原标签：{case["original_label_name"]}；p(pos)：主融合 {case["p_positive"]["main"]:.6f}，2048 {case["p_positive_2048"]:.6f}，DeBERTa {case["p_positive"]["deberta"]:.6f}，固定融合 {case["p_positive"]["fusion"]:.6f}。',
            f'原 Laya 截断：{case["laya_truncated"]}；DeBERTa 完整 review tokens：{case["deberta_review_tokens"]}。','',case['text'],'']
    (out/'common-47.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for name,m in prepared.items():ax.plot(np.arange(1,2501)/25,100*m['risk'],label=name)
    ax.set(xlabel='Coverage (%)',ylabel='Selective error rate (%)',title='Frozen development predictions (n=2500); lower is better')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'risk-coverage.png',dpi=180);fig.savefig(out/'risk-coverage.svg');plt.close(fig)
    lines=['# 异构融合：开发集校准与风险—覆盖','',
        '使用现有 CUDA BF16 DeBERTa 及原冻结 Laya 概率，不增加推理或拟合，不读取官方测试。以下为探索性开发结果。','',
        '| 方案 | Accuracy | NLL | ECE | AUROC | AURC |','|---|---:|---:|---:|---:|---:|']
    for name,m in all_new.items():lines.append(f'| {name} | {m["accuracy"]*100:.2f}% | {m["nll"]:.6f} | {m["ece_15_equal_width_top_label"]*100:.3f}% | {m["auroc"]:.8f} | {prepared[name]["aurc"]:.8f} |')
    lines+=['','| 覆盖率 | 原主融合 | DeBERTa | 固定融合 | 融合相对主方案的差及逐点 95% 区间 |','|---|---:|---:|---:|---|']
    for point in points:
        values=[point['methods'][name]['accuracy']*100 for name in probs];ci=point['delta_ci95_pp']
        lines.append(f'| {point["coverage"]*100:.0f}% | {values[0]:.3f}% | {values[1]:.3f}% | {values[2]:.3f}% | {point["delta_fusion_minus_main_pp"]:+.3f} pp [{ci[0]:+.3f}, {ci[1]:+.3f}] |')
    lines+=['',f'融合相对主方案的 AURC 差为 {summary["delta_aurc_fusion_minus_main"]:.8f}，条件 95% 区间 {summary["delta_aurc_ci95"]}；负数有利于融合。',
        f'全部 2,500 个非空前缀中，融合风险高于原主融合的前缀有 {summary["fusion_worse_prefixes_than_main"]} 个，不能仅凭若干点更好就声称整条曲线严格占优。',
        '',f'在明确列出的既有 512/2048、A/B/C、主/辅助等权方案与本轮方案池中，最佳判断为：`{json.dumps(best,ensure_ascii=False)}`。该池不代表仓库所有历史实验。',
        '','区间采用 20,000 次类别分层配对 bootstrap，同一轮为各方案重排同一抽样多重集合；为逐点区间，非同时置信带。风险由各自排序和保留集合计算，不能把同覆盖率的不同保留集直接当同一样本做 McNemar。区间不含训练 seed、校准/阈值估计、反复开发选择或分布变化的不确定性。',
        '','400 条校准集的 kth-largest 阈值另存 calibration-thresholds.json；阈值包含边界，不承诺测试覆盖率与校准目标相同，也没有有限样本风险保证。47 条共错全文见 common-47.md 和 common-47.json；原标签均保留。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'metrics':{name:{k:m[k] for k in ['accuracy','nll','ece_15_equal_width_top_label','auroc']} for name,m in all_new.items()},'coverage_points':points,'aurc':summary['aurc'],'delta_aurc_ci95':summary['delta_aurc_ci95'],'best_in_pool':best,'common_cases':len(cases)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
