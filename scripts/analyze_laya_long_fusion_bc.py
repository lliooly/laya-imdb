"""Evaluate only the declared B/C mixtures, using cached development probabilities."""
import json
from pathlib import Path

import numpy as np

from common import ROOT, file_digest, write_json
from comparison_metrics import metrics
from analyze_laya_length_router import comparison, wilson
from laya_decision_views import probabilities, probability_margin
from laya_probability_fusion import apply_gated_mixture

STUDY='laya-long-fusion-bc-20261017-v1'
NEW='laya-context-pair-s42-modal-20261016-v1'
OLD='laya-probability-fusion-s43-s42-20261014-v1'
ACTUAL='laya-fusion-inference-cuda-fp32-20261014-v1'


def main():
    config_path=ROOT/'configs'/f'{STUDY}.json'
    c=json.loads(config_path.read_text())
    output=ROOT/'reports'/STUDY
    if output.exists():raise FileExistsError('New run ID required; preserve existing results')
    if (c['new_weight']!=.5 or c['threshold_review_tokens']!=466
            or c['mixture_search'] or c['new_calibration_fit'] or c['test_accessed']):
        raise ValueError('Only the declared equal-weight, long-only mixtures are allowed')
    for rel,sha in c['source_sha256'].items():
        if file_digest(ROOT/rel)!=sha:raise ValueError('Bound input changed: '+rel)
    def read(rel):return json.loads((ROOT/rel).read_text())
    a=read(f'reports/{NEW}/evaluation/512-development.json')
    b=read(f'reports/{NEW}/evaluation/2048-development.json')
    context=read(f'reports/{NEW}/evaluation/summary.json')
    sealed=read(f'reports/{NEW}/evaluation/bundle.json')
    delivered=read(f'reports/{NEW}/local-delivery.json')
    if (delivered['state']!='verified_local' or not delivered['local_delivery_verified']
            or file_digest(ROOT/f'reports/{NEW}/evaluation/bundle.json')!=delivered['bundles']['evaluation']['bundle_sha256']):
        raise ValueError('New predictions have not been strictly delivered')
    for name in ('512-development.json','2048-development.json','summary.json'):
        record=next(f for f in sealed['files'] if f['path']==name)
        p=ROOT/f'reports/{NEW}/evaluation'/name
        if p.stat().st_size!=record['bytes'] or file_digest(p)!=record['sha256']:
            raise ValueError('New sealed prediction changed')
    actual=read(f'reports/{ACTUAL}/validation-actual.json')
    fused=read(f'reports/{OLD}/validation-fused.json')
    frozen=read(f'reports/{OLD}/frozen-fit.json')
    old_summary=read(f'reports/{OLD}/summary.json')
    if (a['test_accessed'] or b['test_accessed'] or fused['test_accessed']
            or not read(f'reports/{ACTUAL}/summary.json')['parity_passed']
            or old_summary['frozen_fit_sha256']!=file_digest(ROOT/f'reports/{OLD}/frozen-fit.json')):
        raise ValueError('Historical provenance differs')
    if (a['sample_ids']!=b['sample_ids'] or a['sample_ids']!=actual['sample_ids']
            or a['sample_ids']!=[r['sample_id'] for r in fused['rows']]
            or len(set(a['sample_ids']))!=2500 or a['true_labels']!=b['true_labels']
            or a['true_labels']!=[r['true_label'] for r in fused['rows']]
            or a['review_token_lengths']!=b['review_token_lengths']
            or a['review_token_lengths']!=[r['review_tokens'] for r in actual['stats']]):
        raise ValueError('IDs, labels or token lengths are not paired')
    y=np.asarray(a['true_labels']);long=np.asarray(a['review_token_lengths'])>466
    if long.tolist()!=[r['truncated'] for r in actual['stats']] or long.tolist()!=[r['truncated'] for r in fused['rows']]:
        raise ValueError('Long-sample mask differs')
    new512=probabilities(a['logits'],context['models']['512']['temperature'])
    new2048=probabilities(b['logits'],context['models']['2048']['temperature'])
    original=np.asarray(actual['primary']);equal=np.asarray(actual['equal'])
    args=[np.asarray([row['probabilities'] for row in actual['models'][name][view]])
          for name in ('baseline-s43','combined-s42') for view in ('head','head_tail')]
    np.testing.assert_array_equal(original,apply_gated_mixture(*args,long,frozen['mixture']['coefficient']))
    np.testing.assert_array_equal(equal,apply_gated_mixture(*args,long,.5))
    np.testing.assert_array_equal(original,[r['probabilities'] for r in fused['rows']])
    def gated(short,full):return np.where(long[:,None],full,short)
    predictions={'new512':new512,'new2048':new2048,'A':gated(new512,new2048),
                 'B':gated(new512,(new512+new2048)/2),'old_primary':original,
                 'C':gated(original,(original+new2048)/2),'old_equal':equal,
                 'C_aux':gated(equal,(equal+new2048)/2)}
    margins={name:probability_margin(p) for name,p in predictions.items()}
    correct={name:(m>0)==y for name,m in margins.items()}
    values={}
    for name,m in margins.items():
        groups={}
        for group,keep in [('all',np.ones(len(y),dtype=bool)),('truncated_at_512',long),('complete_at_512',~long)]:
            value=metrics(y[keep],m[keep])
            value['accuracy_wilson95']=wilson(value['correct'],value['count'])
            groups[group]=value
        values[name]=groups
    if (values['new512']['all']['correct']!=2408 or values['new2048']['all']['correct']!=2408
            or values['A']['all']['correct']!=2414 or values['old_primary']['all']['correct']!=2424
            or values['old_equal']['all']['correct']!=2425):
        raise ValueError('Frozen controls did not reproduce')
    comparisons={name+'_vs_'+base:comparison(y,margins[base],margins[name])
                 for name,base in [('B','new512'),('B','A'),('C','old_primary'),('C_aux','old_equal')]}
    regression=long&correct['new512']&~correct['new2048']
    audit={'B_two_prior_regressions':[{'sample_id':sid,'true_label':int(y[i]),
                'p_positive_512':float(new512[i,1]),'p_positive_2048':float(new2048[i,1]),
                'p_positive_B':float(predictions['B'][i,1]),'B_correct':bool(correct['B'][i])}
                for i,sid in enumerate(a['sample_ids']) if regression[i]]}
    oracle={}
    for name,candidate in [('old_primary','C'),('old_equal','C_aux')]:
        wrong=long&~correct[name]
        possible=wrong&correct['new2048']
        upper=int((correct[name]|(long&correct['new2048'])).sum())
        oracle[name]={'old_long_errors':int(wrong.sum()),'new2048_correct_on_old_long_errors':int(possible.sum()),
                      'long_common_errors':int((wrong&~correct['new2048']).sum()),
                      'upper_bound_correct':upper,'upper_bound_accuracy':upper/len(y),
                      'scope':'short_unchanged_any_long_convex_mix_or_per_sample_choice_of_these_two_probabilities',
                      'uses_true_labels_not_deployable':True}
        audit[name+'_12_long_errors']=[{'sample_id':sid,'true_label':int(y[i]),
            'p_positive_old':float(predictions[name][i,1]),'p_positive_2048':float(new2048[i,1]),
            'p_positive_mix':float(predictions[candidate][i,1]),'new2048_correct':bool(correct['new2048'][i]),
            'mixture_correct':bool(correct[candidate][i])} for i,sid in enumerate(a['sample_ids']) if wrong[i]]
    hashes=dict(c['source_sha256'])
    for rel in ['scripts/analyze_laya_long_fusion_bc.py','scripts/analyze_laya_length_router.py',
                'scripts/comparison_metrics.py','scripts/laya_decision_views.py','scripts/laya_probability_fusion.py',
                'scripts/common.py','configs/'+STUDY+'.json']:
        hashes[rel]=file_digest(ROOT/rel)
    result={'study_id':STUDY,'state':'offline_evaluated','metrics':values,'comparisons':comparisons,
            'oracle_diagnostics':oracle,'B_prior_regressions_recovered':int((regression&correct['B']).sum()),
            'protocol':c,'input_and_source_sha256':hashes,'test_accessed':False,'gpu_called':False,
            'inference_forward_rows':{'A':2500,'B':2907,'old_primary':5814,'C':6221},
            'latency_measured':False,'default_deployment_changed':False,
            'limitations':['posthoc_development_exploration','reused_development','single_new_context_seed',
                           'multiple_comparisons_uncorrected','bootstrap_conditional_on_fixed_predictions']}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'summary.json',result)
    write_json(output/'case-audit.json',audit)
    write_json(output/'predictions.json',{'sample_ids':a['sample_ids'],'true_labels':a['true_labels'],
               'long_mask':long.tolist(),'positive_probabilities':{k:p[:,1].tolist() for k,p in predictions.items()},
               'test_accessed':False})
    lines=['# 长文概率融合 B / C：固定方案离线核算','',
        '本轮按用户提案复用现有预测，未训练、未调用 GPU、未读取官方测试或更换默认部署。先保存固定协议，再计算 B/C 新混合结果；不搜索权重或阈值。方案是在已观察开发分组结果后提出，属于事后探索。', '',
        '## 固定定义','',
        '- 长文为冻结 tokenizer 的完整 review tokens >466；其余为短文。新 512 / 2048 均用本轮 seed 42、epoch 2，先用各自原冻结温度转成概率。',
        '- B：短文新 512；长文新 512 / 新 2048 校准概率各 50%。',
        '- C 主方案：短文保持旧冻结主融合；长文为旧主融合 / 新 2048 各 50%。旧主融合内部 70.2033% baseline-s43 +29.7967% combined-s42 及窗口门控保持。',
        '- C 辅助：同样的长文 50:50 混合，但旧侧使用预先保留的 97.00% 等权方案；单独列示，不据其分数替换旧主方案。', '',
        '## 结果','',
        '| 方案 | 正确数 / 2500 | Accuracy | 长文正确 / 407 | 校准 NLL | ECE |',
        '|---|---:|---:|---:|---:|---:|']
    names={'new512':'全走新 512','new2048':'全走新 2048','A':'A：按长度单选','B':'B：长文双视角等权',
           'old_primary':'旧冻结主融合','C':'C：旧主融合 + 长文 2048','old_equal':'旧等权辅助','C_aux':'C 辅助：旧等权 + 长文 2048'}
    for name in names:
        m=values[name]['all'];l=values[name]['truncated_at_512']
        lines.append(f"| {names[name]} | {m['correct']} | {100*m['accuracy']:.2f}% | {l['correct']} | {m['nll']:.6f} | {m['ece_15_equal_width_top_label']:.6f} |")
    lines += ['', '## 修复、退化与不确定性','']
    for name,p in comparisons.items():
        ci=[100*x for x in p['delta_accuracy_conditional_bootstrap95']]
        lines.append(f"- {name}：修复 {p['b_only_correct']}、退化 {p['a_only_correct']}；Δ={100*p['delta_accuracy']:+.2f} pp，条件配对 bootstrap 95% 区间 [{ci[0]:+.2f}, {ci[1]:+.2f}] pp；精确 McNemar p={p['mcnemar_exact_two_sided_p']:.6g}。")
    lines += ['', 'B 仅救回此前 2048 相对 512 退化的 2 条中的 1 条；相对 A 又丢失 2 条，合计净少对 1 条。概率平均会同时减弱正确和错误模型的置信度，不能保证保留所有原有修复。C 主 / 辅助均修复 3 条同时新增 3 条错误，准确率不变。具体样本概率见 case-audit.json。', '',
        '## 12 条长文错题与 97.2% 目标','']
    for name,o in oracle.items():
        lines.append(f"- {names[name]}：原 12 条长文错误，新 2048 答对 {o['new2048_correct_on_old_long_errors']} 条，两者共同错 {o['long_common_errors']} 条；短文不变时，标签知情的逐条选择上限 {o['upper_bound_correct']}/2500 = {100*o['upper_bound_accuracy']:.2f}%。")
    lines += ['', '二分类中，两份概率都在错误类别一侧时，任何凸概率混合都不能跨到正确类别。因此，对当前固定两份输出，C 主方案无法到 97.2%；辅助方案的 97.20% 只是需要逐条知道正确答案的理想上限，不能作为可部署结果。此界限不适用于新权重、不同输入或非凸二级分类器。', '',
        '## 成本与结论','',
        '- A 对 2500 条样本需要 2500 行模型前向；B 为 2907 行，比 A 多 16.28%。旧主融合需要 5814 行，C 为 6221 行，多约 7.00%，并新增第 3 份模型。以上是行数，不是实测延迟或峰值显存。',
        '- B 相对 A 准确率略低；C 对原主融合无准确率增益。不建议据这轮结果追加训练或搜索融合比例追分。若保留候选，应先明确优化的是概率质量、风险覆盖还是整体准确率，后续在独立数据上验证。',
        '- p 值与 bootstrap 仅描述固定模型的当前开发样本，不覆盖跨种子、方案选择和多次探索的不确定性；官方测试已在先前研究使用，本轮没有把它重新当作未见留出集。', '',
        '## 复现与审计','',
        f'配置：`configs/{STUDY}.json`。运行：`uv run --locked --offline --no-sync python scripts/analyze_laya_long_fusion_bc.py`。输出目录已存在时拒绝覆盖。', '',
        '新预测的封存 SHA、完整交付状态、两次历史融合概率（缓存 / CUDA 实际推理）的完全一致、旧冻结系数重算、2500 条样本 ID / 标签 / 完整 token 长度和长文掩码均已核对。原控制成绩全部复现。完整指标与源码 / 输入 SHA 在 summary.json，样本概率审计在 case-audit.json；历史文件未改写。','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'report':str(output/'REPORT.md'),'accuracy':{k:v['all']['accuracy'] for k,v in values.items()},
                      'C_oracle_upper_bound':oracle['old_primary']['upper_bound_accuracy']},ensure_ascii=False))


if __name__=='__main__':main()
