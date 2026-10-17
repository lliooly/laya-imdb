"""Export only the 94 training candidates with original text and existing review."""
from collections import Counter
import json
from pathlib import Path

from common import ROOT, digest, file_digest, read_jsonl, write_json

STUDY='laya-94-candidates-information-20261017-v1'
PRIOR='laya-train-common-error-review-20261015-v1'
LABEL={0:'负向',1:'正向'}
CATEGORY={'label_conflict_candidate':'标签—文本态度冲突候选',
          'ambiguous_attitude':'态度歧义或缺少本人评价',
          'missing_decisive_context':'决定性证据在截断外',
          'model_error_visible':'可见输入上的模型错误候选'}
ATTITUDE={'positive':'正向','negative':'负向','uncertain':'不确定','ambiguous':'不确定'}
CONFIDENCE={'high':'高','medium':'中','low':'低','legacy_unrated':'旧记录未统一评级'}


def main():
    report=ROOT/'reports'/STUDY
    data=ROOT/'artifacts'/STUDY
    if report.exists() or data.exists():raise FileExistsError('Preserve reports; use a new run ID')
    sources={}
    def read(rel):
        p=ROOT/rel;sources[rel]=file_digest(p)
        return json.loads(p.read_text())
    prior=read(f'reports/{PRIOR}/summary.json')
    ledger=read(f'reports/{PRIOR}/merged-review-ledger.json')['rows']
    boundary=read(f'reports/{PRIOR}/boundary-audit.json')
    manifest=read('reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json')
    source=ROOT/'data/processed/train.jsonl'
    sources[str(source.relative_to(ROOT))]=file_digest(source)
    if sources[str(source.relative_to(ROOT))]!=manifest['train']['source_sha256']:
        raise ValueError('Frozen training data changed')
    by_id={r['id']:r for r in read_jsonl(source)}
    predictions={}
    for model in ('baseline','combined'):
        block=read(f'artifacts/train-hardcase-mining-20261011-v1/{model}-gradient-predictions.json')
        if (block['count']!=22100 or block['split']!='gradient_train' or block['input_rule']!='original_head'
                or block['test_accessed'] or block['training_performed']
                or block['checkpoint_sha256']!=prior['models'][model]['checkpoint_sha256']
                or block['train_source_sha256']!=sources[str(source.relative_to(ROOT))]):
            raise ValueError('Historical model/data provenance differs')
        predictions[model]={r['sample_id']:r for r in block['rows']}
    if (len(ledger)!=94 or len({r['sample_id'] for r in ledger})!=94
            or not set(r['sample_id'] for r in ledger)<=set(manifest['gradient_train_ids'])):
        raise ValueError('The original 94 candidates are not a unique gradient subset')
    rows=[]
    for index,old in enumerate(sorted(ledger,key=lambda r:r['sample_id']),1):
        sid=old['sample_id'];row=by_id[sid];label=row['label']
        if digest(row['text'])!=old['text_sha256'] or row['text_sha256']!=old['text_sha256'] or label!=old['label']:
            raise ValueError('Training text or label differs')
        rating=int(Path(sid).stem.rsplit('_',1)[1])
        if not ((label==0 and 1<=rating<=4) or (label==1 and 7<=rating<=10)):
            raise ValueError('Filename rating differs from original class')
        paired={}
        for model in ('baseline','combined'):
            pp=predictions[model][sid];z=pp['logits'];margin=z[1-label]-z[label]
            if (pp['true_label']!=label or pp['pred_label']!=1-label or margin<=1
                    or pp['input_sha256']!=old['input_sha256']
                    or abs(margin-old['wrong_class_logit_margin'][model])>1e-12):
                raise ValueError('Candidate rule no longer reproduced')
            paired[model]={'pred_label':pp['pred_label'],'pred_label_name':LABEL[pp['pred_label']],
                           'logits_negative_positive':z,'wrong_class_logit_margin':margin,
                           'training_seed':43,'input_rule':'original_head'}
        rows.append({'index':f'C{index:03d}','sample_id':sid,'original_label':label,
             'original_label_name':LABEL[label],'source_rating_from_filename':rating,
             'text':row['text'],'text_sha256':row['text_sha256'],'raw_sha256':row['raw_sha256'],
             'input_sha256':old['input_sha256'],'review_tokens':old['review_tokens'],
             'original_review_budget':466,'original_head_truncated':old['truncated'],
             'predictions':paired,'prior_ai_review':old,'confirmed_label_error':None})
    counts=Counter(r['prior_ai_review']['category'] for r in rows)
    if dict(counts)!=prior['category_counts'] or sum(r['original_head_truncated'] for r in rows)!=16:
        raise ValueError('Existing review totals changed')
    report.mkdir(parents=True);data.mkdir(parents=True)
    intro=['# 94 条训练高冲突候选：完整信息','',
       '只含原梯度训练分区的 94 条候选，没有开发样本。保留原标签与原文；未重新训练或推理。', '',
       '筛选：baseline-s43 与 combined-s43 均预测为给定标签的相反类别，且反向原始 logit margin 均 >1。两者训练 seed 都是 43，属于不同配方，不能称跨种子一致。导出发生在已参与训练的样本上，不是 out-of-fold 预测。', '',
       '模型输入为旧 512 总长度下的原始前缀，评论预算 466 token；16 条在这个规则下截断。其余正文也完整保留供复核。margin 是原始 logits 差，不是校准概率，也不表示不同模型同等置信。', '',
       '原标签：负向 49 条、正向 45 条。文件名原评分与原标签对齐。旧 AI 审阅分类：冲突候选 58、歧义 23、截断缺证据 6、可见模型错误候选 7。旧记录复用 32 条，新 AI 审阅 62 条；尚无独立人工复核，确认错标数量未知。', '',
       '## 清单索引','',
       '| 编号 | 原审阅号 | 样本 ID | 原评分 | 原标签→两模型预测 | tokens | 前缀截断 | baseline / combined 反向 margin | 旧 AI 工作分类 |',
       '|---|---|---|---:|---|---:|---|---|---|']
    for r in rows:
        o=r['prior_ai_review'];m=r['predictions'];idx=r['index']
        intro.append(f"| [{idx}](#{idx.lower()}) | {o['review_alias']} | `{r['sample_id']}` | {r['source_rating_from_filename']} | {r['original_label_name']}→{LABEL[1-r['original_label']]} | {r['review_tokens']} | {'是' if r['original_head_truncated'] else '否'} | {m['baseline']['wrong_class_logit_margin']:.4f} / {m['combined']['wrong_class_logit_margin']:.4f} | {CATEGORY[o['category']]} |")
    full=list(intro)
    for r in rows:
        o=r['prior_ai_review'];m=r['predictions'];full += ['', '## '+r['index'],'',
          f"- 样本：`{r['sample_id']}`；旧审阅号：{o['review_alias']}。",
          f"- 原标签：{r['original_label_name']}；原文件名评分：{r['source_rating_from_filename']}/10；两模型均预测：{LABEL[1-r['original_label']]}。",
          f"- 长度：{r['review_tokens']} review tokens；原 466-token 前缀截断：{'是' if r['original_head_truncated'] else '否'}。",
          f"- baseline-s43 logits（负、正）：{m['baseline']['logits_negative_positive']}；反向 margin：{m['baseline']['wrong_class_logit_margin']:.9f}。",
          f"- combined-s43 logits（负、正）：{m['combined']['logits_negative_positive']}；反向 margin：{m['combined']['wrong_class_logit_margin']:.9f}。",
          f"- 旧 AI 分类：{CATEGORY[o['category']]}；来源：{'复用旧记录' if o['review_origin']!='new_anonymous_text' else '新增文本 AI 审阅'}；置信度：{CONFIDENCE.get(o['confidence'],o['confidence'])}。",
          f"- 旧 AI 全文态度：{ATTITUDE.get(o['full_attitude'],o['full_attitude'])}；前缀态度：{ATTITUDE.get(o['head_attitude'],o['head_attitude'])}。",
          '- 旧 AI 语义标签：'+', '.join(o['semantic_tags'])+'。','', '**已有审阅证据与理由**','']
        for quote in o['full_evidence']:full += ['> '+quote,'']
        full += [o['note'],'','**完整原文（冻结处理后的正文）**','',r['text'],'',
                 '**人工复核待填**：全文态度____；主要评价对象____；原文证据____；评分／文字是否背离____；结论与理由____。']
    (data/'94-candidates-full.md').write_text('\n'.join(full)+'\n')
    write_json(data/'94-candidates.json',{'study_id':STUDY,'count':94,'rows':rows,
               'independent_human_review_completed':False,'labels_changed':False,'test_accessed':False})
    # Compact report contains the entire index; full reviews stay in ignored artifacts.
    report_index=[line.replace('](#',f"]({data/'94-candidates-full.md'}#") for line in intro]
    report_text=report_index+['','## 完整材料','',
       f"- [94 条全文、模型分数与已有 AI 审阅]({data/'94-candidates-full.md'})",
       f"- [完整 JSON 数据]({data/'94-candidates.json'})",'',
       '信息来源为原训练导出及 2026-10-15 已完成的 AI 审阅台账。本次只做整理和一致性核验，未把任何条目标成已确认错标，也未删样本、改标签或分配训练权重。此 94 条是模型冲突富集样本，不能估计整套 IMDB 的标签噪声率。','']
    (report/'REPORT.md').write_text('\n'.join(report_text))
    sources['scripts/export_laya_94_candidate_information.py']=file_digest(Path(__file__))
    result={'study_id':STUDY,'state':'94_training_information_exported','count':94,
            'original_labels':dict(Counter(r['original_label_name'] for r in rows)),
            'existing_ai_category_counts':dict(counts),'truncated_original_prefix':16,
            'independent_human_review_completed':False,'confirmed_label_error_count':None,
            'training_performed':False,'model_inference_performed':False,'development_data_read':False,
            'test_accessed':False,'labels_changed':False,'models':prior['models'],
            'input_and_source_sha256':sources,'output_sha256':{str(p.relative_to(ROOT)):file_digest(p)
                for p in [report/'REPORT.md',data/'94-candidates-full.md',data/'94-candidates.json']},
            'previous_boundary_audit':{'training_input_hashes_verified':boundary['training_input_hashes_verified'],
                'raw_train_members_verified':boundary['raw_train_members_verified']}}
    write_json(report/'summary.json',result)
    print(json.dumps({'count':94,'full_text_file':str(data/'94-candidates-full.md'),
                      'index':str(report/'REPORT.md'),'categories':dict(counts)},ensure_ascii=False))


if __name__=='__main__':main()
