"""Apply explicit user label decisions to a new training-data version only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from common import ROOT, file_digest, write_json

RUN='laya-94-user-label-revision-20261017-v1'
LABEL={0:'neg（负向）',1:'pos（正向）'}
GROUPS=[('疑似反讽','C010 C012 C013 C020 C033 C043 C047'.split()),
        ('烂片趣味 / Cult 欣赏','C054 C057 C077 C078 C089 C093 C094'.split()),
        ('褒贬参半、评价对象或边界态度','C018 C026 C050 C061 C068 C070 C076 C081 C083'.split()),
        ('缺少明确本人评价或评价对象偏移','C014 C041 C084 C086'.split())]


def read(path):return json.loads(Path(path).read_text())


def load():
    c=read(ROOT/'configs'/f'{RUN}.json')
    for rel,sha in c['source_sha256'].items():
        if file_digest(ROOT/rel)!=sha:raise ValueError('Frozen source changed: '+rel)
    original=read(ROOT/c['source_inventory'])['rows']
    candidates={r['index']:r for r in original}
    decisions={r['candidate_id']:r for r in c['decisions']}
    if (len(candidates)!=94 or len(decisions)!=len(c['decisions'])
            or not set(decisions)<=set(candidates)):
        raise ValueError('Repeated or unknown decision ID')
    frozen=read(ROOT/c['source_training_manifest'])
    gradient=set(frozen['gradient_train_ids']);calibration=set(frozen['calibration_ids'])
    dev=set(frozen['validation']['ids'])
    for index,d in decisions.items():
        r=candidates[index]
        if (d['sample_id']!=r['sample_id'] or d['original_label']!=r['original_label']
                or d['sample_id'] not in gradient or d['sample_id'] in calibration|dev
                or d['new_label'] not in (0,1)
                or d['action'] not in ('keep','flip')
                or (d['action']=='flip')!=(d['original_label']!=d['new_label'])):
            raise ValueError('Decision identity, action or data split differs')
    counts={'flip_total':sum(d['action']=='flip' for d in decisions.values()),
            'negative_to_positive':sum(d['original_label']==0 and d['new_label']==1 for d in decisions.values()),
            'positive_to_negative':sum(d['original_label']==1 and d['new_label']==0 for d in decisions.values()),
            'keep':sum(d['action']=='keep' for d in decisions.values()),'unresolved':94-len(decisions)}
    if counts!=c['effective_counts']:raise ValueError('Effective decision counts differ')
    return c,candidates,decisions,frozen


def verify():
    c,candidates,decisions,frozen=load()
    data=ROOT/c['output_train'];manifest=read(data.parent/'manifest.json')
    if (manifest['run_id']!=RUN or manifest['config_sha256']!=file_digest(ROOT/'configs'/f'{RUN}.json')
            or manifest['train']['sha256']!=file_digest(data)):
        raise ValueError('Revised dataset identity or SHA differs')
    changed={};original_counts=Counter();new_counts=Counter();grad_counts=Counter()
    expected={d['sample_id']:d for d in decisions.values() if d['action']=='flip'}
    ids=[]
    source=ROOT/c['source_train']
    gradient=set(frozen['gradient_train_ids']);calibration=set(frozen['calibration_ids'])
    with source.open() as old,data.open() as new:
        for old_line,new_line in zip(old,new,strict=True):
            a,b=json.loads(old_line),json.loads(new_line)
            sid=a['id'];ids.append(sid);original_counts[a['label']]+=1;new_counts[b['label']]+=1
            if sid in gradient:grad_counts[b['label']]+=1
            d=expected.get(sid)
            target=dict(a,label=d['new_label']) if d else a
            if b!=target:raise ValueError('Unexpected text, ID, label or metadata change: '+sid)
            if d:
                if a['label']!=d['original_label']:raise ValueError('Original label changed')
                changed[sid]={'before':a['label'],'after':b['label']}
            elif old_line!=new_line:
                raise ValueError('Untargeted row bytes changed: '+sid)
            if sid in calibration and a!=b:raise ValueError('Calibration row changed')
    if (ids!=frozen['train']['ids'] or len(ids)!=22500 or len(set(ids))!=22500
            or set(changed)!=set(expected) or len(changed)!=58):
        raise ValueError('Changed IDs, counts or original row order differ')
    return {'state':'verified','run_id':RUN,'training_rows':22500,'gradient_rows':22100,
            'calibration_rows_unchanged':400,'changed_labels':58,
            'negative_to_positive':32,'positive_to_negative':26,'confirmed_keep':9,
            'unresolved_retaining_original_label':27,'original_class_counts':dict(original_counts),
            'revised_class_counts':dict(new_counts),'revised_gradient_class_counts':dict(grad_counts),
            'only_label_field_changed':True,'unaffected_rows_byte_identical':True,
            'frozen_sources_sha256_unchanged':True,'row_order_and_all_ids_unchanged':True,
            'validation_body_accessed':False,'test_accessed':False,'training_performed':False,
            'dataset_sha256':file_digest(data)}


def apply():
    c,candidates,decisions,frozen=load()
    data=ROOT/c['output_train'];folder=data.parent;report=ROOT/'reports'/RUN
    if folder.exists() or report.exists():raise FileExistsError('Refuse to overwrite a label-revision version')
    attachment=Path(c['uncertain_attachment']['source_path'])
    if file_digest(attachment)!=c['uncertain_attachment']['sha256']:
        raise ValueError('User attachment differs')
    text=attachment.read_text()
    matches=list(re.finditer(r'(C\d{3})\s*(?:\(\s*)?(train/(?:neg|pos)/\d+_\d+\.txt)',text))
    pending=set(candidates)-set(decisions)
    if set(m[1] for m in matches)!=pending or len(matches)!=27:
        raise ValueError('Attachment and unresolved complement differ')
    excerpts={}
    for i,m in enumerate(matches):
        if candidates[m[1]]['sample_id']!=m[2]:raise ValueError('Attachment source path differs')
        fragment=text[m.start():matches[i+1].start() if i+1<len(matches) else len(text)]
        fragment=re.split(r'类别 [ABC]：|三、|\(注：',fragment)[0].strip()
        excerpts[m[1]]=fragment
    if set(index for _,indices in GROUPS for index in indices)!=pending:
        raise ValueError('Unresolved document grouping differs')
    folder.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    by_id={d['sample_id']:d for d in decisions.values()}
    with (ROOT/c['source_train']).open() as source,data.open('x') as out:
        for line in source:
            r=json.loads(line);d=by_id.get(r['id'])
            if d and d['action']=='flip':
                if r['label']!=d['original_label']:raise ValueError('Original label differs')
                r['label']=d['new_label'];line=json.dumps(r,ensure_ascii=False)+'\n'
            out.write(line)
    ledger=[]
    for index,r in sorted(candidates.items()):
        d=decisions.get(index)
        ledger.append({'candidate_id':index,'sample_id':r['sample_id'],'original_label':r['original_label'],
              'effective_label':d['new_label'] if d else r['original_label'],
              'status':('user_confirmed_'+d['action']) if d else 'unresolved_keep_original_for_now',
              'text_sha256':r['text_sha256'],'source_rating_from_filename':r['source_rating_from_filename']})
    write_json(folder/'label-patch.json',{'run_id':RUN,'authority':'explicit_user_decisions',
                'decisions':c['decisions'],'unresolved_candidate_ids':sorted(pending),
                'declared_counts':c['declared_counts'],'effective_counts':c['effective_counts']})
    write_json(folder/'original-labels-94.json',{'run_id':RUN,'source_train_sha256':c['source_sha256'][c['source_train']],
               'rows':[{'candidate_id':r['index'],'sample_id':r['sample_id'],
                        'original_label':r['original_label'],'text_sha256':r['text_sha256']} for r in candidates.values()]})
    write_json(report/'review-ledger.json',{'run_id':RUN,'rows':ledger})
    split=read(ROOT/c['source_split_manifest'])
    manifest={'format':'laya-user-label-revision-v1','run_id':RUN,
       'config_path':f'configs/{RUN}.json','config_sha256':file_digest(ROOT/'configs'/f'{RUN}.json'),
       'source_train_path':c['source_train'],'source_train_sha256':c['source_sha256'][c['source_train']],
       'source_training_manifest':c['source_training_manifest'],'source_split_manifest':c['source_split_manifest'],
       'train':{'path':c['output_train'],'sha256':file_digest(data),'count':22500,'ids':frozen['train']['ids']},
       'gradient_train_ids':frozen['gradient_train_ids'],'calibration_ids':frozen['calibration_ids'],
       'validation':{'path':'data/processed/validation.jsonl','sha256':split['jsonl_sha256']['validation'],
                     'ids':frozen['validation']['ids'],'changed':False},
       'official_test':{'path':'data/processed/test.jsonl','sha256':split['jsonl_sha256']['test'],
                        'changed':False,'accessed':False},
       'effective_counts':c['effective_counts'],'training_performed':False,'labels_from_user_adjudication':True,
       'id_paths_and_filename_scores_describe_original_source_not_revised_labels':True,
       'requires_new_training_run_id_and_revised_source_hash_contract':True,
       'label_patch_sha256':file_digest(folder/'label-patch.json')}
    write_json(folder/'manifest.json',manifest)
    checked=verify()
    write_json(report/'verification.json',checked)
    pending_doc=['# 94 条训练候选中尚未裁定的 27 条','',
       '本文件只列最终逐条清单中未明确裁定的样本。当前训练修订版仍保留这些条目的原标签；没有删除。附件中的拟定倾向仅供复核，未作为修改指令。', '',
       '疑似反讽 7 条；Cult 欣赏 7 条；混合／对象／边界态度 9 条；缺少明确本人评价或对象偏移 4 条。附件最后一类标题写“5 条”，实际列举 4 条，均已纳入。', '',
       '可对每条选择：保持原标签、翻转标签、继续待定；如要剔除，另行明确。评分与样本路径仍是原始来源信息。','']
    for group,indices in GROUPS:
        pending_doc += ['## '+group+'（'+str(len(indices))+' 条）','']
        for index in indices:
            r=candidates[index];o=r['prior_ai_review']
            pending_doc += ['### '+index+' — '+r['sample_id'],'',
             f"- 原标签及当前有效标签：{LABEL[r['original_label']]}；原文件名评分：{r['source_rating_from_filename']}/10。",
             f"- review 长度：{r['review_tokens']} token；旧 466-token 前缀是否截断：{'是' if r['original_head_truncated'] else '否'}。",
             '- 本轮状态：待定，未修改。','', '**你提供的待审核意见（原文）**','',excerpts[index],'',
             '**先前 AI 审阅记录（不是本轮最终裁定）**','',
             f"- 分类：{o['category']}；全文态度：{o['full_attitude']}；置信度：{o['confidence']}。",o['note'],'',
             '**完整评论正文**','',r['text'],'',
             '**最终决定待填**：保持 / 翻转 / 待定；若删除须另行明确。理由与证据：____。','']
    (report/'unresolved-27.md').write_text('\n'.join(pending_doc))
    write_json(report/'unresolved-27.json',{'run_id':RUN,'count':27,'rows':[
         dict(candidates[index],decision_status='unresolved',current_effective_label=candidates[index]['original_label'],
              attachment_tentative_opinion=excerpts[index]) for index in sorted(pending)]})
    lines=['# 94 条训练候选：按用户逐条裁定修订标签','',
       '**完成：58 条翻转（neg→pos 32、pos→neg 26）、9 条确认保持、27 条单列待定。未启动训练。**', '',
       '逐条列举与标题数量不符：标题写“翻转共54条／负转正29条／正转负22条”，实际明确列出32＋26＝58条。执行以唯一的样本编号及 ID 清单为准，没有增补或漏改。待定27条与附件中实际列出的27个编号完全一致。', '',
       '## 修改后的数据','',
       f'- [修订版 train.jsonl]({data})：22500 条、原顺序及 ID 保持，仅指定 58 行的 label 字段变化。',
       f'- [新数据 manifest]({folder/"manifest.json"})：原 22100 梯度／400 校准分区保持；全部翻转位于梯度训练分区。',
       f'- [完整修改补丁]({folder/"label-patch.json"})；[94 条原标签备份]({folder/"original-labels-94.json"})。',
       f'- 原冻结数据 `{c["source_train"]}` 及其 manifest、历史报告均保留。修订版不直接覆盖旧实验源哈希；下次训练需用新 run_id 和本版本 manifest。',
       '- 样本 ID 中 neg/pos 目录及文件名评分保留为原始出处，不按新语义标签重命名，也不据路径重新推导有效标签。', '',
       '## 单独待定文档','',
       f'- [27 条待定：全文、附件意见及先前 AI 审阅]({report/"unresolved-27.md"})。',
       '- 待定全部暂留原标签。附件中“改为／或保持／或删除”等拟定意见不覆盖用户最终明确清单。', '',
       '## 已执行的 58 条翻转','', '| 编号 | 样本 ID | 原标签 | 修订标签 |','|---|---|---|---|']
    for d in c['decisions']:
        if d['action']=='flip':lines.append(f"| {d['candidate_id']} | `{d['sample_id']}` | {LABEL[d['original_label']]} | {LABEL[d['new_label']]} |")
    lines += ['', '## 确认保持的 9 条','', '| 编号 | 样本 ID | 保持标签 |','|---|---|---|']
    for d in c['decisions']:
        if d['action']=='keep':lines.append(f"| {d['candidate_id']} | `{d['sample_id']}` | {LABEL[d['new_label']]} |")
    lines += ['', '## 核验','',
       f"- 新训练集类别数：负向 {checked['revised_class_counts'][0]}、正向 {checked['revised_class_counts'][1]}。原数据各11250；正向净增6。",
       '- 对22500条逐行比较：仅58个指定label改变，全部正文／ID／文本哈希不变；其余22442行连序列化字节也保持。400条校准不变，开发与官方测试正文未读取。',
       '- 用户裁定作为此次标签修订的依据；不将旧AI分类更新成独立人工一致性证明。原始评分和旧标签已保留用于版本比较。',
       '- 复核命令：`uv run --locked --offline --no-sync python scripts/apply_laya_94_user_label_revision.py --verify`。', '']
    (report/'REPORT.md').write_text('\n'.join(lines))
    summary=dict(checked,config_path=f'configs/{RUN}.json',dataset_path=c['output_train'],
        manifest_path=str((folder/'manifest.json').relative_to(ROOT)),
        unresolved_document=str((report/'unresolved-27.md').relative_to(ROOT)),
        declared_counts=c['declared_counts'],effective_counts=c['effective_counts'],
        script_sha256=file_digest(Path(__file__)),input_sha256=c['source_sha256'],
        output_sha256={str(p.relative_to(ROOT)):file_digest(p) for p in
            [folder/'manifest.json',folder/'label-patch.json',folder/'original-labels-94.json',
             report/'review-ledger.json',report/'unresolved-27.md',report/'unresolved-27.json',report/'REPORT.md']})
    write_json(report/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:print(json.dumps(verify(),ensure_ascii=False))
    else:apply()
