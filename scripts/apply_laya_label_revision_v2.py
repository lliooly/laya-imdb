"""Build a separate reviewed data version, retaining original and v1 datasets."""
import argparse
from collections import Counter
import json
from pathlib import Path

from common import ROOT, file_digest, write_json

RUN='laya-94-user-label-revision-20261017-v2'
NAMES={0:'neg',1:'pos',None:'删除'}
ACTION={'flip':'翻转','keep':'保持','pending':'待定，暂留原标签','delete':'删除'}


def read(path):return json.loads(Path(path).read_text())


def load():
    c=read(ROOT/'configs'/f'{RUN}.json')
    if c['run_id']!=RUN:raise ValueError('Unexpected revision ID')
    for rel,sha in c['source_sha256'].items():
        if file_digest(ROOT/rel)!=sha:raise ValueError('Protected source changed: '+rel)
    candidates={r['index']:r for r in read(ROOT/c['source_inventory'])['rows']}
    previous=read(ROOT/c['prior_config'])
    prior_manifest=read(ROOT/c['prior_manifest'])
    frozen=read(ROOT/c['original_training_manifest'])
    decisions={index:{'candidate_id':index,'sample_id':r['sample_id'],'original_label':r['original_label'],
                        'action':'pending','new_label':r['original_label'],'reason':'尚未裁定'}
               for index,r in candidates.items()}
    for d in previous['decisions']:decisions[d['candidate_id']]=dict(d,reason='第一轮用户逐条人工裁定')
    pending_before={index for index,d in decisions.items() if d['action']=='pending'}
    if (len(c['updates'])!=27 or {d['candidate_id'] for d in c['updates']}!=pending_before
            or len({d['candidate_id'] for d in c['updates']})!=27):
        raise ValueError('Updates must address exactly the previous 27 pending items')
    for d in c['updates']:decisions[d['candidate_id']]=dict(d)
    if len(decisions)!=94:raise ValueError('Candidate count changed')
    for index,d in decisions.items():
        r=candidates[index]
        if (d['sample_id']!=r['sample_id'] or d['original_label']!=r['original_label']
                or d['sample_id'] not in set(frozen['gradient_train_ids']) or d['action'] not in ACTION
                or (d['action']=='delete' and d['new_label'] is not None)
                or (d['action']!='delete' and d['new_label'] not in (0,1))
                or (d['action']=='flip' and d['new_label']!=1-d['original_label'])
                or (d['action'] in ('keep','pending') and d['new_label']!=d['original_label'])):
            raise ValueError('Invalid source identity, action, label or partition')
    counts=Counter(d['action'] for d in decisions.values())
    if dict(counts)!=c['expected_cumulative_counts']:raise ValueError('Cumulative counts differ')
    return c,candidates,decisions,frozen,prior_manifest


def verify():
    c,candidates,decisions,frozen,prior_manifest=load()
    folder=(ROOT/c['output_train']).parent
    m=read(folder/'manifest.json')
    if (m['config_sha256']!=file_digest(ROOT/'configs'/f'{RUN}.json')
            or m['train']['sha256']!=file_digest(ROOT/c['output_train'])):
        raise ValueError('Revised dataset or bound configuration differs')
    by_id={d['sample_id']:d for d in decisions.values()}
    dropped={d['sample_id'] for d in decisions.values() if d['action']=='delete'}
    changed=[];ids=[];labels=Counter();gradient_labels=Counter();removed=[]
    with (ROOT/c['original_train']).open() as source,(ROOT/c['output_train']).open() as output:
        for original_line in source:
            a=json.loads(original_line);d=by_id.get(a['id'])
            if a['id'] in dropped:
                removed.append(a['id']);continue
            line=next(output,None)
            if line is None:raise ValueError('Revised file ended early')
            b=json.loads(line)
            expected=dict(a,label=d['new_label']) if d else a
            if b!=expected:raise ValueError('Unrequested text, label, ID or metadata change')
            if a['label']!=b['label']:changed.append(a['id'])
            elif line!=original_line:raise ValueError('Unchanged row bytes differ')
            ids.append(b['id']);labels[b['label']]+=1
            if b['id'] in set(m['gradient_train_ids']):gradient_labels[b['label']]+=1
        if next(output,None) is not None:raise ValueError('Unexpected extra revised rows')
    expected_ids=[sid for sid in frozen['train']['ids'] if sid not in dropped]
    if (ids!=expected_ids or ids!=m['train']['ids'] or len(ids)!=22497 or len(set(ids))!=22497
            or len(changed)!=61 or set(removed)!=dropped
            or m['gradient_train_ids']!=[sid for sid in frozen['gradient_train_ids'] if sid not in dropped]
            or len(m['gradient_train_ids'])!=22097 or m['calibration_ids']!=frozen['calibration_ids']):
        raise ValueError('Counts, membership or preserved order differ')
    # Compare directly to v1 as well as the original. Only three new flips/deletions.
    delta=[]
    with (ROOT/c['prior_train']).open() as source,(ROOT/c['output_train']).open() as output:
        for old_line in source:
            old=json.loads(old_line)
            if old['id'] in dropped:continue
            new_line=next(output);new=json.loads(new_line)
            if old['label']!=new['label']:delta.append(old['id'])
            elif old_line!=new_line:raise ValueError('Unchanged v1 row bytes differ')
    if set(delta)!={d['sample_id'] for d in c['updates'] if d['action']=='flip'}:
        raise ValueError('Incremental flips differ')
    return {'state':'verified','run_id':RUN,'train_count':22497,'gradient_count':22097,
            'calibration_count':400,'labels_flipped_vs_original':61,'labels_flipped_vs_v1':3,
            'negative_to_positive_vs_original':33,'positive_to_negative_vs_original':28,
            'confirmed_keep':23,'pending_kept_original':7,'deleted':3,
            'class_counts':dict(labels),'gradient_class_counts':dict(gradient_labels),
            'original_and_v1_source_hashes_unchanged':True,'retained_text_and_id_fields_unchanged':True,
            'untouched_rows_byte_identical':True,'remaining_order_preserved':True,
            'calibration_membership_and_labels_unchanged':True,'new_partition_or_resampling':False,
            'validation_body_accessed':False,'test_accessed':False,'training_performed':False,
            'dataset_sha256':file_digest(ROOT/c['output_train'])}


def apply():
    c,candidates,decisions,frozen,prior_manifest=load()
    folder=(ROOT/c['output_train']).parent;report=ROOT/'reports'/RUN
    if folder.exists() or report.exists():raise FileExistsError('Never overwrite an existing dataset revision')
    folder.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    by_id={d['sample_id']:d for d in decisions.values()}
    removed=[];ids=[]
    with (ROOT/c['prior_train']).open() as source,(ROOT/c['output_train']).open('x') as output:
        for line in source:
            row=json.loads(line);d=by_id.get(row['id'])
            if d and d['action']=='delete':removed.append(row);continue
            if d and row['label']!=d['new_label']:
                row['label']=d['new_label'];line=json.dumps(row,ensure_ascii=False)+'\n'
            output.write(line);ids.append(row['id'])
    dropped={r['id'] for r in removed}
    # Removed rows are retained as a reversible backup, not sent to another partition.
    with (folder/'removed-training-rows.jsonl').open('x') as out:
        for row in removed:out.write(json.dumps(row,ensure_ascii=False)+'\n')
    ledger=[]
    for index,d in sorted(decisions.items()):
        ledger.append(dict(d,retained_in_training=d['action']!='delete',
                           text_sha256=candidates[index]['text_sha256']))
    write_json(folder/'cumulative-label-patch.json',{'run_id':RUN,'prior_run_id':c['prior_run_id'],
                'authority':'user_provided_manual_adjudication','rows':ledger})
    write_json(folder/'original-labels-94.json',{'run_id':RUN,'rows':[
        {'candidate_id':index,'sample_id':r['sample_id'],'original_label':r['original_label'],
         'text_sha256':r['text_sha256']} for index,r in sorted(candidates.items())]})
    m={'format':'laya-user-label-revision-v2','run_id':RUN,'prior_run_id':c['prior_run_id'],
       'config_path':f'configs/{RUN}.json','config_sha256':file_digest(ROOT/'configs'/f'{RUN}.json'),
       'original_train':c['original_train'],'original_train_sha256':c['source_sha256'][c['original_train']],
       'prior_train':c['prior_train'],'prior_train_sha256':c['source_sha256'][c['prior_train']],
       'train':{'path':c['output_train'],'sha256':file_digest(ROOT/c['output_train']),'count':22497,'ids':ids},
       'gradient_train_ids':[sid for sid in frozen['gradient_train_ids'] if sid not in dropped],
       'calibration_ids':frozen['calibration_ids'],'validation':prior_manifest['validation'],
       'official_test':prior_manifest['official_test'],'deleted_gradient_ids':sorted(dropped),
       'effective_counts':c['expected_cumulative_counts'],'manual_decision_provider':'user',
       'training_performed':False,'requires_new_training_run_id_and_revised_source_hash_contract':True,
       'source_paths_and_scores_preserved_as_original_provenance':True}
    write_json(folder/'manifest.json',m)
    checked=verify();write_json(report/'verification.json',checked)
    write_json(report/'review-ledger.json',{'run_id':RUN,'authority':'user_provided_manual_adjudication','rows':ledger})
    pending=[d for d in ledger if d['action']=='pending']
    md=['# 剩余 7 条待定训练样本','',
        '最终裁定由用户提供。本轮继续待定的 7 条暂留原标签并保留训练成员身份，没有自动删除或改成中性类。','']
    for d in pending:
        r=candidates[d['candidate_id']]
        md += ['## '+d['candidate_id']+' — '+d['sample_id'],'',
               f"原标签 / 当前标签：{NAMES[d['original_label']]}；原评分 {r['source_rating_from_filename']}/10。",'',
               '**用户本轮待定理由**：'+d['reason'],'','**全文**','',r['text'],'',
               '**后续决定**：保持 / 翻转 / 删除 / 继续待定；理由：____。','']
    (report/'unresolved-7.md').write_text('\n'.join(md))
    write_json(report/'unresolved-7.json',{'run_id':RUN,'rows':[dict(d,text=candidates[d['candidate_id']]['text']) for d in pending]})
    lines=['# 用户人工裁定：训练数据修订 v2','',
       '原始训练集、上一版修订均保留；本版按用户两轮裁定累计翻转 61 条、确认保持 23 条、删除 3 条、待定 7 条。最终决定归属于用户人工裁定，先前 AI 注释仅为背景资料。', '',
       '**C012 已澄清：保持 neg。** 用户选择“按反讽判断，保持 neg”，覆盖先前“翻转”字样。C010 按用户明确的反讽负向理由保持 neg。','',
       '## 本轮新增操作','',
       '| 操作 | 数量 | 编号 |','|---|---:|---|']
    for action in ('flip','keep','delete','pending'):
        rows=[d for d in c['updates'] if d['action']==action]
        lines.append(f"| {ACTION[action]} | {len(rows)} | {', '.join(d['candidate_id'] for d in rows)} |")
    lines += ['', '## 训练版本与备份','',
       f"- [新训练数据]({ROOT/c['output_train']})；[新 manifest]({folder/'manifest.json'})。",
       f"- [原始训练数据]({ROOT/c['original_train']})：22500 条；[上一版]({ROOT/c['prior_train']})：22500 条。本版保留 22497 条。",
       '- 原梯度 22100 条中移除 3 条，现为 22097 条；原 400 条校准完整保留，开发 / 测试不改、不重新划分。',
       f"- [删除的 3 条正文备份]({folder/'removed-training-rows.jsonl'})；[全部 94 条原标签备份]({folder/'original-labels-94.json'})。",
       f"- [累计修改补丁]({folder/'cumulative-label-patch.json'})；[剩余 7 条单独文档]({report/'unresolved-7.md'})。", '',
       '## 本轮逐条裁定','', '| 编号 | 样本 ID | 操作 | 原标签 → 本版 | 用户理由 |', '|---|---|---|---|---|']
    for d in c['updates']:
        reason=d['reason'].replace('|','\\|')
        lines.append(f"| {d['candidate_id']} | `{d['sample_id']}` | {ACTION[d['action']]} | {NAMES[d['original_label']]} → {NAMES[d['new_label']]} | {reason} |")
    lines += ['', '## 核验与后续使用','',
       '- 原始版与 v1 的文件哈希均未变化。相对原始版，只翻转指定 61 条标签、删除指定 3 条；相对 v1，仅新增 3 条翻转和 3 条删除。其余行字节一致、保留样本顺序与正文不变。',
       f"- 本版类别数：neg {checked['class_counts'][0]}、pos {checked['class_counts'][1]}；梯度部分 neg {checked['gradient_class_counts'][0]}、pos {checked['gradient_class_counts'][1]}。",
       '- 校准 400 条不受修改；开发与官方测试正文未读取。待定 7 条原样保留。样本 ID 和文件名评分仍表示原始来源，训练标签以本版 label 字段为准。',
       '- 已具备后续原版 / 改版对照所需的数据版本与出处；下次须用新 run_id 绑定本版 manifest。本轮没有启动训练，也没有因本次修改重算历史模型成绩。',
       '- 复核：`uv run --locked --offline --no-sync python scripts/apply_laya_label_revision_v2.py --verify`。','']
    (report/'REPORT.md').write_text('\n'.join(lines))
    result=dict(checked,dataset_path=c['output_train'],manifest_path=str((folder/'manifest.json').relative_to(ROOT)),
                manual_decision_provider='user',C012_final_label='neg',source_sha256=c['source_sha256'],
                script_sha256=file_digest(Path(__file__)),output_sha256={str(p.relative_to(ROOT)):file_digest(p) for p in
                [folder/'manifest.json',folder/'cumulative-label-patch.json',folder/'original-labels-94.json',
                 folder/'removed-training-rows.jsonl',report/'REPORT.md',report/'review-ledger.json',report/'unresolved-7.md',report/'unresolved-7.json']})
    write_json(report/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','output_sha256')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:print(json.dumps(verify(),ensure_ascii=False))
    else:apply()
