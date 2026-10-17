"""Frozen partitions and tokenizer windows; preprocessing does not require torch."""
import json
from pathlib import Path

from common import read_jsonl,file_digest,write_json,distribution

STUDY='deberta-heterogeneous-s42-modal-20261017-v2'
VOLUME='end-imdb-laya-robust-deferred-20261015-v2'
BASE='reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json'
ROOT=Path(__file__).resolve().parents[1]
RECIPE=dict(seed=42,epochs=2,max_length=512,micro_batch=8,accumulation=4,
            learning_rate=1e-5,weight_decay=.01,adam_epsilon=1e-6,
            betas=[.9,.999],gradient_clip=1.,warmup_steps=83,total_steps=1382,
            precision='float32',tf32=False,gradient_checkpointing=True)


def read(path):return json.loads(Path(path).read_text())


def encode(tok,text):
    full=tok.encode(text,add_special_tokens=False,truncation=False)
    budget=512-tok.num_special_tokens_to_add(pair=False)
    if budget<=0:raise ValueError('Special tokens exhaust review budget')
    chosen=full if len(full)<=budget else full[:(budget+1)//2]+full[-(budget//2):]
    ids=tok.build_inputs_with_special_tokens(chosen)
    if len(ids)>512 or len(ids)!=len(chosen)+512-budget:
        raise ValueError('Unexpected special-token layout')
    return dict(input_ids=ids,review_tokens=len(full),truncated=len(full)>budget)


def training_rows(root=ROOT):
    split=read(root/BASE);path=root/'data/processed/train.jsonl'
    if file_digest(path)!=split['train']['source_sha256']:raise ValueError('Original data changed')
    rows=list(read_jsonl(path));ids=[r['id'] for r in rows]
    gradient,calibration,development=[set(x) for x in (split['gradient_train_ids'],split['calibration_ids'],split['validation']['ids'])]
    if (ids!=split['train']['ids'] or len(set(ids))!=22500 or len(gradient)!=22100
            or len(calibration)!=400 or len(development)!=2500 or gradient&calibration
            or gradient&development or calibration&development or gradient|calibration!=set(ids)):
        raise ValueError('Frozen partitions differ or overlap')
    if any(type(r['label']) is not int or r['label'] not in (0,1) for r in rows):raise ValueError('Invalid binary labels')
    return {r['id']:r for r in rows},split


def prepare(root=ROOT):
    from transformers import AutoTokenizer
    lock=read(root/'configs/deberta_model_lock.json');source=root/lock['local_dir']
    for name,sha in lock['sha256'].items():
        if file_digest(source/name)!=sha:raise ValueError('Official asset changed: '+name)
    rows,split=training_rows(root)
    folder=root/'artifacts'/STUDY
    if (folder/'train-tokens.jsonl').exists():raise FileExistsError('Keep previous prepared inputs')
    folder.mkdir(parents=True,exist_ok=True)
    # Preserve official SentencePiece byte fallback; the converted fast tokenizer warns it drops it.
    tok=AutoTokenizer.from_pretrained(str(source),local_files_only=True,use_fast=False)
    tok.save_pretrained(folder/'tokenizer')
    lengths=[];records=[]
    with (folder/'train-tokens.jsonl').open('x') as out:
        for sid in split['gradient_train_ids']:
            row=rows[sid];item=dict(id=sid,label=row['label'],**encode(tok,row['text']))
            out.write(json.dumps(item,separators=(',',':'))+'\n');records.append(item);lengths.append(item['review_tokens'])
    order=list(range(22100));__import__('random').Random(42).shuffle(order)
    write_json(folder/'input-summary.json',dict(study_id=STUDY,count=22100,pad_token_id=tok.pad_token_id,
        special_tokens=tok.num_special_tokens_to_add(pair=False),lengths=distribution(lengths),
        truncated=sum(x['truncated'] for x in records),
        pilot_representative=order[:128],pilot_stress=sorted(range(22100),key=lambda i:(-lengths[i],i))[:32],
        calibration_tokenized=False,development_accessed=False,test_accessed=False,
        train_tokens_sha256=file_digest(folder/'train-tokens.jsonl')))
    print(json.dumps(read(folder/'input-summary.json'),ensure_ascii=False),flush=True)


if __name__=='__main__':prepare()
