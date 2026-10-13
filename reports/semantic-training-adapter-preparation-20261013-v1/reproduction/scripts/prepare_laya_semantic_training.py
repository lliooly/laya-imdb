"""Bind a human-reviewed semantic pack to strict A/B/C configs and exposure contracts."""
import argparse
import json
from pathlib import Path

from common import ROOT, file_digest, write_json
from laya_semantic_training import load_semantic, schedule_digest
from semantic_pair_pilot import make_schedule, materialize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=ROOT/'reports/laya-l512-s42-ce/manifest.json')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(42,43),default=42)
    args=parser.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(ROOT/'reports') or out.exists():
        raise ValueError('Use a new repository reports preparation directory')
    options={'arm':'A'}
    for name,path in (('pack',args.pack),('review',args.review),('manifest',args.manifest)):
        path=path.resolve()
        options[name+'_file']=str(path.relative_to(ROOT))
        options[name+'_sha256']=file_digest(path)
    # All review, source and byte checks precede creating any output directory.
    bundle=load_semantic(options)
    pack=bundle['pack']
    labels={i:bundle['source'][i]['label'] for i in bundle['gradient_ids']}
    slots=make_schedule(bundle['gradient_ids'],labels,[r['parent_id'] for r in pack['rows']],seed=args.seed)
    base=json.loads((ROOT/'configs/laya-rtx4090-ce-resampling-control-pool45-s43-epochs2-strict-20261013-v1.json').read_text())
    base.pop('sampling');base['training']['seed']=args.seed
    arms={a:materialize(slots,bundle['source'],pack['rows'],a) for a in ('A','B','C')}
    for epochs in zip(*(arms[a] for a in ('A','B','C'))):
        for a,b,c in zip(*epochs):
            if (a['parent_id'],a['label'])!=(b['parent_id'],b['label']) or (a['parent_id'],a['label'])!=(c['parent_id'],c['label']):
                raise ValueError('Semantic exposure alignment mismatch')
            if not a['augmented'] and not a==b==c:
                raise ValueError('Semantic nonaugmented positions differ')
    lock=json.loads((ROOT/'configs/laya_model_lock.json').read_text())
    if pack['tokenizer_sha256']!=lock['sha256']['tokenizer/tokenizer.json']:
        raise ValueError('Semantic pack tokenizer mismatch')
    out.mkdir()
    for arm in ('A','B','C'):
        spec=json.loads(json.dumps(base));spec['run_id']=out.name+'-'+arm
        spec['semantic_pairs']=dict(options,arm=arm)
        write_json(out/f'{arm}-config.json',spec)
        smoke=json.loads(json.dumps(spec));smoke['run_id']+='-smoke';smoke['validation_limit']=32
        smoke['semantic_pairs']['smoke_items']=128
        write_json(out/f'{arm}-smoke-config.json',smoke)
    write_json(out/'execution-contract.json',{'state':'human_review_bound_configs_prepared',
        'formal_training_ready':False,'remaining_admission':['near_duplicate_audit','full_encoder_cuda_smoke',
            'complete_prior_weight_backup_and_disk_check','semantic_validation_protocol_freeze','current_cost_estimate'],
        'source_count':len(pack['rows']),'seed':args.seed,'epochs':2,'expected_exposures':sum(map(len,slots)),
        'expected_schedule_sha256':[schedule_digest(e) for e in slots],
        'expected_updates':2*((len(bundle['gradient_ids'])+7)//8),
        'bindings':options,'test_accessed':False,'training_performed':False,
        'note':'Reviewed configs do not establish semantic quality or GPU admission; pilot-only pool must be expanded before formal trial.'})
    print('Prepared human-review-bound A/B/C configs; remaining admission is explicit.')


if __name__=='__main__':main()
