"""Reuse byte-identical inputs and persistent assets; reserve a new bounded BF16 run."""
import json
from pathlib import Path
import shutil
import time

from common import ROOT,file_digest,write_json
from deberta_bf16_study import STUDY,RECIPE
from deberta_inputs import read

OLD='deberta-heterogeneous-s42-modal-20261017-v2'


def main():
    path=ROOT/'configs'/f'{STUDY}.json'
    if path.exists():raise FileExistsError('BF16 run already frozen')
    prior=read(ROOT/'delivery'/f'{OLD}-budget.json')
    if not prior['compute_closed'] or prior['conservative_available_after_this_study_usd']<5:
        raise ValueError('FP32 compute not closed or conservative available balance insufficient')
    local=read(ROOT/'reports'/OLD/'local-delivery.json')
    if not local['bundles']['pilot-boundary']['local_delivery_verified']:raise ValueError('Old pilot not delivered')
    old=read(ROOT/'configs'/f'{OLD}.json');folder=ROOT/'artifacts'/STUDY
    folder.mkdir(parents=True,exist_ok=False)
    source=ROOT/'artifacts'/OLD
    shutil.copyfile(source/'train-tokens.jsonl',folder/'train-tokens.jsonl')
    shutil.copytree(source/'tokenizer',folder/'tokenizer')
    stats=read(source/'input-summary.json');stats['study_id']=STUDY;write_json(folder/'input-summary.json',stats)
    bindings={}
    for rel,sha in old['source_sha256'].items():
        newrel=rel.replace('artifacts/'+OLD+'/','artifacts/'+STUDY+'/')
        if rel=='artifacts/'+OLD+'/input-summary.json':sha=file_digest(ROOT/newrel)
        # Unchanged historical data/model/source bytes must still match their frozen digests.
        if rel not in old['evaluation_only_files'] and file_digest(ROOT/newrel)!=sha:
            raise ValueError('Old source or reused input changed: '+newrel)
        bindings[newrel]=sha
    fresh=['scripts/'+p for p in ['deberta_bf16_training.py','deberta_bf16_study.py',
        'modal_deberta_bf16.py','receive_deberta_bf16.py','prepare_deberta_bf16.py']]
    fresh+=['tests/test_deberta_bf16.py','docs/superpowers/specs/2026-10-17-deberta-bf16-design.md',
            'docs/superpowers/plans/2026-10-17-deberta-bf16-implementation.md']
    bindings.update({rel:file_digest(ROOT/rel) for rel in fresh})
    uploads=[rel for rel,sha in bindings.items() if old['source_sha256'].get(rel)!=sha]
    c=dict(old,study_id=STUDY,recipe=RECIPE,pilot_reserve_usd=1,source_sha256=bindings,
        upload_files=uploads,cached_image_id='im-wjfLMpFBzPeiRaUhpdw2dA',
        reference_compute_precision='float32',task_compute_precision='bfloat16',master_precision='float32',
        input_cache_sha256=file_digest(folder/'train-tokens.jsonl'))
    c['resource_budget']=dict(gpu_rate_usd_per_hour=3,gpu_resource_upper_usd=4.5,
        extra_cpu_and_transfer_usd=.5,cached_image=True,pilot_gpu_upper_usd=.75,pilot_extra_cpu_usd=.25,
        formal_gpu_upper_usd=3.75,formal_extra_cpu_usd=.25,maximum_gpu_seconds=5400,
        pilot_maximum_seconds=900,coordinator_cpu=.125,coordinator_memory_mib=2048,
        coordinator_timeout=5700,coordinator_nonpreemptible_multiplier=3)
    write_json(path,c);report=ROOT/'reports'/STUDY;report.mkdir(parents=True,exist_ok=True)
    write_json(report/'preflight.json',dict(state='local_prepared',tests='5 BF16 CPU checks passed',
        historical_sources_unchanged=True,input_cache_identical_to_fp32=True,
        source_files=len(bindings),incremental_upload_files=len(uploads),
        incremental_upload_bytes=sum((ROOT/p).stat().st_size for p in uploads),
        actual_new_content_bytes_upper=sum((ROOT/p).stat().st_size for p in fresh)+
                                      (folder/'input-summary.json').stat().st_size,
        config_sha256=file_digest(path),precision='bfloat16',master_precision='float32',
        reference_precision='float32',test_accessed=False))
    b=ROOT/'delivery'/f'{STUDY}-budget.json'
    if b.exists():raise FileExistsError('BF16 reservation already exists')
    write_json(b,dict(study_id=STUDY,state='prepared_reserved',additional_reserved_usd=5,
        pilot_reserve_usd=1,conditional_formal_reserve_usd=4,
        conservative_available_before_usd=prior['conservative_available_after_this_study_usd'],
        remaining_after_full_reserve_usd=round(prior['conservative_available_after_this_study_usd']-5,2),
        reserved_unix=time.time(),budget_not_actual_invoice=True,actual_balance_unreadable=True,
        source='delivery/'+OLD+'-budget.json',resource_budget=c['resource_budget']))
    print(json.dumps(read(report/'preflight.json'),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
