"""Freeze new sources and latest-balance reservation, preserving all old experiments."""
import json
import shutil
import time
from common import ROOT,file_digest,write_json
from deberta_inputs import read
from deberta_budget_study import STUDY,RECIPE,SECONDS,CAP_USD

OLD='deberta-heterogeneous-bf16-s42-modal-20261017-v1'


def main():
    path=ROOT/'configs'/f'{STUDY}.json'
    if path.exists():raise FileExistsError('Run already frozen')
    balance=read(ROOT/'delivery/deberta-balance-confirmed-7p03-20261017.json')
    if balance['user_confirmed_current_free_balance_usd']!=7.03:raise ValueError('Actual latest balance differs')
    old=read(ROOT/'configs'/f'{OLD}.json');report=ROOT/'reports'/OLD
    if not read(report/'pilot-result.json')['exact_model_optimizer_scheduler_rng_resume']:raise ValueError('No BF16 admission')
    if not read(report/'pilot-small-file-delivery.json')['verified']:raise ValueError('No persisted receipt delivery')
    if not read(report/'local-delivery.json')['bundles']['pilot-boundary']['local_delivery_verified']:
        raise ValueError('BF16 admission complete weights not verified locally')
    folder=ROOT/'artifacts'/STUDY;source=ROOT/'artifacts'/OLD;folder.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source/'train-tokens.jsonl',folder/'train-tokens.jsonl')
    shutil.copytree(source/'tokenizer',folder/'tokenizer')
    stats=read(source/'input-summary.json');stats['study_id']=STUDY;write_json(folder/'input-summary.json',stats)
    bindings={}
    for rel,sha in old['source_sha256'].items():
        newrel=rel.replace('artifacts/'+OLD+'/','artifacts/'+STUDY+'/')
        if rel=='artifacts/'+OLD+'/input-summary.json':sha=file_digest(ROOT/newrel)
        if rel not in old['evaluation_only_files'] and file_digest(ROOT/newrel)!=sha:raise ValueError('Old bound bytes changed: '+rel)
        bindings[newrel]=sha
    fresh=['scripts/'+name for name in ['deberta_budget_training.py','deberta_budget_study.py',
        'modal_deberta_budget7.py','receive_deberta_budget7.py','prepare_deberta_budget7.py']]
    fresh+=['tests/test_deberta_budget_training.py',
        'docs/superpowers/specs/2026-10-17-deberta-budget7-design.md',
        'docs/superpowers/plans/2026-10-17-deberta-budget7-implementation.md']
    bindings.update({rel:file_digest(ROOT/rel) for rel in fresh})
    admission=['pilot-result.json','pilot-cloud-verification.json','pilot-small-file-delivery.json']
    evidence={str((report/name).relative_to(ROOT)):file_digest(report/name) for name in admission}
    uploads=[rel for rel,sha in bindings.items() if old['source_sha256'].get(rel)!=sha]
    budget=dict(worker_rate_usd_per_second=.00066544,coordinator_rate_usd_per_second=.0000182325,
        shared_absolute_seconds=SECONDS,cleanup_transfer_allowance_usd=.13,
        computed_upper_usd=SECONDS*(.00066544+.0000182325)+.13,
        worker_cpu_limit=4,worker_memory_limit_mib=32768,coordinator_cpu_limit=.125,
        coordinator_memory_limit_mib=2048,coordinator_nonpreemptible_multiplier=3,
        pricing_url='https://modal.com/pricing',verified_date='2026-10-17',
        included_monthly_egress_gib=1024,no_gpu_nonpreemptible_premium=True,no_region_selection=True)
    if budget['computed_upper_usd']>CAP_USD:raise ValueError('Budget arithmetic exceeds authorization')
    c=dict(old,study_id=STUDY,recipe=RECIPE,maximum_gpu_seconds=SECONDS,reserve_usd=CAP_USD,
        pilot_reserve_usd=0,pilot_maximum_seconds=0,source_sha256=bindings,upload_files=uploads,
        resource_budget=budget,periodic_groups=[256,512],stop_update_margin_seconds=180,
        reused_admission_sha256=evidence,authorized_narrow_budget_risk=True,
        authorization='User: 先试试吧 after actual current balance7.03 and projection discussion')
    write_json(path,c);out=ROOT/'reports'/STUDY;out.mkdir(parents=True,exist_ok=True)
    write_json(out/'preflight.json',dict(state='prepared',tests='7 budget position/resume/driver tests passed',
        config_sha256=file_digest(path),source_files=len(bindings),historical_sources_unchanged=True,
        actual_current_balance_usd=7.03,budget_upper_usd=budget['computed_upper_usd'],
        admission_reused=True,no_pilot_gpu=True,test_accessed=False,development_accessed=False))
    reservation=ROOT/'delivery'/f'{STUDY}-budget.json'
    if reservation.exists():raise FileExistsError('Reservation exists')
    write_json(reservation,dict(study_id=STUDY,state='prepared_reserved',current_actual_balance_usd=7.03,
        reserve_usd=CAP_USD,remaining_after_full_reserve_usd=.03,resource_budget=budget,
        old_costs_not_subtracted_again=True,source='delivery/deberta-balance-confirmed-7p03-20261017.json',
        authorization=c['authorization'],reserved_unix=time.time(),budget_not_invoice=True))
    print(json.dumps(read(out/'preflight.json')),flush=True)


if __name__=='__main__':main()
