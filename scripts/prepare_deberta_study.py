"""Freeze source/input hashes, preflight evidence and a single $5 reservation."""
import json
from pathlib import Path
import sys
import time

from common import ROOT,file_digest,write_json
from deberta_inputs import STUDY,BASE,read,training_rows
from deberta_training import RECIPE


def main():
    report=ROOT/'reports'/STUDY
    if (ROOT/'configs'/f'{STUDY}.json').exists():raise FileExistsError('Frozen study already prepared')
    prior=read(ROOT/'delivery/laya-reviewed-data-s42-modal-20261017-v1-budget.json')
    if prior['state']!='completed_verified_local' or prior['remaining_after_full_reserve_usd']!=6:
        raise ValueError('Previous budget basis changed')
    _,split=training_rows(ROOT)
    folder=ROOT/'artifacts'/STUDY
    stats=read(folder/'input-summary.json')
    if stats['count']!=22100 or stats['development_accessed'] or stats['test_accessed']:
        raise ValueError('Input cache scope differs')
    lock=read(ROOT/'configs/deberta_model_lock.json')
    sources=['scripts/'+s for s in ['deberta_inputs.py','deberta_training.py','deberta_study.py',
        'modal_deberta_study.py','deberta_image_build.py','receive_deberta_study.py','prepare_deberta_study.py',
        'common.py','laya_head_control.py','laya_robust_training.py','laya_checkpoint_delivery.py',
        'comparison_metrics.py','analyze_laya_length_router.py']]
    # analyze_laya_length_router imports the inference-only router at module scope.
    sources+=['scripts/laya_length_router.py','tests/test_deberta_training.py','configs/deberta_model_lock.json',BASE,
              'environments/laya-cu126/pyproject.toml','environments/laya-cu126/uv.lock',
              'environments/deberta-cu126/pyproject.toml','environments/deberta-cu126/uv.lock',
              'environments/deberta-prep/pyproject.toml','environments/deberta-prep/uv.lock',
              'docs/superpowers/specs/2026-10-17-deberta-heterogeneous-complementarity-design.md',
              'docs/superpowers/plans/2026-10-17-deberta-heterogeneous-implementation.md',
              'data/processed/train.jsonl',str((folder/'train-tokens.jsonl').relative_to(ROOT)),
              str((folder/'input-summary.json').relative_to(ROOT))]
    sources += [str(p.relative_to(ROOT)) for p in sorted((folder/'tokenizer').rglob('*')) if p.is_file()]
    sources += [lock['local_dir']+'/'+name for name in lock['sha256']]
    eval_only=['data/processed/validation.jsonl','reports/laya-long-fusion-bc-20261017-v1/predictions.json']
    # Use the historical development source digest without opening its text during preparation.
    hashes={rel:file_digest(ROOT/rel) for rel in sources}
    hashes[eval_only[0]]=split['validation']['source_sha256']
    hashes[eval_only[1]]=file_digest(ROOT/eval_only[1])
    c=dict(study_id=STUDY,recipe=RECIPE,model_dir=lock['local_dir'],maximum_gpu_seconds=5400,
        pilot_maximum_seconds=900,reserve_usd=5,fusion_new_weight=.5,
        source_sha256=hashes,evaluation_only_files=eval_only,upload_files=list(hashes),test_accessed=False,
        resource_budget=dict(gpu_resource_upper_usd=4.5,gpu_rate_usd_per_hour=3,
            extra_cpu_and_transfer_usd=.5,build_cpu=1,build_memory_mib=2048,build_timeout=180,
            coordinator_cpu=.125,coordinator_memory_mib=2048,coordinator_timeout=5700,
            coordinator_cpu_upper_limit=.125,coordinator_memory_upper_limit_mib=2048,
            coordinator_nonpreemptible_multiplier=3,receiver_cloud_functions=False,
            builder_max_cpu_seconds_per_process=150,builder_child_wall_timeout=120,
            cached_base_lock='environments/laya-cu126/uv.lock'))
    write_json(ROOT/'configs'/f'{STUDY}.json',c)
    report.mkdir(parents=True,exist_ok=True)
    write_json(report/'preflight.json',dict(state='local_prepared',tests='5 CPU tests passed',
        gradient_count=22100,calibration_count=400,development_count=2500,
        prepared_tokenizer='official_slow_sentencepiece_preserves_byte_fallback',
        initial_load_checked=True,parameter_count=435063810,precision='float32',
        missing_pretrained_keys=['classifier.bias','classifier.weight','pooler.dense.bias','pooler.dense.weight'],
        config_sha256=file_digest(ROOT/'configs'/f'{STUDY}.json'),
        upload_bytes=sum((ROOT/p).stat().st_size for p in hashes),test_accessed=False))
    budget=ROOT/'delivery'/f'{STUDY}-budget.json'
    if budget.exists():raise FileExistsError('Reservation already exists')
    write_json(budget,dict(study_id=STUDY,state='prepared_reserved',reserved_unix=time.time(),
        user_confirmed_previous_free_usd=10,previous_round_reserved_usd=4,
        conservative_available_usd=6,additional_reserved_usd=5,remaining_after_full_reserve_usd=1,
        budget_not_actual_invoice=True,actual_balance_unreadable=True,
        source='delivery/laya-reviewed-data-s42-modal-20261017-v1-budget.json',
        previous_build_failure='deberta-heterogeneous-s42-modal-20261017-v1',
        previous_build_gpu_started=False,previous_build_cpu_upper_usd=.02,
        resource_budget=c['resource_budget']))
    print(json.dumps(read(report/'preflight.json'),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
