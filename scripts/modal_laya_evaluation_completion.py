"""Complete five fixed CUDA evaluations; no admission, optimizer, or model selection."""
import json
from pathlib import Path
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path('/workspace/End-IMDB')
sys.path.insert(0, str(ROOT/'scripts'))
from laya_evaluation_completion import COMPLETION, JOBS, CAP, RATE, reserve
from modal_laya_recovery import image, volume

app = modal.App(COMPLETION)


@app.function(image=image, gpu='L40S', cpu=4, memory=32768,
              volumes={'/workspace': volume}, timeout=1020, startup_timeout=120,
              retries=0, max_containers=1, scaledown_window=2,
              include_source=False, serialized=True)
def evaluate_remaining(submitted_unix: float, deadline_unix: float):
    from modal_laya_recovery import gpu_resume
    from laya_checkpoint_delivery import atomic_json
    from laya_evaluation_completion import verify_scope
    from laya_modal_recovery import BudgetExpired, job_done, read
    root = Path('/workspace/End-IMDB')
    volume.reload()
    if deadline_unix != submitted_unix + 1080 or time.time() >= deadline_unix-60:
        raise BudgetExpired('Evaluation reservation expired before startup')
    verify_scope(root)
    path = root/'reports'/COMPLETION/'state.json'
    state = read(path) if path.exists() else {
        'completion_id': COMPLETION, 'submitted_unix': submitted_unix,
        'deadline_unix': deadline_unix, 'completed': {}, 'test_accessed': False}
    if state['submitted_unix'] != submitted_unix or state['deadline_unix'] != deadline_unix:
        raise ValueError('Cannot extend existing evaluation reservation')
    def save():
        state['updated_unix'] = time.time()
        state['conservative_total_upper_usd'] = min(CAP, 24.+RATE*(time.time()-submitted_unix+60)/3600)
        atomic_json(path, state)
        volume.commit()
    try:
        for job in JOBS:
            if job['evaluate'] is not True:
                raise ValueError('Training is forbidden in the completion job')
            if time.time() >= deadline_unix-120:
                raise BudgetExpired('No time remains for another evaluation')
            key = f"{job['seed']}-{job['arm']}"
            state.update(state='evaluating', active=job)
            save()
            # Reuse the frozen worker and its existing cloud durability checks.
            # With evaluate=True, its planner can only evaluate or skip a sealed result.
            result = gpu_resume.local(job, deadline_unix)
            if not job_done(root, job):
                raise ValueError('Evaluation returned without a verified result')
            state['completed'][key] = result
            state.update(active=None)
            save()
        state.update(state='completed_cloud_pending_local_download', active=None)
        save()
        return state
    except BaseException as exc:
        state.update(state='stopped', error=repr(exc))
        save()
        raise


@app.local_entrypoint()
def main():
    from laya_checkpoint_delivery import atomic_json, digest, validate_receipt
    from laya_modal_recovery import read, verify_sources
    root = ROOT
    verify_sources(root, read(root/'configs/laya-robust-loss-modal-recovery-20261016-v1.json'))
    config_path = root/'configs'/f'{COMPLETION}.json'
    config = read(config_path)
    for rel, sha in config['operational_source_sha256'].items():
        if digest(root/rel) != sha:
            raise ValueError('Operational source changed: '+rel)
    delivery = read(root/'reports/laya-robust-loss-modal-recovery-20261016-v1/local-final-delivery.json')
    if delivery['state'] != 'six_final_models_verified_local' or len(delivery['models']) != 6:
        raise ValueError('Complete local final-model backup is required')
    from laya_modal_deferred import name
    from laya_checkpoint_delivery import request_sha
    for seed in (42,43,44):
        for arm in ('ce','mixed'):
            run = name(seed,arm)
            for stage in ('epoch-002','frozen'):
                req = read(root/'delivery/requests'/f'{run}.{stage}.json')
                validate_receipt(req, read(root/'delivery/receipts'/f'{request_sha(req)}.json'))
    ledger_path = root/'delivery/modal-budget.json'
    ledger = read(ledger_path)
    record = reserve(ledger)
    with volume.batch_upload() as batch:
        for rel in config['operational_source_sha256']:
            batch.put_file(root/rel, '/End-IMDB/'+rel)
        batch.put_file(config_path, '/End-IMDB/configs/'+config_path.name)
    submitted = time.time()
    record['submitted_unix'] = submitted
    atomic_json(ledger_path, ledger)
    call = evaluate_remaining.spawn(submitted, submitted+1080)
    record['function_call_id'] = call.object_id
    atomic_json(ledger_path, ledger)
    info = {'completion_id': COMPLETION, 'scientific_study_id': config['scientific_study_id'],
            'call_id': call.object_id, 'submitted_unix': submitted, 'deadline_unix': submitted+1080,
            'maximum_additional_compute_usd': 1., 'maximum_total_compute_usd': CAP,
            'state': 'submitted_detached', 'test_accessed': False}
    atomic_json(root/'reports'/COMPLETION/'submission.json', info)
    print(json.dumps(info), flush=True)
