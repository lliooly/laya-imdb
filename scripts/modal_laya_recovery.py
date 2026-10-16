"""Resume the original six-model study with durable orchestration and deadlines."""
import json
from pathlib import Path
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path('/workspace/End-IMDB')
sys.path.insert(0, str(ROOT/'scripts'))
from laya_modal_recovery import RECOVERY, STUDY, VOLUME

app = modal.App(RECOVERY)
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_sync(str(ROOT/'environments/laya-cu126'), frozen=True, uv_version='0.12.15')
         .env({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': '/opt/modal_job',
               'PYTHONUNBUFFERED': '1'})
         .add_local_dir(str(ROOT/'scripts'), '/opt/modal_job', ignore=['__pycache__/**'])) if modal.is_local() else None


@app.function(image=image, gpu='L40S', cpu=4, memory=32768,
              volumes={'/workspace': volume}, timeout=7200, startup_timeout=600,
              retries=0, max_containers=1, scaledown_window=2,
              include_source=False, serialized=True)
def gpu_resume(job: dict, deadline_unix: float):
    import os
    import signal
    import subprocess
    from laya_checkpoint_delivery import atomic_json, request_sha
    from laya_modal_deferred import name, validate_cloud_receipt
    from laya_modal_recovery import (accepted, job_done, plan, read, sealed,
                                     verify_sources, BudgetExpired)
    root = Path('/workspace/End-IMDB')
    volume.reload()
    if time.time() >= deadline_unix-60:
        raise BudgetExpired('Budget expired before starting the GPU worker')
    verify_sources(root, read(root/'configs'/f'{RECOVERY}.json'))
    run = name(job['seed'], job['arm'])
    python = '/.uv/.venv/bin/python'
    driver = root/'scripts/laya_modal_deferred.py'
    started = time.time()
    process = None

    def persist(req):
        sha = request_sha(req)
        paths = [root/'delivery'/channel/f'{sha}.json' for channel in ('receipts', 'acks')]
        previous = []
        for path in paths:
            if path.exists():
                receipt = read(path)
                validate_cloud_receipt(req, receipt)
                previous.append(receipt)
        if previous:
            receipt = previous[0]
            if any(r['verification'] != receipt['verification'] for r in previous):
                raise ValueError('Cloud receipt and ACK verification differ')
        else:
            volume.commit()
            request_file = root/'delivery/received-requests'/f'{sha}.json'
            atomic_json(request_file, req)
            remaining = deadline_unix-time.time()-60
            if remaining < 30:
                raise BudgetExpired('No budget remains for strict checkpoint verification')
            checked = subprocess.check_output(
                [python, str(driver), '--verify-cloud', str(request_file)],
                cwd=root, timeout=min(300, remaining))
            receipt = {'schema': 'laya-cloud-durability-v1',
                'state': 'verified_committed_volume', 'modal_volume': VOLUME,
                'local_delivery_verified': False, 'identity': req['identity'],
                'stage': req['stage'], 'request_sha256': sha, 'files': req['files'],
                'verification': json.loads(checked)}
            validate_cloud_receipt(req, receipt)
        for path in paths:
            if not path.exists():
                atomic_json(path, receipt)
        volume.commit()
        print('CLOUD_DURABLE', run, req['stage'], flush=True)

    try:
        # Recovery of a sealed write whose verifier/ACK was interrupted happens
        # before phase planning. Corrupt sealed files fail instead of being erased.
        for stage in ('epoch-001', 'epoch-002', 'frozen', 'evaluated'):
            req = sealed(root, run, stage)
            if req is not None and not accepted(root, run, stage):
                persist(req)
        commands = plan(root, job)
        print('RECOVERY_PLAN', run, commands, flush=True)
        log = root/'reports'/RECOVERY/'controllers'/(run+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', buffering=1) as out:
            for mode, resume in commands:
                command = [python, '-u', str(driver), '--seed', str(job['seed']),
                           '--arm', job['arm'], '--delivery-timeout', '600', '--mode', mode]
                if resume:
                    command.append('--resume')
                print('START', run, mode, 'resume='+str(resume), flush=True)
                process = subprocess.Popen(command, cwd=root, stdout=out,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    if time.time() >= deadline_unix-60:
                        raise BudgetExpired('Original absolute compute deadline reached')
                    pending = root/'delivery/pending.json'
                    if pending.exists():
                        req = read(pending)
                        if req['identity']['run_id'] == run:
                            sha = request_sha(req)
                            if not (root/'delivery/acks'/f'{sha}.json').exists():
                                # The producer closes its sealed bundle before publishing.
                                persist(req)
                    time.sleep(1)
                if process.returncode:
                    print(log.read_text()[-8000:], flush=True)
                    raise RuntimeError(f'{run} {mode} exited {process.returncode}')
        if not job_done(root, job):
            raise ValueError('Missing verified final bundle after recovery')
        volume.commit()
        return {'run_id': run, 'state': 'evaluated_cloud' if job['evaluate'] else 'frozen_cloud',
                'worker_seconds': time.time()-started, 'commands': commands,
                'local_delivery_verified': False}
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        volume.commit()
        raise


@app.function(image=image, cpu=.125, memory=2048, nonpreemptible=True,
              volumes={'/workspace': volume}, timeout=28800, retries=0,
              max_containers=1, scaledown_window=2, include_source=False, serialized=True)
def coordinate_recovery(gpu_handle):
    from laya_checkpoint_delivery import atomic_json
    from laya_modal_recovery import (CAP, RATE, BudgetExpired, check_state, execute_queue,
                                     initial_state, job_done, read, verify_sources)
    root = Path('/workspace/End-IMDB')
    volume.reload()
    config = read(root/'configs'/f'{RECOVERY}.json')
    verify_sources(root, config)
    import common
    import laya_modal_deferred
    common.ROOT = root.resolve()
    laya_modal_deferred.ROOT = root.resolve()
    runner, study = laya_modal_deferred.install()
    runner.ROOT = study.ROOT = root.resolve()
    report = root/'reports'/RECOVERY
    path = report/'state.json'
    state = read(path) if path.exists() else initial_state(config, time.time())
    check_state(state, config)

    def save(value):
        value['updated_unix'] = time.time()
        value['conservative_upper_usd'] = min(CAP, value['previous_upper_usd'] +
                                            (time.time()-value['started_unix'])*RATE/3600)
        atomic_json(path, value)
        volume.commit()

    def done(job):
        # Only the coordinator reloads its own mount while workers run. It never
        # has the worker's training files open.
        volume.reload()
        return job_done(root, job)

    def spawn(job, deadline):
        return gpu_handle.spawn(job, deadline)

    try:
        save(state)
        admission = study.check_admission()
        if not (report/'verified-existing-admission.json').exists():
            atomic_json(report/'verified-existing-admission.json', admission)
            volume.commit()
        training = [{'seed': seed, 'arm': arm, 'evaluate': False} for seed, arm in runner.PAIRS]
        evaluate = [{**job, 'evaluate': True} for job in training]
        for queue in (training, evaluate):
            if queue is evaluate:
                study.freeze_all()
                volume.commit()
            execute_queue(queue, state, save=save, spawn=spawn,
                          attach=modal.FunctionCall.from_id, done=done)
        state.update(state='completed_cloud_pending_local_download', test_accessed=False)
        save(state)
        return {'state': state['state'], 'local_delivery_verified': False}
    except KeyboardInterrupt:
        state['state'] = 'coordinator_interrupted'
        save(state)
        raise
    except BaseException as exc:
        state.update(state='stopped_budget' if isinstance(exc, BudgetExpired) else 'stopped_error',
                     error=repr(exc), test_accessed=False)
        save(state)
        raise


@app.local_entrypoint()
def main():
    from laya_checkpoint_delivery import atomic_json
    from laya_modal_recovery import CAP, read, verify_sources
    config_path = ROOT/'configs'/f'{RECOVERY}.json'
    config = read(config_path)
    verify_sources(ROOT, config)
    ledger_path = ROOT/'delivery/modal-budget.json'
    ledger = read(ledger_path)
    if any(job['state'] == 'reserved' for job in ledger['jobs']):
        raise RuntimeError('A compute reservation is already active; refuse duplicate launch')
    if ledger['upper_bound_used_usd'] != config['previous_upper_usd']:
        raise ValueError('Budget changed after preparing this recovery configuration')
    with volume.batch_upload() as batch:
        for relative in sorted(config['recovery_source_sha256']):
            batch.put_file(ROOT/relative, '/End-IMDB/'+relative)
        batch.put_file(config_path, '/End-IMDB/configs/'+config_path.name)
    record = {'run_id': RECOVERY, 'state': 'reserved', 'mode': 'checkpoint_recovery',
              'previous_budget_usd': config['previous_upper_usd'],
              'reserved_usd': CAP-config['previous_upper_usd']}
    ledger['jobs'].append(record)
    ledger['upper_bound_used_usd'] = CAP
    atomic_json(ledger_path, ledger)
    call = coordinate_recovery.spawn(gpu_resume)
    record['function_call_id'] = call.object_id
    atomic_json(ledger_path, ledger)
    info = {'recovery_id': RECOVERY, 'scientific_study_id': STUDY,
            'coordinator_call_id': call.object_id, 'gpu_function_id': gpu_resume.object_id,
            'modal_volume': VOLUME, 'state': 'submitted_detached',
            'local_delivery_verified': False}
    atomic_json(ROOT/'reports'/RECOVERY/'submission.json', info)
    print(json.dumps(info), flush=True)
