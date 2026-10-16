"""Single-GPU Modal study with persistent checkpoints and synchronous local delivery.

Run with the already authenticated Modal 1.2.6 CLI from the repository root.
Only an explicit --phase all/formal can start the six formal models.
"""
import json
import os
from pathlib import Path
import sys
import time

import modal

LOCAL = modal.is_local()
ROOT = Path(__file__).resolve().parents[1] if LOCAL else Path('/workspace/End-IMDB')
sys.path.insert(0, str(ROOT/'scripts'))
from laya_modal_runtime import STUDY, VOLUME, MINIMUM_FREE, CAMPAIGN, name, poll_call

app = modal.App(STUDY)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True, version=1)
events = modal.Dict.from_name(STUDY+'-delivery', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_sync(str(ROOT/'environments/laya-cu126'), frozen=True, uv_version='0.12.15')
         .env({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1',
               'PYTHONPATH': '/opt/modal_job'})
         .add_local_dir(str(ROOT/'scripts'), '/opt/modal_job', ignore=['__pycache__/**'])) if LOCAL else None


@app.function(image=image, cpu=2, memory=8192, timeout=300,
              scaledown_window=2, max_containers=1, retries=0,
              include_source=False, serialized=True)
def runtime_probe():
    from importlib.metadata import version
    import torch
    result = {'torch': version('torch'), 'laya': version('laya'),
              'cuda_build': torch.version.cuda, 'gpu_requested': False}
    if result['torch'] != '2.14.1+cu126' or result['laya'] != '0.4.1':
        raise ValueError('Modal runtime differs from locked CUDA environment')
    return result


@app.function(image=image, gpu='L40S', cpu=4, memory=32768,
              volumes={'/workspace': volume}, max_containers=1, retries=0,
              timeout=7200, startup_timeout=600, scaledown_window=2,
              include_source=False, serialized=True)
def gpu_job(seed: int, arm: str, repeat: int = 0, job_seconds: int = 3600,
            evaluate_only: bool = False):
    sys.path.insert(0, '/workspace/End-IMDB/scripts')
    channel = modal.Dict.from_name(STUDY+'-delivery', create_if_missing=True)
    import signal
    import subprocess
    from laya_checkpoint_delivery import atomic_json, request_sha
    # Module globals are serialized from the client. Never use its local path.
    root = Path('/workspace/End-IMDB')
    volume.reload()  # No training files are open yet. Never reload inside a callback.
    run = name(seed, arm, repeat or None)
    key = run + ('.evaluate' if evaluate_only else '.train')
    started = time.monotonic()
    deadline = started + job_seconds
    process = None
    known = set()
    # The scientific runner rejects an existing per-run report directory.
    # Controller logs must live outside that directory before training starts.
    log = root/'reports'/STUDY/'controllers'/(run+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    python = '/.uv/.venv/bin/python'
    base = [python, '-u', str(root/'scripts/laya_modal_runtime.py'),
            '--seed', str(seed), '--arm', arm, '--delivery-timeout', '900']
    if repeat:
        base += ['--repeat', str(repeat)]
    commands = []
    if evaluate_only:
        commands.append([*base, '--mode', 'evaluate'])
    elif repeat == 2:
        commands += [[*base, '--mode', 'train', '--admission-checkpoint-exit'],
                     [*base, '--mode', 'train', '--resume']]
    else:
        commands.append([*base, '--mode', 'train'])
    if not evaluate_only:
        commands.append([*base, '--mode', 'reload'])
    channel[key+'.status'] = {'state': 'running', 'run_id': run}
    try:
        with log.open('a', buffering=1) as out:
            for command in commands:
                print('Starting:', ' '.join(command), flush=True)
                process = subprocess.Popen(command, cwd=root, stdout=out,
                    stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Compute budget/runtime deadline reached; keep Volume checkpoints')
                    pending = root/'delivery/pending.json'
                    if pending.exists():
                        req = json.loads(pending.read_text())
                        if req['identity']['run_id'] == run:
                            sha = request_sha(req)
                            if sha not in known:
                                # A committed snapshot must exist before the client can fetch bytes.
                                volume.commit()
                                channel[key+'.pending'] = req
                                known.add(sha)
                            ack = channel.get(sha)
                            ack_path = root/'delivery/acks'/f'{sha}.json'
                            if ack is not None and not ack_path.exists():
                                if ack.get('state') != 'verified_local':
                                    atomic_json(root/'delivery/failures'/f'{sha}.json', ack)
                                    raise RuntimeError('Local receiver failed; no further updates allowed')
                                atomic_json(ack_path, ack)
                                volume.commit()
                                channel[key+'.pending'] = None
                    time.sleep(2)
                if process.returncode:
                    tail = log.read_text()[-12000:]
                    print(tail, flush=True)
                    raise RuntimeError(f'{run}: child exited {process.returncode}')
        volume.commit()
        result = {'state': 'evaluated_delivered' if evaluate_only else 'frozen_delivered',
                  'run_id': run, 'worker_seconds': time.monotonic()-started,
                  'test_accessed': False}
        channel[key+'.status'] = result
        return result
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        volume.commit()
        channel[key+'.status'] = {'state': 'stopped', 'run_id': run, 'error': repr(exc)}
        raise


def upload_inputs(contract):
    from common import file_digest
    from laya_modal_delivery import volume_path, read_json
    paths = set(contract['source_sha256']) | {'configs/'+STUDY+'.json'}
    lock = json.loads((ROOT/'configs/laya_model_lock.json').read_text())
    paths |= {lock['local_dir']+'/'+p for p in lock['sha256']}
    manifest = {'study_id': STUDY, 'sha256': {p: file_digest(ROOT/p) for p in sorted(paths)}}
    try:
        existing = read_json(volume, 'modal-inputs.json')
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing != manifest:
            raise ValueError('Existing Modal input snapshot differs; use a new study/Volume')
        print('Reusing immutable Modal input snapshot', flush=True)
        return
    print(f'Uploading {len(paths)} allowlisted files; official test and credentials excluded', flush=True)
    with volume.batch_upload() as batch:
        for relative in sorted(paths):
            batch.put_file(ROOT/relative, volume_path(relative))
    import io
    with volume.batch_upload() as batch:
        batch.put_file(io.BytesIO(json.dumps(manifest, sort_keys=True).encode()), volume_path('modal-inputs.json'))


@app.local_entrypoint()
def main(phase: str = 'admission'):
    import shutil
    from common import file_digest
    from laya_checkpoint_delivery import atomic_json, request_sha
    from laya_modal_delivery import receive
    from laya_modal_runtime import install, contract_path
    if phase == 'runtime':
        print(json.dumps(runtime_probe.remote(), sort_keys=True), flush=True)
        return
    if phase not in ('admission', 'formal', 'evaluate', 'all'):
        raise ValueError('phase must be runtime, admission, formal, evaluate or all')
    runner, study = install()
    contract, _ = runner.verify_contract()
    report = ROOT/'reports'/STUDY
    report.mkdir(parents=True, exist_ok=True)
    ledger_path = ROOT/'delivery/modal-budget.json'
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger.get('campaign_id', ledger['study_id'].rsplit('-v', 1)[0]) != CAMPAIGN:
            raise ValueError('Budget ledger belongs to another study')
        ledger['campaign_id'] = CAMPAIGN
        ledger['study_ids'] = list(dict.fromkeys([*ledger.get('study_ids', [ledger['study_id']]), STUDY]))
    else:
        ledger = {'study_id': STUDY, 'maximum_compute_usd': 24, 'reserved_build_usd': 1,
                  'upper_bound_used_usd': 1., 'jobs': []}
        atomic_json(ledger_path, ledger)
    if any(j['state'] == 'reserved' for j in ledger['jobs']):
        raise RuntimeError('Previous job has unresolved reserved credit; inspect Modal before resuming')
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE:
        raise OSError('Require 100 GiB local free space')
    verifier = ROOT/'.venv/bin/python'
    if not verifier.exists():
        raise FileNotFoundError('Local locked CPU verification environment is missing')
    upload_inputs(contract)
    # Write small gates through the Volume API between GPU containers, never
    # reload a mounted Volume while the child still has open audit handles.
    def upload_json(relative, value):
        import io
        from laya_modal_delivery import volume_path
        atomic_json(ROOT/relative, value)
        with volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(json.dumps(value, sort_keys=True).encode()), volume_path(relative))
    def execute(seed, arm, repeat=None, evaluate=False):
        run = name(seed, arm, repeat)
        stage = 'evaluated' if evaluate else 'frozen'
        request_file = ROOT/'delivery/requests'/f'{run}.{stage}.json'
        if request_file.exists():
            request = json.loads(request_file.read_text())
            from laya_checkpoint_delivery import verify_files
            verify_files(ROOT, request)
            runner.production_receipt(request, 0)
            print('Already locally verified:', run, stage, flush=True)
            return
        left = ledger['maximum_compute_usd']-ledger['upper_bound_used_usd']
        cap = 2700 if repeat or evaluate else 6500
        # 600 seconds reserve covers startup; billing rate is deliberately above
        # published L40S + 4 CPU + 32 GiB. Six dollars remain outside this ledger.
        seconds = min(cap, int(left/3*3600)-600)
        if seconds < (600 if repeat or evaluate else 1800):
            raise RuntimeError('Free-credit budget cannot cover another job; stop before launch')
        record = {'run_id': run, 'evaluate': evaluate, 'state': 'reserved',
                  'submitted_wall_time': time.time(),
                  'maximum_seconds': seconds, 'reserved_usd': (seconds+600)*3/3600}
        ledger['jobs'].append(record)
        ledger['upper_bound_used_usd'] += record['reserved_usd']
        atomic_json(ledger_path, ledger)
        key = run + ('.evaluate' if evaluate else '.train')
        events[key+'.pending'] = None
        submitted = time.monotonic()
        call = gpu_job.spawn(seed, arm, repeat or 0, seconds, evaluate)
        record['function_call_id'] = call.object_id
        atomic_json(ledger_path, ledger)
        seen = set()
        try:
            while True:
                req = events.get(key+'.pending')
                if req is not None and request_sha(req) not in seen:
                    if req['identity'] != runner.identity(seed, arm, repeat):
                        raise ValueError('Worker requested another experiment identity')
                    print('Receiving:', run, req['stage'],
                          sum(f['bytes'] for f in req['files']), 'bytes', flush=True)
                    receipt = receive(volume, events, ROOT, req, verifier)
                    atomic_json(ROOT/'delivery/requests'/f'{run}.{req["stage"]}.json', req)
                    atomic_json(ROOT/'delivery/acks'/f'{request_sha(req)}.json', receipt)
                    seen.add(request_sha(req))
                    print('Locally loaded and verified:', run, req['stage'], flush=True)
                finished, result = poll_call(call)
                if finished:
                    break
                worker_status = events.get(key+'.status')
                if worker_status and worker_status['state'] == 'stopped':
                    raise RuntimeError(worker_status.get('error', 'Modal worker stopped'))
                if time.monotonic()-submitted > seconds+600:
                    raise TimeoutError('Local controller budget deadline reached')
                time.sleep(3)
        except BaseException:
            call.cancel(terminate_containers=True)
            # Retain the full reservation if stopping failed or billing is delayed.
            record['state'] = 'stopped_reserved'
            atomic_json(ledger_path, ledger)
            raise
        elapsed = time.monotonic()-submitted
        charged_upper = min(record['reserved_usd'], (elapsed+30)*3/3600)
        ledger['upper_bound_used_usd'] += charged_upper-record['reserved_usd']
        record.update(state='completed', wall_seconds=elapsed, charged_upper_usd=charged_upper)
        atomic_json(ledger_path, ledger)
        atomic_json(report/(run+('.evaluation' if evaluate else '')+'.json'), result)
        print('Completed:', run, 'conservative compute USD:', round(ledger['upper_bound_used_usd'], 3), flush=True)
    try:
        if phase in ('admission', 'all'):
            for arm in ('ce', 'mixed'):
                for repeat in (1, 2):
                    execute(42, arm, repeat)
            admission = study.check_admission()
            upload_json('reports/'+STUDY+'/admission.json', admission)
            print('CUDA, exact resume, and local weight delivery admission PASSED', flush=True)
        if phase in ('formal', 'all'):
            admission = study.check_admission()
            upload_json('reports/'+STUDY+'/admission.json', admission)
            for seed, arm in runner.PAIRS:
                execute(seed, arm)
            study.freeze_all()
            upload_json('reports/'+STUDY+'/all-models-frozen.json',
                        json.loads((report/'all-models-frozen.json').read_text()))
        if phase in ('evaluate', 'all'):
            study.freeze_all()
            for seed, arm in runner.PAIRS:
                execute(seed, arm, evaluate=True)
        atomic_json(report/'status.json', {'state': 'completed_phase', 'phase': phase,
                    'contract_sha256': file_digest(contract_path()), 'test_accessed': False})
    except BaseException as exc:
        atomic_json(report/'status.json', {'state': 'stopped', 'phase': phase,
                    'error': repr(exc), 'test_accessed': False})
        raise
