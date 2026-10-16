"""Single GPU queue, absolute spend deadline, durable epoch acceptance."""
import json
from pathlib import Path
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path('/workspace/End-IMDB')
sys.path.insert(0, str(ROOT/'scripts'))
from laya_context_pair import STUDY, VOLUME, SECONDS, CAP_USD
from modal_laya_recovery import image, volume

app = modal.App(STUDY)


@app.function(image=image, gpu='L40S', cpu=4, memory=32768,
              volumes={'/workspace':volume}, timeout=6000, startup_timeout=120,
              retries=0, max_containers=1, scaledown_window=2,
              include_source=False, serialized=True)
def gpu_job(job, deadline):
    import os
    import signal
    import subprocess
    from common import file_digest
    from laya_checkpoint_delivery import atomic_json
    from laya_context_pair import config, bundle, receipt, identity, check_bundle, read
    root = Path('/workspace/End-IMDB')
    volume.reload(); config(root)
    if time.time() >= deadline-180:
        raise TimeoutError('No remaining study budget')
    report = root/'reports'/STUDY
    report.mkdir(parents=True, exist_ok=True)
    driver = root/'scripts/laya_context_pair.py'
    python = '/.uv/.venv/bin/python'
    mode = job['mode']
    tag = mode if mode != 'train' else f"{job['cap']}-epoch-{job['epoch']+1}"
    log = report/'logs'/(tag+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [python,'-u',str(driver),mode,'--deadline',str(deadline)]
    if mode=='train':command += ['--cap',str(job['cap']),'--epoch',str(job['epoch'])]
    process = None
    try:
        print('START',tag,flush=True)
        with log.open('x',buffering=1) as out:
            process = subprocess.Popen(command,cwd=root,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            last = time.time()
            while process.poll() is None:
                if time.time() >= deadline-60:
                    raise TimeoutError('Absolute 13500-second compute deadline reached')
                if time.time()-last>=60:
                    lines=log.read_text().splitlines()
                    print('PROGRESS',tag,lines[-1][-500:] if lines else 'loading',flush=True)
                    last=time.time()
                time.sleep(1)
        if process.returncode:
            print(log.read_text()[-8000:],flush=True)
            raise RuntimeError(f'{tag} exited {process.returncode}')
        # All producers have closed files before a Volume commit and a new verifier.
        volume.commit()
        if mode in ('admission','train'):
            folder = (root/'checkpoints'/STUDY/'admission-2048') if mode=='admission' else bundle(job['cap'],job['epoch'],root)
            output = report/'cloud-verification'/(tag+'.json')
            remaining=deadline-time.time()-60
            if remaining<30:raise TimeoutError('No verification time remains')
            subprocess.run([python,str(driver),'verify','--bundle',str(folder),'--output',str(output)],
                           cwd=root,check=True,timeout=min(240,remaining))
            manifest=check_bundle(folder)
            r={'state':'verified_committed_volume','identity':manifest['identity'],
               'bundle_sha256':file_digest(folder/'bundle.json'),'bundle_manifest':manifest,
               'verification':read(output),'local_delivery_verified':False,'modal_volume':VOLUME}
            path=(report/'admission-cloud-receipt.json') if mode=='admission' else receipt(job['cap'],job['epoch'],root)
            atomic_json(path,r)
            volume.commit()
            print('CLOUD_DURABLE',tag,flush=True)
        else:
            manifest=check_bundle(report/'evaluation')
            atomic_json(report/'evaluation-cloud-receipt.json',
                        {'state':'verified_committed_volume','bundle_manifest':manifest,
                         'bundle_sha256':file_digest(report/'evaluation/bundle.json'),
                         'local_delivery_verified':False,'test_accessed':False})
            volume.commit()
        return {'job':job,'state':'cloud_verified','test_accessed':False}
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
        volume.commit()
        raise


@app.function(image=image,cpu=.125,memory=2048,nonpreemptible=True,
              volumes={'/workspace':volume},timeout=13800,retries=0,
              max_containers=1,scaledown_window=2,include_source=False,serialized=True)
def coordinate(gpu_handle, submitted, deadline):
    from laya_checkpoint_delivery import atomic_json
    from laya_context_pair import config, freeze, read
    root=Path('/workspace/End-IMDB');volume.reload();config(root)
    path=root/'reports'/STUDY/'state.json'
    if path.exists():raise FileExistsError('Duplicate coordinator')
    state={'study_id':STUDY,'state':'running','submitted_unix':submitted,'absolute_deadline_unix':deadline,
           'reserve_usd':CAP_USD,'completed':[],'test_accessed':False,'local_delivery_verified':False}
    def save():
        state['updated_unix']=time.time()
        state['conservative_elapsed_gpu_upper_usd']=min(11.25,(time.time()-submitted)*3/3600)
        atomic_json(path,state);volume.commit()
    active=None
    try:
        save()
        jobs=[{'mode':'admission'}]+[{'mode':'train','cap':cap,'epoch':epoch} for cap in (512,2048) for epoch in (0,1)]+[{'mode':'evaluate'}]
        for job in jobs:
            volume.reload()
            if time.time()>=deadline-180:raise TimeoutError('Study deadline exhausted before next job')
            if job['mode']=='evaluate':freeze(root);volume.commit()
            state['active_job']=job;save()
            active=gpu_handle.spawn(job,deadline)
            state['active_call_id']=active.object_id;save()
            returned=active.get(timeout=min(6180,max(1,deadline-time.time())))
            active=None
            volume.reload()
            state['completed'].append(returned)
            state.pop('active_call_id',None);save()
        state['state']='completed_cloud_pending_local';state.pop('active_job',None);save()
        return state
    except BaseException as exc:
        if active is not None:
            active.cancel(terminate_containers=True)
        volume.reload()
        state.update(state='stopped_budget' if isinstance(exc,TimeoutError) else 'stopped_error',error=repr(exc))
        save()
        raise


@app.local_entrypoint()
def main():
    import shutil
    from common import file_digest
    from laya_checkpoint_delivery import atomic_json
    from laya_context_pair import config, read
    c=config(ROOT)
    if shutil.disk_usage(ROOT).free<50*1024**3:raise OSError('Need 50 GiB free before paired delivery')
    report=ROOT/'reports'/STUDY
    if (report/'submission.json').exists():raise FileExistsError('Already submitted; never auto-resubmit')
    old=read(ROOT/'delivery/modal-budget.json')
    if any(j['state']=='reserved' for j in old['jobs']):raise ValueError('An older compute reservation remains active')
    balance=read(ROOT/'delivery/laya-long-context-budget-20261016.json')
    if balance['user_confirmed_current_free_balance_usd']!=15 or balance['pilot_reserved_usd']!=1:
        raise ValueError('Confirmed balance changed')
    budget=ROOT/'delivery'/f'{STUDY}-budget.json'
    if budget.exists():raise FileExistsError('Reservation exists; inspect before a manual retry')
    with volume.batch_upload(force=True) as batch:
        for rel in c['upload_files']:batch.put_file(ROOT/rel,'/End-IMDB/'+rel)
        batch.put_file(ROOT/'configs'/f'{STUDY}.json','/End-IMDB/configs/'+STUDY+'.json')
    submitted=time.time();deadline=submitted+SECONDS
    record={'study_id':STUDY,'state':'reserved','user_confirmed_free_usd':15,'prior_pilot_reserve_usd':1,
            'additional_reserved_usd':CAP_USD,'remaining_after_full_reserve_usd':2,
            'submitted_unix':submitted,'absolute_deadline_unix':deadline,
            'balance_source':'delivery/laya-long-context-budget-20261016.json',
            'historical_upper_bound_ledger':'delivery/modal-budget.json','billing_balance_unreadable_via_cli':True}
    atomic_json(budget,record)
    call=coordinate.spawn(gpu_job,submitted,deadline)
    record['coordinator_call_id']=call.object_id;atomic_json(budget,record)
    info=dict(record,state='submitted_detached',modal_volume=VOLUME,local_delivery_verified=False,
              contract_sha256=file_digest(ROOT/'configs'/f'{STUDY}.json'))
    atomic_json(report/'submission.json',info)
    print(json.dumps(info),flush=True)
