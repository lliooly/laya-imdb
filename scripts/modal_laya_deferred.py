"""Detached cloud coordinator. Local delivery is explicitly deferred by the user."""
import json
from pathlib import Path
import sys
import time
import modal

ROOT=Path(__file__).resolve().parents[1] if modal.is_local() else Path('/workspace/End-IMDB')
sys.path.insert(0,str(ROOT/'scripts'))
from laya_modal_deferred import STUDY,VOLUME,name

app=modal.App(STUDY)
volume=modal.Volume.from_name(VOLUME,create_if_missing=True,version=1)
image=(modal.Image.debian_slim(python_version='3.12')
       .uv_sync(str(ROOT/'environments/laya-cu126'),frozen=True,uv_version='0.12.15')
       .env({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
             'PYTHONPATH':'/opt/modal_job','PYTHONUNBUFFERED':'1'})
       .add_local_dir(str(ROOT/'scripts'),'/opt/modal_job',ignore=['__pycache__/**'])) if modal.is_local() else None


@app.function(image=image,gpu='L40S',cpu=4,memory=32768,volumes={'/workspace':volume},
              timeout=7200,startup_timeout=600,retries=0,max_containers=1,
              scaledown_window=2,include_source=False,serialized=True)
def gpu_run(seed:int,arm:str,repeat:int=0,seconds:int=6000,evaluate:bool=False):
    import os,signal,subprocess
    root=Path('/workspace/End-IMDB');sys.path.insert(0,str(root/'scripts'))
    from laya_checkpoint_delivery import atomic_json,request_sha
    from laya_modal_deferred import validate_cloud_receipt
    volume.reload()
    run=name(seed,arm,repeat or None)
    log=root/'reports'/STUDY/'controllers'/(run+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    python='/.uv/.venv/bin/python';driver=root/'scripts/laya_modal_deferred.py'
    base=[python,'-u',str(driver),'--seed',str(seed),'--arm',arm,'--delivery-timeout','600']
    if repeat:base+=['--repeat',str(repeat)]
    commands=[]
    if evaluate:commands.append([*base,'--mode','evaluate'])
    elif repeat==2:commands += [[*base,'--mode','train','--admission-checkpoint-exit'],[*base,'--mode','train','--resume']]
    else:commands.append([*base,'--mode','train'])
    if not evaluate:commands.append([*base,'--mode','reload'])
    started=time.monotonic();process=None
    try:
        with log.open('a',buffering=1) as out:
            for command in commands:
                print('Starting',run,command[-1],flush=True)
                process=subprocess.Popen(command,cwd=root,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                while process.poll() is None:
                    if time.monotonic()-started>seconds:raise TimeoutError('Compute budget deadline')
                    pending=root/'delivery/pending.json'
                    if pending.exists():
                        req=json.loads(pending.read_text());sha=request_sha(req)
                        ack=root/'delivery/acks'/f'{sha}.json'
                        if req['identity']['run_id']==run and not ack.exists():
                            volume.commit()
                            request_file=root/'delivery/received-requests'/f'{sha}.json'
                            atomic_json(request_file,req)
                            remaining=seconds-(time.monotonic()-started)
                            if remaining<30:raise TimeoutError('No budget for independent checkpoint verification')
                            checked=subprocess.check_output([python,str(driver),'--verify-cloud',str(request_file)],
                                cwd=root,timeout=min(300,remaining))
                            receipt={'schema':'laya-cloud-durability-v1','state':'verified_committed_volume',
                                'modal_volume':VOLUME,'local_delivery_verified':False,'identity':req['identity'],
                                'stage':req['stage'],'request_sha256':sha,'files':req['files'],
                                'verification':json.loads(checked)}
                            validate_cloud_receipt(req,receipt)
                            atomic_json(root/'delivery/receipts'/f'{sha}.json',receipt)
                            atomic_json(ack,receipt);volume.commit()
                            print('Persisted and independently loaded in cloud:',run,req['stage'],flush=True)
                    time.sleep(1)
                if process.returncode:
                    print(log.read_text()[-10000:],flush=True)
                    raise RuntimeError(f'{run} child exited {process.returncode}')
        volume.commit()
        return {'run_id':run,'state':'evaluated_cloud' if evaluate else 'frozen_cloud',
                'worker_seconds':time.monotonic()-started,'local_delivery_verified':False}
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        volume.commit();raise


@app.function(image=image,cpu=.125,memory=1024,volumes={'/workspace':volume},
              timeout=28800,retries=0,max_containers=1,scaledown_window=2,
              include_source=False,serialized=True)
def coordinate(gpu_handle,previous_budget_usd:float):
    root=Path('/workspace/End-IMDB');sys.path.insert(0,str(root/'scripts'))
    # Serialized helpers can be imported from /opt/modal_job before the mounted
    # project enters sys.path. Bind the coordinator to the actual Volume root.
    import common,laya_modal_deferred
    common.ROOT=root.resolve();laya_modal_deferred.ROOT=root.resolve()
    from laya_checkpoint_delivery import atomic_json
    from laya_modal_deferred import install
    runner,study=install();runner.ROOT=root.resolve();study.ROOT=root.resolve();volume.reload()
    report=root/'reports'/STUDY;report.mkdir(parents=True,exist_ok=True)
    marker=report/'coordinator-started.json'
    if marker.exists():raise RuntimeError('Workflow already started; refuse automatic duplicate/retraining')
    atomic_json(marker,{'started_unix':time.time(),'previous_budget_usd':previous_budget_usd})
    used=previous_budget_usd;records=[];gpu=gpu_handle
    def status(state,**kw):
        atomic_json(report/'status.json',{'state':state,'local_delivery_verified':False,
            'maximum_compute_usd':24,'conservative_used_usd':used,'updated_unix':time.time(),**kw})
        volume.commit()
    def execute(seed,arm,repeat=None,evaluate=False):
        nonlocal used
        seconds=min(2700 if repeat or evaluate else 6500,int((24-used)/3*3600)-600)
        if seconds<(300 if repeat or evaluate else 1800):raise RuntimeError('Free-credit compute budget insufficient; stop')
        reserve=(seconds+600)*3/3600
        status('running',run_id=name(seed,arm,repeat),evaluate=evaluate,reserved_usd=reserve)
        submitted=time.monotonic();call=gpu.spawn(seed,arm,repeat or 0,seconds,evaluate)
        atomic_json(report/'active-call.json',{'id':call.object_id,'reserved_usd':reserve})
        volume.commit()
        try:result=call.get(timeout=seconds+600)
        except BaseException:
            call.cancel(terminate_containers=True);used+=reserve;volume.reload();raise
        elapsed=time.monotonic()-submitted;used+=min(reserve,(elapsed+30)*3/3600)
        volume.reload()
        records.append({**result,'wall_seconds':elapsed})
        atomic_json(report/'runs.json',records)
        status('between_runs')
    try:
        status('admission')
        for arm in ('ce','mixed'):
            for repeat in (1,2):execute(42,arm,repeat)
        atomic_json(report/'admission.json',study.check_admission());volume.commit()
        for seed,arm in runner.PAIRS:execute(seed,arm)
        study.freeze_all();volume.commit()
        for seed,arm in runner.PAIRS:execute(seed,arm,evaluate=True)
        status('completed_cloud_pending_local_download',runs=records,test_accessed=False)
        return {'state':'completed_cloud_pending_local_download','local_delivery_verified':False}
    except BaseException as exc:
        status('stopped',error=repr(exc),runs=records,test_accessed=False);raise


@app.local_entrypoint()
def main():
    import io
    from common import file_digest
    from laya_checkpoint_delivery import atomic_json
    from laya_modal_deferred import contract_path
    contract=json.loads(contract_path().read_text())
    for rel,sha in contract['source_sha256'].items():
        if file_digest(ROOT/rel)!=sha:raise ValueError('Bound source changed: '+rel)
    ledger_path=ROOT/'delivery/modal-budget.json';ledger=json.loads(ledger_path.read_text())
    if any(r['state']=='reserved' for r in ledger['jobs']):raise RuntimeError('Previous compute reservation is unresolved')
    previous=ledger['upper_bound_used_usd']
    if previous>=20:raise RuntimeError('Insufficient remaining free-credit budget')
    paths=set(contract['source_sha256'])|{str(contract_path().relative_to(ROOT))}
    lock=json.loads((ROOT/'configs/laya_model_lock.json').read_text())
    paths|={lock['local_dir']+'/'+p for p in lock['sha256']}
    with volume.batch_upload() as batch:
        for rel in sorted(paths):batch.put_file(ROOT/rel,'/End-IMDB/'+rel)
    remaining=24-previous
    record={'run_id':STUDY,'state':'reserved','reserved_usd':remaining,
            'mode':'detached_cloud_coordinator','previous_budget_usd':previous}
    ledger['jobs'].append(record);ledger['upper_bound_used_usd']=24
    atomic_json(ledger_path,ledger)
    # The already hydrated Function handle can be serialized as an argument.
    # Modal 1.2.6 has FunctionCall.from_id, but no Function.from_id.
    call=coordinate.spawn(gpu_run,previous)
    record['function_call_id']=call.object_id;atomic_json(ledger_path,ledger)
    info={'study_id':STUDY,'modal_volume':VOLUME,'coordinator_call_id':call.object_id,
          'gpu_function_id':gpu_run.object_id,'local_delivery_verified':False,
          'state':'submitted_detached','previous_budget_usd':previous}
    atomic_json(ROOT/'reports'/STUDY/'submission.json',info)
    print(json.dumps(info,sort_keys=True),flush=True)
