"""One user-authorized BF16 attempt; shared absolute compute window, no retries."""
import json
from pathlib import Path
import sys
import time
import modal

ROOT=Path(__file__).resolve().parents[1] if modal.is_local() else Path('/workspace/End-IMDB')
sys.path.insert(0,str(ROOT/'scripts'))
from deberta_budget_study import STUDY,VOLUME,SECONDS,CAP_USD

app=modal.App(STUDY)
volume=modal.Volume.from_name(VOLUME,create_if_missing=False)
if modal.is_local():
    image=(modal.Image.from_id('im-wjfLMpFBzPeiRaUhpdw2dA')
           .add_local_dir(str(ROOT/'scripts'),'/opt/modal_job',ignore=['__pycache__/**']))
else:image=None


@app.function(image=image,gpu='L40S',cpu=(4,4),memory=(32768,32768),
    volumes={'/workspace':volume},timeout=10000,startup_timeout=120,retries=0,
    max_containers=1,scaledown_window=2,include_source=False,serialized=True)
def gpu_job(mode,epoch,deadline):
    import os
    import signal
    import subprocess
    from common import file_digest
    from laya_checkpoint_delivery import atomic_json
    from deberta_budget_study import config,read,install
    root=Path('/workspace/End-IMDB');volume.reload();config(root,evaluation=mode=='evaluate')
    study=install(root);report=root/'reports'/STUDY;python='/.uv/.venv/bin/python'
    driver=root/'scripts/deberta_budget_study.py';tag=f'epoch-{epoch+1:03d}' if mode=='train' else 'evaluate'
    log=report/'logs'/(tag+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    process=None
    def accept(request):
        stage=request['stage'];folder=root/request['bundle'];ack=report/'acks'/(stage+'.json')
        if ack.exists():return
        volume.commit()  # Child is paused with immutable model/optimizer/source files closed.
        if request['bundle_sha256']!=file_digest(folder/'bundle.json'):raise ValueError('Pending bundle changed')
        checked=report/'cloud-verification'/(stage+'.json');remaining=deadline-time.time()-25
        if remaining<15:raise TimeoutError('Checkpoint verifier absolute deadline')
        subprocess.run([python,str(driver),'verify','--bundle',str(folder),'--output',str(checked)],
                       check=True,cwd=root,timeout=min(300,remaining))
        manifest=study.check_bundle(folder);verification=read(checked)
        receipt=dict(state='verified_committed_volume',bundle_manifest=manifest,
            bundle_sha256=request['bundle_sha256'],verification=verification,modal_volume=VOLUME,
            local_delivery_verified=False,test_accessed=False)
        atomic_json(report/'cloud-delivery'/(stage+'.json'),receipt)
        index_path=report/'checkpoint-index.json'
        index=read(index_path) if index_path.exists() else {'receipts':{}}
        index['receipts'][stage]=f'cloud-delivery/{stage}.json';index['latest_checkpoint']=request
        atomic_json(index_path,index);volume.commit() # Receipt + checkpoint durable before ACK.
        atomic_json(ack,dict(bundle_sha256=request['bundle_sha256'],verified=True));volume.commit()
        print('CLOUD_VERIFIED',stage,flush=True)
    try:
        command=[python,'-u',str(driver),mode,'--deadline',str(deadline)]
        if mode=='train':command+=['--epoch',str(epoch)]
        with log.open('x',buffering=1) as out:
            process=subprocess.Popen(command,cwd=root,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            last=0.
            while process.poll() is None:
                if time.time()>deadline-20:raise TimeoutError('Paid study absolute deadline')
                pending=report/'pending-checkpoint.json'
                if pending.exists():accept(read(pending))
                if time.time()-last>30:
                    lines=log.read_text().splitlines();print('PROGRESS',tag,lines[-1][-500:] if lines else 'loading',flush=True)
                    # Journal/log visibility without pausing an update; only sealed checkpoints are deliverable.
                    volume.commit();last=time.time()
                time.sleep(1)
        volume.commit()
        if process.returncode==42:
            return dict(mode=mode,epoch=epoch,**read(report/'budget-stop.json'))
        if process.returncode:
            print(log.read_text()[-10000:],flush=True);raise RuntimeError(f'{tag} exited {process.returncode}')
        if mode=='evaluate':
            folder=report/'evaluation';manifest=study.check_bundle(folder)
            receipt=dict(state='verified_committed_volume',bundle_manifest=manifest,
                bundle_sha256=file_digest(folder/'bundle.json'),verification={'all_file_sha256_verified':True,'test_accessed':False},
                modal_volume=VOLUME,local_delivery_verified=False,test_accessed=False)
            atomic_json(report/'cloud-delivery/evaluate.json',receipt)
            index=read(report/'checkpoint-index.json');index['receipts']['evaluation']='cloud-delivery/evaluate.json'
            atomic_json(report/'checkpoint-index.json',index);volume.commit()
        return dict(mode=mode,epoch=epoch,state='cloud_verified',test_accessed=False)
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        volume.commit();raise


@app.function(image=image,cpu=(.125,.125),memory=(2048,2048),nonpreemptible=True,
    volumes={'/workspace':volume},timeout=10120,retries=0,max_containers=1,
    scaledown_window=2,include_source=False,serialized=True)
def coordinate(gpu_handle,submitted,deadline):
    from laya_checkpoint_delivery import atomic_json
    from deberta_inputs import read
    root=Path('/workspace/End-IMDB');volume.reload();report=root/'reports'/STUDY
    path=report/'state.json';state=read(path);active=None
    if state['state']!='admission_reused_authorized':raise ValueError('Need authorized reused admission')
    def save():state.update(updated_unix=time.time());atomic_json(path,state);volume.commit()
    try:
        state['state']='running';save()
        for mode,epoch in [('train',0),('train',1),('evaluate',None)]:
            if time.time()>deadline-180:raise TimeoutError('Shared deadline before next stage')
            state.update(active_job={'mode':mode,'epoch':epoch});save()
            active=gpu_handle.spawn(mode,epoch,deadline);state['active_call_id']=active.object_id;save()
            result=active.get(timeout=max(1,deadline-time.time()-15));active=None
            volume.reload();state.setdefault('completed',[]).append(result);state.pop('active_call_id',None)
            if result['state']=='budget_stopped_checkpointed':
                state.update(state='stopped_budget_checkpointed',latest_checkpoint=result['latest_checkpoint']);save();return state
            save()
        state.update(state='completed_cloud_pending_local',test_accessed=False);save();return state
    except BaseException as exc:
        if active:active.cancel(terminate_containers=True)
        volume.reload();state.update(state='stopped_budget' if isinstance(exc,TimeoutError) else 'stopped_error',error=repr(exc));save();raise


@app.local_entrypoint()
def main():
    import shutil
    from laya_checkpoint_delivery import atomic_json
    from deberta_budget_study import config,read
    c=config(ROOT);report=ROOT/'reports'/STUDY
    if (report/'submission.json').exists():raise FileExistsError('No automatic resubmission')
    if shutil.disk_usage(ROOT).free<90*1024**3:raise OSError('Need90GiB for full and partial delivery')
    reservation=read(ROOT/'delivery'/f'{STUDY}-budget.json')
    if reservation['state']!='prepared_reserved' or reservation['current_actual_balance_usd']!=7.03:
        raise ValueError('Reservation/balance differs')
    old=ROOT/'reports/deberta-heterogeneous-bf16-s42-modal-20261017-v1'
    if not read(old/'pilot-result.json')['exact_model_optimizer_scheduler_rng_resume']:
        raise ValueError('Reused BF16 admission not passed')
    if not read(old/'pilot-small-file-delivery.json')['verified']:
        raise ValueError('Reused persistent receipt not received')
    print('UPLOADING_FROZEN_NEW_INPUTS',len(c['upload_files']),flush=True)
    with volume.batch_upload(force=True) as batch:
        for rel in c['upload_files']:batch.put_file(ROOT/rel,'/End-IMDB/'+rel)
        batch.put_file(ROOT/'configs'/f'{STUDY}.json','/End-IMDB/configs/'+STUDY+'.json')
    submitted=time.time();deadline=submitted+SECONDS
    submission=dict(study_id=STUDY,submitted_unix=submitted,absolute_deadline_unix=deadline,
                    reserve_usd=CAP_USD,admission='reused_passed_bf16',retries=0)
    atomic_json(report/'submission.json',submission)
    state=dict(**submission,state='admission_reused_authorized',completed=[],local_delivery_verified=False,test_accessed=False)
    local_state=report/'state.json';atomic_json(local_state,state)
    with volume.batch_upload(force=False) as batch:batch.put_file(local_state,'/End-IMDB/reports/'+STUDY+'/state.json')
    running=coordinate.spawn(gpu_job,submitted,deadline)
    submission['coordinator_call_id']=running.object_id;atomic_json(report/'submission.json',submission)
    reservation.update(state='running',**submission);atomic_json(ROOT/'delivery'/f'{STUDY}-budget.json',reservation)
    print(json.dumps(dict(state='submitted_formal',**submission)),flush=True)
