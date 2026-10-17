"""One bounded DeBERTa admission and optional two-epoch run, never auto-retried."""
import json
from pathlib import Path
import sys
import time

import modal

ROOT=Path(__file__).resolve().parents[1] if modal.is_local() else Path('/workspace/End-IMDB')
sys.path.insert(0,str(ROOT/'scripts'))
from deberta_bf16_study import STUDY,VOLUME
from deberta_image_build import install_tokenizer_dependencies

app=modal.App(STUDY)
volume=modal.Volume.from_name(VOLUME,create_if_missing=False)


if modal.is_local():
    image=(modal.Image.from_id('im-wjfLMpFBzPeiRaUhpdw2dA')
           .add_local_dir(str(ROOT/'scripts'),'/opt/modal_job',ignore=['__pycache__/**']))
else:image=None


@app.function(image=image,gpu='L40S',cpu=(4,4),memory=(32768,32768),
              volumes={'/workspace':volume},timeout=5400,startup_timeout=120,
              retries=0,max_containers=1,scaledown_window=2,include_source=False,serialized=True)
def gpu_job(mode,epoch,deadline,pilot_deadline=None):
    import os
    import signal
    import subprocess
    from common import file_digest
    from laya_checkpoint_delivery import atomic_json
    from deberta_bf16_study import config,read,install
    root=Path('/workspace/End-IMDB');volume.reload();config(root,evaluation=mode=='evaluate')
    study=install(root)
    bundle,check_bundle=study.bundle,study.check_bundle
    report=root/'reports'/STUDY;report.mkdir(parents=True,exist_ok=True)
    python='/.uv/.venv/bin/python';driver=root/'scripts/deberta_bf16_study.py'
    tag=f'epoch-{epoch+1:03d}' if mode=='train' else mode
    log=report/'logs'/(tag+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    end=min(deadline,pilot_deadline) if pilot_deadline else deadline
    process=None
    def run(command,suffix=''):
        nonlocal process
        target=log if not suffix else log.with_name(tag+suffix+'.log')
        with target.open('x',buffering=1) as out:
            process=subprocess.Popen(command,cwd=root,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            last=0.
            while process.poll() is None:
                if time.time()>end-45:raise TimeoutError('Paid study absolute deadline')
                if time.time()-last>30:
                    lines=target.read_text().splitlines();print('PROGRESS',tag,lines[-1][-500:] if lines else 'loading',flush=True);last=time.time()
                time.sleep(1)
        if process.returncode:
            print(target.read_text()[-10000:],flush=True);raise RuntimeError(f'{tag}{suffix} exited {process.returncode}')
    try:
        command=[python,'-u',str(driver),mode,'--deadline',str(end)]
        if mode=='train':command+=['--epoch',str(epoch)]
        run(command);volume.commit()
        if mode=='pilot':
            run([python,'-u',str(driver),'pilot-resume','--deadline',str(end)],'-resume')
            folder=root/'checkpoints'/STUDY/'pilot-boundary'
        elif mode=='train':folder=bundle(epoch+1)
        else:folder=report/'evaluation'
        if mode in ('pilot','train'):
            checked=report/'cloud-verification'/(tag+'.json')
            remaining=end-time.time()-30
            if remaining<30:raise TimeoutError('No verification time remains')
            subprocess.run([python,str(driver),'verify','--bundle',str(folder),'--output',str(checked)],
                           check=True,cwd=root,timeout=min(300,remaining))
            verification=read(checked)
        else:verification={'all_file_sha256_verified':True,'test_accessed':False}
        manifest=check_bundle(folder)
        record=dict(state='verified_committed_volume',bundle_manifest=manifest,
                    bundle_sha256=file_digest(folder/'bundle.json'),verification=verification,
                    modal_volume=VOLUME,local_delivery_verified=False,test_accessed=False)
        atomic_json(report/'cloud-delivery'/(tag+'.json'),record);volume.commit()
        if mode=='pilot':return dict(receipt=record,pilot=read(report/'pilot-result.json'))
        return dict(mode=mode,epoch=epoch,state='cloud_verified',test_accessed=False)
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        volume.commit();raise


@app.function(image=image,cpu=(.125,.125),memory=(2048,2048),nonpreemptible=True,
              volumes={'/workspace':volume},timeout=5700,retries=0,max_containers=1,
              scaledown_window=2,include_source=False,serialized=True)
def coordinate(gpu_handle,submitted,deadline):
    from laya_checkpoint_delivery import atomic_json
    from deberta_inputs import read
    root=Path('/workspace/End-IMDB');volume.reload();report=root/'reports'/STUDY
    path=report/'state.json';state=read(path)
    if state['state']!='admission_passed':raise ValueError('No approved admission and small-file delivery')
    active=None
    def save():state.update(updated_unix=time.time());atomic_json(path,state);volume.commit()
    try:
        state['state']='running';save()
        for mode,epoch in [('train',0),('train',1),('evaluate',None)]:
            if time.time()>deadline-180:raise TimeoutError('Study deadline before next stage')
            state.update(active_job={'mode':mode,'epoch':epoch});save()
            active=gpu_handle.spawn(mode,epoch,deadline)
            state['active_call_id']=active.object_id;save()
            result=active.get(timeout=max(1,deadline-time.time()-30));active=None
            volume.reload();state.setdefault('completed',[]).append(result);state.pop('active_call_id',None);save()
        state.update(state='completed_cloud_pending_local',test_accessed=False);save();return state
    except BaseException as exc:
        if active:active.cancel(terminate_containers=True)
        volume.reload();state.update(state='stopped_budget' if isinstance(exc,TimeoutError) else 'stopped_error',error=repr(exc));save();raise


@app.local_entrypoint()
def main():
    import shutil
    from common import file_digest
    from laya_checkpoint_delivery import atomic_json
    from deberta_inputs import read
    from deberta_bf16_study import config,SECONDS,CAP_USD
    c=config(ROOT);report=ROOT/'reports'/STUDY
    if (report/'submission.json').exists():raise FileExistsError('Never auto-resubmit this study')
    if shutil.disk_usage(ROOT).free<65*1024**3:raise OSError('Need65GiB for complete delivery')
    prior=read(ROOT/'delivery/deberta-heterogeneous-s42-modal-20261017-v2-budget.json')
    if not prior['compute_closed'] or prior['conservative_available_after_this_study_usd']<CAP_USD:
        raise ValueError('Prior reservation not closed or balance basis changed')
    budget=ROOT/'delivery'/f'{STUDY}-budget.json'
    reservation=read(budget)
    if reservation['state']!='prepared_reserved':raise ValueError('Need one pre-build reservation')
    with volume.batch_upload(force=True) as batch:
        for rel in c['upload_files']:batch.put_file(ROOT/rel,'/End-IMDB/'+rel)
        batch.put_file(ROOT/'configs'/f'{STUDY}.json','/End-IMDB/configs/'+STUDY+'.json')
    submitted=time.time();deadline=submitted+SECONDS
    submission=dict(study_id=STUDY,submitted_unix=submitted,absolute_deadline_unix=deadline,
                    pilot_deadline_unix=submitted+900,reserve_usd=CAP_USD)
    atomic_json(report/'submission.json',submission)
    state=dict(study_id=STUDY,state='pilot_running',**{k:v for k,v in submission.items() if k!='study_id'},
               completed=[],local_delivery_verified=False,test_accessed=False)
    local_state=report/'state.json';atomic_json(local_state,state)
    with volume.batch_upload(force=False) as batch:batch.put_file(local_state,'/End-IMDB/reports/'+STUDY+'/state.json')
    reservation.update(state='pilot_running',**submission);atomic_json(budget,reservation)
    call=gpu_job.spawn('pilot',None,deadline,submitted+900)
    submission['pilot_call_id']=call.object_id;atomic_json(report/'submission.json',submission)
    try:
        returned=call.get(timeout=900)
    except BaseException as exc:
        call.cancel(terminate_containers=True);state.update(state='stopped_pilot_error',error=repr(exc))
        atomic_json(local_state,state);reservation['state']=state['state'];atomic_json(budget,reservation)
        with volume.batch_upload(force=True) as batch:batch.put_file(local_state,'/End-IMDB/reports/'+STUDY+'/state.json')
        raise
    pilot=returned['pilot'];atomic_json(report/'pilot-result.json',pilot)
    # Download and verify exact persisted small-file bytes before permitting formal training.
    remote='/End-IMDB/reports/'+STUDY+'/pilot-result.json'
    data=b''.join(volume.read_file(remote))
    decoded=json.loads(data)
    if decoded!=pilot or not decoded['exact_model_optimizer_scheduler_rng_resume']:
        raise ValueError('Pilot receipt or CPU-only small-file download failed')
    atomic_json(report/'pilot-small-file-delivery.json',dict(verified=True,sha256=__import__('hashlib').sha256(data).hexdigest(),gpu_called=False))
    timings=[x['seconds'] for x in pilot['representative']]
    # Conservative time guard: 25% on worst observed representative update,
    # 50% inference margin and 10 minutes for saves, cold reloads and independent verification.
    estimate=1382*max(timings)*1.25+(2900/128)*pilot['inference_seconds']*1.5+max(600,4*pilot['save_seconds'])
    remaining=deadline-time.time()-120
    gate=dict(estimated_formal_seconds=estimate,remaining_compute_seconds=remaining,
              exact_resume=True,small_file_delivery=True,
              representative_worst_update_seconds=max(timings),
              state='passed' if estimate<remaining else 'stopped_budget_projection')
    atomic_json(report/'admission.json',gate);state['admission']=gate
    if gate['state']!='passed':
        state.update(state='stopped_budget_projection',test_accessed=False)
        atomic_json(local_state,state);reservation['state']=state['state'];atomic_json(budget,reservation)
        with volume.batch_upload(force=True) as batch:batch.put_file(local_state,'/End-IMDB/reports/'+STUDY+'/state.json')
        print(json.dumps(state),flush=True);return
    state['state']='admission_passed';atomic_json(local_state,state)
    with volume.batch_upload(force=True) as batch:batch.put_file(local_state,'/End-IMDB/reports/'+STUDY+'/state.json')
    running=coordinate.spawn(gpu_job,submitted,deadline)
    submission.update(coordinator_call_id=running.object_id);atomic_json(report/'submission.json',submission)
    reservation.update(state='running',coordinator_call_id=running.object_id);atomic_json(budget,reservation)
    print(json.dumps(dict(state='submitted_formal',**submission,admission=gate)),flush=True)
