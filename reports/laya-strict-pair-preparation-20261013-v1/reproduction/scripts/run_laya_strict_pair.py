"""Bounded CUDA smoke + seed43 paired experiment; never reads the official test body."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

from common import ROOT, file_digest, write_json
from comparison_metrics import paired


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds', type=int, default=7200)
    args = parser.parse_args()
    if not 600 <= args.max_seconds <= 7200:
        raise ValueError('Runtime must be bounded to at most two hours')
    modes = ('control', 'resampled')
    full = [ROOT / f'configs/laya-rtx4090-ce-resampling-{mode}-pool45-s43-epochs2-strict-20261013-v1.json' for mode in modes]
    smokes = [ROOT / f'configs/laya-rtx4090-smoke-ce-resampling-{mode}-pool45-s43-epochs2-strict-20261013-v1.json' for mode in modes]
    specs = [json.loads(p.read_text()) for p in full]
    contract = json.loads((ROOT/'reports/laya-resampling-45-preparation-v2/execution-contract.json').read_text())
    for spec in specs:
        if spec.get('runtime_profile') != 'strict':
            raise ValueError('Both paired runs require strict runtime')
        if spec['sampling']['candidate_sha256'] != contract['candidate_sha256'] or spec['sampling']['candidate_file'] != contract['candidate_file']:
            raise ValueError('Selected 45-item candidate binding mismatch')
    # Refuse unintended changes to any paired factor outside sampling mode.
    factor = [{k:v for k,v in s.items() if k not in ('run_id','sampling')} for s in specs]
    if factor[0] != factor[1]:
        raise ValueError('Paired configurations differ outside sampling')
    for paths in (full, smokes):
        for path in paths:
            s=json.loads(path.read_text())
            if s['training']['seed']!=43 or s['training']['epochs']!=2 or not s['epoch_checkpoints']:
                raise ValueError('Seed43, two epochs and checkpoint retention required')
            for prefix in ('reports','checkpoints'):
                if (ROOT/prefix/s['run_id']).exists():raise FileExistsError(s['run_id'])
    OUT=ROOT/'artifacts/resampling45-seed43-strict-20261013-v1'
    OUT.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+args.max_seconds
    state={'state':'running','started_utc':now(),'max_seconds':args.max_seconds,'training_performed':False,
           'test_accessed':False,'runner_sha256':file_digest(Path(__file__)),'stages':[]}
    def status():write_json(OUT/'status.json',state)
    def run(stage,command,cap):
        seconds=min(cap,deadline-time.monotonic())
        if seconds<=0:raise TimeoutError('Paired experiment runtime ceiling reached')
        entry={'stage':stage,'started_utc':now(),'command':command,'cap_seconds':seconds}
        state['stages'].append(entry);status()
        with (OUT/(stage+'.log')).open('w') as log:
            process=subprocess.Popen([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry['pid']=process.pid;status()
            try:code=process.wait(timeout=seconds)
            except BaseException:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
        entry.update(returncode=code,finished_utc=now());status()
        if code:raise RuntimeError(stage+' failed; inspect '+str(OUT/(stage+'.log')))
    def audit_training(s,expected_count):
        report=ROOT/'reports'/s['run_id']
        manifest=json.loads((report/'manifest.json').read_text())
        runtime=manifest['runtime']
        if (runtime['profile'] != 'strict' or runtime['cublas_workspace_config'] != ':4096:8'
                or not runtime['deterministic_algorithms'] or runtime['warn_only']
                or not runtime['cudnn_deterministic'] or runtime['cudnn_benchmark']):
            raise ValueError('Actual runtime does not satisfy strict protocol')
        sampling=json.loads((report/'sampling_audit.json').read_text())
        optimization=json.loads((report/'optimization_diagnostics.json').read_text())
        expected_updates=2*((expected_count+7)//8)
        if expected_count == 22100:
            if sampling['epoch_ordered_id_sha256'] != contract['epoch_ordered_id_sha256'][s['sampling']['mode']]:
                raise ValueError('Actual full epoch schedule differs from predeclared 45-item plan')
            if any(sampling['exposure_counts'][i] != 2 for i in contract['quarantined_ids']):
                raise ValueError('Boundary samples must retain their normal two exposures')
        if sampling['state']!='completed' or sampling['actual_train_count']!=expected_count or sampling['epochs_observed']!=[1,2]:
            raise ValueError('Sampling epoch/count audit failed')
        if optimization['state']!='completed' or optimization['completed_optimizer_updates']!=expected_updates or sampling['expected_updates']!=expected_updates:
            raise ValueError('Actual update count mismatch')
        epochs=json.loads((report/'epoch_checkpoints.json').read_text())['epochs']
        if [e['epoch'] for e in epochs]!=[1,2]:raise ValueError('Both epoch checkpoints required')
        checkpoint=ROOT/'checkpoints'/s['run_id']
        for entry in epochs:
            if file_digest(checkpoint/('epoch-%03d'%entry['epoch'])/'model.safetensors')!=entry['weights_sha256']:
                raise ValueError('Epoch checkpoint hash mismatch')
        return sampling,epochs
    try:
        status()
        run('cpu-resampling-contract',['-m','unittest','discover','-s','tests','-p','test_laya_resampling.py','-v'],120)
        run('cpu-runtime-contract',['-m','unittest','discover','-s','tests','-p','test_laya_runtime.py','-v'],120)
        admission=[]
        for repeat in (1,2):
            config=dict(specs[0],run_id=OUT.name+'-admission-'+str(repeat))
            path=OUT/('admission-'+str(repeat)+'-config.json')
            write_json(path,config)
            trace=ROOT/'reports'/(config['run_id']+'-trace')
            admission.append(trace)
            run('strict-admission-'+str(repeat),['scripts/run_laya_strict_trace.py','--config',str(path),'--out',str(trace)],300)
        reference=ROOT/'reports/laya-determinism-cuda-20261012-v1/strict-control-1'
        comparisons={}
        for label,left,right in [('current-repeat',admission[0],admission[1]),
                                 ('validated-reference',admission[0],reference)]:
            rows=[[json.loads(line) for line in (folder/'trace/updates.jsonl').read_text().splitlines()] for folder in (left,right)]
            gradients=[json.loads((folder/'trace/first-gradients.json').read_text()) for folder in (left,right)]
            identical=len(rows[0])==len(rows[1])==16 and rows[0]==rows[1] and gradients[0]==gradients[1]
            comparisons[label]={'identical':identical,'updates':len(rows[0]),'parameter_count':len(gradients[0])}
            if not identical:
                write_json(OUT/'admission-comparison.json',comparisons)
                raise ValueError('Strict admission divergence: '+label)
        write_json(OUT/'admission-comparison.json',comparisons)
        for mode,path in zip(modes,smokes):
            run(mode+'-cuda-smoke',['scripts/audit_cuda_training.py','--config',str(path)],300)
            s=json.loads(path.read_text());audit_training(s,128)
            numerical=json.loads((ROOT/'reports'/s['run_id']/'numerical_audit.json').read_text())
            if not numerical.get('passed') or len(numerical['optimizer_steps'])!=32:raise ValueError('CUDA smoke did not pass')
        # Compare smoke subset identities; only exposure schedules may differ.
        smoke_sampling=[json.loads((ROOT/'reports'/json.loads(p.read_text())['run_id']/'sampling_audit.json').read_text()) for p in smokes]
        if smoke_sampling[0]['actual_gradient_ids']!=smoke_sampling[1]['actual_gradient_ids']:raise ValueError('Smoke subsets differ')
        state['training_performed']=True;status()
        for mode,path,s in zip(modes,full,specs):
            run(mode+'-train',['scripts/train_laya.py','--config',str(path)],3300)
            sampling,epochs=audit_training(s,22100)
            report=ROOT/'reports'/s['run_id'];checkpoint=ROOT/'checkpoints'/s['run_id']
            curve=[]
            for epoch in epochs:
                destination=report/('cuda-saved-validation-epoch%d.json'%epoch['epoch'])
                run(mode+'-epoch%d-validation'%epoch['epoch'],['scripts/export_cuda_laya_predictions.py','--checkpoint',str(checkpoint/('epoch-%03d'%epoch['epoch'])),'--out',str(destination)],300)
                curve.append(dict(epoch,raw_metrics=json.loads(destination.read_text())['raw_metrics']))
            run(mode+'-final-validation',['scripts/export_cuda_laya_predictions.py','--checkpoint',str(checkpoint),'--out',str(report/'cuda-saved-validation.json')],300)
            write_json(report/'epoch_validation_curve.json',{'epochs':curve,'primary_metric':'final_saved_checkpoint_accuracy','best_epoch':max(curve,key=lambda e:(e['raw_metrics']['accuracy'],e['epoch']))['epoch'],'test_accessed':False})
        reports=[ROOT/'reports'/s['run_id'] for s in specs]
        manifests=[json.loads((r/'manifest.json').read_text()) for r in reports]
        for key in ('gradient_train_ids','calibration_ids','validation','model_lock','optimization','runtime','runtime_adapter_sha256'):
            if manifests[0][key]!=manifests[1][key]:raise ValueError('Paired mismatch: '+key)
        exports=[json.loads((r/'cuda-saved-validation.json').read_text()) for r in reports]
        predictions=[e['validation_predictions'] for e in exports]
        inputs=[[(p['sample_id'],p['true_label'],p['input_sha256']) for p in rows] for rows in predictions]
        if inputs[0]!=inputs[1] or len(inputs[0])!=2500:raise ValueError('Paired full-validation input mismatch')
        comparison={'seed':43,'test_accessed':False,'input_hashes_all_match':True,
                    'metrics':{mode:e['raw_metrics'] for mode,e in zip(modes,exports)},
                    'paired':paired([p['true_label'] for p in predictions[0]],
                        [p['logits'][1]-p['logits'][0] for p in predictions[0]],
                        [p['logits'][1]-p['logits'][0] for p in predictions[1]])}
        write_json(OUT/'comparison.json',comparison)
        state.update(state='completed',finished_utc=now(),gpu_work_finished=True);status()
    except BaseException as exc:
        state.update(state='failed',error=repr(exc),finished_utc=now(),gpu_work_finished=True);status();raise
    finally:
        status()
        # Logs and diagnostics only; raw typed review copies and weights excluded.
        bundle=OUT.with_name(OUT.name+'-reports.tar.gz')
        with tarfile.open(bundle,'w:gz') as tar:
            tar.add(OUT,arcname=str(OUT.relative_to(ROOT)))
            for config_path in OUT.glob('admission-*-config.json'):
                admission_spec=json.loads(config_path.read_text())
                for folder in (ROOT/'reports'/admission_spec['run_id'],ROOT/'reports'/(admission_spec['run_id']+'-trace')):
                    if folder.exists():
                        for f in folder.rglob('*'):
                            if f.is_file() and 'data' not in f.relative_to(folder).parts:
                                tar.add(f,arcname=str(f.relative_to(ROOT)))
            for path in full+smokes:
                s=json.loads(path.read_text());folder=ROOT/'reports'/s['run_id']
                if folder.exists():
                    for f in folder.rglob('*'):
                        if f.is_file() and 'data' not in f.relative_to(folder).parts:
                            tar.add(f,arcname=str(f.relative_to(ROOT)))
        write_json(OUT.with_name(OUT.name+'-bundle.json'),{'path':str(bundle.relative_to(ROOT)),'sha256':file_digest(bundle),'size':bundle.stat().st_size})


if __name__=='__main__':main()
