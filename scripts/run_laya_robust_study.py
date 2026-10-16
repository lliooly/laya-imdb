"""Admit, train six paired models, and gate development on verified local delivery."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import ROOT, file_digest
from laya_checkpoint_delivery import atomic_json, request_sha, wait_for_receipt, verify_files
from run_laya_robust_loss import STUDY, PAIRS, name, verify_contract, contract_path, request_for, production_receipt


def check_admission():
    summaries = []
    for arm in ('ce','mixed'):
        records = []
        for repeat in (1,2):
            run = name(42, arm, repeat); base = ROOT/'checkpoints'/run
            req = request_for(run, 'frozen'); receipt = production_receipt(req, 0)
            epoch_req = request_for(run, 'epoch-001'); transfer = production_receipt(epoch_req, 0)
            if transfer.get('minimum_free_bytes',0) < 100*1024**3:
                raise ValueError('Receiver did not enforce 100 GiB disk reserve')
            audit = json.loads((base/'epoch-002/training_audit.json').read_text())
            manifest = json.loads((base/'epoch-002/manifest.json').read_text())
            result = json.loads((base/'epoch-002/result.json').read_text())
            if result.get('restored_epoch1') is not (repeat == 2):
                raise ValueError('Admission repeat2 must test real CUDA epoch recovery')
            updates = [json.loads(line) for line in (base/'epoch-002/updates.jsonl').read_text().splitlines()]
            if audit['updates'] != 32 or audit['gradient_exposures'] != 256 or len(updates) != 32:
                raise ValueError('Admission requires 32 actual updates and 256 exposures')
            records.append({'run_id':run, 'initial_state':manifest['initial_state_sha256'],
                'input_sha':audit['input_stream_sha256'], 'updates':updates,
                'final_sha':receipt['verification']['state_sha256'],
                'freeze_request_sha256':request_sha(req), 'receive_seconds':transfer['receive_seconds'],
                'bundle_bytes':transfer['bundle_bytes']})
        a,b = records
        for field in ('initial_state','input_sha','updates','final_sha'):
            if a[field] != b[field]:
                raise ValueError('Repeated same-arm admission differs: ' + field)
        summaries.append({'arm':arm, 'repeat_equal':True, 'records':records})
    if summaries[0]['records'][0]['initial_state'] != summaries[1]['records'][0]['initial_state']:
        raise ValueError('Admission initial weights differ across loss arms')
    if summaries[0]['records'][0]['input_sha'] != summaries[1]['records'][0]['input_sha']:
        raise ValueError('Loss arm changed actual gradient input schedule')
    if summaries[0]['records'][0]['final_sha'] == summaries[1]['records'][0]['final_sha']:
        raise ValueError('Loss intervention did not change any learned parameter')
    return {'state':'passed', 'arms':summaries, 'cuda_and_local_delivery_passed':True,
            'contract_sha256':file_digest(contract_path()), 'test_accessed':False}


def freeze_all():
    runs = [name(s,a) for s,a in PAIRS]
    hashes = {}
    for seed in (42,43,44):
        pair = []
        for arm in ('ce','mixed'):
            run = name(seed,arm); base = ROOT/'checkpoints'/run
            req = request_for(run,'frozen'); verify_files(ROOT,req); production_receipt(req,0)
            audit = json.loads((base/'epoch-002/training_audit.json').read_text())
            manifest = json.loads((base/'epoch-002/manifest.json').read_text())
            if (audit['updates'],audit['gradient_exposures']) != (5526,44200):
                raise ValueError('Formal training budget incomplete')
            pair.append((audit['input_stream_sha256'],manifest['initial_state_sha256']))
            hashes[run] = request_sha(req)
        if pair[0] != pair[1]:
            raise ValueError('Formal paired input streams/initial weights differ')
    gate = {'run_ids':runs, 'freeze_request_sha256':hashes,'contract_sha256':file_digest(contract_path()),
            'development_body_read':False,'test_accessed':False}
    atomic_json(ROOT/'reports'/STUDY/'all-models-frozen.json',gate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('admission','formal','evaluate'),required=True)
    p.add_argument('--delivery-timeout',type=float,default=10800)
    p.add_argument('--resume-run',help='Explicitly recover only the matching interrupted run')
    args = p.parse_args()
    contract,_ = verify_contract()
    report = ROOT/'reports'/STUDY;report.mkdir(parents=True,exist_ok=True)
    def status(state,**extra):
        atomic_json(report/'status.json',{'state':state,'phase':args.phase,**extra})
    def run(mode,seed,arm,repeat=None,checkpoint_exit=False,force_resume=False):
        run_id = name(seed,arm,repeat)
        if mode == 'train' and (ROOT/'checkpoints'/run_id/'frozen').exists():
            production_receipt(request_for(run_id,'frozen'),0)
            return
        command = [sys.executable,'-u',str(ROOT/'scripts/run_laya_robust_loss.py'),
                   '--mode',mode,'--seed',str(seed),'--arm',arm,'--delivery-timeout',str(args.delivery_timeout)]
        if repeat:
            command += ['--repeat',str(repeat)]
        if mode == 'train' and (args.resume_run == run_id or force_resume):
            command.append('--resume')
        if checkpoint_exit:
            command.append('--admission-checkpoint-exit')
        status('running',run_id=run_id,mode=mode)
        process = subprocess.Popen(command,cwd=ROOT,start_new_session=True)
        try:
            code = process.wait(timeout=3*args.delivery_timeout+7200)
        except BaseException:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        if code:
            raise RuntimeError(f'{run_id} {mode} failed; stop without automatic retraining')
    try:
        if args.phase in ('admission','formal'):
            if shutil.disk_usage(ROOT).free < contract['minimum_free_bytes']:
                raise OSError('Require 100 GiB free on producer')
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise RuntimeError('GPU already has compute processes')
        if args.phase == 'admission':
            for arm in ('ce','mixed'):
                for repeat in (1,2):
                    if repeat == 2:
                        run('train',42,arm,repeat,checkpoint_exit=True)
                        run('train',42,arm,repeat,force_resume=True)
                    else:
                        run('train',42,arm,repeat)
                    run('reload',42,arm,repeat)
            atomic_json(report/'admission.json',check_admission())
        elif args.phase == 'formal':
            admission = json.loads((report/'admission.json').read_text())
            if (admission['state'] != 'passed' or admission['contract_sha256'] != file_digest(contract_path())):
                raise ValueError('Missing current CUDA and local transfer admission')
            check_admission()  # Recheck real frozen models and receipts, not a boolean alone.
            for seed,arm in PAIRS:
                run('train',seed,arm);run('reload',seed,arm)
            freeze_all()
        else:
            freeze_all()
            for seed,arm in PAIRS:
                run('evaluate',seed,arm)
        status('completed_phase',formal_training_started=args.phase=='formal',test_accessed=False)
    except BaseException as exc:
        status('stopped',error=repr(exc),test_accessed=False)
        raise


if __name__ == '__main__':
    main()
