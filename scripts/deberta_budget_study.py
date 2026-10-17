"""Budget-bound BF16 training with durable intermediate checkpoints and exact resume."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

from common import file_digest,write_json
from deberta_inputs import ROOT,VOLUME,read
from deberta_bf16_study import RECIPE
from deberta_budget_training import position,validate_progress,can_start_update

STUDY='deberta-heterogeneous-bf16-budget7-s42-20261017-v1'
SECONDS=10000
CAP_USD=7


def config(root=ROOT,evaluation=False):
    c=read(root/'configs'/f'{STUDY}.json')
    if (c['study_id']!=STUDY or c['recipe']!=RECIPE or c['maximum_gpu_seconds']!=SECONDS
            or c['reserve_usd']!=CAP_USD or c['fusion_new_weight']!=.5
            or c['periodic_groups']!=[256,512] or c['stop_update_margin_seconds']!=180):
        raise ValueError('Approved budget7 scope differs')
    for rel,sha in c['source_sha256'].items():
        if not evaluation and rel in c['evaluation_only_files']:continue
        if file_digest(root/rel)!=sha:raise ValueError('Bound source changed: '+rel)
    return c


def install(root=ROOT):
    from deberta_bf16_study import install as old_install
    cp=old_install(root);cp.ROOT=root;cp.STUDY=STUDY
    cp.config=lambda root=root,evaluation=False:config(root,evaluation)
    previous=cp.seal
    def seal(folder,stage):
        if (folder/'training.json').exists():
            progress=read(folder/'training.json')['progress'];validate_progress(progress)
            if not (folder/'assets/updates.jsonl').exists():
                shutil.copyfile(root/progress['audit_path'],folder/'assets/updates.jsonl')
        return previous(folder,stage)
    cp.seal=seal
    return cp


def check_bundle(folder):
    from laya_checkpoint_delivery import within
    m=read(folder/'bundle.json')
    if m['format']!='deberta-bundle-v1' or m['identity']['study_id']!=STUDY:raise ValueError('Unexpected budget7 bundle')
    if len({f['path'] for f in m['files']})!=len(m['files']):raise ValueError('Repeated bundle files')
    for f in m['files']:
        p=within(folder,f['path'])
        if p.stat().st_size!=f['bytes'] or file_digest(p)!=f['sha256']:raise ValueError('Corrupt budget7 bundle')
    return m


def train(cp,epoch,deadline,resume_bundle=None):
    import torch
    import deberta_bf16_training as core
    from laya_head_control import state_digest
    from laya_checkpoint_delivery import atomic_json
    if epoch not in (0,1):raise ValueError('Only epochs0,1')
    core.seed_all();items,stats=cp.training_inputs();device=torch.device('cuda')
    report=cp.ROOT/'reports'/STUDY;folder=resume_bundle
    if folder is None and epoch:folder=cp.bundle(epoch)
    if folder is not None:
        cp.check_bundle(folder);progress=read(folder/'training.json')['progress'];validate_progress(progress)
        if progress['next_epoch']!=epoch:raise ValueError('Resume belongs to different epoch')
        stage=read(folder/'bundle.json')['stage'];receipt=read(report/'cloud-delivery'/(stage+'.json'))
        if not receipt['verification']['strict_load'] or receipt['bundle_sha256']!=file_digest(folder/'bundle.json'):
            raise ValueError('Resume checkpoint lacks independent durable receipt')
        model=core.load_saved(folder)
    else:
        model=core.load_initial(cp.ROOT/config(cp.ROOT)['model_dir'])
        progress=dict(**position(0,0),initial_state_sha256=state_digest(model.state_dict()),history=[],
                      epoch_loss_sum=0.,epoch_seconds=0.)
    model=core.setup_model(model,device);opt,sched=core.optimizer_for(model)
    if folder is not None:
        state=torch.load(folder/'optimizer_rng.pt',map_location='cpu',weights_only=True)
        core.restore(state,model,opt,sched,device,cp.identity());del state;gc.collect()
    else:core.seed_all()
    journal=report/'audit'/f'epoch-{epoch+1:03d}.jsonl';journal.parent.mkdir(parents=True,exist_ok=True)
    if progress['next_group']:
        # An abandoned attempt can contain uncheckpointed updates; retain it, then restore the sealed journal.
        if journal.exists():journal.rename(journal.with_name(journal.name+f'.abandoned-{time.time_ns()}'))
        shutil.copyfile(folder/'assets/updates.jsonl',journal)
    elif journal.exists():raise FileExistsError('Never overwrite an epoch audit')
    groups=core.groups_for(len(items),epoch);last_saved=None
    periodic_groups=config(cp.ROOT)['periodic_groups']
    def save(stage,target,out):
        nonlocal last_saved
        out.flush();os.fsync(out.fileno());progress['audit_path']=str(journal.relative_to(cp.ROOT))
        validate_progress(progress)
        cp.save(model,opt,sched,device,progress,items,stats['pad_token_id'],target,stage)
        request=dict(stage=stage,bundle=str(target.relative_to(cp.ROOT)),
                     bundle_sha256=file_digest(target/'bundle.json'),progress=dict(progress))
        atomic_json(report/'requests'/(stage+'.json'),request)
        atomic_json(report/'pending-checkpoint.json',request)
        ack=report/'acks'/(stage+'.json')
        while not ack.exists():
            if time.time()>deadline-25:raise TimeoutError('No time for durable independent checkpoint ACK')
            time.sleep(.5)
        if read(ack)['bundle_sha256']!=request['bundle_sha256']:raise ValueError('Checkpoint ACK differs')
        (report/'pending-checkpoint.json').unlink(missing_ok=True)
        last_saved=request
        print('CHECKPOINT_ACCEPTED',stage,flush=True)
    with journal.open('a' if progress['next_group'] else 'x',buffering=1) as out:
        for step in range(progress['next_group'],len(groups)):
            if not can_start_update(time.time(),deadline):
                if last_saved is None or last_saved['progress']['updates']!=progress['updates']:
                    stage=f'budget-stop-epoch-{epoch+1:03d}-step-{step:04d}'
                    save(stage,cp.ROOT/'checkpoints'/STUDY/'periodic'/stage,out)
                atomic_json(report/'budget-stop.json',dict(state='budget_stopped_checkpointed',
                    latest_checkpoint=last_saved,test_accessed=False,development_accessed=False))
                return False
            indices=groups[step]
            value=core.update(model,[items[i] for i in indices],stats['pad_token_id'],opt,sched,device,deadline)
            progress['epoch_loss_sum']+=value['loss']*len(indices);progress['epoch_seconds']+=value['seconds']
            out.write(json.dumps(dict(epoch=epoch+1,update=epoch*len(groups)+step+1,
                ids=[items[i]['id'] for i in indices],**value))+'\n')
            if step+1==len(groups):
                out.flush();os.fsync(out.fileno())
                progress['history'].append(dict(epoch=epoch+1,updates=len(groups),exposures=len(items),
                    mean_loss=progress['epoch_loss_sum']/len(items),seconds=progress['epoch_seconds'],
                    journal_sha256=file_digest(journal)))
                progress.update(**position(epoch+1,0),epoch_loss_sum=0.,epoch_seconds=0.)
                save(f'epoch-{epoch+1:03d}',cp.bundle(epoch+1),out)
            else:
                progress.update(position(epoch,step+1))
                if (step+1)%16==0:print(f'epoch={epoch+1} update={step+1}/691 loss={value["loss"]:.6f} training_seconds={progress["epoch_seconds"]:.1f}',flush=True)
                if step+1 in periodic_groups:
                    stage=f'periodic-epoch-{epoch+1:03d}-step-{step+1:04d}'
                    save(stage,cp.ROOT/'checkpoints'/STUDY/'periodic'/stage,out)
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate','verify'])
    p.add_argument('--epoch',type=int);p.add_argument('--deadline',type=float,default=float('inf'))
    p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path);p.add_argument('--resume-bundle',type=Path)
    a=p.parse_args();cp=install()
    if a.mode=='verify':
        progress=read(a.bundle/'training.json')['progress'];validate_progress(progress);cp.verify(a.bundle,a.output)
    elif a.mode=='evaluate':cp.evaluate(a.deadline)
    elif not train(cp,a.epoch,a.deadline,a.resume_bundle):raise SystemExit(42)


if __name__=='__main__':main()
