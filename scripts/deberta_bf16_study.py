"""Isolated BF16 adapter; all historical FP32 modules and bundles stay unchanged."""
import argparse
import importlib.util
from pathlib import Path
import sys

from common import file_digest,write_json
from deberta_inputs import ROOT,VOLUME,read,RECIPE as FP32_RECIPE

STUDY='deberta-heterogeneous-bf16-s42-modal-20261017-v1'
RECIPE=dict(FP32_RECIPE,precision='bfloat16')
SECONDS=5400
CAP_USD=5


def config(root=ROOT,evaluation=False):
    c=read(root/'configs'/f'{STUDY}.json')
    if (c['study_id']!=STUDY or c['recipe']!=RECIPE or c['maximum_gpu_seconds']!=SECONDS
            or c['reserve_usd']!=5 or c['pilot_reserve_usd']!=1 or c['pilot_maximum_seconds']!=900
            or c['fusion_new_weight']!=.5):raise ValueError('Approved BF16 scope differs')
    for rel,sha in c['source_sha256'].items():
        if not evaluation and rel in c['evaluation_only_files']:continue
        if file_digest(root/rel)!=sha:raise ValueError('Bound source changed: '+rel)
    return c


def install(root=ROOT):
    import deberta_bf16_training as core
    # Deferred imports in the frozen study bind only inside this process to this private core.
    # The historical fp32 object held by core.fp32 is neither mutated nor replaced in memory.
    sys.modules['deberta_training']=core
    spec=importlib.util.spec_from_file_location('deberta_bf16_frozen_study',root/'scripts/deberta_study.py')
    cp=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cp;spec.loader.exec_module(cp)
    cp.ROOT=root;cp.STUDY=STUDY;cp.config=lambda root=root,evaluation=False:config(root,evaluation)
    original_save,original_seal,original_verify=cp.save,cp.seal,cp.verify
    def save(model,*args,**kwargs):
        previous=getattr(model,'_fp32_storage_reference',False)
        model._fp32_storage_reference=True
        try:return original_save(model,*args,**kwargs)
        finally:model._fp32_storage_reference=previous
    def seal(folder,stage):
        runtime=folder/'assets/runtime.json'
        if runtime.exists():
            value=read(runtime);value.update(amp=True,amp_dtype='bfloat16',grad_scaler=False,
                master_precision='float32',optimizer_precision='float32',storage_reference_precision='float32',
                task_inference_precision='bfloat16')
            write_json(runtime,value)
            reference=folder/'reference.json'
            if reference.exists():
                v=read(reference);v['compute_precision']='float32';write_json(reference,v)
        return original_seal(folder,stage)
    def verify(folder,output):
        runtime=read(folder/'assets/runtime.json')
        if (not runtime['amp'] or runtime['amp_dtype']!='bfloat16'
                or runtime['master_precision']!='float32'
                or read(folder/'reference.json')['compute_precision']!='float32'):
            raise ValueError('Storage/reference precision metadata differs')
        original_verify(folder,output)
        v=read(output);v.update(training_compute_precision='bfloat16',master_precision='float32',
                              reference_compute_precision='float32')
        write_json(output,v)
    cp.save,cp.seal,cp.verify=save,seal,verify
    return cp


def check_bundle(folder):
    from laya_checkpoint_delivery import within
    m=read(folder/'bundle.json')
    if m['format']!='deberta-bundle-v1' or m['identity']['study_id']!=STUDY:
        raise ValueError('Unexpected BF16 bundle study')
    if len({f['path'] for f in m['files']})!=len(m['files']):raise ValueError('Repeated file')
    for f in m['files']:
        p=within(folder,f['path'])
        if p.stat().st_size!=f['bytes'] or file_digest(p)!=f['sha256']:raise ValueError('Corrupt BF16 bundle')
    return m


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','pilot-resume','train','evaluate','verify'])
    p.add_argument('--epoch',type=int);p.add_argument('--deadline',type=float,default=float('inf'))
    p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();cp=install()
    if a.mode=='verify':cp.verify(a.bundle,a.output)
    elif a.mode=='pilot':cp.pilot(a.deadline)
    elif a.mode=='pilot-resume':cp.pilot_resume(a.deadline)
    elif a.mode=='train':cp.train(a.epoch,a.deadline)
    else:cp.evaluate(a.deadline)


if __name__=='__main__':main()
