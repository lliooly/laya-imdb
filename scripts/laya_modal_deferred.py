"""User-authorized deferred local delivery; cloud receipts never claim local backup."""
import argparse
import json
import sys
import time

from common import ROOT, file_digest
from laya_checkpoint_delivery import atomic_json, request_sha, verify_files

STUDY = 'laya-robust-loss-modal-deferred-20261015-v2'
VOLUME = 'end-imdb-laya-robust-deferred-20261015-v2'


def name(seed, arm, repeat=None):
    run = f'laya-robust-loss-modal-deferred-{arm}-s{seed}-strict-fp32-20261015-v2'
    return run if repeat is None else run+f'-admission-r{repeat}'


def contract_path():
    return ROOT/'configs'/f'{STUDY}.json'


def validate_cloud_receipt(req, receipt):
    if (receipt.get('schema') != 'laya-cloud-durability-v1'
            or receipt.get('state') != 'verified_committed_volume'
            or receipt.get('local_delivery_verified') is not False
            or receipt.get('modal_volume') != VOLUME
            or receipt.get('request_sha256') != request_sha(req)
            or receipt.get('identity') != req['identity'] or receipt.get('stage') != req['stage']
            or receipt.get('files') != req['files']
            or receipt.get('verification', {}).get('strict_load') is not True
            or not receipt.get('verification', {}).get('state_sha256')):
        raise ValueError('Missing or invalid cloud durability receipt; no local delivery claimed')


def production_receipt(req, timeout):
    deadline = time.monotonic()+timeout
    path = ROOT/'delivery/acks'/f'{request_sha(req)}.json'
    while True:
        if path.exists():
            receipt = json.loads(path.read_text())
            validate_cloud_receipt(req, receipt)
            return receipt
        if time.monotonic() >= deadline:
            raise TimeoutError('Cloud commit/independent reload did not complete')
        time.sleep(1)


def check_admission():
    records = {}
    for arm in ('ce','mixed'):
        for repeat in (1,2):
            run = name(42,arm,repeat); base = ROOT/'checkpoints'/run
            def request(stage):
                return json.loads((ROOT/'delivery/requests'/f'{run}.{stage}.json').read_text())
            frozen = request('frozen'); receipt = production_receipt(frozen,0)
            for stage in ('epoch-001','epoch-002'):
                req = request(stage);verify_files(ROOT,req);production_receipt(req,0)
            audit = json.loads((base/'epoch-002/training_audit.json').read_text())
            result = json.loads((base/'epoch-002/result.json').read_text())
            manifest = json.loads((base/'epoch-002/manifest.json').read_text())
            updates = [json.loads(x) for x in (base/'epoch-002/updates.jsonl').read_text().splitlines()]
            if ((audit['updates'],audit['gradient_exposures'],len(updates)) != (32,256,32)
                    or result['restored_epoch1'] is not (repeat==2)):
                raise ValueError('Incomplete real CUDA admission/recovery')
            records[(arm,repeat)] = {'initial':manifest['initial_state_sha256'],
                'input':audit['input_stream_sha256'],'updates':updates,
                'final':receipt['verification']['state_sha256']}
        if records[(arm,1)] != records[(arm,2)]:
            raise ValueError('Continuous and resumed CUDA admission differ: '+arm)
    a,b = records[('ce',1)],records[('mixed',1)]
    if a['initial'] != b['initial'] or a['input'] != b['input'] or a['final'] == b['final']:
        raise ValueError('Loss intervention changed initialization/input or had no effect')
    return {'state':'passed','contract_sha256':file_digest(contract_path()),
            'cuda_and_cloud_durability_passed':True,'local_delivery_verified':False,
            'records':{f'{a}-r{r}':v for (a,r),v in records.items()},'test_accessed':False}


def install():
    import laya_checkpoint_delivery as delivery
    import run_laya_robust_loss as runner
    import run_laya_robust_study as study
    for module in (runner,study):
        module.STUDY=STUDY;module.name=name;module.contract_path=contract_path
        module.production_receipt=production_receipt
    runner.validate_receipt=validate_cloud_receipt
    delivery.validate_receipt=validate_cloud_receipt
    study.check_admission=check_admission
    return runner,study


def prepare():
    parent = ROOT/'configs/laya-robust-loss-modal-20261015-v3.json'
    d=json.loads(parent.read_text())
    for rel,sha in d['source_sha256'].items():
        if file_digest(ROOT/rel)!=sha:raise ValueError('Parent source changed: '+rel)
    extra=[parent,ROOT/'scripts/laya_modal_deferred.py',ROOT/'scripts/modal_laya_deferred.py',
           ROOT/'tests/test_laya_modal_deferred.py']
    for p in extra:d['source_sha256'][str(p.relative_to(ROOT))]=file_digest(p)
    d.update(study_id=STUDY,modal_volume=VOLUME,parent_contract_sha256=file_digest(parent),
             delivery_mode='cloud_commit_and_independent_reload_then_deferred_local_download',
             local_delivery_required=False,local_delivery_verified=False,
             user_authorized_deferred_delivery=True,state='prepared_pending_cuda_admission')
    if contract_path().exists():
        if json.loads(contract_path().read_text())!=d:raise FileExistsError('Create a new protocol version')
    else:atomic_json(contract_path(),d)
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--verify-cloud')
    args,rest=p.parse_known_args()
    if args.prepare:
        print(json.dumps({'study':STUDY,'bound_files':len(prepare()['source_sha256'])}));return
    runner,_=install()
    if args.verify_cloud:
        from laya_checkpoint_delivery import verify_checkpoint
        req=json.loads(open(args.verify_cloud).read())
        print(json.dumps(verify_checkpoint(ROOT,req)));return
    sys.argv=[sys.argv[0],*rest]
    runner.main()


if __name__=='__main__':main()
