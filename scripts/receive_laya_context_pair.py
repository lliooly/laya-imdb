"""CPU-only background delivery; all Modal operations use one native async loop."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import sys
import time

import aiohttp
import certifi
from modal.volume import _Volume
from modal_proto import api_pb2

from common import file_digest
from laya_checkpoint_delivery import atomic_json, within, fsync_dir
from laya_context_pair import STUDY, VOLUME

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 8*1024*1024
MIN_FREE = 50*1024**3


async def download(v, session, sem, relative, record):
    dest=within(ROOT,relative)
    if dest.exists():
        if dest.stat().st_size!=record['bytes'] or file_digest(dest)!=record['sha256']:
            raise ValueError('Existing local bytes differ: '+relative)
        return
    if shutil.disk_usage(ROOT).free<MIN_FREE+record['bytes']:
        raise OSError('Insufficient disk with 50 GiB reserve')
    dest.parent.mkdir(parents=True,exist_ok=True)
    part=dest.with_name(dest.name+'.download.partial')
    sidecar=part.with_name(part.name+'.blocks.json')
    response=await asyncio.wait_for(v._client.stub.VolumeGetFile2(
        api_pb2.VolumeGetFile2Request(volume_id=v.object_id,path='/End-IMDB/'+relative)),90)
    if len(response.get_urls)!=(record['bytes']+BLOCK-1)//BLOCK:
        raise ValueError('Unexpected download block count')
    fd=os.open(part,os.O_CREAT|os.O_RDWR,0o600)
    state={'sha256':record['sha256'],'completed':{}}
    try:
        if sidecar.exists():
            state=json.loads(sidecar.read_text())
            if state['sha256']!=record['sha256']:raise ValueError('Partial file identity differs')
        for key,sha in list(state['completed'].items()):
            i=int(key);n=min(BLOCK,record['bytes']-i*BLOCK)
            if i<0 or n<=0 or hashlib.sha256(os.pread(fd,n,i*BLOCK)).hexdigest()!=sha:
                del state['completed'][key]
        atomic_json(sidecar,state)
        per_file=asyncio.Semaphore(8)
        async def one(i):
            async with per_file,sem:
                for attempt in range(5):
                    try:
                        async with session.get(response.get_urls[i]) as r:
                            r.raise_for_status();data=await r.read()
                        if len(data)!=min(BLOCK,record['bytes']-i*BLOCK):raise ValueError('Block size differs')
                        done=0
                        while done<len(data):done+=os.pwrite(fd,data[done:],i*BLOCK+done)
                        os.fsync(fd)
                        state['completed'][str(i)]=hashlib.sha256(data).hexdigest();atomic_json(sidecar,state)
                        if record['bytes']>1024**3 and len(state['completed'])%40==0:
                            print('BLOCKS',relative,len(state['completed']),'/',len(response.get_urls),flush=True)
                        return
                    except Exception:
                        if attempt==4:raise RuntimeError('Bounded block retries exhausted: '+relative) from None
                        await asyncio.sleep(min(8,2**attempt))
        await asyncio.gather(*(one(i) for i in range(len(response.get_urls)) if str(i) not in state['completed']))
        os.ftruncate(fd,record['bytes']);os.fsync(fd)
    finally:os.close(fd)
    if part.stat().st_size!=record['bytes'] or file_digest(part)!=record['sha256']:
        raise ValueError('Whole-file SHA mismatch: '+relative)
    os.replace(part,dest);fsync_dir(dest.parent)
    sidecar.unlink(missing_ok=True)


async def cloud_json(v,relative,missing=False):
    async def get():
        data=b''.join([chunk async for chunk in v.read_file('/End-IMDB/'+relative)])
        return json.loads(data)
    try:return await asyncio.wait_for(get(),60)
    except FileNotFoundError:
        if missing:return None
        raise


async def receive(v,session,sem,r,key,result):
    manifest=r['bundle_manifest'];relative=manifest['bundle']
    # Compare the manifest bytes to the independent cloud receipt before trusting paths.
    if not relative.startswith(('checkpoints/'+STUDY+'/', 'reports/'+STUDY+'/')):
        raise ValueError('Delivery path escapes study')
    async def get_manifest():
        return b''.join([chunk async for chunk in v.read_file('/End-IMDB/'+relative+'/bundle.json')])
    raw=await asyncio.wait_for(get_manifest(),60)
    if hashlib.sha256(raw).hexdigest()!=r['bundle_sha256'] or json.loads(raw)!=manifest:
        raise ValueError('Cloud bundle manifest differs')
    for start in range(0,len(manifest['files']),4):
        await asyncio.gather(*(download(v,session,sem,relative+'/'+f['path'],f) for f in manifest['files'][start:start+4]))
    folder=within(ROOT,relative)
    # Preserve the exact cloud serialization, which differs from atomic_json's format.
    target=folder/'bundle.json'
    temporary=target.with_name('bundle.json.partial')
    with temporary.open('wb') as out:
        out.write(raw);out.flush();os.fsync(out.fileno())
    os.replace(temporary,target);fsync_dir(folder)
    if file_digest(folder/'bundle.json')!=r['bundle_sha256']:
        raise ValueError('Manifest serialization changed')
    checked=ROOT/'reports'/STUDY/'local-verification'/(key+'.json')
    if manifest['stage']=='evaluated':
        # Evaluation bundles contain predictions and reports, not another model copy.
        from laya_context_pair import check_bundle
        check_bundle(folder)
        verification={'all_file_sha256_verified':True,'test_accessed':False}
    else:
        process=await asyncio.create_subprocess_exec(str(ROOT/'.venv/bin/python'),
            str(ROOT/'scripts/laya_context_pair.py'),'verify','--bundle',str(folder),'--output',str(checked),
            cwd=ROOT)
        try:code=await asyncio.wait_for(process.wait(),300)
        except BaseException:
            process.kill();await process.wait();raise
        if code:raise RuntimeError('Independent local CPU verifier failed')
        verification=json.loads(checked.read_text())
    result['bundles'][key]={'state':'verified_local','local_delivery_verified':True,
        'bundle':relative,'bundle_sha256':r['bundle_sha256'],'verification':verification,'verified_unix':time.time()}
    atomic_json(ROOT/'reports'/STUDY/'local-delivery.json',result)
    print('LOCAL_VERIFIED',key,flush=True)


async def main():
    report=ROOT/'reports'/STUDY
    submission=json.loads((report/'submission.json').read_text())
    result={'study_id':STUDY,'state':'receiving','bundles':{},'local_delivery_verified':False,'test_accessed':False}
    path=report/'local-delivery.json'
    if path.exists():result=json.loads(path.read_text())
    atomic_json(path,result)
    v=_Volume.from_name(VOLUME);await v.hydrate()
    sem=asyncio.Semaphore(16)
    connector=aiohttp.TCPConnector(ssl=ssl.create_default_context(cafile=certifi.where()),limit=16)
    expected={f'context-{c}-s42-epoch-{e:03d}':f'cloud-delivery/context-{c}-s42-epoch-{e:03d}.json' for c in (512,2048) for e in (1,2)}
    expected['evaluation']='evaluation-cloud-receipt.json'
    async with aiohttp.ClientSession(connector=connector,timeout=aiohttp.ClientTimeout(total=120,connect=20,sock_read=30)) as session:
        # Extra time permits delayed downloads after all paid GPU work has stopped.
        while time.time()<submission['absolute_deadline_unix']+86400:
            state=await cloud_json(v,f'reports/{STUDY}/state.json',missing=True)
            if state:atomic_json(report/'cloud-state.json',state)
            for key,rel in expected.items():
                if key in result['bundles']:continue
                r=await cloud_json(v,f'reports/{STUDY}/'+rel,missing=True)
                if r:
                    print('RECEIVING',key,flush=True)
                    await receive(v,session,sem,r,key,result)
                    atomic_json(report/'cloud-receipts'/(key+'.json'),r)
            if len(result['bundles'])==len(expected):
                result.update(state='verified_local',local_delivery_verified=True,finished_unix=time.time())
                atomic_json(path,result)
                budget=ROOT/'delivery'/f'{STUDY}-budget.json'
                record=json.loads(budget.read_text());record['state']='completed_verified_local'
                atomic_json(budget,record)
                print('PAIR_AND_EVALUATION_VERIFIED_LOCAL',flush=True)
                return
            if state and state['state'].startswith('stopped'):
                result.update(state=state['state'],error=state.get('error'),local_delivery_verified=False)
                atomic_json(path,result)
                print('CLOUD_STOPPED',state['state'],flush=True);return
            await asyncio.sleep(30)
    raise TimeoutError('Local receiver deadline exceeded; cloud copies remain persistent')


if __name__=='__main__':asyncio.run(main())
