"""Existing-weight training-pool diagnostics; no gradient updates."""
import datetime,hashlib,json,pathlib,subprocess,time,tarfile
ROOT=pathlib.Path('/root/sj-tmp/End-IMDB'); OUT=ROOT/'artifacts/train-hardcase-mining-20261011-v1'; OUT.mkdir(exist_ok=False)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
state={'state':'running','started_utc':now(),'training_performed':False,'test_accessed':False,'models':{},'runner_sha256':sha(pathlib.Path(__file__))}
def status():(OUT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
def run(args,log,timeout):
 with (OUT/log).open('w') as f:
  p=subprocess.run([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 assert p.returncode==0,(args,p.returncode)
try:
 status();run(['-m','unittest','discover','-s','tests','-p','test_laya_train_candidates.py','-v'],'preflight.log',120)
 for model in ('baseline','combined'):
  run_id=f'laya-rtx4090-ce-{model}-s43-epochs2-fp32-v1';folder=ROOT/'reports'/run_id;checkpoint=ROOT/'checkpoints'/run_id
  expected=json.loads((folder/'cuda-saved-validation.json').read_text())['checkpoint_sha256'];weight=checkpoint/'model.safetensors'
  state['models'][model]={'state':'waiting_weight','expected_sha256':expected};status();deadline=time.monotonic()+1500
  while not weight.exists():
   assert time.monotonic()<deadline,'Weight wait timeout';time.sleep(2)
  assert sha(weight)==expected
  state['models'][model].update(state='exporting',started_utc=now());status()
  run(['scripts/export_laya_train_candidates.py','--checkpoint',str(checkpoint),'--manifest',str(folder/'manifest.json'),'--out',str(OUT/(model+'-gradient-predictions.json'))],model+'-export.log',900)
  result=json.loads((OUT/(model+'-gradient-predictions.json')).read_text());assert result['count']==22100 and result['state']=='completed'
  assert sha(weight)==expected;weight.unlink()
  state['models'][model].update(state='completed',finished_utc=now(),elapsed_inference_seconds=result['elapsed_seconds'],temporary_weight_removed=True);status()
 state.update(state='completed',finished_utc=now());status()
 bundle=ROOT/'artifacts/train-hardcase-mining-20261011-v1-results.tar.gz';assert not bundle.exists()
 with tarfile.open(bundle,'w:gz') as t:t.add(OUT,arcname=str(OUT.relative_to(ROOT)))
except BaseException as e:
 state.update(state='failed',finished_utc=now(),error=repr(e));status();raise
