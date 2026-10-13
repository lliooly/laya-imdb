"""Bounded reviewed 30-source exploratory A/B/C; frozen official test remains sealed."""
import argparse,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from common import ROOT,file_digest,write_json
from comparison_metrics import paired

def now():return datetime.now(timezone.utc).isoformat()
def main():
 p=argparse.ArgumentParser();p.add_argument('--configs',required=True);p.add_argument('--smoke-only',action='store_true');p.add_argument('--backup',default='artifacts/strict-seed43-20261013-backup/verified.json');a=p.parse_args()
 folder=ROOT/a.configs;contract=json.loads((folder/'execution-contract.json').read_text());out=ROOT/'artifacts'/(folder.name+('-admission' if a.smoke_only else '-execution'));out.mkdir(parents=True,exist_ok=False)
 state={'state':'running','started_utc':now(),'test_accessed':False,'exploratory_pool_size':30,'primary_metric':'final_saved_checkpoint_accuracy','hourly_price_yuan':1.24,'price_basis':'previous_user_quote','max_seconds':1800 if a.smoke_only else 10800,'maximum_cost_yuan':0.62 if a.smoke_only else 3.72,'stages':[],'runner_sha256':file_digest(Path(__file__))};deadline=time.monotonic()+state['max_seconds']
 def status():write_json(out/'status.json',state)
 def run(label,cmd,cap):
  e={'stage':label,'started_utc':now(),'command':cmd};state['stages'].append(e);status()
  seconds=min(cap,deadline-time.monotonic())
  if seconds<=0:raise TimeoutError('Runtime bound reached')
  with (out/(label+'.log')).open('w') as log:
   child=subprocess.Popen([sys.executable,*cmd],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);e['pid']=child.pid;status()
   try:code=child.wait(timeout=seconds)
   except BaseException:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise
  e.update(returncode=code,finished_utc=now());status()
  if code:raise RuntimeError(label+' failed')
 def audit(spec,n):
  r=ROOT/'reports'/spec['run_id'];s=json.loads((r/'semantic_audit.json').read_text());o=json.loads((r/'optimization_diagnostics.json').read_text());m=json.loads((r/'manifest.json').read_text())
  assert s['state']=='completed' and s['actual_train_count']==n and s['epochs_observed']==[1,2] and s['consumed_counts']==[n,n]
  assert s['expected_schedule_sha256']==s['actual_schedule_sha256']
  assert o['state']=='completed' and o['completed_optimizer_updates']==2*((n+7)//8)
  if n==22100:assert s['actual_schedule_sha256']==contract['expected_schedule_sha256'] and s['augmented_slots']==2210
  runtime=m['runtime'];assert runtime['profile']=='strict' and runtime['deterministic_algorithms'] and not runtime['warn_only'] and runtime['cublas_workspace_config']==':4096:8'
  epochs=json.loads((r/'epoch_checkpoints.json').read_text())['epochs'];assert [e['epoch'] for e in epochs]==[1,2]
  for e in epochs:assert file_digest(ROOT/'checkpoints'/spec['run_id']/('epoch-%03d'%e['epoch'])/'model.safetensors')==e['weights_sha256']
  return s,epochs
 def deduplicate(spec):
  known={};links=[]
  for f in sorted((ROOT/'checkpoints'/spec['run_id']).rglob('model.safetensors')):
   sha=file_digest(f)
   if sha in known and f.stat().st_ino!=known[sha].stat().st_ino:
    tmp=f.with_name('model.safetensors.hardlink');os.link(known[sha],tmp);os.replace(tmp,f);links.append(str(f.relative_to(ROOT)))
   else:known[sha]=f
  write_json(out/(spec['run_id']+'-dedup.json'),{'verified_links':links,'retained_all_checkpoint_paths':True})
 try:
  status();specs=[json.loads((folder/(arm+('-smoke' if a.smoke_only else '')+'-config.json')).read_text()) for arm in 'ABC']
  factors=[{k:v for k,v in s.items() if k not in ('run_id','semantic_pairs')} for s in specs];assert factors[0]==factors[1]==factors[2]
  bindings=[{k:v for k,v in s['semantic_pairs'].items() if k!='arm'} for s in specs];assert bindings[0]==bindings[1]==bindings[2]
  if not a.smoke_only:
   admission=json.loads((ROOT/'artifacts'/(folder.name+'-admission')/'status.json').read_text());assert admission['state']=='completed'
   backup=json.loads((ROOT/a.backup).read_text());assert backup['completed'] and len(backup['verified'])>0
   if __import__('shutil').disk_usage(ROOT).free<8*1024**3:raise ValueError('Require 8GiB free before full experiment')
  audits=[]
  for arm,s in zip('ABC',specs):
   path=folder/(arm+('-smoke' if a.smoke_only else '')+'-config.json')
   run(arm+'-train',['scripts/audit_cuda_training.py' if a.smoke_only else 'scripts/train_laya.py','--config',str(path)],450 if a.smoke_only else 3000)
   sem,epochs=audit(s,128 if a.smoke_only else 22100);audits.append(sem);deduplicate(s)
   report=ROOT/'reports'/s['run_id'];checkpoint=ROOT/'checkpoints'/s['run_id']
   if a.smoke_only:
    numerical=json.loads((report/'numerical_audit.json').read_text());assert numerical['passed'] and len(numerical['optimizer_steps'])==32
   else:
    curve=[]
    for e in epochs:
     dest=report/('cuda-saved-validation-epoch%d.json'%e['epoch']);run(arm+'-epoch%d-validation'%e['epoch'],['scripts/export_cuda_laya_predictions.py','--checkpoint',str(checkpoint/('epoch-%03d'%e['epoch'])),'--out',str(dest)],300);curve.append(dict(e,raw_metrics=json.loads(dest.read_text())['raw_metrics']))
    run(arm+'-final-validation',['scripts/export_cuda_laya_predictions.py','--checkpoint',str(checkpoint),'--out',str(report/'cuda-saved-validation.json')],300);write_json(report/'epoch_validation_curve.json',{'epochs':curve,'test_accessed':False})
  assert audits[0]['actual_gradient_ids']==audits[1]['actual_gradient_ids']==audits[2]['actual_gradient_ids']
  if not a.smoke_only:
   exports=[json.loads((ROOT/'reports'/s['run_id']/'cuda-saved-validation.json').read_text()) for s in specs];rows=[e['validation_predictions'] for e in exports];inputs=[[(r['sample_id'],r['true_label'],r['input_sha256']) for r in rs] for rs in rows];assert inputs[0]==inputs[1]==inputs[2] and len(inputs[0])==2500
   write_json(out/'comparison.json',{'metrics':{arm:e['raw_metrics'] for arm,e in zip('ABC',exports)},'paired':{str(i)+'-'+str(j):paired([r['true_label'] for r in rows[i]],[r['logits'][1]-r['logits'][0] for r in rows[i]],[r['logits'][1]-r['logits'][0] for r in rows[j]]) for i,j in [(0,1),(0,2),(1,2)]},'test_accessed':False,'exploratory_pool_size':30})
  state.update(state='completed',finished_utc=now());status()
 except BaseException as e:state.update(state='failed',error=repr(e),finished_utc=now());status();raise
if __name__=='__main__':main()
