"""Fixed-weight diagnostic; no optimization steps or official-test access."""
import datetime,hashlib,json,pathlib,subprocess,time
import numpy as np
ROOT=pathlib.Path('/root/sj-tmp/End-IMDB');OUT=ROOT/'artifacts/posthoc-seed42-20261011-v1';OUT.mkdir(exist_ok=False)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
state={'state':'running','started_utc':now(),'training_performed':False,'test_accessed':False,'hourly_price_yuan':1.24,'user_confirmed_replenished':True,'balance_amount_unknown':True,'runner_sha256':sha(pathlib.Path(__file__)),'models':{},'stages':[]}
def status():save(OUT/'status.json',state)
def run(name,args,timeout):
 status();stage={'name':name,'started_utc':now(),'command':args};state['stages'].append(stage);status()
 with (OUT/(name+'.log')).open('w') as f:
  p=subprocess.run([str(ROOT/'.venv/bin/python'),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 stage.update(returncode=p.returncode,finished_utc=now());status();assert p.returncode==0,(name,p.returncode)
try:
 run('preflight',['-m','unittest','discover','-s','tests','-p','test_laya_posthoc*.py','-v'],120)
 for model in ['baseline','combined']:
  run_id={'baseline':'laya-rtx4090-ce-epochs2-l512-s42-fp32-v1','combined':'laya-rtx4090-ce-epochs2-warmup005-ls005-l512-s42-fp32-v1'}[model];folder=ROOT/'reports'/run_id;checkpoint=ROOT/'checkpoints'/run_id;reference=json.loads((folder/'cuda-saved-validation.json').read_text());expected=reference['checkpoint_sha256'];weight=checkpoint/'model.safetensors'
  state['models'][model]={'state':'waiting_verified_weight','expected_weight_sha256':expected};status();deadline=time.monotonic()+1800
  while not weight.exists():
   assert time.monotonic()<deadline,'Weight restoration timeout';time.sleep(2)
  assert sha(weight)==expected
  export=ROOT/'reports'/('laya-posthoc-'+model+'-s42-fp32-v1')
  state['models'][model].update(state='exporting',export_path=str(export));status()
  run(model+'-export',['scripts/export_laya_posthoc.py','--checkpoint',str(checkpoint),'--manifest',str(folder/'manifest.json'),'--expected-weight-sha256',expected,'--out',str(export)],600)
  head=json.loads((export/'validation-head.json').read_text())['rows'];old=reference['validation_predictions']
  assert [(x['sample_id'],x['true_label'],x['input_sha256']) for x in head]==[(x['sample_id'],x['true_label'],x['input_sha256']) for x in old]
  delta=float(np.max(np.abs(np.asarray([x['logits'] for x in head])-np.asarray([x['logits'] for x in old]))));matches=all(x['pred_label']==y['pred_label'] for x,y in zip(head,old));parity={'all_ids_labels_input_hashes_match':True,'all_predictions_match':matches,'max_abs_logit_delta':delta,'tolerance':1e-4};save(export/'head-parity.json',parity);assert matches and delta<=1e-4,parity
  run(model+'-analyze',['scripts/laya_posthoc.py','--export-dir',str(export),'--manifest',str(folder/'manifest.json'),'--split-manifest',str(ROOT/'reports/split_manifest.json'),'--train-source',str(ROOT/'data/processed/train.jsonl'),'--validation-source',str(ROOT/'data/processed/validation.jsonl'),'--out',str(export/'analysis.json')],120)
  analysis=json.loads((export/'analysis.json').read_text());summary={'head_parity':parity,'fits':{k:v['fit'] for k,v in analysis['rules'].items()},'metrics':{k:v['validation_metrics'] for k,v in analysis['rules'].items()},'head_tail_vs_head':analysis['head_tail_vs_head']}
  rows={rule:json.loads((export/('validation-'+rule+'.json')).read_text())['rows'] for rule in ['head','head_tail']};labels=np.asarray([x['true_label'] for x in head]);truncated=np.asarray([x['truncated'] for x in head]);summary['groups']={}
  for rule in ['head','head_tail']:
   margin=np.asarray([x['logits'][1]-x['logits'][0] for x in rows[rule]]);fit=summary['fits'][rule]
   for cal,z in [('raw',margin),('temperature_bias',margin/fit['temperature']+fit['bias'])]:
    summary['groups'][rule+'_'+cal]={}
    for group,keep in [('truncated',truncated),('untruncated',~truncated)]:
     summary['groups'][rule+'_'+cal][group]={'count':int(keep.sum()),'errors':int(((z>0)!=labels)[keep].sum())}
  save(export/'summary.json',summary)
  assert sha(weight)==expected;weight.unlink()
  state['models'][model].update(state='completed',summary=summary,temporary_remote_weight_removed=True);status()
 state.update(state='completed',finished_utc=now());status()
except Exception as e:
 state.update(state='failed',finished_utc=now(),error=repr(e));status();raise
