import sys,json,csv,hashlib,itertools
from pathlib import Path
sys.path.insert(0,'scripts')
import numpy as np
from common import ROOT,read_jsonl,file_digest
from comparison_metrics import metrics,paired
from transformers import AutoTokenizer
import train_laya
train_laya.block_network()
out=ROOT/'reports/laya-mac-ce-comparison-v1'
out.mkdir(exist_ok=False)
runs=['laya-l512-s42-ce','laya-mps-ce-epochs2-l512-s42-fp32-v1','laya-mps-ce-epochs2-lr06-l512-s42-fp32-v1']
rows=list(read_jsonl(ROOT/'data/processed/validation.jsonl')); ids=[r['id'] for r in rows]; labels=np.array([r['label'] for r in rows])
split=json.loads((ROOT/'reports/split_manifest.json').read_text());assert file_digest(ROOT/'data/processed/validation.jsonl')==split['jsonl_sha256']['validation']
oldmanifest=json.loads((ROOT/'reports'/runs[0]/'manifest.json').read_text())
tok=AutoTokenizer.from_pretrained(str(ROOT/'.cache/laya/english/tokenizer'),local_files_only=True)
tok.backend_tokenizer.no_truncation();tok.backend_tokenizer.no_padding()
lengths=np.array([len(tok.encode(r['text'].replace(tok.mask_token,' '),add_special_tokens=False)) for r in rows])
exports={}; margins={}; result={}; inputs={}
for idx,run in enumerate(runs):
 folder=ROOT/'reports/model-comparison/laya_ce_full' if idx==0 else ROOT/'reports'/run/'saved-validation'
 ps=list(read_jsonl(folder/'raw.jsonl'));meta=json.loads((folder/'export.json').read_text())
 assert file_digest(folder/'raw.jsonl')==meta['predictions_sha256']
 assert [p['sample_id'] for p in ps]==ids and [p['true_label'] for p in ps]==labels.tolist()
 assert file_digest(ROOT/'checkpoints'/run/'model.safetensors')==meta['checkpoint_sha256']
 m=json.loads((ROOT/'reports'/run/'manifest.json').read_text())
 for k in ['gradient_train_ids','calibration_ids','question','model_lock','versions','upstream_train_sha256']:assert m[k]==oldmanifest[k],k
 assert m['validation']['ids']==oldmanifest['validation']['ids']
 status=json.loads((ROOT/'reports'/run/'status.json').read_text()); assert status['state']=='completed'
 margins[run]=np.array([p['logits'][1]-p['logits'][0] for p in ps])
 met=metrics(labels,margins[run],meta['native_temperature'])
 result[run]={'metrics':met,'training_seconds':status['elapsed_seconds'],'config':m['spec'], 'checkpoint_sha256':meta['checkpoint_sha256'],'strata':{n:{'count':int(keep.sum()),'errors':int(np.sum(((margins[run]>0)!=labels)&keep))} for n,keep in [('not_truncated',lengths<=466),('truncated',lengths>466)]},'source_changes':{k:{'historical':oldmanifest[k],'current':m[k]} for k in ['uv_lock_sha256','script_sha256'] if oldmanifest[k]!=m[k]}}
 if idx:
  mlx=json.loads((ROOT/'reports'/run/'mlx-validation/mlx-fp16-eval.json').read_text())
  mp=list(read_jsonl(ROOT/'reports'/run/'mlx-validation/mlx-fp16-predictions.jsonl'))
  assert mlx['checkpoint_sha256']==meta['checkpoint_sha256'] and [p['sample_id'] for p in mp]==ids
  probs=np.array([p['prob_positive'] for p in mp]);ref=1/(1+np.exp(-margins[run]/meta['native_temperature']))
  result[run]['mlx']={'metrics':mlx['native_calibrated_metrics'],'label_agreement':float(np.mean(np.array([p['pred_label'] for p in mp])==(margins[run]>0))),'inputs_match':mlx['identical_inputs'],'max_abs_probability_delta':float(np.max(np.abs(probs-ref))),'batch_size':8}
 for f in [folder/'raw.jsonl',folder/'export.json',ROOT/'reports'/run/'result.json',ROOT/'reports'/run/'manifest.json',ROOT/'reports'/run/'status.json']:inputs[str(f.relative_to(ROOT))]=file_digest(f)
comparisons={a+' vs '+b:paired(labels,margins[a],margins[b]) for a,b in itertools.combinations(runs,2)}
summary={'test_accessed':False,'validation_count':2500,'validation_sha256':split['jsonl_sha256']['validation'],'runs':result,'paired':comparisons,'target_98_allowed_errors':50,'best_run':runs[1],'remaining_net_corrections_to_98':2500-result[runs[1]]['metrics']['correct']-50,'input_sha256':inputs,'analysis_script_sha256':file_digest(Path(__file__)),'limitations':['Single seed, development validation used for selection','Epoch count also changes cosine scheduling horizon','Historical lock and adapter hashes changed; upstream and core versions match','Training report uses in-memory FP32; primary analysis uses saved FP16 weights reloaded FP32']}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'metrics.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['run','accuracy','errors','macro_f1','nll','ece','brier_multiclass','training_seconds'])
 for run,v in result.items():
  m=v['metrics'];w.writerow([run,m['accuracy'],2500-m['correct'],m['macro_f1'],m['nll'],m['ece_15_equal_width_top_label'],m['brier_multiclass'],v['training_seconds']])
with (out/'prediction_transitions.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['sample_id','true_label','review_tokens','historical_pred','epochs2_pred','lr06_pred'])
 for i,r in enumerate(rows): w.writerow([r['id'],r['label'],lengths[i],*[int(margins[run][i]>0) for run in runs]])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
names=['MPS: 1 epoch','MPS: 2 epochs','MPS: 2 epochs, LR x0.6']
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,key,title,scale in [(axs[0],'accuracy','Validation accuracy (%)',100),(axs[1],'nll','Calibrated NLL',1)]:
 vals=[result[r]['metrics'][key]*scale for r in runs];ax.bar(range(3),vals,color=['#7c93b8','#2c9c76','#ca8b61']);ax.set_xticks(range(3),names,rotation=15,ha='right');ax.set_title(title);ax.spines[['top','right']].set_visible(False)
 if key=='accuracy':ax.set_ylim(94,99);ax.axhline(98,color='gray',ls='--',lw=1,label='Target: 98%');ax.legend(frameon=False)
 for i,v in enumerate(vals):ax.text(i,v+(0.08 if key=='accuracy' else 0.001),f'{v:.2f}' if key=='accuracy' else f'{v:.4f}',ha='center')
fig.suptitle('Laya-CE: frozen development validation (n=2500, seed=42)')
fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
(out/'analysis.py').write_text(Path(__file__).read_text())
checks={str(p.relative_to(out)):file_digest(p) for p in out.iterdir() if p.is_file()};(out/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'paired':comparisons,'strata':{r:v['strata'] for r,v in result.items()},'mlx':{r:v.get('mlx') for r,v in result.items()}},indent=2))
