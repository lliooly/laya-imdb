import sys,json,time
from pathlib import Path
sys.path.insert(0,'scripts')
import day2_runtime,train_laya
train_laya.block_network()
from common import ROOT,read_jsonl,file_digest
from comparison_metrics import metrics
import mlx.core as mx
import numpy as np
from laya_mlx import Agent
from laya_mlx.agent import collate_items
run='laya-mps-ce-epochs2-l512-s42-fp32-v1'
folder=ROOT/'reports'/run; out=folder/'mlx-validation/mlx-fp16-batch1.json';assert not out.exists()
rows=list(read_jsonl(ROOT/'data/processed/validation.jsonl'));ref=list(read_jsonl(folder/'saved-validation/raw.jsonl'))
assert [r['id'] for r in rows]==[r['sample_id'] for r in ref]
meta=json.loads((folder/'saved-validation/export.json').read_text());chk=ROOT/'checkpoints'/run
assert file_digest(chk/'model.safetensors')==meta['checkpoint_sha256']
mx.set_cache_limit(2*1024**3)
a=Agent(str(chk),dtype='float16',device='gpu',compile=False,cache_prompts=False)
T=meta['native_temperature'];samples=[];began=time.monotonic()
for i,(row,r) in enumerate(zip(rows,ref)):
 item=a.prepare(row['text'],{'sentiment':train_laya.QUESTION})[0][0]
 z=np.asarray(a.forward(collate_items([item],a.tok.pad_token_id))[0])[0].astype(float)
 assert np.isfinite(z).all()
 p=np.exp((z-z.max())/T);p/=p.sum()
 refz=np.array(r['logits']);rp=np.exp((refz-refz.max())/T);rp/=rp.sum()
 samples.append({'sample_id':row['id'],'true_label':row['label'],'logits':z.tolist(),'pred_label':int(z.argmax()),'mps_pred_label':int(refz.argmax()),'max_abs_probability_delta':float(np.max(np.abs(p-rp)))})
 if (i+1)%500==0: print('Best MLX batch1',i+1,'/2500',flush=True)
maxdelta=max(s['max_abs_probability_delta'] for s in samples);agreement=sum(s['pred_label']==s['mps_pred_label'] for s in samples)/len(samples)
report={'checkpoint_sha256':meta['checkpoint_sha256'],'validation_sha256':file_digest(ROOT/'data/processed/validation.jsonl'),'test_accessed':False,'batch_size':1,'dtype':'float16','scope':'MLX batch1 vs saved-checkpoint MPS FP32 batch1; prepared-input equality independently verified by batch8 worker','elapsed_seconds':time.monotonic()-began,'label_agreement':agreement,'max_abs_probability_delta':maxdelta,'probability_tolerance':0.01,'passed':agreement==1 and maxdelta<=0.01,'native_calibrated_metrics':metrics([s['true_label'] for s in samples],[s['logits'][1]-s['logits'][0] for s in samples],T),'source_sha256':file_digest(Path(__file__)),'samples':samples}
out.write_text(json.dumps(report,indent=2)+'\n');(out.parent/'batch1_analysis.py').write_text(Path(__file__).read_text())
print({k:report[k] for k in ['passed','label_agreement','max_abs_probability_delta']})
