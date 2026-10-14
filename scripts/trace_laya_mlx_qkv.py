"""Localize one preselected worst FP32 discrepancy using CPU FP32 layer outputs."""
import json
from pathlib import Path
import time
import numpy as np
import train_laya
from common import ROOT, file_digest, read_jsonl, write_json
from export_laya_posthoc import prepare_rules, input_digest
from diagnose_laya_mlx_precision import probability


def main():
    out = ROOT/'reports/laya-mlx-qkv-control-20261014-v1'
    out.mkdir(exist_ok=False)
    source = ROOT/'reports/laya-mlx-precision-diagnosis-20261014-v1/summary.json'
    diagnosis = json.loads(source.read_text())
    selected = max((r for r in diagnosis['records'] if r['dtype']=='float32' and r['batch']=='original_batch8'),
                   key=lambda r:r['max_probability_delta_cuda'])
    write_json(out/'protocol.json', {'selected_record': selected, 'diagnosis_sha256': file_digest(source),
        'source_sha256':file_digest(Path(__file__)), 'posthoc_diagnostic_only':True,
        'scope':'Worst MLX FP32 probability discrepancy, original batch; CPU reference and intermediate stages',
        'test_accessed':False, 'deployment_changed':False})
    train_laya.block_network()
    import torch
    import mlx.core as mx
    import laya.train as upstream
    from laya_mlx import Agent
    from laya_mlx.agent import collate_items
    from laya_mlx.model import attention_masks
    from laya_mlx_fusion_inference import MLXFusionPredictor
    torch.set_num_threads(4)
    spec=json.loads((ROOT/'configs/laya-fusion-inference-mlx-fp16-20261014-v1.json').read_text())
    fit=json.loads((ROOT/spec['frozen_fit']).read_text())
    checker=MLXFusionPredictor(ROOT,spec['checkpoint_root'],spec['models'],fit)
    model,tok,cfg=upstream.load_checkpoint(str(checker.folder(selected['model'])))
    model.float().eval()
    by_id={r['id']:r for r in read_jsonl(ROOT/'data/processed/validation.jsonl')}
    rows=[by_id[i] for i in selected['batch_sample_ids']]
    head,tail,_,_,_=prepare_rules(tok,rows)
    items=head if selected['effective_view']=='head' else tail
    prepared=[upstream.encode_item(tok,i,512,192,[0,1],parallel=False) for i in items]
    slot=selected['batch_sample_ids'].index(selected['sample_id'])
    assert input_digest(prepared[slot])==selected['input_sha256']
    valid=len(prepared[slot]['ids'])
    captured={};handles=[];embedding_input=[]
    def capture(name):
        def hook(module,args,output):
            value=output[0] if isinstance(output,tuple) else output
            captured[name]=value[slot,:valid].detach().cpu().numpy().copy()
            if name=='embeddings':embedding_input.append(value.detach().cpu().numpy().copy())
        return hook
    modules=[('embeddings',model.encoder.embeddings)]
    modules += [('encoder/'+str(i),layer) for i,layer in enumerate(model.encoder.layers)]
    modules += [('final_norm',model.encoder.final_norm)]
    first=model.encoder.layers[0]
    modules += [('op/qkv',first.attn.Wqkv),('op/attention',first.attn),
                ('op/mlp_norm',first.mlp_norm),('op/mlp_in',first.mlp.Wi),('op/mlp_out',first.mlp.Wo)]
    modules += [('head/'+str(i),layer) for i,layer in enumerate(model.head.layers)]
    for name,module in modules:handles.append(module.register_forward_hook(capture(name)))
    started=time.monotonic()
    with torch.inference_mode():
        cpu_logits=upstream._forward(model,upstream.collate_items([prepared],tok.pad_token_id),torch.device('cpu'),False,False).numpy()[slot]
    for handle in handles:handle.remove()
    print('CPU FP32 reference captured',flush=True)
    agent=Agent(str(checker.folder(selected['model'])),device='gpu',dtype='float32',compile=False,cache_prompts=False)
    batch={k:mx.array(v) for k,v in collate_items(prepared,tok.pad_token_id).items()}
    stages=[]
    weight_t=model.encoder.layers[0].attn.Wqkv.weight.detach().cpu().numpy()
    weight_m=np.asarray(agent.model.encoder.layers[0].attn.Wqkv.weight)
    assert np.array_equal(weight_t,weight_m), 'QKV weights differ'
    controlled={}
    for dev in (mx.cpu,mx.gpu):
        with mx.stream(dev):
            xx=mx.array(embedding_input[0]);ww=mx.array(weight_m)
            yy=xx@ww.T;mx.eval(yy)
            delta=np.abs(np.asarray(yy)[slot,:valid]-captured['op/qkv'])
            controlled[str(dev)]={'max_abs_delta':float(delta.max()),'rms_delta':float(np.sqrt(np.mean(delta**2)))}
    controlled['weights_exactly_equal']=True
    controlled['inputs']='Exact copied PyTorch FP32 embedding output; identical across devices'

    def compare(name,x):
        mx.eval(x)
        a=np.asarray(x)[slot,:valid];b=captured[name]
        delta=np.abs(a-b)
        stages.append({'stage':name,'max_abs_delta':float(delta.max()),'rms_delta':float(np.sqrt(np.mean(delta**2))),
                       'reference_rms':float(np.sqrt(np.mean(b**2)))})
    with mx.stream(mx.gpu):
        m=agent.model
        x=m.encoder.embeddings(batch['input_ids']);compare('embeddings',x)
        masks=attention_masks(batch['attention_mask'],m.encoder.config.local_attention)
        first=m.encoder.layers[0]
        compare('op/qkv',first.attn.Wqkv(x))
        attn=first.attn(x,masks[first.attention_type]);compare('op/attention',attn)
        normalized=first.mlp_norm(x+attn);compare('op/mlp_norm',normalized)
        pre=first.mlp.Wi(normalized);compare('op/mlp_in',pre)
        import mlx.nn as nn
        v,g=mx.split(pre,2,axis=-1)
        compare('op/mlp_out',first.mlp.Wo(nn.gelu(v)*g))
        for i,layer in enumerate(m.encoder.layers):
            x=layer(x,masks[layer.attention_type]);compare('encoder/'+str(i),x)
        x=m.encoder.final_norm(x);compare('final_norm',x)
        x=x+m.type_emb(batch['qtype'])[:,None,:]
        for i,layer in enumerate(m.head.layers):
            x=layer(x,batch['attention_mask'][:,None,None,:].astype(mx.bool_));compare('head/'+str(i),x)
        markers=x[mx.arange(x.shape[0])[:,None],batch['marker_pos']]
        logits=m.scorer(markers).squeeze(-1);mx.eval(logits)
        mlx_logits=np.asarray(logits)[slot].astype(float)
    ref=json.loads((ROOT/'reports/laya-fusion-inference-cuda-fp32-20261014-v1/validation-actual.json').read_text())
    idx=ref['sample_ids'].index(selected['sample_id'])
    cuda=ref['models'][selected['model']][selected['view']][idx]
    temperature=cfg['temperature'][0]
    result={'selected_record':selected,'stages':stages,'controlled_qkv':controlled,'cpu_logits':cpu_logits.tolist(),'mlx_logits':mlx_logits.tolist(),
      'cuda_logits':cuda['logits'], 'cpu_probability_delta_cuda':float(np.max(abs(probability(cpu_logits,temperature)-cuda['probabilities']))),
      'mlx_probability_delta_cuda':float(np.max(abs(probability(mlx_logits,temperature)-cuda['probabilities']))),
      'mlx_replay_logit_delta':float(np.max(abs(mlx_logits-selected['logits']))),
      'elapsed_seconds':time.monotonic()-started,'test_accessed':False,'posthoc_diagnostic_only':True}
    write_json(out/'summary.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
