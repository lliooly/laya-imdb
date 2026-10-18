"""Independent NumPy/SciPy forward and metric checks for token query heads."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf, expit, softmax
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from common import file_digest, write_json
from run_final_cpu_alternatives_v2 import numpy_head_state
from run_token_query_study import ROOT, OUT, ART, KINDS, check_frozen, guard


def load(name):
    return json.loads((OUT/name).read_text())


def forward(h, markers, state, kind):
    def linear(x, name):
        return x@state[name+'.weight'].T+state[name+'.bias']
    def norm(x, name):
        centered=x-x.mean(-1,keepdims=True)
        return centered/np.sqrt((centered**2).mean(-1,keepdims=True)+1e-5)*state[name+'.weight']+state[name+'.bias']
    h=norm(h.astype(np.float64),'input_norm')
    v=linear(h,'tokens.0')
    v=.5*v*(1+erf(v/np.sqrt(2.)))
    q=linear(h[markers],'marker_projection')+state['queries']
    if kind=='mean':
        weights=np.full((2,len(h)),1/len(h))
    else:
        scores=q@v.T/np.sqrt(q.shape[-1])
        if kind=='top32' and len(h)>32:
            keep=np.argsort(scores,axis=-1)[:,-32:]
            allowed=np.zeros(scores.shape,dtype=bool)
            np.put_along_axis(allowed,keep,True,axis=-1)
            scores[~allowed]=-np.inf
        weights=softmax(scores,axis=-1)
    return linear(norm(q+weights@v,'output_norm'),'scorer').ravel()


def main():
    guard('evaluate')
    frozen, protocol=check_frozen()
    cal=load('heads-frozen.json')
    dev=load('development-heads.json')
    probs=load('development-probabilities.json')
    summary=load('summary.json')
    cy,y=np.asarray(cal['calibration_labels']),np.asarray(dev['true_labels'])
    assert len(cy)==400 and len(y)==2500 and len(set(dev['sample_ids']))==2500
    assert not set(cal['calibration_ids'])&set(dev['sample_ids'])
    for split,sha in frozen['token_index_sha256'].items():
        index_path=ART/(split+'-index.json')
        assert file_digest(index_path)==sha
        cache_index=json.loads(index_path.read_text())
        assert file_digest(ART/(split+'-tokens.npy'))==cache_index['cache_sha256']
    index=json.loads((ART/'calibration-index.json').read_text())
    values=np.load(ART/'calibration-tokens.npy',mmap_mode='r',allow_pickle=False)
    forward_errors={}
    for kind in KINDS:
        for seed in protocol['config']['seeds']:
            name=f'{kind}-s{seed}'
            record=frozen['records'][name]
            assert len(record['history'])==5
            assert sum(e['exposures'] for e in record['history'])==110500
            assert sum(e['updates'] for e in record['history'])==3455
            assert all(np.isfinite(e['mean_loss']) for e in record['history']) and record['reload_exact']
            state=numpy_head_state(ART/(name+'.safetensors'))
            z=np.asarray(cal['calibration'][name]['calibration_logits'])
            errors=[]
            for i in range(8):
                a,b=index['offsets'][i:i+2]
                independent=forward(values[a:b],np.asarray(index['markers'][i]),state,kind)
                errors.append(float(np.max(np.abs(independent-z[i]))))
                assert np.array_equal(independent.argmax(),z[i].argmax())
            assert max(errors)<.001, (name,errors)
            forward_errors[name]=max(errors)
            margin=z[:,1]-z[:,0]
            t=cal['calibration'][name]['temperature']
            objective=lambda inverse:float(np.mean(np.logaddexp(0,margin*inverse)-cy*margin*inverse))
            result=minimize_scalar(objective,bounds=(.2,2.),method='bounded',options=dict(xatol=1e-12))
            best=min([.2,result.x,2.],key=objective)
            assert abs(objective(best)-objective(1/t))<1e-10
    for seed in protocol['config']['seeds']:
        seed_records=[frozen['records'][f'{k}-s{seed}']['initial_state_sha256'] for k in KINDS]
        assert len(set(seed_records))==1
    baseline=np.asarray(probs['baseline'])
    assert accuracy_score(y,baseline>.5)==summary['baseline']['accuracy']==.9728
    pair_p={}
    for kind in KINDS:
        cp=np.asarray(cal['calibration_branch_probabilities'][kind])
        reconstructed_cal=[]
        reconstructed_dev=[]
        for seed in protocol['config']['seeds']:
            name=f'{kind}-s{seed}'
            cz=np.asarray(cal['calibration'][name]['calibration_logits'])
            dz=np.asarray(dev['logits'][name])
            temp=cal['calibration'][name]['temperature']
            reconstructed_cal.append(expit((cz[:,1]-cz[:,0])/temp))
            reconstructed_dev.append(expit((dz[:,1]-dz[:,0])/temp))
        np.testing.assert_allclose(cp,np.mean(reconstructed_cal,axis=0),atol=1e-15,rtol=0)
        np.testing.assert_allclose(probs['predictions'][kind]['branch'],np.mean(reconstructed_dev,axis=0),atol=1e-15,rtol=0)
        ca=np.asarray(cal['baseline'])
        w=cal['mixtures'][kind]['sparse_weight']
        objective=lambda weight:log_loss(cy,ca+weight*(cp-ca),labels=[0,1])
        result=minimize_scalar(objective,bounds=(0,.25),method='bounded',options=dict(xatol=1e-12))
        best=min([0,result.x,.25],key=objective)
        assert abs(best-w)<1e-6 and abs(objective(best)-objective(w))<1e-11
        for entry in ('branch','fusion'):
            pp=np.asarray(probs['predictions'][kind][entry])
            saved=summary['results'][kind][entry]
            assert accuracy_score(y,pp>.5)==saved['accuracy']
            assert abs(log_loss(y,pp,labels=[0,1])-saved['nll'])<1e-10
            assert abs(roc_auc_score(y,pp)-saved['auroc'])<1e-12
        pp=np.asarray(probs['predictions'][kind]['fusion'])
        branch=np.asarray(probs['predictions'][kind]['branch'])
        np.testing.assert_allclose(pp,(1-w)*baseline+w*branch,atol=1e-15,rtol=0)
        a,b=(baseline>.5)==y,(pp>.5)==y
        repaired,damaged=int((~a&b).sum()),int((a&~b).sum())
        pair=summary['results'][kind]['fusion_vs_baseline']
        assert (repaired,damaged,repaired-damaged)==(pair['repaired'],pair['damaged'],pair['net'])
        p=binomtest(min(repaired,damaged),repaired+damaged).pvalue if repaired+damaged else 1.
        assert p==pair['mcnemar_exact_two_sided_p']
        pair_p[kind]=p
    previous=0.
    for rank,kind in enumerate(sorted(pair_p,key=pair_p.get)):
        previous=max(previous,min(1.,(3-rank)*pair_p[kind]))
        assert summary['results'][kind]['fusion_holm_p']==previous
    for stage in ('extract','train','develop','evaluate'):
        audit=load(stage+'-access-audit.json')
        assert not audit['test_accessed']
        if stage in ('extract','train'):
            assert not any(Path(p).name.startswith('validation') or Path(p).name=='predictions.json'
                           or Path(p).name.endswith('-development.json') for p in audit['read_paths'])
    dependencies=load('dependency-source-audit.json')
    packages=ROOT/'.venv/lib/python3.12/site-packages'
    assert file_digest(ROOT/'uv.lock')==dependencies['uv_lock_sha256']
    for path,sha in dependencies['installed_source_sha256'].items():
        assert file_digest(packages/path)==sha
    output=dict(passed=True,numpy_float64_forward_max_error=forward_errors,
        independent_temperature_and_mixture_optimizers=True,sklearn_metrics=True,exact_tests_and_holm=True,
        training_exposure_and_same_initialization_verified=True,access_audit_verified=True,
        installed_dependencies_unchanged=True,ensemble_reconstruction_verified=True,
        verifier_sha256=file_digest(Path(__file__)),
        frozen_sha256=file_digest(OUT/'heads-frozen.json'),summary_sha256=file_digest(OUT/'summary.json'))
    write_json(OUT/'independent-verification.json',output)
    print(json.dumps(output))


if __name__=='__main__':
    main()
