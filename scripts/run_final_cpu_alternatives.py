"""Bounded four-candidate CPU study; frozen feature heads and held-out fusion."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from common import ROOT, file_digest, write_json
from comparison_metrics import fit_temperature, metrics, sigmoid

RUN = 'final-cpu-alternatives-20261018-v1'
OUT = ROOT/'reports'/RUN
ART = ROOT/'artifacts'/RUN
CONFIG = 'configs/'+RUN+'.json'
PROBE = 'laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1'
PREVIOUS = 'reports/laya-deberta-calibration-weight-cpu-20261018-v1'
CONTEXT = 'reports/laya-context-pair-s42-modal-20261016-v1/evaluation'
NEURAL = 'reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation'
MANIFEST = 'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json'
READS = set()


def read(path):
    return json.loads((ROOT/path).read_text())


def save(name, data):
    write_json(OUT/name, data)


def emit(stage, **kwargs):
    print(json.dumps(dict(stage=stage, **kwargs), ensure_ascii=False), flush=True)


def install_guard(stage):
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        mode, flags = args[1], args[2]
        reading = ('r' in mode or '+' in mode) if isinstance(mode, str) else not (flags & os.O_WRONLY)
        if not reading:
            return
        name = str(path)
        if any(x in name for x in ('/data/raw/', '/data/processed/test', '/laya-frozen-official-test-', '/deberta-laya-final-cpu-evaluation-')):
            raise PermissionError('Official-test file prohibited: '+name)
        if stage in ('heads', 'fit') and (path.name.startswith('validation') or path.name.endswith('-development.json')
            or path.name in ('predictions.json', 'development.json', 'validation.jsonl', 'development-heads.json')):
            raise PermissionError('Development file prohibited until all candidates frozen: '+name)
        if name.startswith(str(ROOT)+'/') and '/.venv/' not in name:
            READS.add(str(path.relative_to(ROOT)))
    sys.addaudithook(guard)


def met(y, p):
    q = np.clip(p, 1e-15, 1-1e-15)
    return metrics(y, np.log(q)-np.log1p(-q))


def feature_block(split, config):
    path = ROOT/'artifacts'/PROBE/(split+'.npz')
    index_path = path.with_name(split+'-index.json')
    if split == 'validation':
        expected, index_expected = config['feature_validation_cache_expected_sha256'], config['feature_validation_index_expected_sha256']
    else:
        meta = read(f'reports/{PROBE}/{split}-extraction.json')
        expected, index_expected = meta['cache_sha256'], meta['index_sha256']
        assert meta['cache'] == str(path.relative_to(ROOT)) and not meta['test_accessed']
    assert file_digest(path) == expected and file_digest(index_path) == index_expected
    index = json.loads(index_path.read_text())
    assert index['split'] == split and not index['test_accessed']
    ids = [r['sample_id'] for r in index['rows']]
    with np.load(path, allow_pickle=False) as z:
        x, y = z['encoder'], z['labels']
        extra = dict(truncated=z['truncated'], full_logits=z['full_logits'])
    assert x.shape == (len(ids), 1024) and np.isfinite(x).all()
    assert len(set(ids)) == len(ids) and np.array_equal(y, [r['true_label'] for r in index['rows']])
    manifest = read(MANIFEST)
    expected_ids = manifest['gradient_train_ids' if split == 'gradient_train' else 'calibration_ids'] if split != 'validation' else manifest['validation']['ids']
    assert ids == expected_ids
    return x, y, ids, extra


def make_head(torch, config):
    from torch import nn
    h = config['heads']
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Linear(1024, h['hidden']), nn.GELU(), nn.LayerNorm(h['hidden']))
            self.dropout = nn.Dropout(h['dropout'])
            self.scorer = nn.Linear(h['hidden'], 1)
        def forward(self, x):
            return self.scorer(self.dropout(self.features(x))).squeeze(-1)
    return Head().float().cpu()


def torch_logits(torch, model, values):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(values[i:i+512])).numpy() for i in range(0, len(values), 512)])


def heads():
    import torch
    from torch.nn import functional as F
    from safetensors.torch import load_file, save_file
    if OUT.exists() or ART.exists():
        raise FileExistsError('Preserve completed or partial run')
    c = read(CONFIG)
    h = c['heads']
    torch.set_num_threads(h['torch_cpu_threads'])
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True)
    ART.mkdir(parents=True)
    train, y, train_ids, _ = feature_block('gradient_train', c)
    cal, cy, cal_ids, _ = feature_block('calibration', c)
    assert not set(train_ids) & set(cal_ids)
    mean = train.mean(axis=0, dtype=np.float64)
    scale = train.std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.
    np.savez(ART/'normalization.npz', mean=mean, scale=scale)
    train = np.asarray((train-mean)/scale, dtype=np.float32)
    cal = np.asarray((cal-mean)/scale, dtype=np.float32)
    tx, ty = torch.from_numpy(train), torch.tensor(y, dtype=torch.float32)
    outputs, records = {}, {}
    initialization = None
    for samples in h['dropout_samples']:
        name = f'mlp_dropout{samples}'
        started = time.perf_counter()
        torch.manual_seed(h['initialization_seed'])
        model = make_head(torch, c)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if initialization is None:
            initialization = state
        else:
            assert all(torch.equal(v, initialization[k]) for k, v in state.items())
        optimizer = torch.optim.AdamW(model.parameters(), lr=h['learning_rate'], weight_decay=h['weight_decay'])
        rng = np.random.default_rng(h['order_seed'])
        history = []
        for epoch in range(h['epochs']):
            model.train()
            order = rng.permutation(len(y))
            summed = 0.
            updates = 0
            for start in range(0, len(y), h['batch_size']):
                ii = order[start:start+h['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                hidden = model.features(tx[ii])
                loss = torch.stack([F.binary_cross_entropy_with_logits(model.scorer(model.dropout(hidden)).squeeze(-1), ty[ii]) for _ in range(samples)]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite readout loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite readout gradient')
                optimizer.step()
                summed += float(loss.detach())*len(ii)
                updates += 1
            history.append(dict(epoch=epoch+1, mean_loss=summed/len(y), updates=updates, exposures=len(y)))
            emit('head_epoch', candidate=name, **history[-1])
        model_path = ART/(name+'.safetensors')
        save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, model_path)
        z = torch_logits(torch, model, cal)
        restored = make_head(torch, c)
        restored.load_state_dict(load_file(model_path), strict=True)
        np.testing.assert_array_equal(z, torch_logits(torch, restored, cal))
        outputs[name] = z.tolist()
        records[name] = dict(model_path=str(model_path.relative_to(ROOT)), model_sha256=file_digest(model_path),
            dropout_samples=samples, history=history, seconds=time.perf_counter()-started,
            parameter_count=sum(p.numel() for p in model.parameters()), calibration_reload_exact=True)
    source_paths = [CONFIG, MANIFEST, 'uv.lock', 'scripts/run_final_cpu_alternatives.py',
        f'artifacts/{PROBE}/gradient_train.npz', f'artifacts/{PROBE}/gradient_train-index.json',
        f'artifacts/{PROBE}/calibration.npz', f'artifacts/{PROBE}/calibration-index.json']
    save('heads-frozen.json', dict(config_sha256=file_digest(ROOT/CONFIG),
        sources_sha256={p: file_digest(ROOT/p) for p in source_paths}, records=records,
        normalization_sha256=file_digest(ART/'normalization.npz'), cal_ids=cal_ids, cal_labels=cy.tolist(),
        calibration_margins=outputs, initialization_equal=True, order_seed_shared=True,
        versions=dict(torch=torch.__version__, numpy=np.__version__),
        frozen_utc=datetime.now(timezone.utc).isoformat(), backbone_updated=False))
    save('heads-access-audit.json', dict(read_paths=sorted(READS), development_opened=False, test_opened=False))
    emit('heads_completed', candidates=list(records))


def cal_neural():
    previous = read(PREVIOUS+'/frozen.json')
    receipt = read(PREVIOUS+'/fit-access-audit.json')
    assert file_digest(ROOT/(PREVIOUS+'/frozen.json')) == receipt['frozen_sha256']
    path = PREVIOUS+'/calibration-predictions.json'
    assert file_digest(ROOT/path) == previous['calibration_predictions_sha256']
    data = read(path)
    return np.asarray(data['p_laya']), np.asarray(data['p_deberta']), np.asarray(data['true_labels']), data['sample_ids']


def bounded_minimum(derivative, upper):
    lo, hi = 0., upper
    if derivative(lo) >= 0:
        return lo
    if derivative(hi) <= 0:
        return hi
    for _ in range(80):
        mid = (lo+hi)/2
        if derivative(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2


def logit_features(a, b, clip=12.):
    def convert(p):
        p = np.clip(p, 1e-12, 1-1e-12)
        return np.clip(np.log(p)-np.log1p(-p), -clip, clip)
    return np.column_stack([convert(a), convert(b), np.ones(len(a))])


def fit():
    from scipy.optimize import minimize
    from bow_nb_branch import fit_sparse_weight
    c, hf = read(CONFIG), read(str((OUT/'heads-frozen.json').relative_to(ROOT)))
    if (OUT/'frozen.json').exists():
        raise FileExistsError('Fit already frozen')
    for p, sha in hf['sources_sha256'].items():
        assert file_digest(ROOT/p) == sha
    a, b, y, ids = cal_neural()
    assert ids == hf['cal_ids'] and np.array_equal(y, hf['cal_labels'])
    neural = (a+b)/2
    x = logit_features(a, b)
    prior = np.array(c['logit_model']['prior'])
    penalty = c['logit_model']['l2']
    def objective(w):
        z = x@w
        residual = sigmoid(z)-y
        return float(np.mean(np.logaddexp(0, z)-y*z)+penalty*np.sum((w-prior)**2)), x.T@residual/len(y)+2*penalty*(w-prior)
    result = minimize(objective, prior, jac=True, method='L-BFGS-B', bounds=[(0, 2), (0, 2), (-1, 1)],
        options=dict(ftol=1e-13, gtol=1e-9, maxiter=1000))
    if not result.success:
        raise RuntimeError(str(result.message))
    fits = dict(regularized_logit=dict(coefficients=result.x.tolist(), objective=float(result.fun), optimizer_converged=True))
    probabilities = dict(regularized_logit=sigmoid(x@result.x))
    bundle = read(CONTEXT+'/bundle.json')
    expected = {v['path']: v['sha256'] for v in bundle['files']}
    assert file_digest(ROOT/(CONTEXT+'/2048-calibration.json')) == expected['2048-calibration.json']
    context = read(CONTEXT+'/2048-calibration.json')
    assert context['sample_ids'] == ids and np.array_equal(context['true_labels'], y) and not context['test_accessed']
    z = np.asarray(context['logits'])
    temperature = fit_temperature(z[:, 1]-z[:, 0], y, bounds=(.5, 5.))
    context_p = sigmoid((z[:, 1]-z[:, 0])/temperature)
    mask = read(f'reports/laya-posthoc-baseline-s43-fp32-v1/calibration-head_tail.json')
    assert [r['sample_id'] for r in mask['rows']] == ids and np.array_equal([r['true_label'] for r in mask['rows']], y)
    long = np.asarray([r['truncated'] for r in mask['rows']], dtype=bool)
    assert long.sum() == 66
    difference = long*(context_p-neural)
    def derivative(w):
        p = np.clip(neural+w*difference, 1e-12, 1-1e-12)
        return float(np.mean((p-y)*difference/(p*(1-p)))+2*c['long_context']['l2_to_zero']*w)
    weight = bounded_minimum(derivative, .5)
    fits['long_context'] = dict(weight=weight, temperature=temperature, calibration_long_count=int(long.sum()))
    probabilities['long_context'] = neural+weight*difference
    for name in ('mlp_dropout1', 'mlp_dropout4'):
        margin = np.asarray(hf['calibration_margins'][name])
        temp = fit_temperature(margin, y, bounds=(.5, 5.))
        p = sigmoid(margin/temp)
        blend_fit = fit_sparse_weight(neural, p, y)
        fits[name] = dict(temperature=temp, mixture=blend_fit)
        probabilities[name] = (1-blend_fit['sparse_weight'])*neural+blend_fit['sparse_weight']*p
    losses = {k: met(y, p)['nll'] for k, p in probabilities.items()}
    primary = min(c['candidates'], key=lambda k: losses[k])
    provenance = read(PREVIOUS+'/protocol.json')
    sources = [CONFIG, MANIFEST, PREVIOUS+'/protocol.json', PREVIOUS+'/frozen.json', PREVIOUS+'/fit-access-audit.json',
        PREVIOUS+'/calibration-predictions.json', CONTEXT+'/bundle.json', CONTEXT+'/2048-calibration.json',
        f'reports/laya-posthoc-baseline-s43-fp32-v1/calibration-head_tail.json',
        str((OUT/'heads-frozen.json').relative_to(ROOT)), 'scripts/run_final_cpu_alternatives.py',
        'scripts/common.py', 'scripts/comparison_metrics.py', 'scripts/bow_nb_branch.py',
        'scripts/run_bow_nb_study.py', 'environments/bow-cpu/uv.lock']
    protocol = dict(config=c, sources_sha256={p: file_digest(ROOT/p) for p in sources},
        expected_neural_development_sha256=provenance['development_cache_expected_sha256'],
        expected_context_development_sha256=expected['2048-development.json'],
        environment='heads: main uv.lock CPU torch; fusion/evaluation: environments/bow-cpu/uv.lock',
        continuation_rule_fixed_before_development=True, no_attention_pooling_tested=True)
    save('protocol.json', protocol)
    save('calibration-predictions.json', dict(sample_ids=ids, true_labels=y.tolist(), neural=neural.tolist(),
        p_laya=a.tolist(), p_deberta=b.tolist(), context2048=context_p.tolist(), laya_long=long.tolist(),
        predictions={k: p.tolist() for k, p in probabilities.items()}))
    save('frozen.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=file_digest(OUT/'protocol.json'),
        primary=primary, fits=fits, calibration_nll=losses,
        development_ids=read(MANIFEST)['validation']['ids'],
        heads_frozen_sha256=file_digest(OUT/'heads-frozen.json'),
        calibration_predictions_sha256=file_digest(OUT/'calibration-predictions.json'),
        development_opened=False, test_opened=False))
    save('fit-access-audit.json', dict(read_paths=sorted(READS), frozen_sha256=file_digest(OUT/'frozen.json'), development_opened=False, test_opened=False))
    emit('all_candidates_frozen', primary=primary, fits=fits, calibration_nll=losses)


def check_frozen():
    f = read(str((OUT/'frozen.json').relative_to(ROOT)))
    p = read(str((OUT/'protocol.json').relative_to(ROOT)))
    receipt = read(str((OUT/'fit-access-audit.json').relative_to(ROOT)))
    assert file_digest(OUT/'frozen.json') == receipt['frozen_sha256']
    assert file_digest(OUT/'protocol.json') == f['protocol_sha256']
    assert file_digest(OUT/'heads-frozen.json') == f['heads_frozen_sha256']
    assert file_digest(OUT/'calibration-predictions.json') == f['calibration_predictions_sha256']
    for path, sha in p['sources_sha256'].items():
        assert file_digest(ROOT/path) == sha, path
    hf = read(str((OUT/'heads-frozen.json').relative_to(ROOT)))
    for path, sha in hf['sources_sha256'].items():
        assert file_digest(ROOT/path) == sha, path
    assert file_digest(ART/'normalization.npz') == hf['normalization_sha256']
    for record in hf['records'].values():
        assert file_digest(ROOT/record['model_path']) == record['model_sha256']
    return f, p, hf


def evaluate_heads():
    import torch
    from safetensors.torch import load_file
    f, p, hf = check_frozen()
    if (OUT/'development-heads.json').exists():
        raise FileExistsError('Preserve head evaluation')
    c = p['config']
    torch.set_num_threads(c['heads']['torch_cpu_threads'])
    x, y, ids, extra = feature_block('validation', c)
    assert ids == f['development_ids']
    with np.load(ART/'normalization.npz', allow_pickle=False) as z:
        x = np.asarray((x-z['mean'])/z['scale'], dtype=np.float32)
    predictions = {}
    for name, record in hf['records'].items():
        model = make_head(torch, c)
        model.load_state_dict(load_file(ROOT/record['model_path']), strict=True)
        predictions[name] = torch_logits(torch, model, x).tolist()
    save('development-heads.json', dict(sample_ids=ids, true_labels=y.tolist(), margins=predictions,
        laya_long=extra['truncated'].tolist(), source_full_logits=extra['full_logits'].tolist(),
        frozen_sha256=file_digest(OUT/'frozen.json')))
    save('evaluate-heads-access-audit.json', dict(read_paths=sorted(READS), test_opened=False, backbone_updated=False))
    emit('heads_development_complete')


def holm(pvalues):
    result, previous = {}, 0.
    for rank, name in enumerate(sorted(pvalues, key=pvalues.get)):
        previous = max(previous, min(1., (len(pvalues)-rank)*pvalues[name]))
        result[name] = previous
    return result


def numpy_head_state(path):
    """Read this study's own FP32 safetensors without a second runtime dependency."""
    with Path(path).open('rb') as handle:
        size = int.from_bytes(handle.read(8), 'little')
        if not 0 < size < 1000000:
            raise ValueError('Invalid model header size')
        header = json.loads(handle.read(size))
        data = handle.read()
    state = {}
    for name, item in header.items():
        if name == '__metadata__':
            continue
        start, stop = item['data_offsets']
        if item['dtype'] != 'F32' or not 0 <= start <= stop <= len(data) or stop-start != 4*int(np.prod(item['shape'])):
            raise ValueError('Unexpected tensor format')
        state[name] = np.frombuffer(data[start:stop], dtype='<f4').reshape(item['shape'])
    return state


def evaluate():
    from scipy.optimize import minimize_scalar
    from scipy.special import erf
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    import run_bow_nb_study as stats
    f, protocol, hf = check_frozen()
    if (OUT/'summary.json').exists():
        raise FileExistsError('Preserve final evaluation')
    path = NEURAL+'/predictions.json'
    assert file_digest(ROOT/path) == protocol['expected_neural_development_sha256']
    neural_data = read(path)
    ids, y = neural_data['sample_ids'], np.asarray(neural_data['true_labels'])
    assert ids == f['development_ids'] and not neural_data['test_accessed']
    pp = neural_data['positive_probabilities']
    a, b, neural = [np.asarray(pp[k]) for k in ('frozen_main', 'deberta', 'fusion_50_50')]
    np.testing.assert_array_equal(neural, (a+b)/2)
    assert int(((neural > .5) == y).sum()) == 2432
    heads_dev = read(str((OUT/'development-heads.json').relative_to(ROOT)))
    assert heads_dev['sample_ids'] == ids and np.array_equal(heads_dev['true_labels'], y)
    assert heads_dev['frozen_sha256'] == file_digest(OUT/'frozen.json')
    long = np.asarray(heads_dev['laya_long'], dtype=bool)
    assert long.sum() == 407
    path = CONTEXT+'/2048-development.json'
    assert file_digest(ROOT/path) == protocol['expected_context_development_sha256']
    context = read(path)
    assert context['sample_ids'] == ids and np.array_equal(context['true_labels'], y) and not context['test_accessed']
    z = np.asarray(context['logits'])
    context_p = sigmoid((z[:, 1]-z[:, 0])/f['fits']['long_context']['temperature'])
    predictions = dict(regularized_logit=sigmoid(logit_features(a, b)@np.asarray(f['fits']['regularized_logit']['coefficients'])),
        long_context=neural+f['fits']['long_context']['weight']*long*(context_p-neural))
    records = {}
    # Validate saved head inference with an independent NumPy/SciPy implementation.
    vx, vy, vids, _ = feature_block('validation', protocol['config'])
    with np.load(ART/'normalization.npz', allow_pickle=False) as norm:
        vx = np.asarray((vx-norm['mean'])/norm['scale'], dtype=np.float32)
    independent_max = {}
    for name in ('mlp_dropout1', 'mlp_dropout4'):
        margin = np.asarray(heads_dev['margins'][name])
        state = numpy_head_state(ROOT/hf['records'][name]['model_path'])
        h = vx@state['features.0.weight'].T+state['features.0.bias']
        h = .5*h*(1+erf(h/np.sqrt(2.)))
        h = (h-h.mean(axis=1, keepdims=True))/np.sqrt(h.var(axis=1, keepdims=True)+1e-5)
        h = h*state['features.2.weight']+state['features.2.bias']
        independent = (h@state['scorer.weight'].T+state['scorer.bias']).ravel()
        independent_max[name] = float(np.max(np.abs(independent-margin)))
        np.testing.assert_allclose(independent, margin, atol=1e-4, rtol=1e-4)
        hp = sigmoid(margin/f['fits'][name]['temperature'])
        w = f['fits'][name]['mixture']['sparse_weight']
        predictions[name] = (1-w)*neural+w*hp
        records[name] = dict(standalone=met(y, hp), versus_original_full_model=stats.compare(y,
            sigmoid(np.diff(np.asarray(heads_dev['source_full_logits']), axis=1).ravel()), hp))
    head_comparison = stats.compare(y, sigmoid(np.asarray(heads_dev['margins']['mlp_dropout1'])), sigmoid(np.asarray(heads_dev['margins']['mlp_dropout4'])))
    results = {}
    for name in protocol['config']['candidates']:
        probability = predictions[name]
        m = met(y, probability)
        assert accuracy_score(y, probability > .5) == m['accuracy']
        assert abs(log_loss(y, probability, labels=[0, 1])-m['nll']) < 1e-12
        assert abs(roc_auc_score(y, probability)-m['auroc']) < 1e-12
        results[name] = dict(metrics=m, paired=stats.compare(y, neural, probability),
            groups={group:dict(count=int(mask.sum()), correct=int(((probability[mask] > .5) == y[mask]).sum()),
                baseline_correct=int(((neural[mask] > .5) == y[mask]).sum())) for group, mask in [('laya_long', long), ('laya_short', ~long)]})
    adjusted = holm({k: r['paired']['mcnemar_exact_two_sided_p'] for k, r in results.items()})
    rule = protocol['config']['continuation_reference']
    for name, r in results.items():
        r['holm_adjusted_p'] = adjusted[name]
        r['passes_continuation_reference'] = bool(r['paired']['delta_accuracy_pp'] >= rule['minimum_development_delta_pp']
            and r['paired']['conditional_stratified_bootstrap95_pp'][0] > 0 and adjusted[name] < rule['holm_adjusted_p_below'])
    summary = dict(run_id=RUN, primary=f['primary'], baseline=met(y, neural), results=results,
        standalone_heads=records, msd_minus_single_dropout=head_comparison,
        continue_research_candidate=[k for k, r in results.items() if r['passes_continuation_reference']],
        frozen_sha256=file_digest(OUT/'frozen.json'), fits=f['fits'],
        limitations=['reused calibration/development', 'frozen encoder previously fitted on gradient training',
            'not OOF stacking', 'single head initialization', 'no end-to-end attention pooling or encoder retraining',
            'sample intervals exclude training/calibration/model-selection uncertainty'],
        official_test_opened=False, official_test_distribution_used=False, cloud_compute=False,
        backbone_updated=False, default_deployment_changed=False)
    save('development-predictions.json', dict(sample_ids=ids, true_labels=y.tolist(), neural=neural.tolist(),
        predictions={k: v.tolist() for k, v in predictions.items()}))
    save('summary.json', summary)
    report(summary, hf)
    save('evaluate-access-audit.json', dict(read_paths=sorted(READS), test_opened=False))
    save('verification.json', dict(passed=True, independent_numpy_head_max_abs_logit_difference=independent_max,
        head_initialization_equal=hf['initialization_equal'], head_reload_exact=True,
        source_model_partition_sha256_verified=True, sklearn_metrics_recomputed=True,
        original_neural_probabilities_bitwise_equal=True,
        report_sha256=file_digest(OUT/'REPORT.md'), summary_sha256=file_digest(OUT/'summary.json')))
    emit('final_results', primary=f['primary'], results={k:dict(accuracy=r['metrics']['accuracy'], net=r['paired']['net'],
        repaired=r['paired']['repaired'], damaged=r['paired']['damaged']) for k, r in results.items()},
        standalone_heads={k:r['standalone']['accuracy'] for k,r in records.items()},
        continuation=summary['continue_research_candidate'])


def report(s, hf):
    lines = ['# 最后一轮本地 CPU 候选实验', '',
        '四个候选的配置、训练轮数、参数范围和继续投入参考在开发评估前固定；两个 MLP 在 22100 条冻结训练表示上学习，其余参数仅用固定 400 条校准拟合。全部模型与参数共同封存后，独立进程读取 2500 条开发缓存。未读取原始评论、官方测试正文/预测/分布，未调用云端或 GPU，未修改默认部署。', '',
        f'校准 NLL 预选主方案：`{s["primary"]}`。原 50:50 神经融合 2432/2500 = 97.28% 已逐样本复现。', '',
        '| 方案 | 正确 / 2500 | Accuracy | NLL | 修复 / 退化 | 净变化 |', '|---|---:|---:|---:|---:|---:|',
        f'| 原神经融合 | 2432 | 97.28% | {s["baseline"]["nll"]:.6f} | — | — |']
    for name, r in s['results'].items():
        m, p = r['metrics'], r['paired']
        lines.append(f'| {name} | {m["correct"]} | {100*m["accuracy"]:.2f}% | {m["nll"]:.6f} | {p["repaired"]} / {p["damaged"]} | {p["net"]:+d} |')
    lines += ['', '## 固定方法与参数', '',
        '- regularized_logit：对两路原校准概率取 logit、裁剪到 ±12，拟合两个非负系数及截距；系数范围 [0,2]、截距 [-1,1]、固定 L2=0.01，向 (0.5,0.5,0) 收缩。是 400 条留出集融合，不是 OOF stacking。',
        '- long_context：短文维持原神经融合，仅 Laya 原始评论 token>466 时加入冻结 2048/seed42/epoch2 概率。400 条校准重新拟合该分支温度，单权重范围 [0,0.5]，NLL+0.01*w²。466 为 Laya 路由标准，不声称等于 DeBERTa 的截断阈值。',
        '- mlp_dropout1/4：同一 baseline-s43 编码器缓存的正负候选标记之差（1024 维），训练集标准化后接 Linear(1024,64)→GELU→LayerNorm→Dropout(0.2)→Linear(64,1)。仅新头训练，原编码器不更新。两组起始参数逐位相同，训练顺序使用独立且相同的随机流。',
        '- 两个头均 CPU、seed42、AdamW lr=0.001/weight_decay=0.01、batch256、固定10轮；普通头每批一次 dropout，多样本头共享前面表示，对末层采样4次 dropout 并平均四次 BCE。每组870更新、221000样本曝光，不按校准或开发选择 epoch。',
        '- 两个头各用校准拟合正温度，再拟合与原神经融合的单个概率系数 [0,0.25]，可为零。未搜索隐藏宽度、dropout率、种子或损失参数。', '',
        '冻结参数：', '', '```json', json.dumps(s['fits'], ensure_ascii=False, indent=2), '```', '',
        '## 冻结表示分类头对照', '',
        '| 头 | 单独开发 Accuracy | 参数数 | CPU训练/保存/重载秒 |', '|---|---:|---:|---:|']
    for name, r in s['standalone_heads'].items():
        record = hf['records'][name]
        lines.append(f'| {name} | {100*r["standalone"]["accuracy"]:.2f}% | {record["parameter_count"]} | {record["seconds"]:.2f} |')
    p = s['msd_minus_single_dropout']
    lines += ['', f'多样本相对单次 Dropout：净 {p["net"]:+d} 条、差 {p["delta_accuracy_pp"]:+.3f} pp，条件区间 {p["conditional_stratified_bootstrap95_pp"]} pp。单个初始化只描述这次配对；不证明跨 seed 效果。', '',
        '特征是候选标记表示之差，不是完整 token 序列，因此本轮没有测试 attention pooling；也没有进行端到端头/编码器共同微调。训练特征来自已在同一梯度分区训练的编码器，属于固定表示新读出，不是折外表示。不能用本轮结果宣称注意力池化无效或所有结构方法均无空间。', '',
        '## 不确定性与分组', '',
        '| 方案 | 差值条件95%区间（pp） | 精确McNemar p | 四候选Holm调整p | 长文净变化 | 短文净变化 |', '|---|---|---:|---:|---:|---:|']
    for name, r in s['results'].items():
        p = r['paired']; ci = p['conditional_stratified_bootstrap95_pp']
        g = r['groups']
        lines.append(f'| {name} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {p["mcnemar_exact_two_sided_p"]:.6g} | {r["holm_adjusted_p"]:.6g} | {g["laya_long"]["correct"]-g["laya_long"]["baseline_correct"]:+d} | {g["laya_short"]["correct"]-g["laya_short"]["baseline_correct"]:+d} |')
    lines += ['', '配对 bootstrap 共20000次，类别分层、seed=20261018，固定现有模型与拟合参数。区间不包括校准拟合、训练随机性、方案选择和反复开发使用的乐观性。四个最终融合比较的名义 p 另作 Holm 校正；两个头间对照属于次级诊断，未合并进该四项族。', '',
        '## 是否继续', '',
        '预声明的投入参考：开发提升至少0.20 pp（净5条）、条件区间下界>0、四候选Holm调整p<0.05。它是是否值得独立确认的参考，不是泛化保证，也不能替代更多种子。', '',
        f'通过参考的候选：{s["continue_research_candidate"]}。'+('建议为通过者另立独立确认实验，不立即部署。' if s['continue_research_candidate'] else '本轮没有达到继续投入参考的候选，建议结束当前增益搜索并准备课程交付。'), '',
        '收束只针对当前预算和已验证候选，不证明全局最优。原神经融合的97.28%开发与历史97.028%正式测试成绩继续保留；本轮新候选未进行官方测试。此前TF-IDF融合97.32%仅净多一条、无稳定收益，仍独立保留为探索记录。', '',
        '## 验证', '',
        '原特征缓存与索引 SHA、来源 ID/标签、源码/配置、两头相同初始化、连续有限更新及曝光数、模型独立重载、共同冻结契约均核验。开发头 logits 另用 NumPy/SciPy 重写 GELU/LayerNorm/线性前向核对；全部最终预测的 Accuracy/NLL/AUROC 用 sklearn 复算。访问记录分阶段保存；模型在 artifacts/'+RUN+'/，报告、参数、逐样本预测和 verification.json 在本目录。', '',
        '入口依次运行 heads（主环境）、fit（bow-cpu环境）、evaluate_heads（主环境）、evaluate（bow-cpu环境）。拒绝覆盖已有运行，不自动追加候选。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('heads', 'fit', 'evaluate_heads', 'evaluate'))
    args = parser.parse_args()
    install_guard(args.stage)
    {'heads': heads, 'fit': fit, 'evaluate_heads': evaluate_heads, 'evaluate': evaluate}[args.stage]()
