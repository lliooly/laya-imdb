"""Independent optimization and saved-metric verification of the bounded study."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from common import file_digest, write_json
from run_final_cpu_alternatives_v2 import OUT, ROOT, check_frozen, install_guard, holm


def load(name):
    return json.loads((OUT/name).read_text())


def main():
    install_guard('evaluate')
    frozen, protocol, heads = check_frozen()
    config = protocol['config']
    cal, dev, summary = load('calibration-predictions.json'), load('development-predictions.json'), load('summary.json')
    y, cy = np.asarray(dev['true_labels']), np.asarray(cal['true_labels'])
    assert len(set(dev['sample_ids'])) == len(y) == 2500
    assert len(set(cal['sample_ids'])) == len(cy) == 400
    assert not set(dev['sample_ids']) & set(cal['sample_ids'])
    a, b = np.asarray(cal['p_laya']), np.asarray(cal['p_deberta'])
    transform = lambda p: np.clip(np.log(np.clip(p, 1e-12, 1-1e-12))-np.log1p(-np.clip(p, 1e-12, 1-1e-12)), -12, 12)
    x = np.column_stack([transform(a), transform(b), np.ones(len(a))])
    prior = np.array([.5, .5, 0.])
    objective = lambda w: float(np.mean(np.logaddexp(0, x@w)-cy*(x@w))+.01*np.sum((w-prior)**2))
    independent = minimize(objective, [.4, .6, .1], method='SLSQP', bounds=[(0, 2), (0, 2), (-1, 1)],
        options=dict(ftol=1e-13, maxiter=1000))
    assert independent.success
    weights = np.asarray(frozen['fits']['regularized_logit']['coefficients'])
    assert np.max(np.abs(independent.x-weights)) < 1e-5
    residual = x.T@(expit(x@weights)-cy)/len(cy)+.02*(weights-prior)
    assert np.max(np.abs(residual)) < 1e-7
    np.testing.assert_allclose(expit(x@weights), cal['predictions']['regularized_logit'], atol=1e-14, rtol=0)
    neural = np.asarray(cal['neural'])
    long = np.asarray(cal['laya_long'])
    extra = np.asarray(cal['context2048'])
    long_objective = lambda w: log_loss(cy, neural+w*long*(extra-neural), labels=[0, 1])+.01*w*w
    candidate = minimize_scalar(long_objective, bounds=(0, .5), method='bounded')
    long_best = min([0., candidate.x, .5], key=long_objective)
    assert abs(long_best-frozen['fits']['long_context']['weight']) < 1e-7
    for name in ('mlp_dropout1', 'mlp_dropout4'):
        margin = np.asarray(heads['calibration_margins'][name])
        fitted = frozen['fits'][name]
        p = expit(margin/fitted['temperature'])
        objective_mix = lambda w: log_loss(cy, (1-w)*neural+w*p, labels=[0, 1])
        independent_mix = minimize_scalar(objective_mix, bounds=(0, .25), method='bounded', options=dict(xatol=1e-12))
        best = min([0., independent_mix.x, .25], key=objective_mix)
        frozen_weight = fitted['mixture']['sparse_weight']
        assert abs(best-frozen_weight) < 1e-7
        assert abs(objective_mix(best)-objective_mix(frozen_weight)) < 1e-12
        np.testing.assert_allclose((1-frozen_weight)*neural+frozen_weight*p, cal['predictions'][name], atol=1e-14, rtol=0)
        history = heads['records'][name]['history']
        assert len(history) == 10 and sum(v['updates'] for v in history) == 870
        assert sum(v['exposures'] for v in history) == 221000
        assert all(np.isfinite(v['mean_loss']) for v in history)
    losses = {name: log_loss(cy, p, labels=[0, 1]) for name, p in cal['predictions'].items()}
    assert min(config['candidates'], key=lambda name: losses[name]) == frozen['primary'] == summary['primary']
    baseline = np.asarray(dev['neural'])
    assert int(((baseline > .5) == y).sum()) == 2432
    pair_checks = {}
    for name, p in dev['predictions'].items():
        p = np.asarray(p)
        result = summary['results'][name]
        assert accuracy_score(y, p > .5) == result['metrics']['accuracy']
        assert abs(log_loss(y, p, labels=[0, 1])-result['metrics']['nll']) < 1e-12
        assert abs(roc_auc_score(y, p)-result['metrics']['auroc']) < 1e-12
        good, changed = (baseline > .5) == y, (p > .5) == y
        repaired, damaged = int((~good & changed).sum()), int((good & ~changed).sum())
        pair = result['paired']
        assert (repaired, damaged, repaired-damaged) == (pair['repaired'], pair['damaged'], pair['net'])
        expected_p = binomtest(min(repaired, damaged), repaired+damaged).pvalue if repaired+damaged else 1.
        assert expected_p == pair['mcnemar_exact_two_sided_p']
        pair_checks[name] = dict(repaired=repaired, damaged=damaged, net=repaired-damaged)
    expected_holm = holm({k: r['paired']['mcnemar_exact_two_sided_p'] for k, r in summary['results'].items()})
    assert all(r['holm_adjusted_p'] == expected_holm[k] for k, r in summary['results'].items())
    for name in ('heads-access-audit.json', 'fit-access-audit.json', 'evaluate-heads-access-audit.json', 'evaluate-access-audit.json'):
        for path in load(name)['read_paths']:
            assert not ('/data/raw/' in path or path.endswith('/test.jsonl') or 'final-cpu-evaluation' in path or 'frozen-official-test' in path)
            if name in ('heads-access-audit.json', 'fit-access-audit.json'):
                assert not (Path(path).name in ('validation.npz', 'validation-index.json', 'predictions.json', 'development-heads.json') or path.endswith('-development.json'))
    output = dict(passed=True, independent_optimizer='SLSQP numerical derivative versus original bounded L-BFGS-B analytic derivative',
        independent_logit_coefficients=independent.x.tolist(), logit_max_abs_gradient=float(np.max(np.abs(residual))),
        independent_context_and_head_mixture_optima=True, primary_calibration_selection_verified=True,
        training_exposures_and_updates_verified=True, sklearn_metrics_verified=True,
        paired_counts_and_holm_verified=True, access_audit_verified=True,
        pair_counts=pair_checks, frozen_sha256=file_digest(OUT/'frozen.json'),
        summary_sha256=file_digest(OUT/'summary.json'), verifier_sha256=file_digest(Path(__file__)))
    write_json(OUT/'independent-verification.json', output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
