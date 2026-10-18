"""Independent saved-output checks; never fit or access official-test files."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from common import ROOT, file_digest, write_json
from run_bow_nb_study import ART, KINDS, OUT, install_guard


def read(name):
    return json.loads((OUT/name).read_text())


def main():
    install_guard('evaluate')
    summary, frozen, pred = read('summary.json'), read('frozen.json'), read('development-predictions.json')
    receipt, protocol = read('fit-access-audit.json'), read('protocol.json')
    frozen_sha = file_digest(OUT/'frozen.json')
    assert summary['frozen_sha256'] == receipt['frozen_sha256'] == frozen_sha
    assert frozen['protocol_sha256'] == file_digest(OUT/'protocol.json')
    for path, sha in frozen['source_sha256'].items():
        assert file_digest(ROOT/path) == sha, path
    for path, sha in summary['evaluation_source_sha256'].items():
        assert file_digest(ROOT/path) == sha, path
    y = np.asarray(pred['true_labels'])
    ids = pred['sample_ids']
    assert len(ids) == len(set(ids)) == len(y) == 2500
    assert ids == frozen['development_ids']
    gradient, calibration, development = [set(frozen[k+'_ids']) for k in ('gradient', 'calibration', 'development')]
    assert not gradient & calibration and not gradient & development and not calibration & development
    neural = np.asarray(pred['predictions']['neural'])
    assert int(((neural > .5) == y).sum()) == 2432
    primary_calibration_losses = {}
    checks = {}
    for kind in KINDS:
        info = frozen['candidates'][kind]
        assert file_digest(ROOT/info['model_path']) == info['model_sha256']
        cal = read(kind+'-calibration.json')
        assert cal['sample_ids'] == frozen['calibration_ids']
        cy = np.asarray(cal['true_labels'])
        cp = expit(info['sigmoid']['slope']*np.asarray(cal['margins'])+info['sigmoid']['intercept'])
        np.testing.assert_allclose(cp, cal['positive_probabilities'], atol=1e-15, rtol=0)
        cn = np.asarray(cal['neural_positive_probabilities'])
        objective = lambda w: log_loss(cy, (1-w)*cn+w*cp, labels=[0, 1])
        opt = minimize_scalar(objective, bounds=(0., .25), method='bounded', options={'xatol': 1e-12})
        candidates = [0., opt.x, .25]
        optimum = min(candidates, key=objective)
        w = info['fusion']['sparse_weight']
        assert abs(w-optimum) < 1e-7
        assert objective(w) <= objective(0)+1e-12
        assert abs(objective(w)-info['fusion']['calibration_nll']) < 1e-12
        primary_calibration_losses[kind] = objective(w)
        block = pred['predictions'][kind]
        sp = expit(info['sigmoid']['slope']*np.asarray(block['raw_margins'])+info['sigmoid']['intercept'])
        np.testing.assert_allclose(sp, block['calibrated_positive_probabilities'], atol=1e-15, rtol=0)
        fused = (1-w)*neural+w*sp
        np.testing.assert_allclose(fused, block['fusion_positive_probabilities'], atol=1e-15, rtol=0)
        result = summary['results'][kind]
        for name, values in [('sparse', sp), ('fusion', fused)]:
            expected = result[name]
            assert accuracy_score(y, values > .5) == expected['accuracy']
            assert abs(log_loss(y, values, labels=[0, 1])-expected['nll']) < 1e-12
            assert abs(roc_auc_score(y, values)-expected['auroc']) < 1e-12
        old_correct, new_correct = (neural > .5) == y, (fused > .5) == y
        repaired = int((~old_correct & new_correct).sum())
        damaged = int((old_correct & ~new_correct).sum())
        pair = result['fusion_pair']
        assert repaired == pair['repaired'] and damaged == pair['damaged']
        assert pair['net'] == repaired-damaged
        assert pair['mcnemar_exact_two_sided_p'] == (binomtest(min(repaired, damaged), repaired+damaged).pvalue if repaired+damaged else 1.)
        complement = result['complementarity']
        assert complement['sparse_fixes_neural_errors'] == int((~old_correct & ((sp > .5) == y)).sum())
        assert not complement['oracle_deployable']
        checks[kind] = dict(calibration_weight_independently_optimized=float(optimum),
            weight_abs_difference=float(abs(w-optimum)), accuracy_nll_auroc_recomputed=True,
            saved_probabilities_reconstructed=True, repaired=repaired, damaged=damaged,
            model_sha256_verified=True)
    assert summary['primary'] == min(KINDS, key=lambda k: primary_calibration_losses[k])
    assert not any('/validation-' in p or p.endswith('/validation.jsonl') or p.endswith('/predictions.json') for p in receipt['read_paths'])
    for name in ('fit-access-audit.json', 'evaluate-access-audit.json'):
        assert not any('/test-' in p or '/data/raw/' in p or p.endswith('/test.jsonl') or 'final-cpu-evaluation' in p for p in read(name)['read_paths'])
    result = dict(passed=True, verifier_sha256=file_digest(Path(__file__)),
        frozen_sha256=frozen_sha, summary_sha256=file_digest(OUT/'summary.json'),
        independent_scipy_scalar_optimization=True, sklearn_metrics_recomputed=True,
        source_model_and_data_sha256_verified=True, partition_ids_disjoint=True,
        fit_reads_exclude_development=True, execution_reads_exclude_official_test=True,
        parent_preparation_deviation=protocol['preparation_deviation'], checks=checks)
    write_json(OUT/'verification.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
