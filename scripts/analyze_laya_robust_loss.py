"""Analyze the fixed three paired seeds only after six local evaluation receipts."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path

import numpy as np

from common import ROOT, write_json
from comparison_metrics import metrics
from laya_checkpoint_delivery import (digest, request_sha, validate_receipt, verify_files)
from laya_modal_deferred import STUDY, name

SEEDS = (42, 43, 44)
ARMS = ('ce', 'mixed')
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20261015


def read(path):
    return json.loads(Path(path).read_text())


def exact_p(repairs, regressions):
    n = repairs + regressions
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(repairs, regressions)+1)) / 2**n) if n else 1.


def paired_summary(labels, ce_correct, mixed_correct, nll_deltas=None, draws=BOOTSTRAPS):
    labels = np.asarray(labels)
    ce_correct, mixed_correct = np.asarray(ce_correct, bool), np.asarray(mixed_correct, bool)
    if ce_correct.shape != mixed_correct.shape or ce_correct.shape != (len(labels), 3):
        raise ValueError('Require identical sample IDs and exactly three paired seeds')
    difference = mixed_correct.astype(float) - ce_correct.astype(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(not len(g) for g in strata):
        raise ValueError('Both fixed label strata must be nonempty')
    samples = np.empty((draws, 3))
    loss_samples = {stage: np.empty(draws) for stage in (nll_deltas or {})}
    for i in range(draws):
        ids = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in strata])
        samples[i] = 100 * difference[ids].mean(axis=0)
        for stage, delta in (nll_deltas or {}).items():
            loss_samples[stage][i] = delta[ids].mean()
    per_seed = []
    for j, seed in enumerate(SEEDS):
        a, b = ce_correct[:, j], mixed_correct[:, j]
        repairs, regressions = int((~a & b).sum()), int((a & ~b).sum())
        per_seed.append({'seed': seed, 'ce_correct': int(a.sum()), 'mixed_correct': int(b.sum()),
                         'ce_accuracy_percent': float(100*a.mean()), 'mixed_accuracy_percent': float(100*b.mean()),
                         'repairs': repairs, 'regressions': regressions,
                         'both_wrong': int((~a & ~b).sum()), 'both_correct': int((a & b).sum()),
                         'delta_pp': float(100*difference[:, j].mean()),
                         'conditional_bootstrap95_pp': np.quantile(samples[:, j], [.025, .975]).tolist(),
                         'mcnemar_exact_nominal_two_sided_p': exact_p(repairs, regressions)})
    effects = 100 * difference.mean(axis=0)
    mean = float(effects.mean())
    sign_distribution = [float(np.mean(effects*np.asarray(signs))) for signs in itertools.product((-1, 1), repeat=3)]
    ci = np.quantile(samples.mean(axis=1), [.025, .975]).tolist()
    return {'count': len(labels), 'per_seed': per_seed,
            'ce_mean_accuracy_percent': float(100*ce_correct.mean()),
            'mixed_mean_accuracy_percent': float(100*mixed_correct.mean()),
            'mean_mixed_minus_ce_pp': mean, 'conditional_shared_id_bootstrap95_pp': ci,
            'conditional_interval_contains': {str(v): bool(ci[0] <= v <= ci[1]) for v in (-.2, 0., .2)},
            'seed_effect_sample_std_pp': float(effects.std(ddof=1)),
            'seed_effect_range_pp': [float(effects.min()), float(effects.max())],
            'leave_one_seed_out_mean_pp': {str(seed): float(np.delete(effects, j).mean()) for j, seed in enumerate(SEEDS)},
            'seed_sign_flip_nominal_two_sided_p': float(np.mean(np.abs(sign_distribution) >= abs(mean)-1e-12)),
            'nll_mixed_minus_ce': {stage: {'mean': float(delta.mean()),
                'conditional_bootstrap95': np.quantile(loss_samples[stage], [.025, .975]).tolist(),
                'secondary_metric_interval': True} for stage, delta in (nll_deltas or {}).items()}}


def continuation(group):
    effects = [r['delta_pp'] for r in group['per_seed']]
    checks = {'mean_at_least_020_pp': group['mean_mixed_minus_ce_pp'] >= .2-1e-12,
              'at_least_two_positive_seeds': sum(d > 1e-12 for d in effects) >= 2,
              'no_seed_below_minus_020_pp': min(effects) >= -.2-1e-12}
    return {'passed': all(checks.values()), 'checks': checks,
            'meaning': 'reference for a new confirmation experiment, not significance or deployment approval'}


def analyze(output):
    contract = read(ROOT/'configs'/f'{STUDY}.json')
    contract_sha = digest(ROOT/'configs'/f'{STUDY}.json')
    for rel, sha in contract['source_sha256'].items():
        if digest(ROOT/rel) != sha:
            raise ValueError('Scientific bound source changed: '+rel)
    gate_path = ROOT/'reports/laya-robust-loss-modal-recovery-20261016-v1/status-check/reports'/STUDY/'all-models-frozen.json'
    gate = read(gate_path)
    payloads, runs, partitions = {}, {}, []
    verified_receipts = 0
    for seed, arm in itertools.product(SEEDS, ARMS):
        run = name(seed, arm)
        base = ROOT/'checkpoints'/run
        stage_receipts = {}
        for stage in ('epoch-002', 'frozen', 'evaluated'):
            req = read(ROOT/'delivery/requests'/f'{run}.{stage}.json')
            receipt = read(ROOT/'delivery/receipts'/f'{request_sha(req)}.json')
            validate_receipt(req, receipt)
            verify_files(ROOT, req)
            if req['identity']['contract_sha256'] != contract_sha:
                raise ValueError('Contract mismatch')
            stage_receipts[stage] = receipt
            verified_receipts += 1
            if stage == 'frozen' and request_sha(req) != gate['freeze_request_sha256'][run]:
                raise ValueError('Frozen gate binding changed')
        manifest = read(base/'epoch-002/manifest.json')
        result = read(base/'epoch-002/result.json')
        audit = read(base/'epoch-002/training_audit.json')
        frozen = read(base/'frozen/frozen.json')
        if (audit['updates'], audit['gradient_exposures']) != (5526, 44200):
            raise ValueError('Incomplete formal training')
        if not frozen['independent_reload_logits_bitwise_equal'] or audit['test_accessed'] or audit['development_body_read']:
            raise ValueError('Freeze/numerics/data access checks failed')
        if frozen['checkpoint_state_sha256'] != stage_receipts['evaluated']['verification']['state_sha256']:
            raise ValueError('Evaluation identity differs from locally loaded model')
        partition = (manifest['gradient_ids'], manifest['calibration_ids'])
        if len(partition[0]) != len(set(partition[0])) or len(partition[1]) != len(set(partition[1])) or set(partition[0]) & set(partition[1]):
            raise ValueError('Invalid gradient/calibration partition')
        partitions.append(partition)
        payloads[run] = read(base/'evaluated/validation.json')
        runs[run] = {'seed': seed, 'arm': arm, 'initial_state_sha256': manifest['initial_state_sha256'],
                     'input_stream_sha256': audit['input_stream_sha256'], 'training_result': result,
                     'development': {k:v for k,v in payloads[run].items() if k not in ('rows','logits')},
                     'frozen_temperature': frozen['temperature_fit']['temperature'],
                     'peak_allocated_gib': result['peak_allocated_bytes']/1024**3,
                     'peak_reserved_gib': result['peak_reserved_bytes']/1024**3}
    if any(p != partitions[0] for p in partitions) or tuple(map(len, partitions[0])) != (22100, 400):
        raise ValueError('Model partitions differ')
    for seed in SEEDS:
        a, b = runs[name(seed, 'ce')], runs[name(seed, 'mixed')]
        if a['initial_state_sha256'] != b['initial_state_sha256'] or a['input_stream_sha256'] != b['input_stream_sha256']:
            raise ValueError('Paired initialization or input differs')
    rows = payloads[name(42, 'ce')]['rows']
    if any(d['rows'] != rows or d['test_accessed'] for d in payloads.values()):
        raise ValueError('Development rows/input digests do not align')
    if len(rows) != 2500 or len({r['sample_id'] for r in rows}) != 2500:
        raise ValueError('Invalid development count')
    if {r['sample_id'] for r in rows} & set(partitions[0][0]+partitions[0][1]):
        raise ValueError('Development and training partitions overlap')
    labels = np.array([r['true_label'] for r in rows])
    truncated = np.array([r['truncated'] for r in rows])
    if not np.array_equal(np.bincount(labels), [1250, 1250]) or truncated.sum() != 407:
        raise ValueError('Unexpected labels or truncation groups')
    masks = {'all': np.ones(2500, bool), 'truncated': truncated, 'untruncated': ~truncated}
    correct, losses = {}, {}
    for run, d in payloads.items():
        logits = np.asarray(d['logits'], np.float64)
        if logits.shape != (2500, 2) or not np.isfinite(logits).all():
            raise ValueError('Invalid cached logits')
        margin = logits[:, 1] - logits[:, 0]
        correct[run] = (margin > 0) == labels
        if d['temperature'] != runs[run]['frozen_temperature']:
            raise ValueError('Frozen calibration temperature changed')
        for stage, temperature in (('raw', 1.), ('calibrated', d['temperature'])):
            losses[run, stage] = np.logaddexp(0., np.where(labels == 1, -margin, margin)/temperature)
            for group, keep in masks.items():
                actual = metrics(labels[keep], margin[keep], temperature)
                saved = d[stage+'_metrics'][group]
                for key in ('accuracy', 'nll', 'brier_positive', 'ece_15_equal_width_top_label', 'auroc'):
                    if not np.isclose(actual[key], saved[key], atol=1e-12, rtol=0.):
                        raise ValueError('Saved metric differs from independent recomputation')
    groups = {}
    for group, keep in masks.items():
        a = np.column_stack([correct[name(s, 'ce')][keep] for s in SEEDS])
        b = np.column_stack([correct[name(s, 'mixed')][keep] for s in SEEDS])
        nll = {stage: np.column_stack([losses[name(s, 'mixed'), stage][keep]-losses[name(s, 'ce'), stage][keep] for s in SEEDS]).mean(axis=1) for stage in ('raw', 'calibrated')}
        groups[group] = paired_summary(labels[keep], a, b, nll)
    wrong = np.column_stack([~correct[name(s, a)] for s, a in itertools.product(SEEDS, ARMS)])
    votes = wrong.sum(axis=1)
    summary = {'state': 'completed_analysis', 'study_id': STUDY, 'analysis_utc': datetime.now(timezone.utc).isoformat(),
               'analysis_source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                   (ROOT/'scripts/analyze_laya_robust_loss.py', ROOT/'tests/test_analyze_laya_robust_loss.py')},
               'scientific_contract_sha256': contract_sha,
               'audit': {'bound_sources_verified': len(contract['source_sha256']), 'local_receipts_verified': verified_receipts,
                         'final_models_verified_local': 6, 'evaluation_models_verified_local': 6,
                         'paired_input_and_initial_states_verified': True, 'development_rows_and_inputs_identical': True,
                         'formal_updates': 33156, 'formal_exposures': 265200, 'test_accessed': False},
               'groups': groups, 'runs': runs, 'continuation_reference': continuation(groups['all']),
               'error_overlap': {'all_six_wrong': int((votes == 6).sum()), 'all_six_wrong_truncated': int(((votes == 6) & truncated).sum()),
                                 'wrong_model_count_histogram': {str(k): int((votes == k).sum()) for k in range(7)},
                                 'descriptive_only_not_training_data': True},
               'uncertainty': {'bootstrap': BOOTSTRAPS, 'bootstrap_seed': BOOTSTRAP_SEED,
                   'shared_id_label_stratified_paired': True, 'conditions_on_six_fixed_models': True,
                   'seed_population_uncertainty_not_covered': True, 'calibration_refit_uncertainty_not_covered': True,
                   'historically_reused_development_bias_not_covered': True, 'seed_count': 3,
                   'minimum_three_seed_two_sided_sign_flip_p': .25, 'multiple_secondary_metrics_not_adjusted': True,
                   'no_equivalence_claim': True, 'no_automatic_extra_training_or_ensemble': True}}
    if output.exists():
        raise FileExistsError('Preserve an existing analysis; choose a new output ID')
    output.mkdir(parents=True)
    write_json(output/'summary.json', summary)
    errors = [{'sample_id':r['sample_id'], 'true_label':r['true_label'], 'truncated':r['truncated'],
               'wrong_models':int(votes[i])} for i,r in enumerate(rows) if votes[i] > 0]
    write_json(output/'error-audit.json', {'scope':'development descriptive only; never new training data','rows':errors})
    plot(summary, output)
    print(json.dumps({'output':str(output), 'groups':groups, 'continuation':summary['continuation_reference']}, indent=2))


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for arm, color in (('ce', '#3564a5'), ('mixed', '#d07824')):
        acc = [summary['runs'][name(s, arm)]['development']['raw_metrics']['all']['accuracy']*100 for s in SEEDS]
        axes[0].plot(SEEDS, acc, 'o-', label=arm, color=color)
    axes[0].set(xticks=SEEDS, xlabel='Training seed', ylabel='Development accuracy (%)', title='Same 2,500 development items')
    axes[0].legend(frameon=False)
    group = summary['groups']['all']
    points = [(r['delta_pp'], r['conditional_bootstrap95_pp']) for r in group['per_seed']]
    points.append((group['mean_mixed_minus_ce_pp'], group['conditional_shared_id_bootstrap95_pp']))
    for i,(mean,(lo,hi)) in enumerate(points):
        axes[1].errorbar(mean, i, xerr=[[max(0.,mean-lo)],[max(0.,hi-mean)]], fmt='o', capsize=3, color='#3564a5' if i < 3 else '#222222')
    axes[1].axvline(0, color='gray', linewidth=1)
    axes[1].set(yticks=range(4), yticklabels=['seed 42','seed 43','seed 44','mean'],
                xlabel='Mixed - CE (percentage points)', title='95% conditional paired sample intervals')
    axes[1].invert_yaxis()
    figure.savefig(output/'comparison.png', dpi=180)
    figure.savefig(output/'comparison.svg')
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/laya-robust-loss-analysis-20261016-v1')
    analyze(parser.parse_args().output)
