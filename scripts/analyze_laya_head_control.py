"""Verify completed L2 receipts and analyze cached logits without new model evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

from common import ROOT, write_json
from comparison_metrics import metrics, sigmoid

STUDY = 'laya-head-control-cu126-20261015-v2'
SEEDS = (42, 43, 44)
HEADS = ('original', 'linear')
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20261015


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def name(seed, head):
    return f'laya-head-control-{head}-s{seed}-strict-fp32-20261015-v2'


def exact_p(a, b):
    n = a+b
    return min(1., 2*sum(math.comb(n, k) for k in range(min(a, b)+1))/2**n) if n else 1.


def analyze(output):
    report = ROOT/'reports'/STUDY
    remote = read(report/'summary.json')
    status = read(report/'status.json')
    gate = read(report/'all-models-frozen.json')
    contract_path = ROOT/'configs'/f'{STUDY}.json'
    contract = read(contract_path)
    receipt = read(ROOT/'artifacts/l2-setup-20261015-v1/completed-results-receipt.json')
    transport = read(ROOT/'artifacts/laya-head-control-completed-20261015-v2/verified.json')
    assert status['state'] == remote['state'] == 'completed'
    assert status['formal_models_frozen'] == status['formal_models_started'] == 6
    assert not status['test_accessed'] and transport['state'] == 'verified'
    partitions = []
    for relative, expected in receipt['files'].items():
        assert digest(ROOT/relative) == expected['sha256'], relative
    for relative, expected in contract['source_sha256'].items():
        assert digest(ROOT/relative) == expected, relative
    payloads, runs, initial_encoders = {}, {}, set()
    for seed, head in itertools.product(SEEDS, HEADS):
        run = name(seed, head)
        folder = ROOT/'reports'/run
        manifest, result, frozen = [read(folder/f) for f in ('manifest.json', 'result.json', 'frozen.json')]
        audit = read(folder/'training_audit.json')
        check = receipt['run_checks'][run]
        assert digest(folder/'frozen.json') == gate['frozen_sha256'][run]
        assert frozen['contract_sha256'] == manifest['contract_sha256'] == gate['contract_sha256'] == digest(contract_path)
        assert frozen['independent_reload_logits_bitwise_equal']
        assert len(manifest['gradient_ids']) == len(set(manifest['gradient_ids'])) == 22100
        assert len(manifest['calibration_ids']) == len(set(manifest['calibration_ids'])) == 400
        assert not set(manifest['gradient_ids']) & set(manifest['calibration_ids'])
        assert manifest['gradient_truncated'] == 3436
        partitions.append((manifest['gradient_ids'], manifest['calibration_ids']))
        assert datetime.fromisoformat(frozen['frozen_utc']) <= datetime.fromisoformat(gate['frozen_utc'])
        assert (audit['updates'], audit['gradient_exposures']) == (5526, 44200)
        assert (check['update_count'], check['actual_input_records']) == (5526, 44200)
        assert check['updates_consecutive'] and check['finite_losses_gradients']
        assert audit['input_stream_sha256'] == check['actual_inputs_uncompressed_sha256']
        for step in ('first_update', 'last_update'):
            assert all(check[step][part+'_gradient_norm_after_clipping'] > 0 for part in ('encoder', 'head'))
        assert result['checkpoint_state_sha256'] == frozen['checkpoint_metadata']['state_sha256'] == check['last_update']['model_sha256']
        assert all(result['initial_state_sha256'][part] != result['final_state_sha256'][part] for part in ('encoder', 'head'))
        initial_encoders.add(manifest['initial_state_sha256']['encoder'])
        reference, reloaded = [read(folder/f'calibration-{part}.json') for part in ('reference', 'reloaded')]
        assert reference['rows'] == reloaded['rows'] and np.array_equal(reference['logits'], reloaded['logits'])
        assert len(reference['rows']) == 400 and sum(r['truncated'] for r in reference['rows']) == 66
        assert digest(folder/'calibration-reloaded.json') == frozen['calibration_sha256']
        payload = read(folder/'validation.json')
        payloads[run] = payload
        runs[run] = {'seed': seed, 'head': head, 'parameters': manifest['parameter_count'],
                     'seconds_train_save_reference': result['seconds'],
                     'peak_allocated_gib': result['peak_allocated_bytes']/1024**3,
                     'peak_reserved_gib': result['peak_reserved_bytes']/1024**3,
                     'epoch_training_loss': result['losses'], 'temperature_fit': frozen['temperature_fit'],
                     'input_stream_sha256': audit['input_stream_sha256'],
                     'development': {k: payload[k] for k in ('raw_metrics', 'calibrated_metrics')}}
    assert len(initial_encoders) == 1
    assert all(p == partitions[0] for p in partitions)
    for seed in SEEDS:
        assert runs[name(seed, 'original')]['input_stream_sha256'] == runs[name(seed, 'linear')]['input_stream_sha256']
    rows = payloads[name(42, 'original')]['rows']
    assert all(p['rows'] == rows for p in payloads.values())
    y = np.array([r['true_label'] for r in rows])
    truncated = np.array([r['truncated'] for r in rows])
    assert len(rows) == len({r['sample_id'] for r in rows}) == 2500 and truncated.sum() == 407
    assert not {r['sample_id'] for r in rows} & set(partitions[0][0]+partitions[0][1])
    assert np.array_equal(np.bincount(y), [1250, 1250])
    masks = {'all': np.ones(len(y), dtype=bool), 'truncated': truncated, 'untruncated': ~truncated}
    correct, nll = {}, {}
    for run, payload in payloads.items():
        z = np.asarray(payload['logits'], dtype=np.float64)
        assert z.shape == (2500, 2) and np.isfinite(z).all()
        margin = z[:, 1]-z[:, 0]
        pred = (margin > 0).astype(int)
        correct[run] = pred == y
        for stage, temperature in [('raw', 1.), ('calibrated', payload['temperature'])]:
            p = sigmoid(margin/temperature)
            saved = np.asarray(payload[stage+'_probabilities'])
            assert np.allclose(saved[:, 1], p, atol=1e-14, rtol=0.)
            nll[run, stage] = np.logaddexp(0., np.where(y == 1, -margin, margin)/temperature)
            for group, keep in masks.items():
                actual = metrics(y[keep], margin[keep], temperature)
                expected = payload[stage+'_metrics'][group]
                for key in ('accuracy', 'nll', 'brier_positive', 'ece_15_equal_width_top_label', 'auroc'):
                    assert np.isclose(actual[key], expected[key], rtol=0., atol=1e-12), (run, stage, group, key)
    groups = {}
    for group, keep in masks.items():
        yy = y[keep]
        difference = np.column_stack([correct[name(s, 'original')][keep].astype(float)-correct[name(s, 'linear')][keep] for s in SEEDS])
        loss_delta = {stage: np.column_stack([nll[name(s, 'original'), stage][keep]-nll[name(s, 'linear'), stage][keep] for s in SEEDS]).mean(axis=1) for stage in ('raw', 'calibrated')}
        strata = [np.flatnonzero(yy == label) for label in (0, 1)]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws, loss_draws = np.empty((BOOTSTRAPS, 3)), {stage: np.empty(BOOTSTRAPS) for stage in loss_delta}
        for i in range(BOOTSTRAPS):
            ids = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in strata])
            draws[i] = 100*difference[ids].mean(axis=0)
            for stage in loss_delta:
                loss_draws[stage][i] = loss_delta[stage][ids].mean()
        per_seed = []
        for i, seed in enumerate(SEEDS):
            ca, cb = correct[name(seed, 'original')][keep], correct[name(seed, 'linear')][keep]
            a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
            ci = np.quantile(draws[:, i], [.025, .975]).tolist()
            entry = {'seed': seed, 'original_correct': int(ca.sum()), 'linear_correct': int(cb.sum()),
                     'original_only_correct': a_only, 'linear_only_correct': b_only,
                     'both_wrong': int((~ca & ~cb).sum()), 'both_correct': int((ca & cb).sum()),
                     'delta_pp': float(100*(ca.mean()-cb.mean())), 'conditional_bootstrap95_pp': ci,
                     'mcnemar_exact_nominal_p': exact_p(a_only, b_only)}
            old = remote['groups'][group]['per_seed'][i]
            assert a_only == old['candidate_only_correct'] and b_only == old['reference_only_correct']
            assert np.isclose(entry['mcnemar_exact_nominal_p'], old['mcnemar_exact_nominal_two_sided_p'])
            assert np.allclose(ci, old['paired_stratified_bootstrap95_pp'], atol=1e-12)
            per_seed.append(entry)
        mean_ci = np.quantile(draws.mean(axis=1), [.025, .975]).tolist()
        assert np.allclose(mean_ci, remote['groups'][group]['shared_id_stratified_bootstrap95_mean_delta_pp'], atol=1e-12)
        groups[group] = {**remote['groups'][group], 'independent_per_seed': per_seed,
                         'nll_original_minus_linear': {stage: {'mean': float(v.mean()),
                            'conditional_bootstrap95': np.quantile(loss_draws[stage], [.025, .975]).tolist(),
                            'posthoc_secondary_metric_interval': True} for stage, v in loss_delta.items()}}
    effects = np.array([p['delta_pp'] for p in groups['all']['independent_per_seed']])
    sign_distribution = np.array([np.mean(effects*np.array(signs)) for signs in itertools.product((-1, 1), repeat=3)])
    sign_p = float(np.mean(np.abs(sign_distribution) >= abs(effects.mean())-1e-12))
    costs = {head: {'mean_training_seconds': float(np.mean([runs[name(s, head)]['seconds_train_save_reference'] for s in SEEDS])),
                    'mean_peak_allocated_gib': float(np.mean([runs[name(s, head)]['peak_allocated_gib'] for s in SEEDS])),
                    'mean_peak_reserved_gib': float(np.mean([runs[name(s, head)]['peak_reserved_gib'] for s in SEEDS])),
                    'parameters': runs[name(42, head)]['parameters']} for head in HEADS}
    costs['linear_reduction_percent'] = {key: 100*(1-costs['linear'][key]/costs['original'][key])
        for key in ('mean_training_seconds', 'mean_peak_allocated_gib', 'parameters')}
    log = (ROOT/'artifacts/l2-setup-20261015-v1/study-v2-isolated-headers.log').read_text()
    starts = [(datetime.fromisoformat(m[1]), m[2], int(m[3]), m[4]) for m in re.finditer(
        r'(\S+) START .* --mode (train|reload|evaluate) --seed (\d+) --head-kind (original|linear)', log)]
    formal_start = next(t for t, mode, seed, head in starts if mode == 'train')
    evaluation_starts = [t for t, mode, _, _ in starts if mode == 'evaluate']
    assert len(evaluation_starts) == 6
    assert datetime.fromisoformat(gate['frozen_utc']) < min(evaluation_starts)
    finished = datetime.fromisoformat(status['finished_utc'])
    costs['formal_train_reload_evaluate_seconds'] = (finished-formal_start).total_seconds()
    costs['whole_admission_and_formal_seconds'] = status['seconds']
    storage = read(ROOT/'artifacts/laya-head-control-completed-20261015-v2/final-storage-status.json')
    costs['formal_checkpoint_bytes'] = sum(p['bytes'] for files in storage['checkpoint_files'].values() for p in files)
    costs['formal_checkpoint_file_count'] = sum(len(files) for files in storage['checkpoint_files'].values())
    costs['post_completion_gpu_snapshot'] = {k: storage[k] for k in ('utc', 'gpu', 'compute_pids')}
    costs['evaluation_process_seconds'] = {}
    for i, (t, mode, seed, head) in enumerate(starts):
        if mode != 'evaluate':continue
        end = starts[i+1][0] if i+1 < len(starts) else finished
        costs['evaluation_process_seconds'][name(seed, head)] = (end-t).total_seconds()
    wrong = np.column_stack([~correct[name(s, h)] for s, h in itertools.product(SEEDS, HEADS)])
    votes = wrong.sum(axis=1)
    error_overlap = {'posthoc_descriptive_only': True,
        'wrong_model_count_histogram': {str(k): int((votes == k).sum()) for k in range(7)},
        'all_six_wrong': int((votes == 6).sum()), 'at_least_one_wrong': int((votes > 0).sum()),
        'all_six_wrong_truncated': int(((votes == 6) & truncated).sum()),
        'all_six_wrong_labels': {str(k): int(((votes == 6) & (y == k)).sum()) for k in (0, 1)}}
    errors = [{'sample_id': r['sample_id'], 'true_label': r['true_label'], 'truncated': r['truncated'],
               'review_tokens': r['review_tokens'], 'wrong_model_count': int(votes[i]),
               'wrong_by_run': {name(s, h): bool(wrong[i, j]) for j, (s, h) in enumerate(itertools.product(SEEDS, HEADS))}}
              for i, r in enumerate(rows) if votes[i] > 0]
    output.mkdir(parents=True, exist_ok=False)
    summary = {'state': 'completed', 'study_id': STUDY, 'analysis_utc': datetime.now(timezone.utc).isoformat(),
        'audit': {'transport_sha256': transport['transport']['sha256'], 'receipt_files_verified': len(receipt['files']),
                  'bound_source_files_verified': len(contract['source_sha256']), 'six_reload_checks_exact': True,
                  'three_paired_input_streams_exact': True, 'all_model_gate_precedes_development': True,
                  'gradient_count': 22100, 'calibration_count': 400, 'development_count': 2500,
                  'truncated_counts': {'gradient': 3436, 'calibration': 66, 'development': 407},
                  'formal_updates': 33156, 'formal_gradient_exposures': 265200,
                  'weights_downloaded': False, 'full_input_or_update_journals_downloaded': False},
        'groups': groups, 'runs': runs, 'costs': costs, 'error_overlap': error_overlap,
        'seed_sign_flip_nominal_two_sided_p': sign_p,
        'inference_limits': {'sample_ci_conditions_on_six_fitted_models': True, 'seed_count': 3,
            'seed_population_interval_not_estimated': True, 'development_historically_reused': True,
            'equivalence_not_established': True, 'no_test_access_in_this_run': True,
            'no_new_forward_passes_in_analysis': True, 'no_ensemble_candidate_created': True,
            'legacy_local_development_access_deviation': contract['local_legacy_input_regression_development_access']}}
    write_json(output/'summary.json', summary)
    write_json(output/'error-audit.json', {'scope': 'development; descriptive review queue, not new training data', 'rows': errors})
    plot(summary, output)
    print(json.dumps({'output': str(output), 'groups': {g: {'delta': x['mean_original_minus_linear_pp'],
        'ci': x['shared_id_stratified_bootstrap95_mean_delta_pp'], 'nll': x['nll_original_minus_linear']} for g, x in groups.items()},
        'costs': costs, 'errors': error_overlap, 'seed_sign_flip_p': sign_p}, indent=2))


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    colors = {'original': '#3564a5', 'linear': '#d07824'}
    for head in HEADS:
        acc = [summary['runs'][name(s, head)]['development']['raw_metrics']['all']['accuracy']*100 for s in SEEDS]
        axes[0].plot(SEEDS, acc, 'o-', color=colors[head], label=head)
    axes[0].set(xticks=SEEDS, xlabel='Training seed', ylabel='Development accuracy (%)', title='Paired accuracy; same 2,500 items')
    axes[0].legend(frameon=False)
    entries = summary['groups']['all']['independent_per_seed']
    means = [p['delta_pp'] for p in entries]+[summary['groups']['all']['mean_original_minus_linear_pp']]
    intervals = [p['conditional_bootstrap95_pp'] for p in entries]+[summary['groups']['all']['shared_id_stratified_bootstrap95_mean_delta_pp']]
    for i, (mean, (lo, hi)) in enumerate(zip(means, intervals, strict=True)):
        axes[1].errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='o', capsize=3, color='#3564a5' if i < 3 else '#222222')
    axes[1].axvline(0, color='gray', linewidth=1)
    axes[1].set(yticks=range(4), yticklabels=['seed 42', 'seed 43', 'seed 44', 'mean'],
                xlabel='Original - linear (percentage points)', title='95% paired sample bootstrap intervals')
    axes[1].invert_yaxis()
    relative = [summary['costs']['linear']['mean_training_seconds']/summary['costs']['original']['mean_training_seconds'],
                summary['costs']['linear']['mean_peak_allocated_gib']/summary['costs']['original']['mean_peak_allocated_gib'],
                summary['costs']['linear']['parameters']/summary['costs']['original']['parameters']]
    axes[2].barh(['Training process time', 'Peak allocated VRAM', 'Parameters'], relative, color=colors['linear'])
    axes[2].axvline(1, color='gray', linewidth=1)
    axes[2].set(xlim=(0, 1.16), xlabel='Linear / original (original = 100%)', title='Observed resource cost')
    axes[2].xaxis.set_major_formatter(PercentFormatter(1))
    for i, value in enumerate(relative):axes[2].text(value+.015, i, f'{100*value:.1f}%', va='center')
    figure.suptitle('Laya L2 head control: no stable accuracy advantage detected', fontsize=13)
    figure.savefig(output/'comparison.png', dpi=180)
    figure.savefig(output/'comparison.svg')
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/laya-head-control-analysis-20261015-v1')
    analyze(parser.parse_args().output)
