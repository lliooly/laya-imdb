"""Describe frozen development predictions; no fitting, inference or test access.

Bootstrap draws are paired by sample ID. Repeating each method's independently
confidence-sorted observations by draw multiplicity is exactly the same as
sorting that bootstrap multiset (ties use original source order).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform

import numpy as np

from laya_probability_fusion import apply_gated_mixture


ROOT = Path(__file__).resolve().parents[1]
RUN = 'laya-risk-coverage-20261015-v1'
OUT = ROOT / 'reports' / RUN
FUSION = 'reports/laya-probability-fusion-s43-s42-20261014-v1'
MODELS = ('baseline-s43', 'combined-s42')
METHODS = ('single', 'fusion')
COVERAGES = (1., .9, .8, .7, .5)
GRID = np.arange(1, 101) / 100
REPLICATES = 20000
SEED = 20261015


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def persist(name, content):
    path = OUT / name
    data = content if isinstance(content, bytes) else content.encode('utf-8')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite different output: {path}')
    else:
        path.write_bytes(data)


def save_json(name, value):
    persist(name, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save_csv(name, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    persist(name, stream.getvalue())


def wilson(correct, count):
    if not count:
        return [None, None]
    z = 1.959963984540054
    p = correct / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    half = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [max(0., center-half), min(1., center+half)]


def measure(mask, correct):
    n = int(mask.sum())
    hits = int(correct[mask].sum())
    return {'count': n, 'correct': hits, 'errors': n-hits,
            'accuracy': hits/n if n else None, 'accuracy_wilson95': wilson(hits, n)}


def load_inputs():
    historical = read(f'{FUSION}/protocol.json')
    expected = historical['source_sha256']
    observed = {}

    def admitted(path, historical_hash=True):
        sha = digest(ROOT / path)
        if historical_hash and sha != expected[path]:
            raise ValueError(f'Historical source hash differs: {path}')
        observed[path] = sha
        return read(path)

    frozen = admitted(f'{FUSION}/frozen-fit.json', False)
    previous = admitted(f'{FUSION}/summary.json', False)
    if observed[f'{FUSION}/frozen-fit.json'] != previous['frozen_fit_sha256']:
        raise ValueError('Frozen fit hash differs from historical summary')
    if frozen['primary_models'] != list(MODELS) or frozen['validation_used_for_fitting']:
        raise ValueError('Unexpected primary models or fitting split')
    observed[f'{FUSION}/protocol.json'] = digest(ROOT / FUSION / 'protocol.json')
    split = admitted('reports/split_manifest.json')
    manifest = admitted('reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json')
    ids = {'calibration': manifest['calibration_ids'], 'validation': split['ids']['validation']}
    if len(ids['calibration']) != 400 or len(ids['validation']) != 2500:
        raise ValueError('Unexpected split size')
    if set(ids['calibration']) & set(ids['validation']):
        raise ValueError('Calibration and development overlap')
    result = {}
    for split_name in ('calibration', 'validation'):
        blocks = {}
        reference = None
        for model in MODELS:
            meta = admitted(f'reports/laya-posthoc-{model}-fp32-v1/metadata.json')
            if meta['test_accessed'] or meta['split_counts'] != {'calibration': 400, 'validation': 2500}:
                raise ValueError('Unexpected export metadata')
            for rule in ('head', 'head_tail'):
                path = f'reports/laya-posthoc-{model}-fp32-v1/{split_name}-{rule}.json'
                block = admitted(path)
                rows = block['rows']
                if block['split'] != split_name or block['test_accessed']:
                    raise ValueError('Unexpected export split')
                if [r['sample_id'] for r in rows] != ids[split_name]:
                    raise ValueError('Source ID order differs from manifest')
                labels = np.array([r['true_label'] for r in rows])
                truncated = np.array([r['truncated'] for r in rows], dtype=bool)
                probability = np.array([r['probabilities'] for r in rows], dtype=float)
                if (probability.shape != (len(rows), 2) or not np.isfinite(probability).all()
                        or not np.isin(labels, [0, 1]).all() or (probability < 0).any()
                        or (probability > 1).any()
                        or not np.allclose(probability.sum(axis=1), 1, atol=1e-14, rtol=0)):
                    raise ValueError('Invalid probabilities or labels')
                logits = np.array([r['logits'] for r in rows]) / meta['native_temperature']
                exp = np.exp(logits-logits.max(axis=1, keepdims=True))
                if not np.allclose(probability, exp/exp.sum(axis=1, keepdims=True), atol=1e-12, rtol=0):
                    raise ValueError('Probability differs from temperature-scaled logits')
                if reference is not None:
                    if not np.array_equal(labels, reference[0]) or not np.array_equal(truncated, reference[1]):
                        raise ValueError('Labels or truncation masks differ')
                reference = (labels, truncated)
                blocks[model, rule] = probability
        labels, truncated = reference
        mixture = apply_gated_mixture(*(blocks[m, r] for m in MODELS for r in ('head', 'head_tail')),
                                      truncated, frozen['mixture']['coefficient'])
        if split_name == 'validation':
            cached = admitted(f'{FUSION}/validation-fused.json', False)
            if cached['split'] != 'validation' or cached['test_accessed']:
                raise ValueError('Unexpected frozen fusion split')
            if [r['sample_id'] for r in cached['rows']] != ids[split_name]:
                raise ValueError('Frozen fusion ID mismatch')
            for r, y, t in zip(cached['rows'], labels, truncated, strict=True):
                if r['true_label'] != y or r['truncated'] != t:
                    raise ValueError('Frozen fusion metadata mismatch')
            if not np.array_equal(mixture, np.array([r['probabilities'] for r in cached['rows']])):
                raise ValueError('Fusion replay differs from frozen cached probabilities')
        result[split_name] = {'labels': labels, 'truncated': truncated, 'ids': ids[split_name],
                             'single': blocks['baseline-s43', 'head_tail'], 'fusion': mixture}
    observed['uv.lock'] = digest(ROOT / 'uv.lock')
    for name in ('run_laya_risk_coverage.py', 'laya_probability_fusion.py', 'laya_decision_views.py'):
        observed[f'scripts/{name}'] = digest(ROOT / 'scripts' / name)
    return result, observed, frozen['mixture']['coefficient']


def prepare(block):
    result = {}
    for name in METHODS:
        prob = block[name]
        confidence = prob.max(axis=1)
        correct = prob.argmax(axis=1) == block['labels']
        order = np.argsort(-confidence, kind='stable')
        prefix = np.cumsum(~correct[order])
        risk = prefix / np.arange(1, len(order)+1)
        result[name] = {'confidence': confidence, 'correct': correct, 'order': order,
                        'risk': risk, 'aurc': float(risk.mean())}
    return result


def bootstrap(prepared, labels, stratified, seed):
    rng = np.random.default_rng(seed)
    n = len(labels)
    k = np.rint(GRID*n).astype(int)
    curves = np.empty((REPLICATES, 2, len(GRID)))
    areas = np.empty((REPLICATES, 2))
    groups = [np.flatnonzero(labels == y) for y in (0, 1)]
    denominator = np.arange(1, n+1)
    for b in range(REPLICATES):
        if stratified:
            indices = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups])
        else:
            indices = rng.integers(n, size=n)
        counts = np.bincount(indices, minlength=n)
        for j, name in enumerate(METHODS):
            method = prepared[name]
            order = method['order']
            # Each method independently reorders the SAME sampled multiset.
            errors = np.repeat(~method['correct'][order], counts[order])
            risk = np.cumsum(errors) / denominator
            curves[b, j] = 1-risk[k-1]
            areas[b, j] = risk.mean()
    ci = lambda a: np.quantile(a, [.025, .975], axis=0).T.tolist()
    delta = curves[:, 1]-curves[:, 0]
    result = {'replicates': REPLICATES, 'seed': seed, 'stratified_by_true_label': stratified,
              'interval': 'two-sided percentile 95%; pointwise; conditional on frozen models and fit',
              'aurc': {name: ci(areas[:, j]) for j, name in enumerate(METHODS)},
              'delta_aurc_fusion_minus_single': ci(areas[:, 1]-areas[:, 0]),
              'grid': [{'coverage': float(c),
                        'single_accuracy_ci95': ci(curves[:, 0])[i],
                        'fusion_accuracy_ci95': ci(curves[:, 1])[i],
                        'delta_accuracy_ci95_pp': (100*np.quantile(delta[:, i], [.025, .975])).tolist()}
                       for i, c in enumerate(GRID)]}
    return result


def make_plot(prepared, bootstrap_result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': RUN})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    x = np.arange(1, 2501)/25
    for name, color, label in [('single', '#667085', 'Single: baseline-s43 head_tail'),
                                ('fusion', '#176B87', 'Frozen two-model fusion')]:
        axes[0].plot(x, 100*prepared[name]['risk'], label=label, color=color, linewidth=1.7)
        band = np.array([row[f'{name}_accuracy_ci95'] for row in bootstrap_result['grid']])
        axes[0].fill_between(GRID*100, 100*(1-band[:, 1]), 100*(1-band[:, 0]), color=color, alpha=.12)
    delta = 100*(prepared['single']['risk']-prepared['fusion']['risk'])
    axes[1].plot(x, delta, color='#176B87', linewidth=1.7)
    band = np.array([row['delta_accuracy_ci95_pp'] for row in bootstrap_result['grid']])
    axes[1].fill_between(GRID*100, band[:, 0], band[:, 1], color='#176B87', alpha=.18)
    axes[1].axhline(0, color='#667085', linewidth=.8, linestyle='--')
    axes[0].set(ylabel='Selective error rate (%)', title='Risk vs coverage; lower is better', ylim=(0, 5))
    axes[1].set(ylabel='Fusion minus single accuracy (pp)', title='Paired difference; higher is better')
    axes[0].legend(loc='upper left', fontsize=8)
    for ax in axes:
        ax.set(xlabel='Coverage (%)', xlim=(50, 100))
        ax.grid(alpha=.2)
    fig.suptitle('Development set, n=2,500 | 20,000 paired stratified bootstrap draws\n'
                 'Shading: pointwise 95% intervals, frozen models and calibration fit', fontsize=11)
    for extension in ('png', 'svg'):
        buffer = io.BytesIO()
        metadata = {'Date': None} if extension == 'svg' else {}
        fig.savefig(buffer, format=extension, dpi=180, metadata=metadata)
        persist(f'risk-coverage.{extension}', buffer.getvalue())
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs, hashes, coefficient = load_inputs()
    protocol = {'run_id': RUN, 'analysis_date': '2026-10-15', 'sources_sha256': hashes,
                'methods': {'single': 'baseline-s43/head_tail', 'fusion': 'frozen two_model_fitted'},
                'frozen_mixing_coefficient': coefficient, 'coverage_points': list(COVERAGES),
                'bootstrap_grid': GRID.tolist(), 'replicates_per_design': REPLICATES,
                'primary_design': 'paired class-stratified bootstrap, seed 20261015',
                'sensitivity_design': 'paired ordinary IID bootstrap, seed 20261016',
                'tie_break': 'original source order; duplicate draws use sample multiplicities',
                'threshold_rule': 'calibration-only kth largest max-probability, inclusive >=; 100% uses 0.5',
                'uncertainty_excluded': ['training seeds', 'fitting/calibration uncertainty',
                                         'development selection/reuse', 'distribution shift'],
                'preregistered': False, 'coverage_points_seen_in_previous_development_analysis': True,
                'official_test_predictions_or_text_read': False, 'model_forward_passes': 0,
                'new_fitting': False, 'python': platform.python_version(), 'numpy': np.__version__}
    # Persist the analysis contract before computing bootstrap intervals.
    save_json('protocol.json', protocol)
    dev = inputs['validation']
    prepared = prepare(dev)
    primary = bootstrap(prepared, dev['labels'], True, SEED)
    sensitivity = bootstrap(prepared, dev['labels'], False, SEED+1)
    save_json('bootstrap.json', {'primary': primary, 'sensitivity': sensitivity})
    grid_lookup = {round(row['coverage'], 2): row for row in primary['grid']}
    n = len(dev['labels'])
    points, full_curve, strata, thresholds = [], [], [], []
    for k in range(1, n+1):
        full_curve.append({'retained': k, 'coverage': k/n,
                           'single_risk': float(prepared['single']['risk'][k-1]),
                           'fusion_risk': float(prepared['fusion']['risk'][k-1]),
                           'delta_accuracy_pp': float(100*(prepared['single']['risk'][k-1]-prepared['fusion']['risk'][k-1]))})
    for c in COVERAGES:
        k = round(c*n)
        masks = {}
        row = {'coverage': c, 'retained': k}
        for name in METHODS:
            method = prepared[name]
            mask = np.zeros(n, dtype=bool)
            mask[method['order'][:k]] = True
            masks[name] = mask
            row[name] = measure(mask, method['correct'])
            row[name]['accuracy_paired_bootstrap95'] = grid_lookup[c][f'{name}_accuracy_ci95']
            row[name]['cutoff_confidence_development_descriptive_only'] = float(method['confidence'][method['order'][k-1]])
            for group, group_mask in [('truncated', dev['truncated']), ('untruncated', ~dev['truncated']),
                                      ('negative', dev['labels'] == 0), ('positive', dev['labels'] == 1)]:
                metrics = measure(mask & group_mask, method['correct'])
                strata.append({'coverage': c, 'method': name, 'group': group,
                               'group_size': int(group_mask.sum()), 'retained_fraction_of_group': metrics['count']/int(group_mask.sum()),
                               **metrics})
        row['delta_accuracy_pp'] = 100*(row['fusion']['accuracy']-row['single']['accuracy'])
        row['delta_accuracy_ci95_pp'] = grid_lookup[c]['delta_accuracy_ci95_pp']
        row['retained_intersection'] = int((masks['single'] & masks['fusion']).sum())
        points.append(row)
    cal = prepare(inputs['calibration'])
    for c in COVERAGES:
        for name in METHODS:
            threshold = .5 if c == 1 else float(cal[name]['confidence'][cal[name]['order'][round(c*400)-1]])
            cm = cal[name]['confidence'] >= threshold
            dm = prepared[name]['confidence'] >= threshold
            metrics = measure(dm, prepared[name]['correct'])
            thresholds.append({'calibration_target_coverage': c, 'method': name,
                               'threshold': threshold, 'calibration_retained': int(cm.sum()),
                               'calibration_size': 400, 'development_coverage': metrics['count']/n,
                               'development_coverage_wilson95': wilson(metrics['count'], n), **metrics})
    difference = prepared['fusion']['risk']-prepared['single']['risk']
    worse = np.flatnonzero(difference > 1e-15)
    summary = {'run_id': RUN, 'development_count': n,
               'development_class_counts': np.bincount(dev['labels']).tolist(),
               'truncated_count': int(dev['truncated'].sum()),
               'exploratory_development_only': True, 'official_test_predictions_or_text_read': False,
               'new_fitting': False, 'new_inference': False,
               'coverage_points': points,
               'aurc': {name: {'estimate': prepared[name]['aurc'], 'paired_stratified_bootstrap95': primary['aurc'][name],
                               'paired_iid_bootstrap95': sensitivity['aurc'][name]} for name in METHODS},
               'delta_aurc_fusion_minus_single': {'estimate': prepared['fusion']['aurc']-prepared['single']['aurc'],
                                                  'paired_stratified_bootstrap95': primary['delta_aurc_fusion_minus_single'],
                                                  'paired_iid_bootstrap95': sensitivity['delta_aurc_fusion_minus_single']},
               'fusion_worse_coverage_points': len(worse),
               'first_worse_retained_count': int(worse[0]+1) if len(worse) else None,
               'maximum_fusion_accuracy_deficit_pp': float(100*difference.max()),
               'strata_at_global_coverage': strata, 'calibration_only_thresholds': thresholds,
               'bootstrap': {'replicates_per_design': REPLICATES, 'primary_seed': SEED, 'sensitivity_seed': SEED+1,
                             'primary': 'paired class-stratified percentile 95%, pointwise',
                             'sensitivity_coverage_points': [row for row in sensitivity['grid'] if round(row['coverage'], 2) in COVERAGES]},
               'limitations': protocol['uncertainty_excluded'], 'protocol_sha256': digest(OUT/'protocol.json')}
    save_csv('full-curve.csv', full_curve)
    save_json('summary.json', summary)
    save_json('strata.json', strata)
    save_json('calibration-thresholds.json', thresholds)
    make_plot(prepared, primary)
    artifacts = {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('REPORT.md', 'verification.json')}
    save_json('verification.json', {'source_hashes_verified': True, 'fusion_replay_bitwise_equal': True,
                                   'id_label_truncation_alignment_verified': True, 'artifacts_sha256': artifacts})
    print(json.dumps({'run_id': RUN, 'coverage_points': [{k: p[k] for k in ('coverage', 'delta_accuracy_pp', 'delta_accuracy_ci95_pp')} for p in points],
                      'aurc': summary['aurc'], 'delta_aurc': summary['delta_aurc_fusion_minus_single']}, indent=2))


if __name__ == '__main__':
    main()
