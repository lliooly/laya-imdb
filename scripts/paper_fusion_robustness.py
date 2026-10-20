"""Local post-hoc uncertainty checks; refit all postprocessing inside each fold."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import ROOT, clean_text, digest, file_digest, read_jsonl, write_json
from comparison_metrics import fit_temperature, sigmoid
from laya_probability_fusion import fit_mixing_coefficient
from paper_fusion_diagnostics import STUDY, SOURCE, INVENTORY_SHA, POLICIES, align, fit_weight, score

PREVIOUS = ROOT/'reports/paper-fusion-diagnostics-20261020-v1'
DESIGN = ROOT/'docs/superpowers/specs/2026-10-20-fusion-weight-robustness-design.md'
SPLITS, PARAM_BOOTSTRAPS, SAMPLE_BOOTSTRAPS, SEED = 10, 300, 10000, 20261020


def grouped_folds(groups, seed, k=5):
    """Random group partition with greedy row-count balancing; no label access."""
    groups = np.asarray(groups)
    unique = sorted(set(groups.tolist()))
    if len(unique) < k:
        raise ValueError('Too few text groups')
    rng = np.random.default_rng(seed)
    bins = [[] for _ in range(k)]
    for g in rng.permutation(unique):
        idx = np.flatnonzero(groups == g)
        smallest = min(range(k), key=lambda j: len(bins[j]))
        bins[smallest].extend(idx.tolist())
    folds = [np.asarray(sorted(v), dtype=int) for v in bins]
    assert sorted(np.concatenate(folds).tolist()) == list(range(len(groups)))
    for test in folds:
        train = np.setdiff1d(np.arange(len(groups)), test)
        assert not set(groups[train]) & set(groups[test])
    return folds


def raw_arrays(blocks, native):
    output = {}
    for kind, block in blocks.items():
        def margin(values):
            z = np.asarray(values, dtype=float)
            if z.shape != (400, 2) or not np.isfinite(z).all():
                raise ValueError('Unexpected raw calibration logits')
            return z[:, 1]-z[:, 0]
        if kind == 'deberta':
            m = margin(block['logits'])
            output[kind] = dict(native=m, head=m, tail=m, truncated=np.asarray(block['truncated'], bool))
        else:
            if native[kind]['sample_ids'] != block['sample_ids'] or native[kind]['test_accessed']:
                raise ValueError('Native calibration does not align')
            output[kind] = dict(native=margin(native[kind]['logits']),
                head=margin(block['head_logits']), tail=margin(block['tail_logits']),
                truncated=np.asarray(block['truncated'], bool))
    return output


def fit_pipeline(data, labels, training_indices):
    """Labels outside training_indices cannot affect any fitted parameter."""
    train = np.asarray(training_indices, dtype=int)
    y = np.asarray(labels)[train]
    temperatures, p = {}, {}
    for kind, raw in data.items():
        t = fit_temperature(raw['native'][train], y, bounds=(.5, 5.))
        temperatures[kind] = t
        head, tail = sigmoid(raw['head']/t), sigmoid(raw['tail']/t)
        p[kind] = np.where(raw['truncated'], (head+tail)/2, head)
    binary = lambda x: np.column_stack((1-x, x))
    mixing = fit_mixing_coefficient(binary(p['baseline'][train]), binary(p['combined'][train]), y)['coefficient']
    laya = mixing*p['baseline']+(1-mixing)*p['combined']
    outer = fit_weight(p['deberta'][train], laya[train], y)
    parameters = dict(temperatures=temperatures, laya_coefficient=mixing, outer_deberta_weight=outer)
    return parameters, p['deberta'], laya


def cluster_interval(differences, groups, samples=SAMPLE_BOOTSTRAPS, seed=SEED):
    """One paired text-group draw shared across all fixed training repetitions."""
    differences, groups = np.asarray(differences), np.asarray(groups)
    if differences.ndim != 2 or differences.shape[1] != len(groups):
        raise ValueError('Differences must be [repetitions, aligned sample IDs]')
    unique = sorted(set(groups.tolist()))
    d = differences.mean(axis=0)
    sums = np.asarray([d[groups == g].sum() for g in unique])
    counts = np.asarray([(groups == g).sum() for g in unique])
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, samples, 100):
        selected = rng.integers(0, len(unique), size=(min(100, samples-start), len(unique)))
        draws.extend((100*sums[selected].sum(1)/counts[selected].sum(1)).tolist())
    return dict(mean_delta_pp=float(100*d.mean()),
        percentile_95_pp=np.quantile(draws, [.025, .975]).tolist(),
        bootstrap_samples=samples, unique_text_groups=len(unique), sample_count=len(groups),
        seed=seed, repetitions_averaged=differences.shape[0],
        fixed_models_and_postprocessing=True, includes_training_uncertainty=False,
        accounts_for_adaptive_research_selection=False, independent_confirmation=False), draws


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(ROOT/'reports'):
        raise ValueError('Use a new report directory')
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    inventory_path = SOURCE/STUDY/'final-report-files.sha256.json'
    if file_digest(inventory_path) != INVENTORY_SHA:
        raise ValueError('Source inventory changed')
    inventory = {r['path']: r for r in json.loads(inventory_path.read_text())['files']}
    inputs = {}
    def read(relative):
        path = SOURCE/relative
        h = file_digest(path)
        if h != inventory[relative]['sha256']:
            raise ValueError('Source prediction changed: '+relative)
        inputs[str(path.relative_to(ROOT))] = h
        return json.loads(path.read_text())
    prior_contract = json.loads((PREVIOUS/'input-contract.json').read_text())
    prior_verification = json.loads((PREVIOUS/'verification.json').read_text())
    for name in ('summary.json', 'weights-frozen-before-development.json'):
        if file_digest(PREVIOUS/name) != prior_verification['output_sha256'][name]:
            raise ValueError('Previous result changed')
        inputs[str((PREVIOUS/name).relative_to(ROOT))] = file_digest(PREVIOUS/name)
    previous = json.loads((PREVIOUS/'summary.json').read_text())
    frozen = read(STUDY+'/all-models-frozen.json')
    loaded = []
    for g in frozen['groups']:
        blocks = {k: read(f'{STUDY}-{k}-s{s}/calibration.json') for k, s in g['seeds'].items()}
        ids, y = align(blocks, 400)
        native = {k: read(f'{STUDY}-{k}-s{g["seeds"][k]}/native-calibration.json') for k in ('baseline', 'combined')}
        loaded.append((ids, y, raw_arrays(blocks, native)))
    ids, y = loaded[0][:2]
    assert all(v[0] == ids and np.array_equal(v[1], y) for v in loaded)
    split_manifest = ROOT/'reports/split_manifest.json'
    split = json.loads(split_manifest.read_text())
    texts = {}
    for name in ('train', 'validation'):
        path = ROOT/f'data/processed/{name}.jsonl'
        if file_digest(path) != split['jsonl_sha256'][name]:
            raise ValueError('Text data changed')
        inputs[str(path.relative_to(ROOT))] = file_digest(path)
        for row in read_jsonl(path):
            if name == 'validation' or row['id'] in set(ids):
                texts[row['id']] = row
    assert [texts[i]['label'] for i in ids] == y.tolist()
    cal_groups = np.asarray([digest(clean_text(texts[i]['text'])) for i in ids])
    partitions = [grouped_folds(cal_groups, SEED+i) for i in range(SPLITS)]
    fold_record = []
    for repeat, folds in enumerate(partitions):
        fold_record.append(dict(split_repeat=repeat+1, seed=SEED+repeat, folds=[dict(
            train_ids=[ids[i] for i in np.setdiff1d(np.arange(400), test)],
            heldout_ids=[ids[i] for i in test], heldout_label_counts=np.bincount(y[test], minlength=2).tolist()) for test in folds]))
    write_json(out/'grouped-folds.json', fold_record)
    cv = []
    bootstrap_parameters = []
    for group_number, (_, labels, data) in enumerate(loaded, 1):
        trial_results, fit_records = [], []
        for split_number, folds in enumerate(partitions, 1):
            predictions = {name: np.empty(400) for name in POLICIES}
            for fold_number, test in enumerate(folds, 1):
                train = np.setdiff1d(np.arange(400), test)
                params, d, l = fit_pipeline(data, labels, train)
                fitted = dict(deberta=1., equal=.5, fixed_deberta_080=.8, calibration_nll=params['outer_deberta_weight'])
                for name, weight in fitted.items():
                    predictions[name][test] = weight*d[test]+(1-weight)*l[test]
                fit_records.append(dict(split_repeat=split_number, fold=fold_number, parameters=params))
            scores = {name: score(labels, v) for name, v in predictions.items()}
            trial_results.append(dict(split_repeat=split_number, policies=scores,
                delta_pp={name: 100*(v['accuracy']-scores['deberta']['accuracy']) for name, v in scores.items()}))
        cv.append(dict(training_repeat=group_number, trials=trial_results, fold_fits=fit_records,
            statistics={name: dict(accuracy_mean=float(np.mean([r['policies'][name]['accuracy'] for r in trial_results])),
                delta_mean_pp=float(np.mean([r['delta_pp'][name] for r in trial_results])),
                delta_range_pp=[min(r['delta_pp'][name] for r in trial_results), max(r['delta_pp'][name] for r in trial_results)],
                nll_mean=float(np.mean([r['policies'][name]['nll'] for r in trial_results]))) for name in POLICIES}))
        group_rows = [np.flatnonzero(cal_groups == key) for key in sorted(set(cal_groups.tolist()))]
        rng = np.random.default_rng(SEED+group_number)
        parameter_samples = []
        for _ in range(PARAM_BOOTSTRAPS):
            chosen = rng.integers(0, len(group_rows), size=len(group_rows))
            indices = np.concatenate([group_rows[i] for i in chosen])
            parameters, _, _ = fit_pipeline(data, labels, indices)
            parameter_samples.append(parameters)
        values = [v['outer_deberta_weight'] for v in parameter_samples]
        bootstrap_parameters.append(dict(training_repeat=group_number, samples=parameter_samples,
            outer_deberta_weight=dict(median=float(np.median(values)), percentile_95=np.quantile(values, [.025, .975]).tolist(),
                frequency_at_zero=float(np.mean(np.asarray(values) == 0)), frequency_at_one=float(np.mean(np.asarray(values) == 1))),
            bootstrap_draws=PARAM_BOOTSTRAPS, unique_text_groups=len(group_rows)))
        print('Completed calibration checks for training repeat', group_number, flush=True)
    write_json(out/'calibration-cross-validation.json', cv)
    write_json(out/'calibration-parameter-bootstrap.json', bootstrap_parameters)
    # Only now read the previous development predictions, with parameters fixed.
    differences = {name: [] for name in POLICIES if name != 'deberta'}
    development_ids = None
    for prior_group in previous['results']:
        repeat = prior_group['repeat']
        pred = read(f'{STUDY}/repeat-{repeat}-predictions.json')
        dev_ids, labels = pred['sample_ids'], np.asarray(pred['true_labels'])
        if development_ids is not None and development_ids != dev_ids:
            raise ValueError('Development alignment changed')
        development_ids = dev_ids
        assert not set(dev_ids) & set(ids)
        assert [texts[i]['label'] for i in dev_ids] == labels.tolist()
        d, l = np.asarray(pred['positive_probabilities']['deberta']), np.asarray(pred['positive_probabilities']['laya'])
        for name in differences:
            w = prior_group['fit']['weights_deberta'][name]
            candidate = w*d+(1-w)*l
            difference = ((candidate > .5) == labels).astype(float)-((d > .5) == labels)
            assert round(difference.sum()) == prior_group['policies'][name]['versus_deberta']['net_correct']
            differences[name].append(difference)
    dev_groups = np.asarray([digest(clean_text(texts[i]['text'])) for i in development_ids])
    intervals, draws = {}, {}
    for name, delta in differences.items():
        intervals[name], draws[name] = cluster_interval(delta, dev_groups)
    write_json(out/'development-bootstrap-samples.json', draws)
    write_json(out/'development-conditional-intervals.json', intervals)
    summary = dict(state='completed', calibration_count=400, calibration_unique_text_groups=len(set(cal_groups.tolist())),
        development_count=2500, development_unique_text_groups=len(set(dev_groups.tolist())),
        calibration_cross_validation=[dict(training_repeat=x['training_repeat'], statistics=x['statistics']) for x in cv],
        calibration_parameter_sensitivity=[dict(training_repeat=x['training_repeat'], outer_deberta_weight=x['outer_deberta_weight']) for x in bootstrap_parameters],
        development_conditional_intervals=intervals, new_training=False, official_test_accessed=False,
        independent_confirmation=False, best_policy_selected=False, duration_seconds=time.monotonic()-began,
        limitations=['historically reused calibration and development data', 'analysis plan informed by previous development results',
            'CV folds refit postprocessing but not base models', 'repeated folds are not independent datasets',
            'conditional development intervals exclude training/fit/adaptive-selection uncertainty',
            'Laya temperature solver changed from LBFGS to convex inverse-temperature bisection with same NLL/bounds'])
    write_json(out/'summary.json', summary)
    write_json(out/'input-contract.json', dict(inputs_sha256=inputs, prior_input_contract_sha256=file_digest(PREVIOUS/'input-contract.json'),
        design_sha256=file_digest(DESIGN), script_sha256=file_digest(Path(__file__)),
        dependency_sha256={p: file_digest(ROOT/'scripts'/p) for p in ('paper_fusion_diagnostics.py','comparison_metrics.py','laya_probability_fusion.py')},
        parameters=dict(cv_splits=5, cv_repeats=SPLITS, parameter_bootstraps=PARAM_BOOTSTRAPS, development_bootstraps=SAMPLE_BOOTSTRAPS, seed=SEED)))
    lines = ['# 融合权重的离线稳健性检查', '', '事后探索；未新增训练或测试数据，未访问官方测试。', '',
        '## 开发集条件配对区间', '', '同一清洗文本组整体重采样，三组模型共享抽样；区间只固定这些模型和后处理参数，不覆盖研究选择偏差。', '',
        '| 方案 | 平均增益（百分点） | 95% 条件 bootstrap 区间 |', '|---|---:|---|']
    for name, record in intervals.items():
        lo, hi = record['percentile_95_pp']
        lines.append(f'| {name} | {record["mean_delta_pp"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines += ['', '## 校准集完整后处理交叉验证', '', '每折重新拟合温度、内部 Laya 系数和外层系数；同一文本不跨折。重复分折不增加独立样本数。', '',
        '| 训练重复 | 方案 | 平均留出 Accuracy | 相对 DeBERTa 平均变化（百分点） | 分折变化范围 |', '|---|---|---:|---:|---|']
    for g in cv:
        for name, v in g['statistics'].items():
            lo, hi = v['delta_range_pp']
            lines.append(f'| {g["training_repeat"]} | {name} | {100*v["accuracy_mean"]:.3f}% | {v["delta_mean_pp"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |')
    lines += ['', '## 校准样本扰动与外层权重', '', '| 训练重复 | DeBERTa 权重中位数 | 95% 扰动范围 | 权重到达1的比例 |', '|---|---:|---|---:|']
    for g in bootstrap_parameters:
        v = g['outer_deberta_weight']; lo, hi = v['percentile_95']
        lines.append(f'| {g["training_repeat"]} | {v["median"]:.4f} | [{lo:.4f}, {hi:.4f}] | {100*v["frequency_at_one"]:.1f}% |')
    lines += ['', '不能在这些结果上挑分折、种子或权重，不能写成独立泛化证明。完整限制和来源见 summary.json、input-contract.json。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
