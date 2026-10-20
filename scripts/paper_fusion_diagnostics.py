"""Offline, post-hoc diagnostics; calibration-only fits and no test/model access."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from common import ROOT, file_digest, read_jsonl, write_json
from comparison_metrics import metrics, paired, sigmoid
from laya_decision_views import probability_margin

STUDY = 'paper-gpu-supplement-20261019-v1'
SOURCE = ROOT/'artifacts'/STUDY/'received-formal-v2'/'reports'
INVENTORY_SHA = '603c9e30b966c977431016e8ba11e08c1e0c58d501407650355dbf9723963d6b'
DESIGN = ROOT/'docs/superpowers/specs/2026-10-20-fusion-error-diagnostics-design.md'
POLICIES = ('deberta', 'equal', 'fixed_deberta_080', 'calibration_nll')


def align(blocks, count):
    reference = next(iter(blocks.values()))
    ids, labels = reference['sample_ids'], reference['true_labels']
    if len(ids) != count or len(set(ids)) != count or len(labels) != count:
        raise ValueError('Unexpected ID inventory')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Nonbinary label')
    for d in blocks.values():
        if d['sample_ids'] != ids or d['true_labels'] != labels or d['test_accessed']:
            raise ValueError('IDs/labels differ, or test was accessed')
        p = np.asarray(d['positive_probabilities'], dtype=float)
        if p.shape != (count,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid prediction probabilities')
    return ids, np.asarray(labels, dtype=int)


def fit_weight(deberta, laya, labels):
    """Minimize calibration NLL of w*DeBERTa+(1-w)*Laya, w in [0,1]."""
    a, b, y = map(np.asarray, (deberta, laya, labels))
    if a.shape != b.shape or a.shape != y.shape or a.ndim != 1 or not len(a):
        raise ValueError('Calibration arrays must align')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid calibration input')
    if np.any((a < 0) | (a > 1) | (b < 0) | (b > 1)):
        raise ValueError('Invalid calibration probabilities')
    a, b = np.clip(a, 1e-12, 1-1e-12), np.clip(b, 1e-12, 1-1e-12)
    if np.array_equal(a, b):
        return 1.
    def derivative(w):
        p = w*a+(1-w)*b
        return float(np.mean((p-y)*(a-b)/(p*(1-p))))
    if derivative(0.) >= 0:
        return 0.
    if derivative(1.) <= 0:
        return 1.
    lo, hi = 0., 1.
    for _ in range(90):
        mid = (lo+hi)/2
        if derivative(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2


def score(y, p):
    return metrics(y, probability_margin(np.column_stack((1-p, p))))


def changes(y, reference, candidate):
    a, b = reference > .5, candidate > .5
    good_a, good_b = a == y, b == y
    repaired, harmed = int((~good_a & good_b).sum()), int((good_a & ~good_b).sum())
    assert repaired-harmed == int(good_b.sum()-good_a.sum())
    return dict(repaired=repaired, harmed=harmed, net_correct=repaired-harmed,
                changed=int((a != b).sum()),
                mcnemar_exact_two_sided_p=paired(y, reference-.5, candidate-.5)['mcnemar_exact_two_sided_p'])


def features(blocks, texts, ids, p):
    d = np.maximum(p['deberta'], 1-p['deberta'])
    result = dict(label=np.where(np.asarray(blocks['deberta']['true_labels']) == 1, 'positive', 'negative'),
                  character_length=np.asarray(['<=1000' if len(texts[i]['text']) <= 1000 else
                      '1001-3000' if len(texts[i]['text']) <= 3000 else '>3000' for i in ids]),
                  deberta_confidence=np.where(d < .7, '<0.7', np.where(d < .9, '0.7-0.9', '>=0.9')),
                  laya_models_disagree=(p['baseline'] > .5) != (p['combined'] > .5))
    for kind, block in blocks.items():
        result[kind+'_truncated'] = np.asarray(block['truncated'], dtype=bool)
        if kind != 'deberta':
            head = np.asarray(block['head_logits']); tail = np.asarray(block['tail_logits'])
            result[kind+'_views_disagree'] = (head[:, 1] > head[:, 0]) != (tail[:, 1] > tail[:, 0])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(ROOT/'reports'):
        raise ValueError('Output must be a new reports directory')
    out.mkdir(parents=True, exist_ok=False)
    inventory = SOURCE/STUDY/'final-report-files.sha256.json'
    if file_digest(inventory) != INVENTORY_SHA:
        raise ValueError('Remote report inventory changed')
    expected = {r['path']: r for r in json.loads(inventory.read_text())['files']}
    inputs = {}
    def read(relative):
        path = SOURCE/relative
        e = expected[relative]
        h = file_digest(path)
        if h != e['sha256'] or path.stat().st_size != e['bytes']:
            raise ValueError('Source differs: '+relative)
        inputs[str(path.relative_to(ROOT))] = h
        return json.loads(path.read_text())
    frozen = read(STUDY+'/all-models-frozen.json')
    weights = []
    # No development prediction or development label is read before these fits.
    calibration_ids = None
    for group in frozen['groups']:
        cal = {k: read(f'{STUDY}-{k}-s{s}/calibration.json') for k, s in group['seeds'].items()}
        ids, y = align(cal, 400)
        if calibration_ids is not None and ids != calibration_ids:
            raise ValueError('Calibration IDs changed across repetitions')
        calibration_ids = ids
        w = group['laya_mixture']['coefficient']
        d = np.asarray(cal['deberta']['positive_probabilities'])
        l = w*np.asarray(cal['baseline']['positive_probabilities'])+(1-w)*np.asarray(cal['combined']['positive_probabilities'])
        fitted = fit_weight(d, l, y)
        fit = dict(repeat=group['repeat'], seeds=group['seeds'], laya_coefficient=w,
                   weights_deberta=dict(deberta=1., equal=.5, fixed_deberta_080=.8, calibration_nll=fitted),
                   calibration_nll={name: score(y, v*d+(1-v)*l)['nll'] for name, v in
                       dict(deberta=1., equal=.5, fixed_deberta_080=.8, calibration_nll=fitted).items()})
        weights.append(fit)
    write_json(out/'weights-frozen-before-development.json', dict(groups=weights,
        calibration_count=400, objective='unpenalized convex probability-mixture NLL',
        bounds=[0, 1], tie_policy='identical predictions select DeBERTa',
        development_used_for_weight_fit=False, post_hoc_exploratory=True, source_inputs_sha256=inputs.copy()))
    weight_sha = file_digest(out/'weights-frozen-before-development.json')
    validation = ROOT/'data/processed/validation.jsonl'
    split = json.loads((ROOT/'reports/split_manifest.json').read_text())
    if file_digest(validation) != split['jsonl_sha256']['validation']:
        raise ValueError('Original development text changed')
    texts = {r['id']: r for r in read_jsonl(validation)}
    inputs[str(validation.relative_to(ROOT))] = file_digest(validation)
    original = read(STUDY+'/summary.json')
    results, strata, cases = [], [], []
    for group, fit in zip(frozen['groups'], weights):
        blocks = {k: read(f'{STUDY}-{k}-s{s}/development.json') for k, s in group['seeds'].items()}
        ids, y = align(blocks, 2500)
        if set(ids) != set(texts) or set(ids) & set(calibration_ids):
            raise ValueError('Development population differs or overlaps calibration')
        if [texts[i]['label'] for i in ids] != y.tolist():
            raise ValueError('Text and prediction labels differ')
        p = {k: np.asarray(v['positive_probabilities']) for k, v in blocks.items()}
        w = fit['laya_coefficient']
        p['laya'] = w*p['baseline']+(1-w)*p['combined']
        candidates = {name: v*p['deberta']+(1-v)*p['laya'] for name, v in fit['weights_deberta'].items()}
        for name, expected_name in [('deberta', 'deberta'), ('equal', 'fusion')]:
            assert score(y, candidates[name])['correct'] == original['results'][group['repeat']-1]['metrics'][expected_name]['correct']
        features_by_name = features(blocks, texts, ids, p)
        cd, cl = (p['deberta'] > .5) == y, (p['laya'] > .5) == y
        pair = dict(deberta_wrong_laya_correct=int((~cd & cl).sum()),
                    deberta_correct_laya_wrong=int((cd & ~cl).sum()),
                    both_wrong=int((~cd & ~cl).sum()), both_correct=int((cd & cl).sum()),
                    oracle_correct_if_true_label_known=int((cd | cl).sum()), oracle_is_deployable=False)
        results.append(dict(repeat=group['repeat'], fit=fit, branch_complementarity=pair,
            policies={name: dict(metrics=score(y, v), versus_deberta=changes(y, p['deberta'], v))
                      for name, v in candidates.items()}))
        changed = (candidates['equal'] > .5) != (p['deberta'] > .5)
        assert not np.any(changed & ((p['deberta'] > .5) == (p['laya'] > .5)))
        for name, values in features_by_name.items():
            for level in sorted(set(values.tolist()), key=str):
                mask = values == level
                strata.append(dict(repeat=group['repeat'], feature=name, level=str(level), count=int(mask.sum()),
                    deberta_accuracy=float(cd[mask].mean()),
                    **changes(y[mask], p['deberta'][mask], candidates['equal'][mask])))
        for i in np.flatnonzero(changed):
            d, l = float(p['deberta'][i]), float(p['laya'][i])
            cases.append(dict(repeat=group['repeat'], sample_id=ids[i], true_label=int(y[i]),
                effect='harmed' if cd[i] else 'repaired',
                positive_probabilities={k: float(v[i]) for k, v in p.items()}, equal_probability=float(candidates['equal'][i]),
                critical_deberta_weight=(.5-l)/(d-l),
                deberta_confidence=max(d, 1-d), laya_confidence=max(l, 1-l),
                features={k: str(v[i]) for k, v in features_by_name.items()},
                text=texts[ids[i]]['text'], excerpt=texts[ids[i]]['text'][:600]))
    statistics = {}
    for name in POLICIES:
        acc = [r['policies'][name]['metrics']['accuracy'] for r in results]
        statistics[name] = dict(accuracy_mean=float(np.mean(acc)), accuracy_sample_sd=float(np.std(acc, ddof=1)),
            nll_mean=float(np.mean([r['policies'][name]['metrics']['nll'] for r in results])),
            delta_pp=[100*(r['policies'][name]['metrics']['accuracy']-r['policies']['deberta']['metrics']['accuracy']) for r in results])
    write_json(out/'changed-cases.json', cases)
    write_json(out/'strata.json', strata)
    write_json(out/'summary.json', dict(state='completed', results=results, statistics=statistics,
        unique_changed_ids=len({r['sample_id'] for r in cases}), changed_case_count=len(cases),
        independent_sample_count=2500, repetitions=3, calibration_count=400,
        post_hoc_exploratory=True, independent_confirmation=False, development_used_for_weight_fit=False,
        best_policy_selected=False, official_test_accessed=False, new_training=False,
        limitations=['previously reused development and calibration sets', 'post-hoc diagnostics after seed results were known',
                     'three repetitions share development IDs', 'NLL fitting does not optimize classification accuracy',
                     'subgroup descriptions do not establish causality', 'no deployment rule fitted on development labels']))
    write_json(out/'input-contract.json', dict(source_report_inventory_sha256=INVENTORY_SHA,
        inputs_sha256=inputs, weights_sha256=weight_sha, design_sha256=file_digest(DESIGN),
        script_sha256=file_digest(Path(__file__)), metrics_script_sha256=file_digest(ROOT/'scripts/comparison_metrics.py')))
    lines = ['# 融合误差诊断与校准权重对照', '', '事后探索分析；只读取原校准/开发预测，不使用官方测试，不新训练。', '',
        '| 方案 | 第一组 Accuracy | 第二组 | 第三组 | 均值 | 相对 DeBERTa 的三组差值（百分点） |',
        '|---|---:|---:|---:|---:|---|']
    for name in POLICIES:
        scores = [100*r['policies'][name]['metrics']['accuracy'] for r in results]
        lines.append('| '+name+' | '+' | '.join(f'{v:.2f}%' for v in scores)+f' | {100*statistics[name]["accuracy_mean"]:.3f}% | '+', '.join(f'{v:+.2f}' for v in statistics[name]['delta_pp'])+' |')
    lines += ['', '## 权重与原等权纠错', '', '| 重复 | 校准拟合的 DeBERTa 权重 | 等权纠正 | 等权带偏 | DeBERTa 错/Laya 对 | DeBERTa 对/Laya 错 |', '|---|---:|---:|---:|---:|---:|']
    for r in results:
        c = r['policies']['equal']['versus_deberta']; pair = r['branch_complementarity']
        lines.append(f'| {r["repeat"]} | {r["fit"]["weights_deberta"]["calibration_nll"]:.6f} | {c["repaired"]} | {c["harmed"]} | {pair["deberta_wrong_laya_correct"]} | {pair["deberta_correct_laya_wrong"]} |')
    lines += ['', '所有候选完整报告；没有按开发成绩选择最优方案。模型温度与内部 Laya 系数沿用旧封存值，只有新外层权重用校准集拟合。',
        '400 条校准集和 2500 条开发集均被历史研究多次使用；不得把本轮分析作为新的独立确认。三次重复不等于 7500 个独立样本。',
        '错误互补的 oracle 指标使用真实标签，只是可见标签后的诊断，不能部署，也不能作为预测规则性能。',
        '全部改变 DeBERTa 类别的案例见 changed-cases.json，固定特征分层见 strata.json；没有按开发标签拟合门控。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(statistics, indent=2))


if __name__ == '__main__':
    main()
