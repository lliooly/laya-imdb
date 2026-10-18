"""One CPU calibration-only probability weight; separate frozen evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from common import ROOT, file_digest, write_json
from comparison_metrics import sigmoid
from laya_probability_fusion import apply_gated_mixture
import run_bow_nb_study as audit

RUN = 'laya-deberta-calibration-weight-cpu-20261018-v1'
OUT = ROOT/'reports'/RUN
CONFIG = 'configs/'+RUN+'.json'
SOURCE = 'reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation'
MANIFEST = 'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json'
PROVENANCE = 'reports/deberta-laya-final-analysis-20261018-v1/protocol.json'
L_FIND = 'reports/laya-probability-fusion-s43-s42-20261014-v1'


def read(relative):
    return json.loads((ROOT/relative).read_text())


def blend(laya, deberta, alpha):
    a, b = np.asarray(laya, dtype=float), np.asarray(deberta, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Invalid probability shape or coefficient')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (a > 1).any() or (b < 0).any() or (b > 1).any():
        raise ValueError('Invalid probabilities')
    return alpha*a+(1-alpha)*b


def fit_weight(laya, deberta, labels):
    a, b, y = np.asarray(laya), np.asarray(deberta), np.asarray(labels)
    blend(a, b, .5)
    if y.shape != a.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must align with probabilities')
    difference = a-b
    def derivative(alpha):
        p = np.clip(blend(a, b, alpha), 1e-12, 1-1e-12)
        return float(np.mean((p-y)*difference/(p*(1-p))))
    lo, hi = 0., 1.
    if np.array_equal(a, b):
        alpha = .5
    elif derivative(lo) >= 0:
        alpha = lo
    elif derivative(hi) <= 0:
        alpha = hi
    else:
        for _ in range(80):
            mid = (lo+hi)/2
            if derivative(mid) > 0:
                hi = mid
            else:
                lo = mid
        alpha = (lo+hi)/2
    return dict(alpha_laya=float(alpha), weight_deberta=float(1-alpha),
        derivative_at_0=derivative(0.), derivative_at_1=derivative(1.),
        derivative_at_solution=derivative(alpha), boundary_hit=alpha in (0., 1.))


def fit():
    if OUT.exists():
        raise FileExistsError('Use a new run_id; preserve this experiment')
    config = read(CONFIG)
    assert config['alpha_bounds'] == [0., 1.] and config['regularization'] == 0
    assert config['bootstrap_replicates'] == 20000 and config['bootstrap_seed'] == 20261018
    manifest, provenance = read(MANIFEST), read(PROVENANCE)
    ids, development_ids = manifest['calibration_ids'], manifest['validation']['ids']
    assert len(ids) == len(set(ids)) == 400 and len(development_ids) == len(set(development_ids)) == 2500
    assert not set(ids) & set(development_ids) and not set(ids) & set(manifest['gradient_train_ids'])
    history, frozen_laya = read(L_FIND+'/protocol.json'), read(L_FIND+'/frozen-fit.json')
    prior_protocol = read('reports/bow-nb-complementarity-cpu-20261018-v1/protocol.json')
    assert file_digest(ROOT/(L_FIND+'/frozen-fit.json')) == prior_protocol['source_sha256'][L_FIND+'/frozen-fit.json']
    assert frozen_laya['primary_models'] == ['baseline-s43', 'combined-s42'] and not frozen_laya['validation_used_for_fitting']
    sources = [CONFIG, MANIFEST, PROVENANCE, L_FIND+'/protocol.json', L_FIND+'/frozen-fit.json',
        'reports/bow-nb-complementarity-cpu-20261018-v1/protocol.json']
    blocks, labels, truncated = [], None, None
    for model in frozen_laya['primary_models']:
        for rule in ('head', 'head_tail'):
            path = f'reports/laya-posthoc-{model}-fp32-v1/calibration-{rule}.json'
            assert file_digest(ROOT/path) == history['source_sha256'][path]
            block = read(path)
            assert block['split'] == 'calibration' and not block['test_accessed']
            rows = block['rows']
            assert [r['sample_id'] for r in rows] == ids
            y = np.asarray([r['true_label'] for r in rows])
            t = np.asarray([r['truncated'] for r in rows], dtype=bool)
            if labels is not None:
                np.testing.assert_array_equal(y, labels)
                np.testing.assert_array_equal(t, truncated)
            labels, truncated = y, t
            blocks.append(np.asarray([r['probabilities'] for r in rows]))
            sources.append(path)
    laya = apply_gated_mixture(*blocks, truncated, frozen_laya['mixture']['coefficient'])[:, 1]
    path = SOURCE+'/calibration.json'
    assert file_digest(ROOT/path) == provenance['source_sha256'][path]
    cal = read(path)
    assert cal['sample_ids'] == ids and not cal['test_accessed']
    np.testing.assert_array_equal(labels, cal['true_labels'])
    z = np.asarray(cal['logits'])
    deberta = sigmoid((z[:, 1]-z[:, 0])/provenance['temperature'])
    sources += [path, 'scripts/run_laya_deberta_calibration_weight.py',
        'scripts/run_bow_nb_study.py', 'scripts/bow_nb_branch.py', 'scripts/common.py',
        'scripts/comparison_metrics.py', 'scripts/laya_probability_fusion.py', 'scripts/laya_decision_views.py',
        'environments/bow-cpu/pyproject.toml', 'environments/bow-cpu/uv.lock']
    OUT.mkdir(parents=True)
    write_json(OUT/'protocol.json', dict(config=config,
        created_utc=datetime.now(timezone.utc).isoformat(),
        sources_sha256={s: file_digest(ROOT/s) for s in sources},
        development_cache_expected_sha256=provenance['source_sha256'][SOURCE+'/predictions.json'],
        laya_internal_coefficient_unchanged=frozen_laya['mixture']['coefficient'],
        deberta_temperature_unchanged=provenance['temperature'],
        fitting_inputs='400 cached probabilities and original labels only',
        no_new_temperature=True, no_new_bias=True, no_ngram_branch=True))
    weight = fit_weight(laya, deberta, labels)
    mixed = blend(laya, deberta, weight['alpha_laya'])
    write_json(OUT/'calibration-predictions.json', dict(sample_ids=ids, true_labels=labels.tolist(),
        p_laya=laya.tolist(), p_deberta=deberta.tolist(), p_fitted=mixed.tolist()))
    frozen = dict(run_id=RUN, frozen_utc=datetime.now(timezone.utc).isoformat(), weight=weight,
        calibration_metrics={name: audit.met(labels, p) for name, p in [('frozen_50_50', blend(laya, deberta, .5)), ('calibration_nll_weight', mixed)]},
        protocol_sha256=file_digest(OUT/'protocol.json'),
        calibration_predictions_sha256=file_digest(OUT/'calibration-predictions.json'),
        calibration_ids=ids, development_ids=development_ids,
        development_loaded=False, official_test_loaded=False)
    write_json(OUT/'frozen.json', frozen)
    write_json(OUT/'fit-access-audit.json', dict(read_paths=sorted(audit.READS),
        frozen_sha256=file_digest(OUT/'frozen.json'), development_opened=False, official_test_opened=False))
    print(json.dumps(dict(stage='fit_frozen', weight=weight,
        calibration_nll={k: v['nll'] for k, v in frozen['calibration_metrics'].items()})))


def evaluate():
    if (OUT/'summary.json').exists():
        raise FileExistsError('Preserve existing evaluation')
    frozen, protocol, receipt = [read(str((OUT/n).relative_to(ROOT))) for n in ('frozen.json', 'protocol.json', 'fit-access-audit.json')]
    assert file_digest(OUT/'frozen.json') == receipt['frozen_sha256']
    assert file_digest(OUT/'protocol.json') == frozen['protocol_sha256']
    assert file_digest(OUT/'calibration-predictions.json') == frozen['calibration_predictions_sha256']
    for path, sha in protocol['sources_sha256'].items():
        assert file_digest(ROOT/path) == sha, path
    path = SOURCE+'/predictions.json'
    assert file_digest(ROOT/path) == protocol['development_cache_expected_sha256']
    dev = read(path)
    assert dev['sample_ids'] == frozen['development_ids'] and not dev['test_accessed']
    y = np.asarray(dev['true_labels'])
    pp = dev['positive_probabilities']
    laya, deberta = np.asarray(pp['frozen_main']), np.asarray(pp['deberta'])
    baseline = blend(laya, deberta, .5)
    np.testing.assert_array_equal(baseline, pp['fusion_50_50'])
    assert int(((baseline > .5) == y).sum()) == 2432
    candidate = blend(laya, deberta, frozen['weight']['alpha_laya'])
    results = {name: audit.met(y, p) for name, p in [('frozen_50_50', baseline), ('calibration_nll_weight', candidate)]}
    paired = audit.compare(y, baseline, candidate)
    # Independent scalar optimizer verifies the already frozen weight, never selects a new candidate.
    cal = read(str((OUT/'calibration-predictions.json').relative_to(ROOT)))
    cy, ca, cb = np.asarray(cal['true_labels']), np.asarray(cal['p_laya']), np.asarray(cal['p_deberta'])
    objective = lambda alpha: log_loss(cy, blend(ca, cb, alpha), labels=[0, 1])
    optimum = minimize_scalar(objective, bounds=(0., 1.), method='bounded', options={'xatol': 1e-12})
    best_check = min((0., float(optimum.x), 1.), key=objective)
    assert abs(best_check-frozen['weight']['alpha_laya']) < 1e-7
    for name, probability in [('frozen_50_50', baseline), ('calibration_nll_weight', candidate)]:
        assert results[name]['accuracy'] == accuracy_score(y, probability > .5)
        assert abs(results[name]['nll']-log_loss(y, probability, labels=[0, 1])) < 1e-12
        assert abs(results[name]['auroc']-roc_auc_score(y, probability)) < 1e-12
    write_json(OUT/'development-predictions.json', dict(sample_ids=dev['sample_ids'], true_labels=y.tolist(),
        p_frozen_50_50=baseline.tolist(), p_calibration_nll_weight=candidate.tolist(),
        changed_sample_ids=[dev['sample_ids'][i] for i in np.flatnonzero((baseline > .5) != (candidate > .5))]))
    summary = dict(run_id=RUN, primary='calibration_nll_weight', reference='frozen_50_50',
        weight=frozen['weight'], development_metrics=results, paired_comparison=paired,
        calibration_metrics=frozen['calibration_metrics'], frozen_sha256=file_digest(OUT/'frozen.json'),
        development_source_sha256=file_digest(ROOT/path), fitting_performed_after_freeze=False,
        official_test_opened=False, official_test_distribution_used=False, gpu_used=False,
        new_model_training=False, new_neural_inference=False, default_deployment_changed=False,
        limitations=['reused calibration/development', 'single existing DeBERTa seed',
            'conditional interval excludes fitting/calibration and historical selection uncertainty',
            'NLL-optimal calibration coefficient does not guarantee accuracy improvement'])
    write_json(OUT/'summary.json', summary)
    ci = paired['conditional_stratified_bootstrap95_pp']
    lines = ['# Laya / DeBERTa：校准集单系数融合实验', '',
        '本地 CPU，仅复用冻结缓存。400 条校准拟合一个系数并封存，独立 evaluate 进程才读取 2500 条开发预测；不读取评论正文、官方测试文件或分布，不进行训练与新模型推理。', '',
        '固定 p=α·p_Laya+(1−α)·p_DeBERTa，α∈[0,1]，只最小化校准 NLL，无正则、偏置、新温度、长度门控或词袋分支。原 Laya 内部融合和 DeBERTa 温度保持不变。DeBERTa 使用产生开发 97.28% 的 CUDA BF16 缓存。', '',
        f'**冻结权重：Laya {100*frozen["weight"]["alpha_laya"]:.4f}%，DeBERTa {100*frozen["weight"]["weight_deberta"]:.4f}%。**', '',
        '| 方案 | 正确 / 2500 | Accuracy | NLL | ECE |', '|---|---:|---:|---:|---:|']
    for name, m in results.items():
        lines.append(f'| {name} | {m["correct"]} | {100*m["accuracy"]:.2f}% | {m["nll"]:.6f} | {100*m["ece_15_equal_width_top_label"]:.3f}% |')
    lines += ['', f'相对原 50:50：修复 {paired["repaired"]} 条，退化 {paired["damaged"]} 条，净变化 {paired["net"]:+d} 条；Accuracy 差 {paired["delta_accuracy_pp"]:+.3f} pp，条件配对 bootstrap 95% 区间 [{ci[0]:+.3f}, {ci[1]:+.3f}] pp；精确 McNemar p={paired["mcnemar_exact_two_sided_p"]:.6g}。', '',
        '## 校准目标与解释', '',
        f'校准 NLL：50:50 为 {frozen["calibration_metrics"]["frozen_50_50"]["nll"]:.8f}，拟合后为 {frozen["calibration_metrics"]["calibration_nll_weight"]["nll"]:.8f}。这只说明在该校准集的预定 NLL 目标改善，不保证开发 Accuracy 增加。', '',
        '全部参数在开发评估前封存，无开发权重扫描、事后择优或追加模型。20,000 次类别分层配对 bootstrap 固定已训练模型和系数，seed=20261018；区间不包含校准估计、训练种子变化或反复开发使用的选择偏差，属于探索性结果。', '',
        '## 验证与产物', '',
        '原融合逐样本概率与 2432/2500 成绩完全复现；输入 ID/标签对齐、来源/源码哈希和冻结契约核验通过。独立 SciPy 标量优化核对校准系数，sklearn 独立复算开发 Accuracy/NLL/AUROC。', '',
        'protocol.json / frozen.json 保存配置、来源 SHA、系数和冻结时间；calibration-predictions.json 与 development-predictions.json 保存对应预测；fit/evaluate-access-audit.json 保存读路径；verification.json 保存独立复算。训练阶段拒绝开发输入，两个进程均禁止官方测试路径。', '',
        '入口：scripts/run_laya_deberta_calibration_weight.py，按 fit、evaluate 顺序分别运行。使用 environments/bow-cpu 的锁定环境，已有报告目录时拒绝覆盖。默认部署未切换，未据此重新打开官方测试。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    write_json(OUT/'evaluate-access-audit.json', dict(read_paths=sorted(audit.READS),
        official_test_opened=False, fitting_performed=False))
    write_json(OUT/'verification.json', dict(passed=True, independent_calibration_alpha=best_check,
        absolute_alpha_difference=abs(best_check-frozen['weight']['alpha_laya']),
        sklearn_metrics_recomputed=True, original_probabilities_bitwise_reproduced=True,
        source_and_frozen_hashes_verified=True, report_sha256=file_digest(OUT/'REPORT.md'),
        summary_sha256=file_digest(OUT/'summary.json')))
    print(json.dumps(dict(stage='evaluated', weight=frozen['weight'],
        accuracy={k: v['accuracy'] for k, v in results.items()}, paired=paired), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('fit', 'evaluate'))
    args = parser.parse_args()
    audit.install_guard(args.stage)
    (fit if args.stage == 'fit' else evaluate)()
