"""Two-process CPU study; no official-test access or development-time fitting."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '6')

import joblib
import numpy as np
import scipy
from scipy.stats import binomtest
import sklearn
from sklearn.exceptions import ConvergenceWarning

from bow_nb_branch import (KINDS, calibrated_probability, decision, fit_sigmoid,
    fit_sparse_weight, mix, probability_nll, train_branch, validate_partitions)
from common import ROOT, file_digest, read_jsonl, write_json
from comparison_metrics import metrics, sigmoid
from laya_probability_fusion import apply_gated_mixture

RUN = 'bow-nb-complementarity-cpu-20261018-v1'
OUT = ROOT/'reports'/RUN
ART = ROOT/'artifacts'/RUN
MANIFEST = 'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json'
NEURAL = 'reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation'
L_FIND = 'reports/laya-probability-fusion-s43-s42-20261014-v1'
DESIGN = 'docs/superpowers/specs/2026-10-18-bow-nb-complementarity-design.md'
READS = set()


def install_guard(stage):
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.fsdecode(args[0])
        full = str(Path(path).absolute())
        mode = args[1]
        flags = args[2] if len(args) > 2 else 0
        reading = ('r' in mode or '+' in mode) if isinstance(mode, str) else not(flags & os.O_WRONLY)
        if not reading:
            return
        if any(part in full for part in ('/data/processed/test', '/data/raw/',
            '/laya-frozen-official-test-', '/deberta-laya-final-cpu-evaluation-',
            '/deberta-laya-final-cpu-evaluation-')):
            raise PermissionError('Official-test access forbidden by study: '+full)
        if stage == 'fit' and (full.endswith('/validation.jsonl') or full.endswith('/predictions.json')
            or '/validation-' in full or full.endswith('/development.json')):
            raise PermissionError('Development access forbidden during fit: '+full)
        if full.startswith(str(ROOT)+'/') and Path(full).suffix in ('.json', '.jsonl', '.toml', '.lock', '.py', '.md') and '/.venv/' not in full:
            READS.add(str(Path(full).relative_to(ROOT)))
    sys.addaudithook(audit)


def read(relative):
    return json.loads((ROOT/relative).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **kwargs):
    print(json.dumps(dict(at=now(), event=event, **kwargs), ensure_ascii=False), flush=True)


def met(y, p):
    p = np.clip(np.asarray(p), 1e-15, 1-1e-15)
    return metrics(y, np.log(p)-np.log1p(-p))


def calibration_neural(ids, labels):
    provenance = read('reports/deberta-laya-final-analysis-20261018-v1/protocol.json')
    history = read(L_FIND+'/protocol.json')
    frozen = read(L_FIND+'/frozen-fit.json')
    if frozen['primary_models'] != ['baseline-s43', 'combined-s42'] or frozen['validation_used_for_fitting']:
        raise ValueError('Historical Laya identity differs')
    blocks = []
    truncated = None
    source_paths = [L_FIND+'/frozen-fit.json', L_FIND+'/protocol.json',
        'reports/deberta-laya-final-analysis-20261018-v1/protocol.json']
    for model in frozen['primary_models']:
        for rule in ('head', 'head_tail'):
            rel = f'reports/laya-posthoc-{model}-fp32-v1/calibration-{rule}.json'
            if file_digest(ROOT/rel) != history['source_sha256'][rel]:
                raise ValueError('Historical calibration SHA mismatch')
            data = read(rel)
            rows = data['rows']
            if data['test_accessed'] or data['split'] != 'calibration':
                raise ValueError('Unexpected cached split')
            if [r['sample_id'] for r in rows] != ids or [r['true_label'] for r in rows] != list(labels):
                raise ValueError('Calibration ID / label mismatch')
            t = np.array([r['truncated'] for r in rows], dtype=bool)
            if truncated is not None and not np.array_equal(t, truncated):
                raise ValueError('Truncation mismatch')
            truncated = t
            blocks.append(np.asarray([r['probabilities'] for r in rows]))
            source_paths.append(rel)
    laya = apply_gated_mixture(*blocks, truncated, frozen['mixture']['coefficient'])[:, 1]
    rel = NEURAL+'/calibration.json'
    if file_digest(ROOT/rel) != provenance['source_sha256'][rel]:
        raise ValueError('DeBERTa calibration SHA mismatch')
    data = read(rel)
    if data['sample_ids'] != ids or data['true_labels'] != list(labels) or data['test_accessed']:
        raise ValueError('DeBERTa calibration mismatch')
    z = np.asarray(data['logits'])
    deberta = sigmoid((z[:, 1]-z[:, 0])/provenance['temperature'])
    source_paths.append(rel)
    return (laya+deberta)/2, source_paths, dict(
        laya_coefficient=frozen['mixture']['coefficient'],
        deberta_temperature=provenance['temperature'],
        expected_development_sha256=provenance['source_sha256'][NEURAL+'/predictions.json'])


def fit():
    if OUT.exists() or ART.exists():
        raise FileExistsError('Preserve previous run; use a new run_id')
    OUT.mkdir(parents=True)
    ART.mkdir(parents=True)
    started = time.perf_counter()
    manifest = read(MANIFEST)
    gradient_ids, calibration_ids = manifest['gradient_train_ids'], manifest['calibration_ids']
    development_ids = manifest['validation']['ids']
    sets = validate_partitions(gradient_ids, calibration_ids, development_ids)
    if list(map(len, sets)) != [22100, 400, 2500]:
        raise ValueError('Frozen partition sizes differ')
    train_path = ROOT/'data/processed/train.jsonl'
    if file_digest(train_path) != manifest['train']['source_sha256']:
        raise ValueError('Original training data SHA differs')
    rows = list(read_jsonl(train_path))
    by_id = {r['id']: r for r in rows}
    if len(rows) != 22500 or len(by_id) != 22500 or set(by_id) != sets[0] | sets[1]:
        raise ValueError('Original training rows differ')
    train_text = [by_id[k]['text'] for k in gradient_ids]
    train_y = np.array([by_id[k]['label'] for k in gradient_ids])
    cal_text = [by_id[k]['text'] for k in calibration_ids]
    cal_y = np.array([by_id[k]['label'] for k in calibration_ids])
    neural, source_paths, identity = calibration_neural(calibration_ids, cal_y)
    source_paths += [MANIFEST, 'data/processed/train.jsonl', DESIGN,
        'scripts/bow_nb_branch.py', 'scripts/run_bow_nb_study.py',
        'scripts/common.py', 'scripts/comparison_metrics.py',
        'scripts/laya_probability_fusion.py', 'scripts/laya_decision_views.py',
        'environments/bow-cpu/pyproject.toml', 'environments/bow-cpu/uv.lock']
    protocol = dict(run_id=RUN, created_utc=now(), design_sha256=file_digest(ROOT/DESIGN),
        source_sha256={p: file_digest(ROOT/p) for p in source_paths},
        gradient_count=22100, calibration_count=400, development_count=2500,
        candidates=list(KINDS), model_hyperparameter_search=False,
        calibration='regularized bounded sigmoid; fixed 400', fusion='convex probability mix; w in [0,0.25]; calibration NLL only',
        primary_selection='lowest calibration mixture NLL; candidate order breaks ties',
        original_neural_identity=identity, official_test_opened=False,
        historical_development_reused=True, preparation_deviation='Parent session inadvertently read historical official-test contract IDs; no test text/predictions/distribution used',
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            sklearn=sklearn.__version__, platform=platform.platform(), cpu_threads=6))
    write_json(OUT/'protocol.json', protocol)
    results = {}
    for kind in KINDS:
        start = time.perf_counter()
        emit('training_start', candidate=kind, gradient_count=len(train_y))
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model, info = train_branch(kind, train_text, train_y)
        margins = decision(model, cal_text)
        calibration = fit_sigmoid(margins, cal_y)
        p = calibrated_probability(margins, calibration)
        mixture = fit_sparse_weight(neural, p, cal_y)
        model_file = ART/(kind+'.joblib')
        joblib.dump(model, model_file, compress=3)
        reloaded = joblib.load(model_file)
        np.testing.assert_array_equal(margins, decision(reloaded, cal_text))
        results[kind] = dict(training=info, sigmoid=calibration, fusion=mixture,
            calibration_metrics=met(cal_y, p), raw_calibration_metrics=metrics(cal_y, margins),
            model_path=str(model_file.relative_to(ROOT)), model_sha256=file_digest(model_file),
            reload_calibration_exact=True, seconds=time.perf_counter()-start)
        write_json(OUT/(kind+'-calibration.json'), dict(sample_ids=calibration_ids,
            true_labels=cal_y.tolist(), margins=margins.tolist(), positive_probabilities=p.tolist(),
            neural_positive_probabilities=neural.tolist(),
            fusion_positive_probabilities=mix(neural, p, mixture['sparse_weight']).tolist()))
        emit('candidate_frozen', candidate=kind, features=info['feature_count'],
            weight=mixture['sparse_weight'], calibration_nll=mixture['calibration_nll'],
            seconds=results[kind]['seconds'])
    primary = min(KINDS, key=lambda name: results[name]['fusion']['calibration_nll'])
    frozen = dict(run_id=RUN, frozen_utc=now(), protocol_sha256=file_digest(OUT/'protocol.json'),
        source_sha256=protocol['source_sha256'], candidates=results, primary=primary,
        neural_calibration_metrics=met(cal_y, neural),
        calibration_ids=calibration_ids, gradient_ids=gradient_ids, development_ids=development_ids,
        expected_development_sha256=identity['expected_development_sha256'],
        expected_development_text_sha256=manifest['validation']['source_sha256'],
        total_fit_seconds=time.perf_counter()-started,
        development_loaded_in_fit=False, official_test_loaded=False)
    write_json(OUT/'frozen.json', frozen)
    write_json(OUT/'fit-access-audit.json', dict(stage='fit', read_paths=sorted(READS),
        forbidden_accesses=0, official_test_opened=False, development_opened=False,
        frozen_sha256=file_digest(OUT/'frozen.json')))
    emit('all_frozen', primary=primary, frozen_sha256=file_digest(OUT/'frozen.json'))


def compare(y, reference, candidate):
    a = (reference > .5) == y
    b = (candidate > .5) == y
    repaired, damaged = int((~a & b).sum()), int((a & ~b).sum())
    rng = np.random.default_rng(20261018)
    diff = b.astype(int)-a.astype(int)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    draws = []
    for _ in range(200):
        sampled = np.concatenate([g[rng.integers(len(g), size=(100, len(g)))] for g in groups], axis=1)
        draws.extend(diff[sampled].mean(axis=1))
    return dict(repaired=repaired, damaged=damaged, net=repaired-damaged,
        delta_accuracy_pp=float(100*diff.mean()),
        conditional_stratified_bootstrap95_pp=(100*np.quantile(draws, [.025, .975])).tolist(),
        bootstrap_replicates=20000, bootstrap_seed=20261018,
        mcnemar_exact_two_sided_p=float(binomtest(min(repaired, damaged), repaired+damaged).pvalue) if repaired+damaged else 1.,
        both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum()))


def evaluate():
    if (OUT/'summary.json').exists():
        raise FileExistsError('Preserve previous evaluation')
    receipt = read(str((OUT/'fit-access-audit.json').relative_to(ROOT)))
    if file_digest(OUT/'frozen.json') != receipt['frozen_sha256']:
        raise ValueError('Frozen contract changed')
    frozen = read(str((OUT/'frozen.json').relative_to(ROOT)))
    if file_digest(OUT/'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('Protocol changed')
    for rel, sha in frozen['source_sha256'].items():
        if file_digest(ROOT/rel) != sha:
            raise ValueError('Fit source changed: '+rel)
    for info in frozen['candidates'].values():
        if file_digest(ROOT/info['model_path']) != info['model_sha256']:
            raise ValueError('Sparse model changed')
    # First development access in this study's execution follows the frozen contract above.
    source = NEURAL+'/predictions.json'
    if file_digest(ROOT/source) != frozen['expected_development_sha256']:
        raise ValueError('Development neural cache SHA differs')
    cache = read(source)
    ids = frozen['development_ids']
    y = np.asarray(cache['true_labels'])
    if cache['sample_ids'] != ids or cache['test_accessed']:
        raise ValueError('Development cache identities differ')
    path = ROOT/'data/processed/validation.jsonl'
    if file_digest(path) != frozen['expected_development_text_sha256']:
        raise ValueError('Development text SHA differs')
    rows = list(read_jsonl(path))
    if len(rows) != 2500 or len({r['id'] for r in rows}) != 2500:
        raise ValueError('Development rows differ')
    by_id = {r['id']: r for r in rows}
    if set(by_id) != set(ids) or not np.array_equal(y, [by_id[k]['label'] for k in ids]):
        raise ValueError('Development ID / label mismatch')
    texts = [by_id[k]['text'] for k in ids]
    pp = cache['positive_probabilities']
    neural = np.asarray(pp['fusion_50_50'])
    np.testing.assert_array_equal(neural, (np.asarray(pp['frozen_main'])+np.asarray(pp['deberta']))/2)
    baseline = met(y, neural)
    if baseline['correct'] != 2432:
        raise ValueError('97.28% baseline did not reproduce')
    # Length groups are inherited Laya budget diagnostics, never fitted routing rules.
    old = read('reports/laya-posthoc-baseline-s43-fp32-v1/validation-head.json')
    if [r['sample_id'] for r in old['rows']] != ids or not np.array_equal(y, [r['true_label'] for r in old['rows']]):
        raise ValueError('Development length metadata differs')
    long = np.array([r['truncated'] for r in old['rows']], dtype=bool)
    if long.sum() != 407:
        raise ValueError('Historical length groups differ')
    results, outputs = {}, dict(neural=neural.tolist())
    for kind in KINDS:
        info = frozen['candidates'][kind]
        model = joblib.load(ROOT/info['model_path'])
        z = decision(model, texts)
        p = calibrated_probability(z, info['sigmoid'])
        fused = mix(neural, p, info['fusion']['sparse_weight'])
        sparse_pair, fusion_pair = compare(y, neural, p), compare(y, neural, fused)
        n_ok = (neural > .5) == y
        s_ok = (p > .5) == y
        complement = dict(neural_errors=int((~n_ok).sum()), sparse_fixes_neural_errors=int((~n_ok & s_ok).sum()),
            sparse_breaks_neural_correct=int((n_ok & ~s_ok).sum()), both_wrong=int((~n_ok & ~s_ok).sum()),
            label_informed_oracle_correct=int((n_ok | s_ok).sum()),
            label_informed_oracle_accuracy=float((n_ok | s_ok).mean()), oracle_deployable=False)
        groups = {}
        for name, mask in [('laya_truncated', long), ('laya_complete', ~long)]:
            groups[name] = dict(count=int(mask.sum()), neural=met(y[mask], neural[mask]),
                sparse=met(y[mask], p[mask]), fusion=met(y[mask], fused[mask]),
                fusion_pair=compare(y[mask], neural[mask], fused[mask]))
        results[kind] = dict(sparse=met(y, p), sparse_uncalibrated=metrics(y, z),
            fusion=met(y, fused), sparse_pair=sparse_pair, fusion_pair=fusion_pair,
            complementarity=complement, groups=groups,
            sparse_weight=info['fusion']['sparse_weight'], selected_by_calibration=kind == frozen['primary'])
        outputs[kind] = dict(raw_margins=z.tolist(), calibrated_positive_probabilities=p.tolist(), fusion_positive_probabilities=fused.tolist())
        emit('development_result', candidate=kind, sparse_accuracy=results[kind]['sparse']['accuracy'],
            fusion_accuracy=results[kind]['fusion']['accuracy'], repaired=fusion_pair['repaired'], damaged=fusion_pair['damaged'])
    write_json(OUT/'development-predictions.json', dict(sample_ids=ids, true_labels=y.tolist(),
        laya_truncated=long.tolist(), predictions=outputs, test_accessed=False))
    # ID-only cases: no textual inspection or post-hoc feature engineering.
    cases = []
    for i, sid in enumerate(ids):
        if (neural[i] > .5) != y[i] or any(((outputs[k]['fusion_positive_probabilities'][i] > .5) == y[i]) != ((neural[i] > .5) == y[i]) for k in KINDS):
            cases.append(dict(sample_id=sid, true_label=int(y[i]), laya_truncated=bool(long[i]),
                neural_positive_probability=float(neural[i]),
                branches={k:dict(sparse_probability=outputs[k]['calibrated_positive_probabilities'][i],
                    fusion_probability=outputs[k]['fusion_positive_probabilities'][i]) for k in KINDS}))
    write_json(OUT/'complementarity-cases.json', dict(cases=cases, raw_text_reviewed=False))
    summary = dict(run_id=RUN, completed_utc=now(), primary=frozen['primary'], baseline=baseline,
        results=results, frozen_sha256=file_digest(OUT/'frozen.json'),
        protocol_sha256=frozen['protocol_sha256'], total_fit_seconds=frozen['total_fit_seconds'],
        official_test_opened=False, official_test_distribution_used=False, gpu_used=False,
        parameters_updated_after_development=False, default_deployment_changed=False,
        evaluation_source_sha256={source: file_digest(ROOT/source),
            'data/processed/validation.jsonl': file_digest(path),
            'reports/laya-posthoc-baseline-s43-fp32-v1/validation-head.json': file_digest(ROOT/'reports/laya-posthoc-baseline-s43-fp32-v1/validation-head.json')},
        limitations=['reused development and calibration', 'single fixed training recipe per branch',
            'conditional intervals exclude calibration fitting/model selection/training variation',
            'secondary candidates and length groups have no multiplicity correction',
            'parent preparation read historical test-contract IDs; no test data/distribution used'])
    write_json(OUT/'summary.json', summary)
    report(summary, frozen)
    write_json(OUT/'evaluate-access-audit.json', dict(stage='evaluate', read_paths=sorted(READS),
        official_test_opened=False, fitting_performed=False, frozen_sha256=file_digest(OUT/'frozen.json')))
    emit('completed', primary=frozen['primary'], report=str(OUT/'REPORT.md'))


def report(summary, frozen):
    lines = ['# 本地 CPU 词袋 / NB-ngram 互补性实验', '',
        '固定 22,100 梯度训练、400 校准、2,500 开发；原标签和分区保持。两个候选的模型、校准参数、融合权重和主方案均在独立 fit 进程封存，evaluate 进程仅推理与统计。官方测试正文、预测和分布未读取。', '',
        f'校准集预选主方案：`{summary["primary"]}`。原 Laya + DeBERTa 固定 50:50 概率缓存准确复现 2432/2500 = 97.28%。DeBERTa 为历史 CUDA BF16 开发缓存，没有新神经网络推理。', '',
        '| 方案 | 正确 / 2500 | Accuracy | NLL | ECE |', '|---|---:|---:|---:|---:|']
    def line(name, m):
        return f'| {name} | {m["correct"]} | {100*m["accuracy"]:.2f}% | {m["nll"]:.6f} | {100*m["ece_15_equal_width_top_label"]:.3f}% |'
    lines.append(line('原神经融合', summary['baseline']))
    for kind, r in summary['results'].items():
        lines += [line(kind+' 单独', r['sparse']), line(kind+' 固定权重融合', r['fusion'])]
    lines += ['', '## 校准和权重', '',
        '不搜索训练超参数。TF-IDF 为 word 1–2 gram、C=4；NB 分支为 binary word 1–3 gram、C=1；两者 min_df=3、最多 300000 特征，全文输入，保留否定词与单字符数字，tokenizer 保留 8/10 评分形式。词表、IDF、NB 类内统计只用梯度训练文本。', '',
        '400 条校准拟合有界 sigmoid，NLL 加固定 0.001 正则；再以校准 NLL 最小化拟合 p=(1-w)p_neural+w*p_sparse，w∈[0,0.25]，可取零。原神经融合内部比例保持不变。主候选按校准融合 NLL 选择，未按开发成绩反选。', '',
        '| 候选 | 特征数 | sigmoid a / b | 稀疏权重 | 校准融合 NLL |', '|---|---:|---:|---:|---:|']
    for kind, info in frozen['candidates'].items():
        lines.append(f'| {kind} | {info["training"]["feature_count"]} | {info["sigmoid"]["slope"]:.6f} / {info["sigmoid"]["intercept"]:.6f} | {100*info["fusion"]["sparse_weight"]:.4f}% | {info["fusion"]["calibration_nll"]:.6f} |')
    lines += ['', '## 实际融合的修复与退化', '',
        '| 候选 | 修复 / 退化 | 净变化 | Accuracy 差值及条件 95% 区间（pp） | 精确 McNemar p |', '|---|---:|---:|---|---:|']
    for kind, r in summary['results'].items():
        p = r['fusion_pair']; ci = p['conditional_stratified_bootstrap95_pp']
        lines.append(f'| {kind} | {p["repaired"]} / {p["damaged"]} | {p["net"]:+d} | {p["delta_accuracy_pp"]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {p["mcnemar_exact_two_sided_p"]:.6g} |')
    lines += ['', '## 独立分支的互补性与 oracle', '',
        '| 分支 | 答对原 68 条错误 | 判错原正确样本 | 共同错误 | 标签知情 oracle |', '|---|---:|---:|---:|---:|']
    for kind, r in summary['results'].items():
        c = r['complementarity']
        lines.append(f'| {kind} | {c["sparse_fixes_neural_errors"]} | {c["sparse_breaks_neural_correct"]} | {c["both_wrong"]} | {100*c["label_informed_oracle_accuracy"]:.2f}% |')
    lines += ['', 'oracle 按已知正确标签逐条选模型，仅描述上限，不能部署。单独分支答对某条错题，不代表小权重融合能跨过 0.5 边界。样本概率见 complementarity-cases.json，未人工查看正文后修改特征。', '',
        '## 长短文分组', '', '| 候选 / 分组 | 样本数 | 原神经准确率 | 融合准确率 | 修复 / 退化 |', '|---|---:|---:|---:|---:|']
    for kind, r in summary['results'].items():
        for group, g in r['groups'].items():
            p = g['fusion_pair']
            lines.append(f'| {kind} / {group} | {g["count"]} | {100*g["neural"]["accuracy"]:.3f}% | {100*g["fusion"]["accuracy"]:.3f}% | {p["repaired"]} / {p["damaged"]} |')
    primary = summary['results'][summary['primary']]
    delta = primary['fusion_pair']['net']
    lines += ['', '## 结论与边界', '',
        f'校准预选主方案在开发集相对 97.28% 净变化 {delta:+d} 条，准确率 {100*primary["fusion"]["accuracy"]:.2f}%。这是本轮预定流程的结果，不把开发上较高的次级候选反选为主方案。', '',
        '20,000 次类别分层、共享 ID 配对 bootstrap（seed=20261018）固定已训练模型与参数，区间不包含校准拟合、模型选择、训练随机性和多轮开发复用的不确定性。两个候选及分组完整报告，次级 p 值未作多重比较校正。NLL 是预定权重目标，不能保证 Accuracy 增加；无新测试或默认部署切换。', '',
        '准备阶段偏离：父会话寻找来源时误读旧测试契约，其中有官方测试样本 ID（文件名含评分）。没有打开测试正文 / 预测，未计算或使用测试分布。因此不声称整个会话未接触测试元数据；fit/evaluate 的测试文件访问禁止与读路径分别记录在 access-audit 文件中。', '',
        f'CPU fit 共 {summary["total_fit_seconds"]:.2f} 秒，包含特征提取、训练、校准、模型保存及重载验证；不包括首次依赖下载。两模型重载后 400 条校准 margins 逐位一致。完整模型与词表在 artifacts/{RUN}/，环境锁在 environments/bow-cpu/uv.lock。', '',
        '复现需新 run_id，拒绝覆盖已有目录。命令：', '', '```bash',
        'UV_CACHE_DIR=/private/tmp/end-imdb-bow-uv-cache uv run --project environments/bow-cpu --locked --offline --no-sync python scripts/run_bow_nb_study.py fit',
        'UV_CACHE_DIR=/private/tmp/end-imdb-bow-uv-cache uv run --project environments/bow-cpu --locked --offline --no-sync python scripts/run_bow_nb_study.py evaluate', '```', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('fit', 'evaluate'))
    args = parser.parse_args()
    install_guard(args.stage)
    (fit if args.stage == 'fit' else evaluate)()
