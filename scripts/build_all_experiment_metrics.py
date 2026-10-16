"""Consolidate saved experiment metrics; never imports models or reads review text."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES: dict[str, str] = {}
CACHE: dict[str, object] = {}
ROWS: list[dict] = []
PAIRS: list[dict] = []


def source(name: str) -> str:
    return name if name.startswith("reports/") else f"reports/{name}/summary.json"


def read(name: str):
    name = source(name)
    if name not in CACHE:
        data = (ROOT / name).read_bytes()
        SOURCES[name] = hashlib.sha256(data).hexdigest()
        CACHE[name] = json.loads(data)
    return CACHE[name]


def get(name: str, *keys):
    obj = read(name)
    for key in keys:
        obj = obj[key]
    return obj


def loc(name: str, keys) -> dict:
    return {"file": source(name), "keys": list(keys)}


def trim(obj: dict) -> dict:
    keep = {"confusion_matrix_actual_rows_predicted_columns", "accuracy_wilson95",
            "accuracy_wilson_95", "accuracy_paired_bootstrap95", "conditional_bootstrap95_pp"}
    return {k: v for k, v in obj.items() if not isinstance(v, (dict, list)) or k in keep}


def add(group, method, seed, protocol, name, keys, *, raw_keys=None,
        probability="报告所用概率", note="", expected=None, split="development", coverage=1.0):
    metrics = trim(get(name, *keys))
    assert "accuracy" in metrics, (name, keys)
    if "count" not in metrics and "correct" in metrics:
        metrics["count"] = round(metrics["correct"] / metrics["accuracy"])
    if "count" in metrics and "correct" in metrics:
        assert abs(metrics["accuracy"] - metrics["correct"] / metrics["count"]) < 1e-12
    if expected is not None:
        assert abs(metrics["accuracy"] * 100 - expected) < 1e-9, (method, metrics["accuracy"], expected)
    row = {
        "id": f"M{len(ROWS)+1:03}", "group": group, "method": method,
        "seed": seed, "protocol": protocol, "split": split, "coverage": coverage,
        "metrics": metrics, "probability_scope": probability, "note": note,
        "source": loc(name, keys),
    }
    if raw_keys is not None:
        row["raw_metrics"] = trim(get(name, *raw_keys))
        row["raw_source"] = loc(name, raw_keys)
        assert abs(row["raw_metrics"]["accuracy"] - metrics["accuracy"]) < 1e-12
    ROWS.append(row)
    return row


def pair(label, reference, candidate, name, keys, *, note="", ci=None):
    p = get(name, *keys)
    entry = {"comparison": label, "reference": reference, "candidate": candidate,
             "statistics": trim(p), "source": loc(name, keys), "note": note}
    if ci is not None:
        entry["conditional_ci95_pp"] = ci
    PAIRS.append(entry)


def build_rows():
    early = "reports/model-comparison/all_metrics.json"
    for key, label, expected in [
        ("modernbert_base", "ModernBERT-base", 94.64),
        ("laya_ce_full", "Laya-CE：Mac 一轮", 95.76),
        ("laya_rlcd_pilot", "Laya-RLCD pilot", 93.72),
        ("laya_ce_pilot", "Laya-CE pilot", 93.52),
    ]:
        calibrated = key != "modernbert_base"
        add("01 早期模型与训练预算", label, 42,
            "MLX；256 token；3 轮；训练 22500" if not calibrated else
            ("MPS FP32；512 token；1 轮；训练 22100" if key == "laya_ce_full" else
             "MPS FP32；512 token；1 轮；仅训练 896、校准 128"),
            early, (key, "full_validation_2500", "native_calibrated" if calibrated else "uncalibrated"),
            raw_keys=(key, "full_validation_2500", "uncalibrated") if calibrated else None,
            probability="原生温度校准" if calibrated else "未校准",
            expected=expected, note="模型规模、输入与训练预算不同，不能归因于单一结构。")
    mac = "laya-mac-ce-comparison-v1"
    for run, label, acc in [
        ("laya-mps-ce-epochs2-l512-s42-fp32-v1", "Laya-CE：Mac 两轮、原学习率", 96.96),
        ("laya-mps-ce-epochs2-lr06-l512-s42-fp32-v1", "Laya-CE：Mac 两轮、学习率 ×0.6", 96.12),
    ]:
        add("01 早期模型与训练预算", label, 42, "MPS FP32；prefix；512 token；2 轮",
            mac, ("runs", run, "metrics"), expected=acc, probability="原生温度校准",
            note="固定保存权重重载；历史单种子开发结果。")
    cuda = "reports/laya-cuda-reference-ce-l512-s42-fp32-v1/cuda-saved-validation.json"
    add("01 早期模型与训练预算", "Laya-CE：RTX 6000D 一轮", 42, "CUDA FP32；prefix；512 token；1 轮",
        cuda, ("native_calibrated_metrics",), raw_keys=("raw_metrics",), expected=96.28,
        probability="原生温度校准", note="跨设备结果，不是与 Mac 的方法单因素对照。")
    reg = "rtx4090-regularization-completed-v1"
    for key, label, acc in [("baseline", "4090 两轮基线", 95.88),
                             ("smoothing", "标签平滑 0.05", 96.16),
                             ("warmup", "warmup 5%", 96.32)]:
        add("02 调度与正则化", label, 42, "RTX 4090；默认运行时；prefix；2 轮",
            reg, ("runs", key, "native_calibrated_metrics"), raw_keys=("runs", key, "raw_metrics"),
            probability="原生温度校准", expected=acc)
    comb = "rtx4090-combined-completed-v1"
    add("02 调度与正则化", "warmup 5% + 标签平滑 0.05", 42, "RTX 4090；默认运行时；prefix；2 轮",
        comb, ("native_calibrated_metrics",), raw_keys=("raw_metrics",), probability="原生温度校准", expected=96.44)
    rd = "rtx4090-rdrop-completed-v1"
    add("02 调度与正则化", "warmup + 平滑 + R-Drop α=1", 42, "RTX 4090；prefix；2 轮；双前向",
        rd, ("validation", "native_calibrated_metrics"), raw_keys=("validation", "raw_metrics"),
        probability="原生温度校准", expected=96.08, note="5526 次更新、11052 次训练前向；未优于组合。")
    for key, label in [("baseline", "4090 两轮基线"), ("combined", "warmup 5% + 标签平滑 0.05")]:
        name = f"reports/laya-rtx4090-ce-{'baseline' if key == 'baseline' else 'combined'}-s43-epochs2-fp32-v1/cuda-saved-validation.json"
        add("02 调度与正则化", label, 43, "RTX 4090；默认运行时；prefix；2 轮", name,
            ("native_calibrated_metrics",), raw_keys=("raw_metrics",), probability="原生温度校准", expected=96.48)
    for key, label, acc in [("control", "45 条重采样：同期控制", 95.96), ("resampled", "45 条重采样：增强", 96.16)]:
        full = f"reports/laya-rtx4090-ce-resampling-{key}-pool45-s43-epochs2-fp32-v2/cuda-saved-validation.json"
        add("03 难例重采样与语义增强", label, 43, "4090；默认运行时；warmup + 平滑；2 轮", full,
            ("native_calibrated_metrics",), raw_keys=("raw_metrics",), expected=acc,
            probability="原生温度校准", note="控制未复现历史 96.48%；来自本地完整重载指标。")
    strict = "laya-strict-pair-result-check-20261013-v1"
    for key, label, acc in [("control", "45 条重采样 strict：同期控制", 96.32), ("resampled", "45 条重采样 strict：增强", 96.40)]:
        add("03 难例重采样与语义增强", label, 43, "4090；strict FP32；warmup + 平滑；2 轮", strict,
            ("runs", key, "final_metrics"), expected=acc, note="冻结摘要只含 Accuracy、计数与 NLL；未补造缺失指标。")
    sem = "semantic30-abc-completed-20261013-v1"
    for arm, label, acc in [("A", "语义 A：原文重复", 96.52), ("B", "语义 B：普通改写", 96.32), ("C", "语义 C：语义改写", 96.44)]:
        add("03 难例重采样与语义增强", label, 42, "strict FP32；30 来源；5% 曝光；2 轮", sem,
            ("metrics", arm), expected=acc, note="A 是同曝光原文重复参照，不是无替换控制。")
    post = "laya-posthoc-consolidated-v2"
    for model, label, seed in [("baseline_s42", "plain", 42), ("baseline_s43", "plain", 43),
                               ("combined_s42", "warmup + 平滑", 42), ("combined_s43", "warmup + 平滑", 43)]:
        for window, mode, title in [("head_tail", "native_temperature", "首尾窗口"),
                                     ("head", "temperature_bias", "前缀 + 温度/偏置"),
                                     ("head_tail", "temperature_bias", "首尾 + 温度/偏置")]:
            original = get(post, "models", model, "metrics", window, mode)
            analysis = f"reports/laya-posthoc-{model.replace('_', '-')}-fp32-v1/analysis.json"
            full_keys = ("rules", window, "validation_metrics", mode)
            full = get(analysis, *full_keys)
            for metric, value in original.items():
                assert abs(full[metric]-value) < 1e-12, (analysis, metric)
            add("04 固定权重输入与校准", f"{label}：{title}", seed, "固定旧权重；仅推理/校准；512 token", analysis,
                full_keys,
                raw_keys=("rules", window, "validation_metrics", "raw") if mode == "native_temperature" else None,
                probability="原生温度校准" if mode == "native_temperature" else "400 条拟合温度 + 偏置",
                note="首尾为前 233 + 后 233；温度本身不改变类别，偏置可以。")
    fusion = "laya-probability-fusion-s43-s42-20261014-v1"
    for model, label, seed in [("baseline-s42", "plain", 42), ("baseline-s43", "plain", 43),
                               ("combined-s42", "warmup + 平滑", 42), ("combined-s43", "warmup + 平滑", 43)]:
        add("05 门控与多模型融合", f"{label}：head/head_tail 门控等权", seed,
            "固定权重；短文 head，长文两窗口概率均值", fusion, ("metrics", f"{model}/gated_equal", "all"),
            probability="原生温度后混合", note="同一权重的多输入，不是新增训练。")
    for key, label, acc, note in [
        ("two_model_fitted", "两模型融合：冻结主方案", 96.96, "400 条校准拟合系数；正式测试采用此方案。"),
        ("two_model_equal", "两模型融合：固定 50:50 辅助对照", 97.00, "预声明辅助结果；没有根据开发集改选为主方案。"),
        ("historical_eight_equal", "历史八视角等权融合：复核值", 96.88, "替代此前未核实口头数字；探索性诊断。"),
        ("historical_eight_ols", "八视角无截距 OLS：诊断", 96.68, "本轮定义的诊断方案，不是最终主模型。"),
    ]:
        add("05 门控与多模型融合", label, "42/43", "固定旧权重；概率融合", fusion,
            ("metrics", key, "all"), probability="校准后融合概率", expected=acc, note=note)
    decision = "laya-decision-study-s43-s42-strict-20261013-v1"
    for model, label, seed in [("baseline-s43", "plain", 43), ("combined-s42", "warmup + 平滑", 42)]:
        for view in ["reverse", "candidate_rerank"]:
            add("06 选项、窗口与重排诊断", f"{label}：{'选项反序' if view == 'reverse' else 'N1 候选重排探针'}", seed,
                "固定旧权重；prefix；无训练", decision, ("models", model, "single_and_fused", view, "all"),
                probability="历史报告概率",
                note="实际 mean/max 集合余弦特征非恒等，拟合系数非零；校准准确率不升，未达训练准入门槛。" if view == "candidate_rerank" else "纯反序与 head 比较；head_tail 与反序还混有窗口变化。")
    win = "laya-decision-execution-20261014-v1-final-study"
    for model, label in [("prefix-trained", "前缀训练控制"), ("random-window-trained", "随机窗口训练")]:
        for view, title in [("head", "前缀推理"), ("head_tail", "首尾推理（主比较）"),
                            ("reverse", "选项反序诊断"), ("gated_equal", "门控等权诊断"),
                            ("candidate_rerank", "N1 候选特征探针")]:
            add("06 选项、窗口与重排诊断", f"{label}：{title}", 42, "strict FP32；2 轮；固定训练预算", win,
                ("models", model, "single_and_fused", view, "all"), probability="历史报告概率",
                note="N4 同期主比较使用双方 head_tail；其余为诊断。" + ("N1 实际特征非恒等，不能按旧文档字符串解释。" if view == "candidate_rerank" else ""))
    avg = "laya-checkpoint-average-prefix-s42-strict-20261014-v1"
    for model, label in [("control", "N8：最终轮控制"), ("average", "N8：epoch 1/2 参数平均 50:50")]:
        add("06 选项、窗口与重排诊断", label, 42, "前缀训练权重；首尾推理；strict FP32；无新更新", avg,
            ("models", model, "all"), probability="各自冻结温度后的概率", expected=96.64,
            note="N4 同一控制 checkpoint/首尾视图，但本轮按首尾重新拟合温度，概率指标不同。")
    probe = "laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1"
    for model, label, acc in [("encoder", "L1：编码器输出线性探针", 96.68), ("post_head", "L1：head 后线性探针", 96.80)]:
        add("07 表示探针与头部训练对照", label, 43, "冻结编码器；训练线性读出；首尾输入", probe,
            ("metrics", model, "all"), expected=acc, probability="探针输出；未另加温度",
            note="不是重新联合微调；完整模型参照 96.84%。")
    for name, group, arm_key in [
        ("laya-head-control-analysis-20261015-v1", "07 表示探针与头部训练对照", "head"),
        ("laya-robust-loss-analysis-20261016-v1", "08 Modal 稳健损失配对训练", "arm"),
    ]:
        for run, obj in get(name, "runs").items():
            arm = obj[arm_key]
            if arm_key == "head":
                label = "L2：重新初始化原始头" if arm == "original" else "L2：共享线性头"
                protocol = "4090；strict CUDA FP32；首尾训练/推理；2 轮"
            else:
                label = "Modal：CE 同期控制" if arm == "ce" else "Modal：0.5 CE + 0.5 GCE (q=0.7)"
                protocol = "L40S；strict CUDA FP32；head/prefix；2 轮"
            row = add(group, label, obj["seed"], protocol, name,
                      ("runs", run, "development", "calibrated_metrics", "all"),
                      raw_keys=("runs", run, "development", "raw_metrics", "all"),
                      probability="固定 400 条温度校准", note="每组 5526 更新；三种子均完整纳入，不能按最优 seed 选点。")
            row["run_id"] = run
            row["strata"] = {k: trim(v) for k, v in obj["development"]["calibrated_metrics"].items() if k != "all"}
    test = "laya-frozen-official-test-20261014-v1"
    for model, label in [("reference", "正式测试：plain-s43 首尾单模型"), ("primary", "正式测试：冻结主融合")]:
        add("09 官方测试", label, "43" if model == "reference" else "42/43", "冻结后一次正式测试；25000 条",
            test, ("metrics", model, "all"), split="official_test", probability="冻结校准/融合概率",
            note="未按测试结果换模型；不与开发集直接作单因素比较。")
    risk = "laya-risk-coverage-20261015-v1"
    for i, pt in enumerate(get(risk, "coverage_points")):
        for method, label in [("single", "首尾单模型"), ("fusion", "冻结主融合")]:
            add("10 风险—覆盖：开发集固定保留比例", label, "43" if method == "single" else "42/43",
                f"按开发置信度保留前 {pt['coverage']:.0%}", risk, ("coverage_points", i, method),
                split="development_selective", coverage=pt["coverage"], probability="冻结概率；保留集合统计",
                note="只报告保留样本准确率；拒答没有节省此前的模型前向。")
    for i, pt in enumerate(get(risk, "calibration_only_thresholds")):
        add("11 风险—覆盖：校准集定阈值", "首尾单模型" if pt["method"] == "single" else "冻结主融合",
            "43" if pt["method"] == "single" else "42/43",
            f"校准目标 {pt['calibration_target_coverage']:.0%}；阈值 {pt['threshold']:.10f}", risk,
            ("calibration_only_thresholds", i), split="development_selective", coverage=pt["development_coverage"],
            probability="校准集阈值固定后开发集统计", note="目标覆盖率与实际覆盖率不同；没有独立部署认证。")
    # Keep the independently held-out common subset separate from full development.
    for key, label in [("modernbert_base", "ModernBERT-base"), ("laya_ce_full", "Laya-CE full"),
                       ("laya_rlcd_pilot", "Laya-RLCD pilot"), ("laya_ce_pilot", "Laya-CE pilot")]:
        add("12 子集诊断", f"{label}：公共 2100 条", 42, "开发集抽出 400 条统一温度拟合，剩余 2100 评估",
            early, (key, "common_evaluation_2100", "calibrated"),
            raw_keys=(key, "common_evaluation_2100", "uncalibrated"), split="development_common2100",
            probability="分析用统一温度", note="不能将此 NLL/ECE 与完整 2500 条准确率拼成一行。")
    for name, models in [(decision, ["baseline-s43", "combined-s42"]),
                          (win, ["prefix-trained", "random-window-trained"])]:
        for model in models:
            for width in ["128", "256", "466"]:
                add("12 子集诊断", f"{model}：EXP-2 滑窗宽度 {width}", 43 if model == "baseline-s43" else 42,
                    "仅 407 条截断评论；predict_long；步长=窗口一半", name,
                    ("models", model, "exp2", width, "metrics"), split="development_truncated407",
                    probability="最置信窗口的原生概率；不是文档级校准", note="不是完整开发成绩；固定短窗口扫描长文，不是新增 >512-token 上下文训练。")
    # Backend parity variants are measured predictions, not independent trained models.
    for backend in ["mps-fp32", "mlx-fp32", "mlx-fp16"]:
        name = f"reports/laya-backend-comparison/{backend}-eval.json"
        obj = read(name)
        metric_key = "native_calibrated_metrics" if "native_calibrated_metrics" in obj else "metrics"
        add("13 工程复核：同权重不同后端", f"Mac 一轮 CE：{backend}", 42, "同一权重；后端核对", name,
            (metric_key,), probability="原生温度校准", expected=95.76, note="三后端标签一致，不能计为三次训练收益。")
    for run in ["laya-mps-ce-epochs2-l512-s42-fp32-v1", "laya-mps-ce-epochs2-lr06-l512-s42-fp32-v1"]:
        add("13 工程复核：同权重不同后端", f"Mac 两轮 {'原 LR' if 'lr06' not in run else 'LR ×0.6'}：MLX FP16",
            42, "同权重 MLX FP16，batch 8", mac, ("runs", run, "mlx", "metrics"),
            probability="原生温度校准", note="标签一致；最大概率差超过 0.01，严格概率核对未通过。")
    for backend, name in [("CUDA FP32", "laya-fusion-inference-cuda-fp32-20261014-v1"),
                           ("MLX FP16", "laya-fusion-inference-mlx-fp16-20261014-v1"),
                           ("MLX precise FP32", "laya-fusion-inference-mlx-fp32-precise-20261014-v1")]:
        for method, label in [("baseline_head_tail", "首尾单模型"), ("primary", "主融合"), ("equal", "50:50 辅助融合")]:
            add("13 工程复核：同权重不同后端", f"{label}：{backend}", "42/43", "已有权重完整开发集工程复核", name,
                ("metrics", "validation", method, "all"), probability="该后端实际校准/融合概率",
                note="CUDA 与 precise FP32 核对通过；MLX FP16 标签一致，但严格概率核对未通过。")


def build_pairs():
    mac = "laya-mac-ce-comparison-v1"
    for key in get(mac, "paired"):
        pair("Mac 轮数/学习率", *key.split(" vs "), mac, ("paired", key), note="单种子探索性配对，未做多重比较校正。")
    reg = "rtx4090-regularization-completed-v1"
    for key in get(reg, "paired"):
        pair("独立正则化", *key.split("__"), reg, ("paired", key))
    comb = "rtx4090-combined-completed-v1"
    for key in ["baseline__combined", "smoothing__combined", "warmup__combined"]:
        pair("组合正则化", *key.split("__"), comb, ("pairs", key))
    rd = "rtx4090-rdrop-completed-v1"
    pair("R-Drop", "warmup+ls", "warmup+ls+rdrop", rd, ("paired_combined_baseline_vs_rdrop",))
    post = "laya-posthoc-consolidated-v2"
    for model in get(post, "models"):
        pair(f"首尾窗口：{model}", "head", "head_tail", post, ("models", model, "raw_transitions", "paired"))
    res = "laya-resampling45-seed43-completed-v1"
    # Some legacy summaries store their p-value alongside the paired counts.
    pair("默认运行时重采样", "control", "resampled", res, ("paired",))
    PAIRS[-1]["statistics"]["mcnemar_exact_two_sided_p"] = get(res, "mcnemar_exact_two_sided_p")
    PAIRS[-1]["additional_p_source"] = loc(res, ("mcnemar_exact_two_sided_p",))
    strict = "laya-strict-pair-result-check-20261013-v1"
    pair("strict 重采样", "control", "resampled", strict, ("paired",))
    sem = "semantic30-abc-completed-20261013-v1"
    for key, a, b in [("0-1", "A", "B"), ("0-2", "A", "C"), ("1-2", "B", "C")]:
        pair("语义增强", a, b, sem, ("paired", key))
    fusion = "laya-probability-fusion-s43-s42-20261014-v1"
    for key in ["two_model_fitted", "two_model_equal", "historical_eight_equal", "historical_eight_ols"]:
        pair("融合", "baseline-s43/head_tail", key, fusion, ("versus_fixed_reference", key, "paired"))
    win = "laya-decision-execution-20261014-v1-final-study"
    pair("N4 随机窗口训练", "prefix-trained/head_tail", "random-window-trained/head_tail", win,
         ("exp3_training_factor", "paired", "head_tail", "paired"))
    avg = "laya-checkpoint-average-prefix-s42-strict-20261014-v1"
    pair("N8 参数平均", "epoch-002", "average-001-002", avg, ("versus_control", "paired"))
    probe = "laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1"
    for model in ["encoder", "post_head"]:
        pair("L1 冻结探针", "full_model", model, probe, ("versus_full_model", model, "all"))
        PAIRS[-1]["conditional_ci95_pp"] = get(probe, "versus_full_model", model, "all", "paired_stratified_bootstrap95_pp")
    head = "laya-head-control-analysis-20261015-v1"
    for i, p in enumerate(get(head, "groups", "all", "independent_per_seed")):
        pair(f"L2 seed {p['seed']}（原始−线性）", "linear", "original", head,
             ("groups", "all", "independent_per_seed", i), ci=p["conditional_bootstrap95_pp"])
    robust = "laya-robust-loss-analysis-20261016-v1"
    for i, p in enumerate(get(robust, "groups", "all", "per_seed")):
        pair(f"稳健损失 seed {p['seed']}", "ce", "mixed", robust,
             ("groups", "all", "per_seed", i), ci=p["conditional_bootstrap95_pp"])
    test = "laya-frozen-official-test-20261014-v1"
    pair("官方测试（冻结后）", "首尾单模型", "主融合", test, ("paired_comparison", "paired"))


def fmt(value, kind="float"):
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value*100:.2f}%"
    if kind == "int":
        return str(int(value))
    return f"{value:.6f}"


def interval(values, digits=3):
    return "[" + ", ".join(f"{x:+.{digits}f}" for x in values) + "]"


def md_table(headers, rows):
    escape = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(escape, headers)) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |"] +
                      ["| " + " | ".join(map(escape, row)) + " |" for row in rows]) + "\n"


def evidence(row):
    name = row["source"]["file"]
    return f"[来源](../../{name})"


def report(output: Path):
    main = [r for r in ROWS if int(r["group"][:2]) <= 8]
    groups = Counter(r["group"] for r in ROWS)
    lines = ["# IMDB 全部实验指标总表", "", "更新：2026-10-16（北京时间）。",
             f"本表从本地冻结报告与机器摘要整理了 **{len(main)} 条完整开发集方案记录**，另列官方测试、拒答、子集和后端复核；共 {len(ROWS)} 条指标记录、{len(PAIRS)} 组配对比较。记录数不是独立训练次数，同权重输入变体和重复后端均已标注。本次没有训练、新推理或访问官方测试逐样本文件。", "",
             "## 1. 先看可用于报告的成绩", "",
             md_table(["用途", "方案", "Accuracy", "范围与边界"], [
                 ["正式测试主成绩", "冻结两模型主融合", "96.344%（24086/25000）", "冻结后正式测试，未按结果改选模型"],
                 ["正式测试单模型参照", "plain-s43 首尾", "96.244%（24061/25000）", "同期冻结参照；主融合净多对 25 条"],
                 ["开发集最高辅助结果", "两模型固定 50:50", "97.00%（2425/2500）", "预声明辅助对照，没有改选成正式主方案"],
                 ["开发集冻结主方案", "两模型拟合单系数融合", "96.96%（2424/2500）", "400 条校准拟合混合系数"],
                 ["历史单模型最高", "Mac 两轮、原学习率、seed42", "96.96%（2424/2500）", "单种子；并非上行融合或同期 CUDA 控制"],
                 ["最新三种子稳健损失", "0.5 CE + 0.5 GCE(q=0.7)", "96.400%（seed 均值）", "CE 同期均值 96.320%；+0.080 pp，条件性区间跨零"],
                 ["开发集选择性成绩", "主融合，保留 90% / 80%", "98.84% / 99.55%", "分别拒绝 10% / 20%，不是全覆盖成绩"],
             ]),
             "## 2. 指标与比较口径", "",
             "Accuracy 为正确率，Macro-F1 为宏平均 F1，AUROC 为排序区分能力，三者越高越好；NLL 为负对数似然，ECE 为校准误差，Brier 为概率平方误差，三者越低越好。F1/AUROC 用 0–1 小数；ECE 在表中转为百分数；pp 表示百分点。Brier 统一展示正类版本，二分类双类平方误差之和恰为其 2 倍，原值同时保留在 JSON 中。", "",
             "主表均为完整开发集 2500 条、100% 覆盖。NLL/ECE 使用该行的「概率口径」；有原始与温度校准两套指标时，表中用箭头表示原始 → 校准。缺失项写「—」，不借用其他分区或其他后端的指标。原生温度、统一分析温度、温度加偏置、融合概率不可混称。", "",
             "不同平台、随机种子、输入策略、头初始化、训练预算和默认/strict 运行时的结果可作历史记录，不能直接作单因素因果比较。开发集反复使用，表中成绩属于探索性结果；历史开发—测试约 0.6 pp 的落差不能作为所有新模型的统一扣减项。", "",
             "## 3. 完整开发集方案总表", ""]
    for group in sorted({r["group"] for r in main}):
        lines.extend([f"### {group}", ""])
        table = []
        for row in [r for r in main if r["group"] == group]:
            m, raw = row["metrics"], row.get("raw_metrics")
            nll = fmt(m.get("nll"))
            ece = fmt(m.get("ece_15_equal_width_top_label"), "pct")
            if raw:
                nll = fmt(raw.get("nll")) + " → " + nll
                ece = fmt(raw.get("ece_15_equal_width_top_label"), "pct") + " → " + ece
            table.append([row["id"], row["method"], row["seed"], fmt(m["accuracy"], "pct"),
                          fmt(m["count"] - m["correct"], "int") if "correct" in m else "—",
                          fmt(m.get("macro_f1")), fmt(m.get("auroc")), nll, ece,
                          fmt(m.get("brier_positive")), evidence(row)])
        lines.extend([md_table(["ID", "方案", "seed", "Accuracy", "错误数", "Macro-F1", "AUROC", "NLL", "ECE", "Brier 正类", "证据"], table), ""])
    lines.extend(["### 主表协议、概率与解释", "",
                  md_table(["ID", "训练/输入/运行协议", "概率口径", "说明"],
                           [[r["id"], r["protocol"], r["probability_scope"], r["note"]] for r in main]), "",
                  "## 4. 同期配对比较与统计证据", "",
                  "单种子精确 McNemar p 只检验当前固定权重在当前样本上的配对差异；通常未做多重比较校正，不证明跨训练稳定。三种子共享 ID bootstrap 区间条件于已训练模型，不包含训练种子总体、校准重拟合和开发选择偏差；区间跨零不证明等效。", ""])
    pairrows = []
    for p in PAIRS:
        s = p["statistics"]
        repairs = s.get("b_only_correct", s.get("candidate_only_correct", s.get("repairs", s.get("original_only_correct", s.get("fixed")))))
        regressions = s.get("a_only_correct", s.get("reference_only_correct", s.get("regressions", s.get("linear_only_correct", s.get("regressed")))))
        delta = (repairs-regressions)/25 if repairs is not None and regressions is not None else s.get("delta_accuracy_pp", s.get("delta_pp"))
        # The official test has ten times as many observations as development.
        if p["comparison"].startswith("官方") and delta is not None:
            delta /= 10
        pv = next((v for k,v in s.items() if "mcnemar" in k and "p" in k), None)
        pairrows.append([p["comparison"], p["reference"] + " → " + p["candidate"],
                         f"{repairs}/{regressions}" if repairs is not None else "—",
                         f"{delta:+.3f}" if delta is not None else "—", fmt(pv),
                         interval(p["conditional_ci95_pp"]) if "conditional_ci95_pp" in p else "—"])
    lines.extend([md_table(["比较", "参照 → 候选", "修复/退化", "Δ Accuracy（pp）", "名义 p", "条件性 95% 区间（pp）"], pairrows), ""])
    head = get("laya-head-control-analysis-20261015-v1", "groups", "all")
    robust = get("laya-robust-loss-analysis-20261016-v1", "groups", "all")
    lines.extend([md_table(["三种子比较", "参照均值", "候选均值", "候选−参照（pp）", "固定六模型共享样本 95% 区间（pp）", "解释"], [
        ["L2：线性 → 原始头", "96.493%", "96.520%", f"{head['mean_original_minus_linear_pp']:+.4f}", interval(head["shared_id_stratified_bootstrap95_mean_delta_pp"]), "没有检出原始头稳定优势，也没有证明两者等效"],
        ["稳健损失：CE → mixed", "96.320%", "96.400%", f"{robust['mean_mixed_minus_ce_pp']:+.4f}", interval(robust["conditional_shared_id_bootstrap95_pp"]), "均值未达预设 +0.20 pp 继续门槛，不自动扩大搜索"],
    ]), "", "## 5. 官方测试完整指标", ""])
    tests = [r for r in ROWS if r["split"] == "official_test"]
    lines.extend([md_table(["方案", "正确/总数", "Accuracy", "Macro-F1", "AUROC", "NLL", "ECE", "Brier 正类", "证据"],
                          [[r["method"], f"{r['metrics']['correct']}/{r['metrics']['count']}", f"{100*r['metrics']['accuracy']:.3f}%",
                            fmt(r["metrics"].get("macro_f1")), fmt(r["metrics"].get("auroc")), fmt(r["metrics"].get("nll")),
                            fmt(r["metrics"].get("ece_15_equal_width_top_label"), "pct"), fmt(r["metrics"].get("brier_positive")), evidence(r)] for r in tests]), "",
                  "正式主融合比单模型多对 25 条（+0.100 pp），精确 McNemar 名义 p 见上表。只报告已归档官方测试摘要，未重新打开官方测试逐条预测。项目仍未达到全覆盖 98%。", "",
                  "## 6. 风险—覆盖：保留正确率必须连同覆盖率报告", ""])
    for group in ["10 风险—覆盖：开发集固定保留比例", "11 风险—覆盖：校准集定阈值"]:
        lines.extend([f"### {group}", ""])
        selected = [r for r in ROWS if r["group"] == group]
        lines.extend([md_table(["方案", "规则", "实际覆盖率", "正确/保留", "错误", "保留 Accuracy", "Wilson 95% 区间"],
                              [[r["method"], r["protocol"], fmt(r["coverage"], "pct"),
                                f"{r['metrics']['correct']}/{r['metrics']['count']}", r["metrics"]["count"]-r["metrics"]["correct"],
                                fmt(r["metrics"]["accuracy"], "pct"),
                                "～".join(fmt(x, "pct") for x in r["metrics"]["accuracy_wilson95"])] for r in selected]), ""])
    risk = "laya-risk-coverage-20261015-v1"
    aurc = get(risk, "aurc")
    lines.extend([f"AURC（越低越好）：单模型 {aurc['single']['estimate']:.8f}，主融合 {aurc['fusion']['estimate']:.8f}；差值的分层配对 95% 区间为 {get(risk, 'delta_aurc_fusion_minus_single')['paired_stratified_bootstrap95']}，跨零。90% 固定覆盖时两者同为 98.84%，不能写成融合独有收益。校准定阈值的覆盖率区间与截断分层见 [风险—覆盖正式报告](../laya-risk-coverage-20261015-v1/REPORT.md)。", "",
                  "## 7. 公共子集与窗口编辑诊断", "",
                  md_table(["ID", "方案", "分区/样本数", "Accuracy", "NLL（报告概率）", "ECE", "证据"],
                           [[r["id"], r["method"], f"{r['split']} / {r['metrics']['count']}", fmt(r["metrics"]["accuracy"], "pct"),
                             fmt(r["metrics"].get("nll")), fmt(r["metrics"].get("ece_15_equal_width_top_label"), "pct"), evidence(r)]
                            for r in ROWS if r["group"] == "12 子集诊断"]), "",
                  "pilot 原生 256 条评估中，CE 和 RLCD 均为 93.75%；这与后来完整 2500 条上的 93.52% / 93.72% 使用不同评估范围。ModernBERT 第 1/2/3 轮开发 Accuracy 为 94.16% / 94.48% / 94.64%，原始 NLL 为 0.219651 / 0.189119 / 0.284442。中间轮数仅作训练轨迹，不冒充独立方法。", "",
                  "## 8. 后端一致性与成本", "",
                  md_table(["ID", "同权重后端结果", "Accuracy", "NLL", "ECE", "证据"],
                           [[r["id"], r["method"], fmt(r["metrics"]["accuracy"], "pct"), fmt(r["metrics"].get("nll")),
                             fmt(r["metrics"].get("ece_15_equal_width_top_label"), "pct"), evidence(r)]
                            for r in ROWS if r["group"].startswith("13")]), ""])
    costs = []
    for backend, name in [("CUDA FP32", "laya-fusion-inference-cuda-fp32-20261014-v1"),
                           ("MLX FP16", "laya-fusion-inference-mlx-fp16-20261014-v1"),
                           ("MLX precise FP32", "laya-fusion-inference-mlx-fp32-precise-20261014-v1")]:
        b = get(name, "benchmark")
        for method in ["single", "fusion"]:
            v = b[method]
            costs.append([backend, method, f"{v['reviews_per_second']:.2f}", f"{v['amortized_milliseconds_per_review']:.2f}",
                          "排除加载；含分词、模板、传输与推理；摊销耗时不是单请求延迟"])
    lines.extend([md_table(["后端", "方案", "吞吐（条/秒）", "摊销毫秒/条", "范围"], costs), "",
                  "Mac 一轮模型的 MLX FP16 端到端推理较 MPS FP32 提速 2.52～3.68 倍（依长度/batch）；这同时改变框架和精度，不能表述为训练提速。两轮 Mac 权重及融合的 MLX FP16 结果虽然类别一致，部分概率差超过 0.01，严格概率验收未通过；explicit precise FP32 的融合路径通过。", "",
                  "L2 原始头 / 线性头三次均值：训练进程 42.21 / 39.34 分钟，峰值 allocated 显存 7.827 / 7.440 GiB，可训练参数 421029889 / 394782720。线性头耗时少 6.81%、显存少 4.94%、参数少 6.23%，尚没有独立部署吞吐提升证据。", "",
                  "## 9. 诊断、未运行方向和完整性边界", "",
                  md_table(["事项", "已有结果/状态", "不能写成"], [
                      ["N1 候选重排", "实际 mean/max 集合余弦特征非恒等、拟合系数非零；校准准确率不升，未达训练门槛", "沿用最早口头结论，称此次实际运行是恒等操作；或外推所有结构均无效"],
                      ["选项反序", "不同权重净方向相反，15 个翻转均为边界样本", "稳定选项顺序偏差主线"],
                      ["oracle 互补上限", "18 个互异视角，40 条全错；98.40%", "已实现的 98.40% 分类器；22 文件不能计作22独立模型"],
                      ["训练共同错误", "94/22100 候选，来自 baseline-s43 与 combined-s43", "跨 seed 一致错标；复核分类不是确认标签噪声"],
                      ["45 条池 / 30 组语义试制", "冻结与审核已完成，完整训练成绩列于主表", "输入内标签一致率等于开发准确率"],
                      ["短程确定性/CUDA 准入", "strict 短程一致；AMP FP16 梯度溢出未通过", "完整开发泛化成绩或所有长程逐位复现保证"],
                      ["Modal 中断/恢复/接收", "本轮6份最终权重和6组开发评估已本地校验", "每次恢复或下载都是一次新的独立模型实验"],
                      ["ModernBERT-base 公平512-token迁移", "尚未取得正式完成结果", "拿历史256-token结果当同预算迁移对照"],
                      ["Laya-RLCD 全量迁移", "未运行；已有的仅为 pilot", "93.72% 是全量RLCD成绩"],
                      ["扩大语义池、蒸馏、新增长上下文训练", "没有已完成正式结果；固定权重滑窗诊断已单列", "把方案或预估收益填入实测成绩表；称滑窗从未运行"],
                  ]), "",
                  "旧 E01–E43 台账的已做/未做状态截至 2026-10-13；其中官方测试、集成与后续训练现已完成相应工作，以本报告和各新报告为准。更长上下文、分块与公平迁移等未完成部分仍然保留，不将一个组合方向整体标成已完成。", "",
                  "## 10. 来源与复算", "",
                  f"机器可读版本：[summary.json](summary.json)。精确数值、每行来源 JSON keys、原始/报告概率、分层、{len(PAIRS)} 组配对结果和 {len(SOURCES)} 份输入文件 SHA-256 均保留。所有来源在输出前后重新计算 SHA，确保整理未改写历史证据。", "",
                  "构建程序：`scripts/build_all_experiment_metrics.py`。仅使用标准库读取已保存摘要；不读评论正文、权重或逐条官方测试预测。原目录存在时拒绝覆盖；重新生成请指定新的输出目录。", "",
                  "```bash", "UV_CACHE_DIR=/private/tmp/end-imdb-review-uv-cache \\",
                  "uv run --locked --offline --no-sync python scripts/build_all_experiment_metrics.py \\",
                  "  --output /private/tmp/end-imdb-all-metrics-review", "```", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return {"main_rows": len(main), "all_rows": len(ROWS), "paired_comparisons": len(PAIRS), "by_group": dict(groups)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/all-experiments-metrics-20261016-v1")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite: {args.output}")
    build_rows()
    build_pairs()
    for row in ROWS:
        m = row["metrics"]
        assert 0 <= m["accuracy"] <= 1
        if int(row["group"][:2]) <= 8:
            assert m["count"] == 2500 and row["coverage"] == 1.0
        for metrics in [m, row.get("raw_metrics", {})]:
            if "brier_positive" in metrics and "brier_multiclass" in metrics:
                assert abs(metrics["brier_positive"]*2 - metrics["brier_multiclass"]) < 1e-12
    args.output.mkdir(parents=True)
    counts = report(args.output)
    for file, sha in SOURCES.items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == sha, file
    result = {"as_of": "2026-10-16", "scope": "saved model, input, calibration, fusion, probe, test and selective results",
              "new_training": False, "new_inference": False, "official_test_sample_files_read": False,
              "counts": counts, "rows": ROWS, "paired_comparisons": PAIRS, "source_sha256": SOURCES,
              "multiple_seed_comparisons": {
                  "head_control": get("laya-head-control-analysis-20261015-v1", "groups"),
                  "robust_loss": get("laya-robust-loss-analysis-20261016-v1", "groups"),
              },
              "n1_actual_nonidentity_fit": get("laya-decision-study-s43-s42-strict-20261013-v1", "n1_calibration_fits"),
              "risk_coverage": {
                  "aurc": get("laya-risk-coverage-20261015-v1", "aurc"),
                  "delta_aurc": get("laya-risk-coverage-20261015-v1", "delta_aurc_fusion_minus_single"),
                  "strata": get("laya-risk-coverage-20261015-v1", "strata_at_global_coverage"),
              },
              "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
