"""Generate a source-backed stage report after full baseline completion."""
import json
from common import ROOT


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    directory = "reports/modernbert-l256-s42"
    result = load(f"{directory}/result.json")
    if result["status"] != "complete":
        raise ValueError("Training not complete")
    resume = load("reports/day3_resume_check.json")
    rust = load("reports/day3_rust_tokenizer_check.json")
    caches = [load(f"reports/rust_cache_{split}_l256.json") for split in ("train", "validation")]
    errors = load(f"{directory}/error_analysis.json")
    inference = load(f"{directory}/inference_benchmark.json")
    lines = ["# Day3 实际结果", "", "阶段编号按进度推进；执行日期：2026-10-09。", "",
        "## 完整基线", "", "ModernBERT-base 全量微调，原配 tokenizer，长度256，batch8，float32，TF32关闭，seed42。",
        "训练22,500条，验证2,500条；按验证Macro-F1选择权重，官方测试集未用于训练、择优或本次错误分析。", "",
        "| Epoch | 在线训练loss | 训练监测Accuracy | 训练监测Macro-F1 | 验证loss | 验证Accuracy | 验证Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["history"]:
        val, train = row["validation"], row["train_monitor_fixed_subset"]
        lines.append(f"| {row['epoch']} | {row['online_train_loss']:.4f} | {train['accuracy']:.2%} | {train['macro_f1']:.4f} | {val['loss']:.4f} | {val['accuracy']:.2%} | {val['macro_f1']:.4f} |")
    best = result["best_validation"]
    selected_train = next(row["train_monitor_fixed_subset"] for row in result["history"] if row["epoch"] == result["best_epoch"])
    lines += ["", f"最佳为epoch {result['best_epoch']}：验证Accuracy **{best['accuracy']:.2%}**，Macro-F1 **{best['macro_f1']:.4f}**，loss {best['loss']:.4f}。",
        f"混淆矩阵（行是真实负/正，列是预测负/正）：`{best['confusion_matrix_rows_true']}`。",
        f"最佳轮次的训练监测Accuracy为{selected_train['accuracy']:.2%}，比验证高{(selected_train['accuracy']-best['accuracy'])*100:.2f}个百分点；训练拟合更好不能单独证明泛化提升。",
        f"完成{result['epochs_completed']}轮；早停触发：{result['early_stopped']}。重载最佳权重后完整验证复核：{result['checkpoint_reload_verified']}。", "",
        f"训练更新累计 {result['elapsed_training_seconds']/60:.1f} 分钟；MLX分配器峰值 {result['mlx_peak_gib']:.2f} GiB。该时间不含验证、检查点I/O、加载和分词，不等同于总墙钟时间。", "",
        "训练监测是固定2,048条均衡训练子集的轮末评估；不是完整训练集指标。在线loss来自不断更新的模型，不能直接当作轮末训练loss。",
        "本次仅一个seed和一个划分，尚无正式测试或多seed泛化结论。后续使用最佳轮次权重，不因最后一轮训练loss更低而覆盖它。", "",
        "三轮上限与patience=2组合不会提前减少本次轮数；本阶段主要通过验证检查点择优保留最终模型。", "",
        "第三轮验证loss从第二轮的0.1891回升至0.2844，Macro-F1仅提高约0.0016（多正确4条）；训练监测Accuracy达到99.51%。出现过拟合或置信度过高的迹象，不能宣称已经做到不过拟合。",
        "本次继续遵守事先规定的Macro-F1择优规则，选第三轮；第二轮权重也保留，其验证loss更低，后续正则化与概率校准对照可据此分析。", "",
        "## 验证集长度分层", "", "| 分层 | 条数 | Accuracy | Macro-F1 |", "|---|---:|---:|---:|"]
    for name in ("fits_256", "truncated_256"):
        metrics = errors["groups"][name]
        lines.append(f"| {'原始长度≤256' if name == 'fits_256' else '原始长度>256、发生截断'} | {metrics['count']} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.4f} |")
    lines += ["", f"验证评论截断比例 {errors['truncated_fraction']:.2%}。分层是描述性结果，评论难度和长度同时变化，不能将差异直接归因为截断。",
        "高置信度错误及文本摘录见 `modernbert-l256-s42/error_analysis.json`；概率未经校准。", "",
        "## 本机推理参照", "", "常驻模型和tokenizer，验证评论，固定padding到256；每种batch预热5批、计时32批，Metal同步。", "",
        "| Batch | 分词及输入数组均值 | 模型及softmax均值 | 端到端均值 | 端到端P95 | 吞吐 | MLX峰值 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for probe in inference["probes"]:
        lines.append(f"| {probe['batch_size']} | {probe['tokenizer_plus_input_arrays']['mean_ms']:.2f}ms | {probe['model_plus_softmax']['mean_ms']:.2f}ms | {probe['end_to_end']['mean_ms']:.2f}ms | {probe['end_to_end']['p95_ms']:.2f}ms | {probe['reviews_per_second']:.1f}条/秒 | {probe['mlx_peak_gib']:.2f}GiB |")
    lines += ["", "不包含进程启动、权重/tokenizer加载、文本预处理和CLI输出。各部分与端到端独立计时，不要求恰好相加；这是单进程短程本机参照，非多次隔离性能结论。", "",
        "## 续训与 Rust 流水线", "", f"含dropout合成网络的下一步恢复核对：{resume['passed']}；loss误差{resume['loss_abs_error']}，参数最大误差{resume['parameter_max_abs_error']}，优化器状态最大误差{resume['optimizer_max_abs_error']}。配置错配拒绝：{resume['identity_mismatch_rejected']}。",
        "完整基线保存模型、优化器、RNG、进度与文件哈希；合成核对不等同于完整基线已做实际中断重启。", "",
        f"Rust Unicode/特殊token/截断/尾批检查通过：{rust['passed']}。", "",
        "| Rust缓存 | 全量核对条数 | 含进程启动及JSONL读写耗时 |", "|---|---:|---:|"]
    for cache in caches:
        lines.append(f"| {cache['identity']['split']}，L256 | {cache['verification']['rows']} | {cache['timing']['wall_seconds']:.2f}秒 |")
    lines += ["", "全量逐条比较id、label、token ID、attention mask、Unicode字符位置，缓存发布后附内容哈希。生成缓存时训练仍在运行，耗时不是隔离测速。Python接口本身也使用Rust库，此核对证明集成一致，不证明独立算法正确或推理更快。",
        "本次训练使用原Python bindings；Rust缓存留给后续接入，未在训练中途改变输入实现。", "",
        "## 后续", "", "先用长度512对照检验信息截断的影响，再结合错误分析决定领域分词适配的范围。蒸馏和多seed验证继续沿用已冻结的数据协议；正式测试仍留到方案冻结。", "",
        "训练配置见 `../configs/day3.json`，方法约定见 `../docs/day3-plan.md`，逐轮原始记录见 `modernbert-l256-s42/result.json`。"]
    (ROOT / "reports/day3.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Wrote reports/day3.md")


if __name__ == "__main__":
    main()
