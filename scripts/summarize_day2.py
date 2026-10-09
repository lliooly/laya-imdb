"""Generate the Day2 readout from measured parity, training and benchmark results."""
import json
from common import ROOT


def main():
    parity = json.loads((ROOT / "reports/day2_parity.json").read_text())
    bench = json.loads((ROOT / "reports/day2_benchmark.json").read_text())
    pilot = json.loads((ROOT / "reports/day2_pilot.json").read_text())
    diagnostic = json.loads((ROOT / "reports/day2_float64_diagnostic.json").read_text())
    if parity["status"] != "passed" or pilot["status"] != "passed" or len(bench["probes"]) != 8:
        raise RuntimeError("Day2 results incomplete")
    lines = ["# Day2 实际结果", "", "执行日期：2026-10-09。Day2是开发阶段编号；用户已要求开始，提前于原计划10/10执行。", "",
        "## 完成情况", "", "原始 ModernBERT-base → 自主MLX编码器 → masked-mean线性分类头 → 真实评论全量微调 → 保存加载 → 手动推理链路已打通。",
        "官方测试集未用于此次训练、验证或选型。", "",
        "## 数值核对", "", f"- Transformers {parity['versions']['transformers']} / PyTorch {parity['versions']['torch']}，CPU eager float32参照；MLX {parity['versions']['mlx']} Metal。",
        "- 显式设置 `MLX_ENABLE_TF32=0`，避免M5默认低精度float32矩阵路径影响核对。",
        f"- 随机3层结构与全部梯度严格逐元素核对：{parity['tiny_architecture']['passed']}。",
        "- 预训练22层有效token表示与自主分类头：", "",
        "| 输入长度 | hidden最大绝对误差 | hidden相对L2误差 | logits最大绝对误差 |",
        "|---:|---:|---:|---:|"]
    for length, r in parity["cases"].items():
        lines.append(f"| {length} | {r['hidden']['max_abs_error']:.3g} | {r['hidden']['relative_l2']:.3g} | {r['logits']['max_abs_error']:.3g} |")
    max_gradient = max(g["relative_l2"] for g in parity["gradients"].values())
    lines += ["", f"长度{parity['gradient_check_length']}的所检查预训练参数梯度最大相对L2误差为 {max_gradient:.3g}；loss、padding扩展和屏蔽ID扰动检查通过。",
        "极短截断样本具有大激活、数值敏感性：长度16/160的hidden逐元素allclose仍有失败位置，但整体相对L2满足3e-4工程标准。",
        "16-token梯度诊断约0.6%相对误差，未宣称该条件反向等价；长度256反向通过1e-3标准。",
        f"同一极短样本的float64仲裁：PyTorch float32相对误差 {diagnostic['results']['torch32']['relative_l2']:.3g}，MLX float32 {diagnostic['results']['mlx32']['relative_l2']:.3g}。",
        "核对标准调整原因、失败诊断及全部误差公开保留于JSON和 `docs/day2-plan.md`；通过关卡不代表逐位相同。", "",
        "## 全量微调测速", "", "每组独立进程；2步预热+5步计时，全部编码器参数参与反向，包含裁剪和AdamW。真实train评论预先分词，固定padding；TF32关闭。", "",
        "| 参数精度 | 长度 | batch | 秒/步 | 评论/秒 | MLX峰值GiB | 22,500条/epoch分钟 | 3 epoch分钟 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in bench["probes"]:
        lines.append(f"| {p['dtype']} | {p['length']} | {p['batch_size']} | {p['mean_step_seconds']:.3f} | {p['reviews_per_second']:.1f} | {p['mlx_peak_gib']:.2f} | {p['train_22500_epoch_minutes']:.1f} | {p['train_3_epochs_minutes']:.1f} |")
    lines += ["", "以上是短程外推，不含验证、检查点I/O和长期热降频；MLX峰值不是系统总内存。当前局部注意力使用稠密mask，尚未实现去padding或稀疏优化。",
        "bfloat16虽然更快且本次梯度有限，但采用低精度参数和优化器状态，未验证有效更新率或收敛；初始完整基线仍用float32。", "",
        "## 真实评论短程训练", "", "固定seed42，train256条、validation128条，类别均衡；float32、长度256、batch8，32步全量微调，固定学习率2e-5。", "",
        "| 阶段 | 划分 | Loss | Accuracy | Macro-F1 |", "|---|---|---:|---:|---:|"]
    for stage in ("before", "after"):
        for split, r in pilot[stage].items():
            lines.append(f"| {stage} | {split} | {r['loss']:.4f} | {r['accuracy']:.2%} | {r['macro_f1']:.4f} |")
    lines += ["", f"末层编码器MLP最大权重变化 {pilot['encoder_last_mlp_max_update']:.3g}，确认编码器参与了学习；梯度有限。保存重载最大输出误差 {pilot['save_reload_max_abs_error']}。",
        "128条验证评论仅是流程诊断，不能当作完整验证集或官方测试准确率，更不能证明已解决过拟合。尚未运行最终训练的warmup、早停或多种子实验。",
        "模型位于 `checkpoints/day2-pilot/`，包含模型参数、配置和运行记录；未保存优化器状态，不能据此精确续训。", "",
        "## Day3建议与服务器判断", "", "目前无需租服务器。优先本机float32、batch8，用256输入打通完整22,500条基线；512输入保留作后续长度对照。",
        "按本次测量，256输入3个epoch训练本体约1小时，512输入约2小时；安排时额外预留至少50%的验证、保存和持续运行缓冲（该缓冲是规划假设，不是实测）。",
        "Day3落实学习率warmup、完整验证集、按Macro-F1选checkpoint、早停与完整实验记录；先单seed，冻结候选后才进行多seed。",
        "若完整epoch实测明显慢于外推或扩展实验累计超出时间预算，再重新评估服务器；本轮未验证CUDA后端及跨设备权重一致性。", ""]
    (ROOT / "reports/day2.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote reports/day2.md")


if __name__ == "__main__":
    main()
