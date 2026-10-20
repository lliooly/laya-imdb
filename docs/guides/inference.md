# 推理与评估入口

以下命令从仓库根目录执行。首次安装见 [环境与复现](../closeout/reproduction.md)，所有推理均需提前准备对应本地模型。

## 入口选择

| 目的 | 入口 | 结果对应 |
| --- | --- | --- |
| Apple Silicon 单条影评预测 | `scripts/predict.py` | 历史单模型，开发 95.76% |
| 双 Laya 多视图推理 | `scripts/predict_laya_fusion.py --config …` | 双 Laya：开发 96.96%、测试 96.344% |
| 最终异构方案研究评估 | `scripts/deberta_final_analysis.py`、`deberta_final_evaluation.py` | 固定融合：开发 97.28%、测试 97.028% |

最终异构方案包含三个 checkpoint。当前仓库内研究评估读取绑定资产和缓存。单独交付的模型包提供 `predict_final.py`，已完成 CPU 单条输入与哈希核验，命令和依赖见随包运行说明；仓库内未收录该包装入口与权重。单模型与双 Laya 命令不会自动调用 DeBERTa。

## Apple Silicon 单模型

```bash
uv sync --locked --extra laya --extra laya-mlx
uv run --locked --offline --no-sync python scripts/predict.py \
  'The acting was excellent, and I loved the ending.'
```

默认模型位于 `checkpoints/laya-l512-s42-ce/`，使用 MLX FP16。输出 JSON 包括类别、保存温度下的概率、原始概率、输入 token 数和截断信息。类别值为 `negative` / `positive`。

```bash
uv run --locked --offline --no-sync python scripts/predict.py \
  'The plot was predictable, but the performances were wonderful.' \
  --checkpoint checkpoints/laya-l512-s42-ce --dtype float32
```

完整目录需包含 `model.safetensors`、`rl_agent_config.json`、`encoder/config.json` 及 `tokenizer/`；不自动下载替代权重。`--checkpoint` 仅接受加载器支持的结构，独立 Top32 头或 DeBERTa 权重不能直接传给它。

## 双 Laya 多视图

入口必须显式指定配置，详细资产要求、CUDA 命令和性能核验见 [双 Laya 推理指南](laya-fusion-inference.md)。配置包含设备、两份 checkpoint、保存温度、冻结权重及哈希；其中一些路径绑定原机器，换机器须重新核验。

Mac 上显式精确 FP32 的既有入口为：

```bash
MLX_ENABLE_TF32=0 uv run --locked --offline --no-sync \
  python scripts/predict_laya_fusion.py \
  'The acting was excellent, and I loved the ending.' \
  --config configs/laya-fusion-inference-mlx-fp32-precise-20261014-v1.json
```

该入口需要根环境的 `laya` 与 `laya-mlx` extras。精确 FP32 已在 2,900 条校准/开发记录上通过原概率门槛。MLX FP16 的类别虽与 CUDA 一致，但严格概率核验未通过，不能将切换 `dtype` 等同于跨后端验证。

## 最终异构评估

冻结方案平均双 Laya 分支与 DeBERTa 的校准正类概率。原测试使用 CPU FP32 DeBERTa 和已冻结的 CUDA FP32 Laya 缓存；开发使用 CUDA BF16 DeBERTa。完整开发集后端核验中最终融合类别一致，但单分支概率并不逐位等价。

[完整测试报告](../../reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md) 记录 97.028%；[三组重训](../../reports/paper-gpu-supplement-20261019-v1/REPORT.md) 与 [稳健性分析](../../reports/paper-fusion-robustness-20261020-v1/REPORT.md) 已纳入最终报告。97.40%、80:20 与校准拟合权重均属于开发探索，没有替换最终冻结规则。

本地演示与常规交付检查无需重新执行官方测试。统一三模型加载、编码、概率一致性和性能仍需独立整合验收，见 [交付状态](../closeout/delivery.md)。
