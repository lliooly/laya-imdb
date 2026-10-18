# 推理入口与成绩对应关系

更新：2026-10-18。[整理前原文](../archive/closeout-20261018/inference-before-closeout.md)保留早期操作记录。

## 默认历史单模型

首次安装与离线运行，从仓库根目录执行：

```bash
uv sync --locked --extra laya-mlx
uv run --locked --offline --no-sync python scripts/predict.py \
  'The acting was excellent, and I loved the ending.'
```

默认读取 checkpoints/laya-l512-s42-ce，历史开发 95.76%。MLX FP16 是该单模型默认部署路线。输出类别、原生温度概率、未校准概率、输入长度和截断信息。

--checkpoint 可指定已存在并核验的同结构本地权重；不会自动下载回退模型。目录需有 model.safetensors、rl_agent_config.json、encoder/config.json 和完整 tokenizer 目录。独立 Top32 头权重不能直接交给旧加载器。

## 旧 Laya 多视图融合

scripts/predict_laya_fusion.py 需要显式--config，属于开发 96.96%、测试 96.344% 的旧主方案。配置及精确 MLX FP32 核验见[Laya 融合指南](laya-fusion-inference.md)。dtype 为 FP32 本身不保证已通过严格概率核对。

## 最新异构研究融合

Laya 主概率与 DeBERTa 按 50:50 平均：开发 97.28%，测试 97.028%。测试使用 CPU FP32 DeBERTa 和旧 CUDA Laya 缓存，开发使用 CUDA BF16 DeBERTa。见[正式报告](../../reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md)。

scripts/deberta_final_analysis.py 与 scripts/deberta_final_evaluation.py 是冻结研究分析/评估入口，并非已交付的通用单条 CLI。两路加载、编码、概率一致性、性能和视频演示仍需整合验收。不要为了演示直接重新运行官方测试。

97.40% 正则化 logit 候选仅有开发成绩。最终交付边界见[交付清单](../closeout/delivery.md)。
