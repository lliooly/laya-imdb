# 环境、数据与复现

使用 Python 3.12、uv 和提交的依赖锁；所有命令从仓库根目录执行。Git 仓库不包含数据、微调权重和完整预测缓存，复现历史成绩前须恢复对应资产。

## 环境安装

| 环境 | 安装命令 | 用途 |
| --- | --- | --- |
| 根环境：Mac Laya / MLX | `uv sync --locked --extra laya --extra laya-mlx` | Apple Silicon 训练、单模型及双 Laya 推理 |
| 根环境：图表 | `uv sync --locked --extra laya --extra laya-mlx --extra analysis` | 在主环境保留模型依赖并增加图表工具 |
| 词袋与统计 | `uv sync --project environments/bow-cpu --locked` | NumPy、SciPy、scikit-learn，独立核验 |
| DeBERTa 准备 | `uv sync --project environments/deberta-prep --locked` | DeBERTa 分词与模型准备 |
| Laya CUDA | `uv sync --project environments/laya-cu126 --locked` | Linux CUDA 12.6；按原实验协议使用 |
| DeBERTa CUDA | `uv sync --project environments/deberta-cu126 --locked` | Linux CUDA 训练；区分 FP32 / BF16 |

首次安装需要网络。已有环境运行时使用 `uv run --locked --offline --no-sync`，不自动安装缺失依赖。CUDA 环境不在 Mac 上安装；其他用途见 [环境说明](../../environments/README.md)。

## 数据与模型

使用 [Stanford IMDB 官方数据](https://ai.stanford.edu/~amaas/data/sentiment/)。原官方训练 25,000 条按清洗文本组固定划分为 22,500 条训练、2,500 条开发；训练部分再留出 400 条校准，实际梯度训练 22,100 条。官方测试为 25,000 条。

首次准备数据及原始 Laya 模型的入口如下；已有冻结资产时应恢复并校验，不重新覆盖已有记录：

```bash
# 会下载缺失的官方数据，并读取测试文本做完整性/重复审计，不执行模型评分
uv run --locked --no-sync python scripts/prepare_data.py

# 下载锁定 revision 的原始 Laya；这不是已微调的交付模型
uv run --locked --no-sync python scripts/prepare_laya.py
```

数据划分身份见 `reports/split_manifest.json`；模型来源和文件哈希见 `configs/laya_model_lock.json`、`configs/deberta_model_lock.json`。微调模型不能由上面的下载命令获得，须从训练输出或对应备份恢复。

单模型推理默认需要 `checkpoints/laya-l512-s42-ce/`，包含 `model.safetensors`、`rl_agent_config.json`、`encoder/config.json` 及完整 `tokenizer/`。其他实验的 checkpoint、逐轮恢复包与预测缓存按原报告的输入契约匹配。

## 配置检查与训练

```bash
# 仅检查配置与已安装训练器，不读取数据或加载模型
uv run --locked --offline --no-sync python scripts/train_laya.py \
  --config configs/laya-full-ce.json --validate-config

# 进一步检查本地资产和 MPS，不启动训练
uv run --locked --offline --no-sync python scripts/train_laya.py \
  --config configs/laya-full-ce.json --check-only
```

`laya-full-ce.json` 是历史一轮 CE 配方，对应开发 95.76% 的单模型。需要重训时，先复制配置、使用新 `run_id`，完成设备准入后，再以 `scripts/train_laya.py --config configs/NEW_RUN.json` 启动；它不会产生最终三模型融合。真实有限梯度、保存重载和同输入验证不能由配置检查替代。

| 路线 | 代码与协议 | 结果 |
| --- | --- | --- |
| 原双 Laya | `train_laya.py`、baseline-s43 / combined-s42 配置；[推理协议](../guides/laya-fusion-inference.md) | 开发 96.96%、测试 96.344% |
| DeBERTa 两轮 | `deberta_budget_study.py` / `deberta_budget_training.py` | [最终评估](../../reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/REPORT.md) |
| 固定异构融合 | `deberta_final_analysis.py` / `deberta_final_evaluation.py` | [开发](../../reports/deberta-laya-final-analysis-20261018-v1/REPORT.md)、[测试](../../reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md) |
| 三组完整重训 | `paper_gpu_supplement.py` / `paper_gpu_evaluate.py`；[设计](../superpowers/specs/2026-10-19-paper-supplementary-validation-design.md) | [完成报告](../../reports/paper-gpu-supplement-20261019-v1/REPORT.md) |
| 权重与稳健性 | `paper_fusion_diagnostics.py` / `paper_fusion_robustness.py` | [诊断](../../reports/paper-fusion-diagnostics-20261020-v1/REPORT.md)、[稳健性](../../reports/paper-fusion-robustness-20261020-v1/REPORT.md) |

这些入口用于定位代码；日期脚本可能硬编码运行身份、资产路径及 SHA，执行前须阅读协议，不能只改配置文件名或批量启动。官方测试结果已完成，常规安装检查不需要重新执行测试集前向。

## 推理、测试与报告

- [推理指南](../guides/inference.md)：历史单模型、双 Laya 与异构评估的实际入口。
- [测试说明](../../tests/README.md)：按模块隔离执行和独立统计环境。
- [报告说明](../../paper/README.md)：编译现有 TeX，或使用完整本地材料重建图表与 Word。

报告的已有 TeX 和图表可独立编译；重新计算图表需要未进入 Git 的完整冻结预测。配置检查、单元测试和论文静态复算均不等于重新通过 GPU 训练或部署验收。

## 历史目录与资料包

`scripts/build_closeout_catalog.py` 的 `build / verify / test / package` 绑定 10 月 18 日整理目录。`build` 会重写历史生成索引，`verify` 和 `test` 会写入该日期的报告位置；日常交付检查优先使用直接 CLI 与独立模块测试。`package` 输出旧日期的源码资料包且不含完整 `paper/`，不能用来代表本次最终材料。

需要重新制作当前版本源码包时，使用新的输出目录与文件名，收录当前文档、源码、锁文件和最终报告，记录清单并逐文件核验；保留历史包与收据。数据、权重、缓存和视频按 [交付说明](delivery.md) 单独处理。
