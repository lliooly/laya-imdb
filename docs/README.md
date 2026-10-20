# 使用与研究文档

本文档对应 2026-10-20 的交付准备版本。首次阅读建议从 [项目首页](../README.md) 和 [最终课程报告](../paper/README.md) 开始。

## 使用与复现

| 文档 | 内容 |
| --- | --- |
| [环境说明](../environments/README.md) | Python、uv、Mac 与 CUDA 依赖环境 |
| [复现指南](closeout/reproduction.md) | 数据、模型、训练、检查和材料范围 |
| [推理指南](guides/inference.md) | 单模型、双 Laya 与最终异构评估的入口区别 |
| [双 Laya 推理](guides/laya-fusion-inference.md) | CUDA FP32、MLX FP16 与精确 FP32 核验 |
| [代码入口](../scripts/README.md) | 各研究方向的主要脚本 |
| [配置说明](../configs/README.md) | 实验配置、模型版本和固定身份 |
| [测试说明](../tests/README.md) | 模块隔离、分析测试与适用边界 |
| [交付说明](closeout/delivery.md) | 源码、报告、权重和视频的状态 |
| [交付准备记录](../reports/delivery-preparation-20261020-v1/REPORT.md) | 当前文档检查与新版本资料包 |

## 最终结果与补实验

| 证据 | 内容 |
| --- | --- |
| [当前结果与限制](experiments/current-status.md) | 固定测试、三组重训、概率质量及工程状态 |
| [完整官方测试](../reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md) | 97.028% Accuracy、后端和冻结规则 |
| [三组完整重训](../reports/paper-gpu-supplement-20261019-v1/REPORT.md) | 九次训练、相同平台测速和回传核验 |
| [完整开发集后端核验](../reports/paper-deberta-cpu-development-parity-20261019-v2/REPORT.md) | DeBERTa CPU FP32 / CUDA BF16 比较 |
| [重复文本敏感性](../reports/paper-duplicate-sensitivity-20261019-v1/REPORT.md) | 重复文本与结果依赖 |
| [融合权重诊断](../reports/paper-fusion-diagnostics-20261020-v1/REPORT.md) | 三组权重候选、纠错与退化 |
| [融合稳健性分析](../reports/paper-fusion-robustness-20261020-v1/REPORT.md) | 条件区间、分组五折与校准样本扰动 |
| [报告核验](../paper/analysis/document_validation.json) | Word/PDF 结构、指标和逐页检查记录 |

## 研究台账与历史记录

[研究总览](closeout/README.md) 汇总最终结论与材料导航。10 月 18 日建立的分类台账保留该阶段的研究范围，后续论文补实验以上表报告为准。

- [24 类研究台账](closeout/research-ledger.md)、[指标索引](closeout/metrics.md) 与 [指标 JSON](closeout/metrics.json)。指标条数不等于独立训练次数。
- [代码与测试索引](closeout/code.md)、[设计与配置索引](closeout/designs.md)、[报告索引](closeout/reports.md)、[文件清单](closeout/files.md)。这些是历史整理工具的生成索引，最新入口另见本页。
- [参考资料与文献审计](closeout/references.md)、[历史复现](guides/historical-reproduction.md)。
- [原训练说明](guides/training.md)、[Modal 训练与回传](guides/laya-modal-training.md)、[CUDA 准备](guides/cuda-preparation.md)。
- [10 月 18 日整理前快照](archive/closeout-20261018/)、`superpowers/` 日期设计与实现计划。

历史报告、manifest、SHA256 和 reproduction 快照按原路径保留，反映记录时的状态。对存在 `evaluation/REPORT.md` 的实验，优先阅读最终评估；准备配置中的“待启动”不代表最终任务仍未完成。
