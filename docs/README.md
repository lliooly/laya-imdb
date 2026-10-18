# 文档索引

更新：2026-10-18。当前状态以[收尾总览](closeout/README.md)为准。日期设计与原报告保留当时状态；顶层启动快照不替代最终 evaluation 结果。

| 入口 | 用途 |
|---|---|
| [收尾总览](closeout/README.md) | 主结论、阅读顺序与交付边界 |
| [24 类研究台账](closeout/research-ledger.md) | 假设、设计、代码、结果与停止原因 |
| [指标索引](closeout/metrics.md) / [JSON](closeout/metrics.json) | 旧 123 条与后续补充；条数不是训练次数 |
| [代码与测试](closeout/code.md) | 模块职责、CLI 和静态局部依赖 |
| [设计、配置与环境](closeout/designs.md) | 日期设计、计划、配置、依赖锁 |
| [报告与验证](closeout/reports.md) | 完整本地报告导航，优先最终 evaluation |
| [文件与本地资产](closeout/files.md) | 源码 SHA、报告元数据、权重存在性 |
| [复现说明](closeout/reproduction.md) | 安装、检查、材料前提与流程 |
| [参考资料](closeout/references.md) | 文献、审计、原提案纠正 |
| [交付清单](closeout/delivery.md) | 源码包与权重边界、后续 Word/视频/模型整合 |
| [当前状态](experiments/current-status.md) | 简明成绩表与完成状态 |

## 最终结果

- 异构融合：[开发](../reports/deberta-laya-final-analysis-20261018-v1/REPORT.md)、[正式测试](../reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md)。
- 长上下文：[最终评估](../reports/laya-context-pair-s42-modal-20261016-v1/evaluation/REPORT.md)；标签修订：[最终评估](../reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation/REPORT.md)。
- 决策头：[三种子分析](../reports/laya-head-control-analysis-20261015-v1/REPORT.md)、[缺失权重说明](../reports/laya-head-control-delivery-status-20261015-v1/REPORT.md)。
- [鲁棒损失](../reports/laya-robust-loss-analysis-20261016-v1/REPORT.md)、[词袋分支](../reports/bow-nb-complementarity-cpu-20261018-v1/REPORT.md)、[最后 CPU 候选](../reports/final-cpu-alternatives-20261018-v2/REPORT.md)。
- [全序列查询头](../reports/laya-token-query-mps-fp32-20261018-v1/REPORT.md)、[BERT+BiLSTM 文献审计](../reports/bert-bilstm-literature-audit-20261018-v1/REPORT.md)。

## 指南与历史

- [训练协议](guides/training.md)、[推理现状](guides/inference.md)、[Laya 融合指南](guides/laya-fusion-inference.md)、[历史复现](guides/historical-reproduction.md)。
- [Modal 训练与回传](guides/laya-modal-training.md)、[鲁棒损失恢复](guides/laya-robust-loss-training.md)、[CUDA 准备](guides/cuda-preparation.md)。
- [10 月 13 日思路台账](experiments/experiment-summary.md)、[10 月 16 日指标快照](../reports/all-experiments-metrics-20261016-v1/REPORT.md)、[原始参考清单](guides/references.md)。
- [本次整理前原文](archive/closeout-20261018/)、[历史提案附件说明](archive/source-notes/README.md)。

run_id、manifest、SHA 和 reproduction 保持原路径；源码整理不重新运行训练或官方测试。完整本地原始材料和精简资料包的范围不同，详见交付清单。
