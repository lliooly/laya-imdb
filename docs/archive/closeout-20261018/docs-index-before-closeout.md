# 文档索引

当前状态及决策以本目录中的汇总文档为入口；日期设计和 reports 保留当时的原始结论，后续更新不能反向覆盖历史证据。

| 文档 | 内容 |
| --- | --- |
| [全部实验指标总表（2026-10-16）](../reports/all-experiments-metrics-20261016-v1/REPORT.md) | 71 条完整开发集方案、123 条总指标记录、34 组配对比较；含正式测试、拒答、F1/AUROC、NLL/ECE/Brier 和来源 |
| [2048 上下文工程短跑](../reports/laya-long-context-pilot-20261016-v1/REPORT.md) | L40S 严格 FP32 四组通过；2048 × batch 8 压力组峰值已分配显存 11.70 GiB，训练影评完整覆盖 99.97%；尚无新准确率结果 |
| [历史实验思路台账（2026-10-13）](experiments/experiment-summary.md) | 当时 43 项工作及方向；后续完成状态和数值见最新总表 |
| [当前状态](experiments/current-status.md) | 最佳成绩、默认部署、严格实验、数据与备份状态 |
| [L2 头部配对训练完整分析](../reports/laya-head-control-analysis-20261015-v1/REPORT.md) | 三个 seed、六次训练的准确率、截断分组、不确定性、校准与实测成本 |
| [鲁棒损失三种子完整分析](../reports/laya-robust-loss-analysis-20261016-v1/REPORT.md) | CE 96.32% / mixed 96.40%，平均 +0.08 pp，条件区间跨零；最终模型已全部本地交付，逐轮恢复包接收中 |
| [训练损失集中度诊断](../reports/laya-training-loss-diagnosis-20261015-v1/REPORT.md) | 94 条候选的旧末期损失占比、鲁棒损失的困难样本风险；仅缓存分析 |
| [下一轮损失与权重交付设计](superpowers/specs/2026-10-15-laya-robust-loss-and-checkpoint-delivery-design.md) | 已批准：三种子同期损失对照、逐 epoch 本地交付门禁与故障验收 |
| [鲁棒损失与交付本地验收](../reports/laya-robust-loss-local-readiness-20261015-v1/REPORT.md) | 历史实现阶段快照：23 项测试通过、111 项绑定；后续 Modal 执行结果见完整分析 |
| [鲁棒损失训练与恢复](guides/laya-robust-loss-training.md) | 接收器、分阶段启动、强制回传、故障恢复与存储预算 |
| [Modal 训练与本地权重交付](guides/laya-modal-training.md) | 用户确认 30 美元赠送额度，单卡预算、持久化 Volume、逐 epoch 本地严格加载与收据 |
| [后续改进路线](experiments/improvement-roadmap.md) | 首尾一致训练／推理、模型迁移、最终评估 |
| [训练协议](guides/training.md) | 冻结划分、训练入口、精度和复现约束 |
| [决策因素实验](guides/laya-decision-study.md) | 提案审查、固定权重诊断、窗口对照、N8参数平均及N2/N6多模型融合 |
| [双模型融合推理](guides/laya-fusion-inference.md) | 冻结融合入口、CUDA与Mac MLX一致性及性能实测 |
| [冻结后官方测试](guides/laya-frozen-evaluation.md) | 固定候选、一次性25000条官方测试与目标判定 |
| [Mac推理](guides/inference.md) | checkpoint结构与部署命令 |
| [CUDA准备](guides/cuda-preparation.md) | 环境、权重锁与训练前检查 |
| [历史复现流程](guides/historical-reproduction.md) | ModernBERT、Rust和旧课程阶段命令 |
| [参考文献](guides/references.md) | 模型、方法和数据来源 |
| [整理与合并记录](../reports/project-consolidation-20261013-v1/REPORT.md) | 路径迁移、代码归档、保护和验证 |

## 命名与归档规则

当前入口采用描述性的 lowercase-kebab-case：`docs/guides/<topic>.md`、`docs/experiments/<topic>.md`。一次性实验协议继续使用已有 `YYYY-MM-DD-<topic>-design.md` 格式；`docs/superpowers/specs/` 和 `plans/` 保留为冻结设计历史，不是当前待办列表。

早期阶段计划移至 `docs/archive/plans/day-NN-*.md`，过期路线移至 `docs/archive/roadmaps/YYYY-MM-DD-*.md`。原 docs 路径只保留兼容跳转，供不可改写的历史报告引用。新引用使用规范路径。

已停用且无模块依赖的临时执行入口移至 `archive/legacy-code/`；不从该目录直接重新训练。`scripts/` 保留当前训练、推理、统计和被测试／复现依赖的模块。负结果和无收益思路以总表中的停止理由归档，不删除实验记录。

报告目录和 run_id 不改名：它们被manifest、SHA和数据绑定引用。历史报告的 REPORT.md 名称也保留。新整理文档提供统一入口，而非重命名冻结证据。大文件、逐条诊断、训练正文和权重留在本地并由Git忽略；必要指标、报告、源码和配置进入版本控制。
