# 文档索引

当前状态及决策以本目录中的汇总文档为入口；日期设计和 reports 保留当时的原始结论，后续更新不能反向覆盖历史证据。

| 文档 | 内容 |
| --- | --- |
| [实验思路与结果总表](experiments/experiment-summary.md) | 43项已完成工作及未完成方向；协议、结果、决策、证据链接 |
| [当前状态](experiments/current-status.md) | 最佳成绩、默认部署、严格实验、数据与备份状态 |
| [后续改进路线](experiments/improvement-roadmap.md) | 首尾一致训练／推理、模型迁移、最终评估 |
| [训练协议](guides/training.md) | 冻结划分、训练入口、精度和复现约束 |
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
