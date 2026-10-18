# 复现、环境与核验

所有命令在仓库根目录执行。使用 Python 3.12、uv 和版本锁。源码包本身不包含数据、模型、全部预测缓存或优化器恢复包；历史实验精确复现需要从对应本地备份恢复协议所绑定的材料。

## 环境分工

| 锁文件 | 用途 | 边界 |
|---|---|---|
| 根 pyproject.toml / uv.lock | Mac PyTorch、Laya、MLX 及主要测试 | laya / laya-mlx 是可选依赖；安装相应 extra |
| environments/bow-cpu/uv.lock | NumPy、SciPy、sklearn 的词袋与统计核验 | 与主环境分开，禁止无记录地混装 NumPy 版本 |
| environments/laya-cu126/uv.lock | Linux CUDA 12.6 L2 对照 | 不在 Mac 安装；L2 权重交付缺口另列 |
| environments/laya-modal/uv.lock | Modal 运行时与权重接收 | 旧预算记录不代表本次获准启动云端 |
| environments/deberta-prep/uv.lock | DeBERTa 准备与分词 | 需要本地锁定模型材料 |
| environments/deberta-cu126/uv.lock | DeBERTa CUDA 训练 | FP32/BF16 按各实验契约，不沿用 Laya 的 AMP 结论 |

首次安装需网络；本次只使用已有环境运行离线检查：

```bash
uv sync --locked --extra laya --extra laya-mlx
uv sync --project environments/bow-cpu --locked
uv run --locked --offline --no-sync python scripts/build_closeout_catalog.py build
uv run --locked --offline --no-sync python scripts/build_closeout_catalog.py verify
uv run --locked --offline --no-sync python scripts/build_closeout_catalog.py test
```

测试按模块各开子进程，因为部分科学边界测试安装永久 Python audit hook。词袋及校准权重模块使用 bow-cpu，其他模块使用主锁。测试日志与计数保存在本次整理报告目录，不能把 CPU 测试通过写成 GPU 训练通过。

## 复现材料先核对

1. 从官方来源获得模型与数据，或恢复既有本地材料。模型 revision、分词器及 SHA 见 configs 中的锁和各原 manifest。
2. 恢复冻结派生划分与 reports/split_manifest.json；使用原预处理方式校验，避免重建得到不同样本或标签顺序。数据文件不在源码包内。
3. 恢复对应 checkpoint、逐轮恢复包或预测缓存，并核对契约所列 SHA。仅有 summary 和源码不保证精确恢复随机状态。
4. 阅读日期设计和最终报告：某些脚本 run_id 硬编码，复制 JSON 并不足以建立新实验。新 run_id、源码及输入需重新绑定，不能覆盖历史目录。
5. 训练前执行该入口的配置和设备准入；纯--validate-config 不代替真实有限梯度、保存重载和同输入核验。

## 按研究路线定位

| 路线 | 核心入口 | 最终证据 |
|---|---|---|
| Laya 训练 | train_laya.py + 配方配置 | [训练指南](../guides/training.md)及[设计索引](designs.md) |
| 长上下文 | laya_context_pair.py / laya_context_training.py | [最终评估](../../reports/laya-context-pair-s42-modal-20261016-v1/evaluation/REPORT.md) |
| 数据修订 | laya_reviewed_data_study.py | [最终评估](../../reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation/REPORT.md) |
| DeBERTa 两轮 | deberta_budget_study.py / deberta_budget_training.py | [最终评估](../../reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/REPORT.md) |
| 固定异构融合 | deberta_final_analysis.py / deberta_final_evaluation.py | [开发](../../reports/deberta-laya-final-analysis-20261018-v1/REPORT.md)、[已完成测试](../../reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md) |
| 词袋互补 | run_bow_nb_study.py / verify_bow_nb_study.py | [报告](../../reports/bow-nb-complementarity-cpu-20261018-v1/REPORT.md) |
| 最后 CPU 候选 | run_final_cpu_alternatives_v2.py | [报告](../../reports/final-cpu-alternatives-20261018-v2/REPORT.md) |
| 全 token 查询头 | run_token_query_study.py | [报告](../../reports/laya-token-query-mps-fp32-20261018-v1/REPORT.md) |

这里列出入口用于定位，不是批量执行命令。官方测试已经完成，本次整理不再次运行；完整历史逐条导出可在本地报告目录查找，源码包只保留选定摘要、来源和验证材料。

## 重建索引与资料包

```bash
uv run --locked --offline --no-sync python scripts/build_closeout_catalog.py build
uv run --locked --offline --no-sync python scripts/build_closeout_catalog.py verify
uv run --locked --offline --no-sync python scripts/build_closeout_catalog.py package
```

build 重新生成本次目录索引，不改历史报告。package 拒绝覆盖同名归档；新的封包应使用新命名。修改源码或文档后先重建 catalog，再核验，最后封包。归档内 PACKAGE-MANIFEST.json 逐文件记录 SHA；这不是可直接部署的模型包。

生成的 `docs/closeout/catalog.json` 与 `local-assets.json` 属于本机清单，不进入 Git；首次检查前先运行 build。
