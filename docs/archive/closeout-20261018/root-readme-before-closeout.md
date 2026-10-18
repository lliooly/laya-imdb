# Laya IMDB

IMDB情感分类课程项目：PyTorch训练，Apple Silicon Mac通过MLX推理。当前主要研究Laya-CE，并计划把有效方法迁移至ModernBERT-base与Laya-RLCD做公平对照。

当前最高历史开发验证Accuracy为 **96.96%**（Mac两轮CE，seed42）。官方测试集尚未正式评估，98%仍是研究目标。最近strict三组原文／普通改写／语义改写结果为 **96.52%／96.32%／96.44%**，未显示增强收益。首尾窗口在四组固定权重上均有净改善，尚未完成一致策略的新训练与Mac部署核对。

所有成绩来自固定2500条开发验证集。不同平台、种子、运行时和输入预算的结果不能直接解释成单因素收益。最高成绩权重也不等同于默认推理checkpoint。

## 研究与文档

- [所有实验思路与结果总表](docs/experiments/experiment-summary.md)：43项工作和未来方向，附证据、限制与停止理由。
- [当前状态](docs/experiments/current-status.md)：候选、部署、备份和数据协议。
- [后续改进路线](docs/experiments/improvement-roadmap.md)：首尾输入、公平迁移与最终交付。
- [完整文档索引](docs/README.md)：训练、推理、历史复现、归档和命名规范。

## 环境和运行

Python3.12，依赖锁定在uv.lock。以下命令在仓库根目录运行；首次依赖与模型下载需要网络，数据和权重不包含在Git中。

```bash
uv sync --locked --extra laya --extra laya-mlx
uv run --locked --offline python scripts/verify_data.py
uv run --locked --offline python scripts/train_laya.py --check-only
uv run --locked --offline python scripts/predict.py \
  'The acting was excellent, and I loved the ending.'
uv run --locked --offline python -m unittest discover -s tests -v
```

默认推理读取`checkpoints/laya-l512-s42-ce/`，对应历史一轮模型；指定其他已核验checkpoint需用`--checkpoint`。CUDA使用FP32，关闭AMP，保留非有限值保护；新的严格对照显式选择strict运行时。CPU单元测试不代替GPU或Mac核对。

冻结划分：原训练22500条，其中梯度22100、校准400；开发2500；官方测试25000。`prepare_data.py`会重建派生材料，已有环境先运行verify_data，不随意重建冻结划分。模型、分词器和数据锁见configs与reports中的manifest。

## 仓库布局

| 目录 | 用途 |
| --- | --- |
| scripts/、tests/ | 当前入口、共享实现、统计和有依赖的历史组件 |
| configs/ | 模型锁与已验证实验配置；重跑需新run_id |
| docs/guides/、docs/experiments/ | 当前协议、实验总览与改进路线 |
| docs/archive/、docs/superpowers/ | 归档计划和冻结实验设计 |
| archive/legacy-code/ | 已停用一次性入口及过期计划配置 |
| reports/ | 原始实验报告、必要指标和不可改写的复现证据 |
| rust-tokenizer/ | 可选Rust分词工程工具 |

权重、训练正文、原始大输出、缓存和传输备份仅保留在本地；Git提交必要源码、配置和精简报告。逐轮检查点不因整理删除。当前语义A/B/C新权重仍在远端，尚未完成本地备份。

## 课程交付与许可证

最终需Word报告、源码压缩包及训练／测试视频（每个不超过5分钟）。仓库不能替代这些交付物。

原创代码使用[MIT](LICENSE)。第三方代码、模型和数据遵循其原条款，见[第三方声明](THIRD_PARTY_NOTICES.md)和[参考文献](docs/guides/references.md)。
