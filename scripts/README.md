# 代码入口

所有命令从仓库根目录运行。先阅读 [环境与复现](../docs/closeout/reproduction.md)，再按任务选择入口。

| 任务 | 入口 | 说明 |
| --- | --- | --- |
| 数据划分 | `prepare_data.py`、`verify_data.py` | 官方数据审计、固定划分与完整性核对 |
| Laya 原始模型 | `prepare_laya.py` | 下载锁定 revision，不生成微调模型 |
| Laya 微调 | `train_laya.py` | 配置校验、设备检查、训练和校准 |
| 单模型预测 | `predict.py` | Mac MLX，默认历史 Laya 单模型 |
| 双 Laya 推理 | `predict_laya_fusion.py` | 显式配置，支持既有多视图规则 |
| 双 Laya 数值核验 | `validate_laya_fusion_inference.py`、`validate_laya_mlx_precise.py` | CUDA / MLX 核验 |
| DeBERTa 微调 | `deberta_budget_study.py`、`deberta_budget_training.py` | 原两轮冻结实验 |
| 最终异构分析 / 评估 | `deberta_final_analysis.py`、`deberta_final_evaluation.py` | 原 97.028% 固定评估，不是通用单条 CLI |
| 三组完整重训 | `paper_gpu_supplement.py` / `paper_gpu_evaluate.py` | 九次训练、独立重载、固定开发评估与测速 |
| 重复文本 / 后端补充 | `paper_supplementary_validation.py`（`analyze` / `cpu-parity`） | 论文补充分析与完整开发集核验 |
| 权重诊断 | `paper_fusion_diagnostics.py` | 三组纠错、退化和权重候选 |
| 稳健性分析 | `paper_fusion_robustness.py` | 文本组重采样、后处理交叉验证 |
| 词袋与查询头 | `run_bow_nb_study.py`、`run_token_query_study.py` | 独立对照，结论见研究台账 |
| 历史资料整理 | `build_closeout_catalog.py` | 绑定 10 月 18 日索引与输出目录 |

日期脚本可能硬编码 `run_id`、原机器路径和输入 SHA256。配置存在不代表实验已执行；已有运行不能覆盖，新实验需完整核对协议。更多历史模块见 [代码索引](../docs/closeout/code.md) 和 [研究台账](../docs/closeout/research-ledger.md)，最新完成状态见 [文档入口](../docs/README.md)。
