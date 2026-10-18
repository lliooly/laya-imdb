# 研究台账：设计、实现、结果与验证

更新：2026-10-18。24类覆盖研究脉络，类数不等于训练次数。历史43条思路与123条指标快照保持原文；补充记录见指标索引。

| 类别 | 状态 | 核心边界 |
|---|---|---|
| C01 数据划分、预处理与分词 | 已验证 | 22100 梯度/400 校准/2500 开发；官方测试 25000。数据目录不进入源码包。 |
| C02 ModernBERT 历史基线 | 历史方案比较 | 94.64% 是 256-token 历史基线；不是与 Laya 等预算消融。 |
| C03 Laya CE / RLCD 先导与首次全量 | 已完成历史训练 | 默认 predict 仍为 95.76% 历史单模型；全量公平 RLCD 迁移未完成。 |
| C04 CUDA 数值准入与跨后端 | 已验证并保留失败 | Laya FP32 准入；AMP FP16 溢出。DeBERTa 后续 BF16 是单独协议，不能混用结论。 |
| C05 warmup、平滑、R-Drop 与种子 | 未检出稳定增益 |  |
| C06 困难样本重采样与严格配对 | 负结果与漂移诊断 |  |
| C07 人工复核与语义增强 | 已完成；收益未复现 |  |
| C08 固定权重首尾视图与校准 | 已完成诊断 | 区分固定权重推理变化与重新训练。 |
| C09 候选顺序、窗口训练、参数平均 | 未达继续门槛 |  |
| C10 Laya 内部多视图与双模型融合 | 历史正式主方案 | 开发 96.96%、旧正式测试 96.344%；97.00% 为预声明辅助方案。 |
| C11 MLX 概率误差定位与精确 FP32 | 数值核验完成 | 类别一致不能替代概率误差准入；吞吐由对应报告定义。 |
| C12 风险—覆盖与拒答 | 已完成；非全覆盖 accuracy | 固定覆盖排序曲线与校准阈值部署结果不同；不能与全覆盖 97.028% 混排。 |
| C13 冻结表示与 L2 决策头配对训练 | 统计完成；L2 权重缺失 | 原始/线性头均值 96.520/96.493%；L2 的 12 份权重未回传，不称可交付。 |
| C14 鲁棒损失与逐轮权重交付 | 六模型结果完成 | CE96.32%、mixed96.40%；最终模型与完整优化器恢复包交付分别认定。 |
| C15 512/2048 长上下文同预算训练 | 已完成最终评估 | 全局同为 96.32%；长组+6、短组-6。顶层 REPORT 为启动快照。 |
| C16 94 条训练复核与用户修订 v2 | 已完成最终评估 | 61 翻标+3 删除；原始划分与原标签文件保留；开发 96.32→96.40%，p=.883。 |
| C17 长度路由与长文融合 | 探索性缓存分析 | 固定路由 96.56%，对 512 净+6、p=.109；不是已验收部署。 |
| C18 DeBERTa-v3 异构训练与恢复 | 两轮最终模型本地交付 | CUDA BF16 开发 97.04%；之前 FP32 中断与试跑记录保留。 |
| C19 Laya+DeBERTa 固定异构融合 | 有开发与正式测试成绩 | 开发 97.28%；正式测试 97.028%，CPU FP32 DeBERTa+旧 CUDA Laya 缓存；不是部署入口成绩。 |
| C20 CPU TF-IDF / NB-ngram 分支 | 互补有限 | TF-IDF 融合 97.32%，净+1，p=1；NB 融合 97.28%。 |
| C21 BERT+BiLSTM 公开文献代码审计 | 静态审计完成 | 公开代码分区不兼容；未证实电影泄漏或序列退化；不据此否定论文 97.67%。 |
| C22 置信度与固定校准权重 | 未获提升 | 校准 NLL 权重开发 97.16%；二模型按绝对置信度选边与 50:50 在二分类下类别相同，不是独立改进。 |
| C23 正则化 logit、长文分支与多样本 Dropout | 探索性最高分，未达继续门槛 | 97.40% 仅开发；相对 97.28 净+3，p=.25，NLL/ECE 变差；无对应正式测试。v1 故障保留。 |
| C24 全 token 情感查询与 Top-32 稀疏聚合 | 九个头完成；没有增益 | 冻结编码器；三种子各 5 轮。稠密 96.84%、融合 97.28%；Top32 融合 97.24%。非 DSA 复现。 |

## C01 数据划分、预处理与分词

状态：已验证。22100 梯度/400 校准/2500 开发；官方测试 25000。数据目录不进入源码包。

设计/计划：以对应报告原协议为准；不补造独立规格。

实现：[scripts/prepare_data.py](../../scripts/prepare_data.py)；[scripts/verify_data.py](../../scripts/verify_data.py)；[scripts/cache_tokens.py](../../scripts/cache_tokens.py)；[scripts/validate_rust_tokenizer.py](../../scripts/validate_rust_tokenizer.py)

最终结果/核验：[reports/project-consolidation-20261013-v1/REPORT.md](../../reports/project-consolidation-20261013-v1/REPORT.md)


## C02 ModernBERT 历史基线

状态：历史方案比较。94.64% 是 256-token 历史基线；不是与 Laya 等预算消融。

设计/计划：以对应报告原协议为准；不补造独立规格。

实现：[scripts/train_baseline.py](../../scripts/train_baseline.py)；[scripts/modernbert_mlx.py](../../scripts/modernbert_mlx.py)；[scripts/validate_modernbert.py](../../scripts/validate_modernbert.py)

最终结果/核验：[reports/model-comparison/REPORT.md](../../reports/model-comparison/REPORT.md)


## C03 Laya CE / RLCD 先导与首次全量

状态：已完成历史训练。默认 predict 仍为 95.76% 历史单模型；全量公平 RLCD 迁移未完成。

设计/计划：以对应报告原协议为准；不补造独立规格。

实现：[scripts/train_laya.py](../../scripts/train_laya.py)；[scripts/predict.py](../../scripts/predict.py)

最终结果/核验：[reports/laya-backend-comparison/REPORT.md](../../reports/laya-backend-comparison/REPORT.md)；[reports/laya-mac-ce-comparison-v1/REPORT.md](../../reports/laya-mac-ce-comparison-v1/REPORT.md)


## C04 CUDA 数值准入与跨后端

状态：已验证并保留失败。Laya FP32 准入；AMP FP16 溢出。DeBERTa 后续 BF16 是单独协议，不能混用结论。

设计/计划：以对应报告原协议为准；不补造独立规格。

实现：[scripts/audit_cuda_training.py](../../scripts/audit_cuda_training.py)；[scripts/cuda_numerics.py](../../scripts/cuda_numerics.py)；[scripts/laya_determinism.py](../../scripts/laya_determinism.py)

最终结果/核验：[reports/cuda-preparation-v1/REPORT.md](../../reports/cuda-preparation-v1/REPORT.md)；[reports/laya-determinism-completed-20261012-v1/REPORT.md](../../reports/laya-determinism-completed-20261012-v1/REPORT.md)


## C05 warmup、平滑、R-Drop 与种子

状态：未检出稳定增益。

设计/计划：[docs/superpowers/specs/2026-10-11-laya-ce-rdrop-design.md](../superpowers/specs/2026-10-11-laya-ce-rdrop-design.md)

实现：[scripts/laya_optimization.py](../../scripts/laya_optimization.py)；[scripts/laya_rdrop.py](../../scripts/laya_rdrop.py)

最终结果/核验：[reports/rtx4090-regularization-completed-v1/REPORT.md](../../reports/rtx4090-regularization-completed-v1/REPORT.md)；[reports/rtx4090-combined-completed-v1/REPORT.md](../../reports/rtx4090-combined-completed-v1/REPORT.md)；[reports/rtx4090-rdrop-completed-v1/REPORT.md](../../reports/rtx4090-rdrop-completed-v1/REPORT.md)


## C06 困难样本重采样与严格配对

状态：负结果与漂移诊断。

设计/计划：[docs/superpowers/specs/2026-10-13-laya-strict-pair-design.md](../superpowers/specs/2026-10-13-laya-strict-pair-design.md)

实现：[scripts/laya_resampling.py](../../scripts/laya_resampling.py)；[scripts/run_laya_strict_pair.py](../../scripts/run_laya_strict_pair.py)

最终结果/核验：[reports/laya-strict-pair-result-check-20261013-v1/REPORT.md](../../reports/laya-strict-pair-result-check-20261013-v1/REPORT.md)；[reports/laya-resampling45-control-drift-audit-v2/REPORT.md](../../reports/laya-resampling45-control-drift-audit-v2/REPORT.md)


## C07 人工复核与语义增强

状态：已完成；收益未复现。

设计/计划：[docs/superpowers/specs/2026-10-13-semantic-training-adapter-design.md](../superpowers/specs/2026-10-13-semantic-training-adapter-design.md)

实现：[scripts/laya_semantic_training.py](../../scripts/laya_semantic_training.py)；[scripts/run_laya_semantic_abc.py](../../scripts/run_laya_semantic_abc.py)

最终结果/核验：[reports/semantic30-abc-completed-20261013-v1/REPORT.md](../../reports/semantic30-abc-completed-20261013-v1/REPORT.md)；[reports/semantic-pair-pilot-20261013-v2/REPORT.md](../../reports/semantic-pair-pilot-20261013-v2/REPORT.md)


## C08 固定权重首尾视图与校准

状态：已完成诊断。区分固定权重推理变化与重新训练。

设计/计划：[docs/superpowers/specs/2026-10-11-laya-posthoc-headtail-design.md](../superpowers/specs/2026-10-11-laya-posthoc-headtail-design.md)

实现：[scripts/export_laya_posthoc.py](../../scripts/export_laya_posthoc.py)；[scripts/laya_posthoc.py](../../scripts/laya_posthoc.py)

最终结果/核验：[reports/laya-posthoc-consolidated-v2/REPORT.md](../../reports/laya-posthoc-consolidated-v2/REPORT.md)


## C09 候选顺序、窗口训练、参数平均

状态：未达继续门槛。

设计/计划：[docs/superpowers/specs/2026-10-13-laya-decision-factor-attribution-design.md](../superpowers/specs/2026-10-13-laya-decision-factor-attribution-design.md)

实现：[scripts/laya_decision_views.py](../../scripts/laya_decision_views.py)；[scripts/run_laya_decision_pipeline.py](../../scripts/run_laya_decision_pipeline.py)；[scripts/run_laya_checkpoint_average.py](../../scripts/run_laya_checkpoint_average.py)

最终结果/核验：[reports/laya-decision-execution-20261014-v1-final-study/summary.json](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json)；[reports/laya-checkpoint-average-prefix-s42-strict-20261014-v1/REPORT.md](../../reports/laya-checkpoint-average-prefix-s42-strict-20261014-v1/REPORT.md)


## C10 Laya 内部多视图与双模型融合

状态：历史正式主方案。开发 96.96%、旧正式测试 96.344%；97.00% 为预声明辅助方案。

设计/计划：[docs/superpowers/plans/2026-10-14-laya-probability-fusion.md](../superpowers/plans/2026-10-14-laya-probability-fusion.md)

实现：[scripts/laya_probability_fusion.py](../../scripts/laya_probability_fusion.py)；[scripts/predict_laya_fusion.py](../../scripts/predict_laya_fusion.py)

最终结果/核验：[reports/laya-probability-fusion-s43-s42-20261014-v1/REPORT.md](../../reports/laya-probability-fusion-s43-s42-20261014-v1/REPORT.md)；[reports/laya-frozen-official-test-20261014-v1/REPORT.md](../../reports/laya-frozen-official-test-20261014-v1/REPORT.md)


## C11 MLX 概率误差定位与精确 FP32

状态：数值核验完成。类别一致不能替代概率误差准入；吞吐由对应报告定义。

设计/计划：[docs/superpowers/plans/2026-10-14-laya-mlx-precise-inference.md](../superpowers/plans/2026-10-14-laya-mlx-precise-inference.md)

实现：[scripts/laya_mlx_precise_inference.py](../../scripts/laya_mlx_precise_inference.py)；[scripts/reproduce_mlx_matmul_precision.py](../../scripts/reproduce_mlx_matmul_precision.py)

最终结果/核验：[reports/laya-mlx-precision-diagnosis-20261014-v1/REPORT.md](../../reports/laya-mlx-precision-diagnosis-20261014-v1/REPORT.md)；[reports/laya-fusion-inference-mlx-fp32-precise-20261014-v1/REPORT.md](../../reports/laya-fusion-inference-mlx-fp32-precise-20261014-v1/REPORT.md)


## C12 风险—覆盖与拒答

状态：已完成；非全覆盖 accuracy。固定覆盖排序曲线与校准阈值部署结果不同；不能与全覆盖 97.028% 混排。

设计/计划：以对应报告原协议为准；不补造独立规格。

实现：[scripts/run_laya_risk_coverage.py](../../scripts/run_laya_risk_coverage.py)

最终结果/核验：[reports/laya-risk-coverage-20261015-v1/REPORT.md](../../reports/laya-risk-coverage-20261015-v1/REPORT.md)


## C13 冻结表示与 L2 决策头配对训练

状态：统计完成；L2 权重缺失。原始/线性头均值 96.520/96.493%；L2 的 12 份权重未回传，不称可交付。

设计/计划：[docs/superpowers/specs/2026-10-15-laya-head-control-design.md](../superpowers/specs/2026-10-15-laya-head-control-design.md)

实现：[scripts/laya_frozen_probe.py](../../scripts/laya_frozen_probe.py)；[scripts/laya_head_control.py](../../scripts/laya_head_control.py)；[scripts/run_laya_head_study.py](../../scripts/run_laya_head_study.py)

最终结果/核验：[reports/laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1/REPORT.md](../../reports/laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1/REPORT.md)；[reports/laya-head-control-analysis-20261015-v1/REPORT.md](../../reports/laya-head-control-analysis-20261015-v1/REPORT.md)；[reports/laya-head-control-delivery-status-20261015-v1/REPORT.md](../../reports/laya-head-control-delivery-status-20261015-v1/REPORT.md)


## C14 鲁棒损失与逐轮权重交付

状态：六模型结果完成。CE96.32%、mixed96.40%；最终模型与完整优化器恢复包交付分别认定。

设计/计划：[docs/superpowers/specs/2026-10-15-laya-robust-loss-and-checkpoint-delivery-design.md](../superpowers/specs/2026-10-15-laya-robust-loss-and-checkpoint-delivery-design.md)

实现：[scripts/laya_robust_training.py](../../scripts/laya_robust_training.py)；[scripts/laya_checkpoint_delivery.py](../../scripts/laya_checkpoint_delivery.py)；[scripts/laya_modal_recovery.py](../../scripts/laya_modal_recovery.py)

最终结果/核验：[reports/laya-robust-loss-analysis-20261016-v1/REPORT.md](../../reports/laya-robust-loss-analysis-20261016-v1/REPORT.md)；[reports/laya-robust-loss-modal-recovery-20261016-v1/REPORT.md](../../reports/laya-robust-loss-modal-recovery-20261016-v1/REPORT.md)


## C15 512/2048 长上下文同预算训练

状态：已完成最终评估。全局同为 96.32%；长组+6、短组-6。顶层 REPORT 为启动快照。

设计/计划：[docs/superpowers/specs/2026-10-16-laya-context-pair-design.md](../superpowers/specs/2026-10-16-laya-context-pair-design.md)

实现：[scripts/laya_context_training.py](../../scripts/laya_context_training.py)；[scripts/laya_context_pair.py](../../scripts/laya_context_pair.py)

最终结果/核验：[reports/laya-context-pair-s42-modal-20261016-v1/evaluation/REPORT.md](../../reports/laya-context-pair-s42-modal-20261016-v1/evaluation/REPORT.md)；[reports/laya-context-pair-s42-modal-20261016-v1/local-delivery.json](../../reports/laya-context-pair-s42-modal-20261016-v1/local-delivery.json)


## C16 94 条训练复核与用户修订 v2

状态：已完成最终评估。61 翻标+3 删除；原始划分与原标签文件保留；开发 96.32→96.40%，p=.883。

设计/计划：[docs/superpowers/specs/2026-10-17-laya-reviewed-data-training-design.md](../superpowers/specs/2026-10-17-laya-reviewed-data-training-design.md)

实现：[scripts/apply_laya_label_revision_v2.py](../../scripts/apply_laya_label_revision_v2.py)；[scripts/laya_reviewed_data_study.py](../../scripts/laya_reviewed_data_study.py)

最终结果/核验：[reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation/REPORT.md](../../reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation/REPORT.md)；[reports/laya-reviewed-data-s42-modal-20261017-v1/local-delivery.json](../../reports/laya-reviewed-data-s42-modal-20261017-v1/local-delivery.json)


## C17 长度路由与长文融合

状态：探索性缓存分析。固定路由 96.56%，对 512 净+6、p=.109；不是已验收部署。

设计/计划：[docs/superpowers/specs/2026-10-17-laya-length-router-design.md](../superpowers/specs/2026-10-17-laya-length-router-design.md)

实现：[scripts/laya_length_router.py](../../scripts/laya_length_router.py)；[scripts/analyze_laya_long_fusion_bc.py](../../scripts/analyze_laya_long_fusion_bc.py)

最终结果/核验：[reports/laya-length-router-20261017-v1/REPORT.md](../../reports/laya-length-router-20261017-v1/REPORT.md)；[reports/laya-long-fusion-bc-20261017-v1/REPORT.md](../../reports/laya-long-fusion-bc-20261017-v1/REPORT.md)


## C18 DeBERTa-v3 异构训练与恢复

状态：两轮最终模型本地交付。CUDA BF16 开发 97.04%；之前 FP32 中断与试跑记录保留。

设计/计划：[docs/superpowers/specs/2026-10-17-deberta-budget7-design.md](../superpowers/specs/2026-10-17-deberta-budget7-design.md)

实现：[scripts/deberta_inputs.py](../../scripts/deberta_inputs.py)；[scripts/deberta_budget_training.py](../../scripts/deberta_budget_training.py)；[scripts/deberta_budget_study.py](../../scripts/deberta_budget_study.py)

最终结果/核验：[reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/REPORT.md](../../reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/REPORT.md)；[reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/local-delivery.json](../../reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/local-delivery.json)


## C19 Laya+DeBERTa 固定异构融合

状态：有开发与正式测试成绩。开发 97.28%；正式测试 97.028%，CPU FP32 DeBERTa+旧 CUDA Laya 缓存；不是部署入口成绩。

设计/计划：[docs/superpowers/specs/2026-10-18-deberta-final-evaluation-design.md](../superpowers/specs/2026-10-18-deberta-final-evaluation-design.md)

实现：[scripts/deberta_final_analysis.py](../../scripts/deberta_final_analysis.py)；[scripts/deberta_final_evaluation.py](../../scripts/deberta_final_evaluation.py)

最终结果/核验：[reports/deberta-laya-final-analysis-20261018-v1/REPORT.md](../../reports/deberta-laya-final-analysis-20261018-v1/REPORT.md)；[reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md](../../reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md)


## C20 CPU TF-IDF / NB-ngram 分支

状态：互补有限。TF-IDF 融合 97.32%，净+1，p=1；NB 融合 97.28%。

设计/计划：[docs/superpowers/specs/2026-10-18-bow-nb-complementarity-design.md](../superpowers/specs/2026-10-18-bow-nb-complementarity-design.md)

实现：[scripts/bow_nb_branch.py](../../scripts/bow_nb_branch.py)；[scripts/run_bow_nb_study.py](../../scripts/run_bow_nb_study.py)；[scripts/verify_bow_nb_study.py](../../scripts/verify_bow_nb_study.py)

最终结果/核验：[reports/bow-nb-complementarity-cpu-20261018-v1/REPORT.md](../../reports/bow-nb-complementarity-cpu-20261018-v1/REPORT.md)


## C21 BERT+BiLSTM 公开文献代码审计

状态：静态审计完成。公开代码分区不兼容；未证实电影泄漏或序列退化；不据此否定论文 97.67%。

设计/计划：以对应报告原协议为准；不补造独立规格。

实现：[scripts/audit_bert_bilstm_literature.py](../../scripts/audit_bert_bilstm_literature.py)

最终结果/核验：[reports/bert-bilstm-literature-audit-20261018-v1/REPORT.md](../../reports/bert-bilstm-literature-audit-20261018-v1/REPORT.md)


## C22 置信度与固定校准权重

状态：未获提升。校准 NLL 权重开发 97.16%；二模型按绝对置信度选边与 50:50 在二分类下类别相同，不是独立改进。

设计/计划：以对应报告原协议为准；不补造独立规格。

实现：[scripts/run_laya_deberta_calibration_weight.py](../../scripts/run_laya_deberta_calibration_weight.py)

最终结果/核验：[reports/laya-deberta-calibration-weight-cpu-20261018-v1/REPORT.md](../../reports/laya-deberta-calibration-weight-cpu-20261018-v1/REPORT.md)


## C23 正则化 logit、长文分支与多样本 Dropout

状态：探索性最高分，未达继续门槛。97.40% 仅开发；相对 97.28 净+3，p=.25，NLL/ECE 变差；无对应正式测试。v1 故障保留。

设计/计划：以对应报告原协议为准；不补造独立规格。

实现：[scripts/run_final_cpu_alternatives_v2.py](../../scripts/run_final_cpu_alternatives_v2.py)；[scripts/verify_final_cpu_alternatives.py](../../scripts/verify_final_cpu_alternatives.py)

最终结果/核验：[reports/final-cpu-alternatives-20261018-v2/REPORT.md](../../reports/final-cpu-alternatives-20261018-v2/REPORT.md)；[reports/final-cpu-alternatives-20261018-v2/independent-verification.json](../../reports/final-cpu-alternatives-20261018-v2/independent-verification.json)


## C24 全 token 情感查询与 Top-32 稀疏聚合

状态：九个头完成；没有增益。冻结编码器；三种子各 5 轮。稠密 96.84%、融合 97.28%；Top32 融合 97.24%。非 DSA 复现。

设计/计划：[docs/superpowers/specs/2026-10-18-token-query-head-design.md](../superpowers/specs/2026-10-18-token-query-head-design.md)

实现：[scripts/token_query_head.py](../../scripts/token_query_head.py)；[scripts/run_token_query_study.py](../../scripts/run_token_query_study.py)；[scripts/verify_token_query_study.py](../../scripts/verify_token_query_study.py)

最终结果/核验：[reports/laya-token-query-mps-fp32-20261018-v1/REPORT.md](../../reports/laya-token-query-mps-fp32-20261018-v1/REPORT.md)；[reports/laya-token-query-mps-fp32-20261018-v1/independent-verification.json](../../reports/laya-token-query-mps-fp32-20261018-v1/independent-verification.json)

## 尚未完成

端到端新头、领域预训练、无标签自训练、第三模型和ModernBERT-base/Laya-RLCD公平迁移未完成；L2缺失权重、历史数据使用及后端差异保留披露。旧提案存在不表示已执行。

[设计与配置](designs.md) · [代码与测试](code.md) · [报告与验证](reports.md) · [指标](metrics.md)
