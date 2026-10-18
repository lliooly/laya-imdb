# 实验指标完整索引

保留 2026-10-16 的全部 123 条记录，并补入后续已完成实验。记录包括同权重多视图、后端复核、分组和选择性覆盖；条数不是独立训练次数。

开发集、正式测试和选择性覆盖必须分别比较。所有新增行来自冻结摘要，未重新计算预测。JSON 保留原指标、来源文件及键路径。

| ID | 分区 / 覆盖 | 方法 | Accuracy | 数量 | 协议 | 证据 |
|---|---|---|---:|---:|---|---|
| H-M001 | development / 1.000 | ModernBERT-base | 94.640% | 2500 | MLX；256 token；3 轮；训练 22500 | [摘要](../../reports/model-comparison/all_metrics.json) |
| H-M002 | development / 1.000 | Laya-CE：Mac 一轮 | 95.760% | 2500 | MPS FP32；512 token；1 轮；训练 22100 | [摘要](../../reports/model-comparison/all_metrics.json) |
| H-M003 | development / 1.000 | Laya-RLCD pilot | 93.720% | 2500 | MPS FP32；512 token；1 轮；仅训练 896、校准 128 | [摘要](../../reports/model-comparison/all_metrics.json) |
| H-M004 | development / 1.000 | Laya-CE pilot | 93.520% | 2500 | MPS FP32；512 token；1 轮；仅训练 896、校准 128 | [摘要](../../reports/model-comparison/all_metrics.json) |
| H-M005 | development / 1.000 | Laya-CE：Mac 两轮、原学习率 | 96.960% | 2500 | MPS FP32；prefix；512 token；2 轮 | [摘要](../../reports/laya-mac-ce-comparison-v1/summary.json) |
| H-M006 | development / 1.000 | Laya-CE：Mac 两轮、学习率 ×0.6 | 96.120% | 2500 | MPS FP32；prefix；512 token；2 轮 | [摘要](../../reports/laya-mac-ce-comparison-v1/summary.json) |
| H-M007 | development / 1.000 | Laya-CE：RTX 6000D 一轮 | 96.280% | 2500 | CUDA FP32；prefix；512 token；1 轮 | [摘要](../../reports/laya-cuda-reference-ce-l512-s42-fp32-v1/cuda-saved-validation.json) |
| H-M008 | development / 1.000 | 4090 两轮基线 | 95.880% | 2500 | RTX 4090；默认运行时；prefix；2 轮 | [摘要](../../reports/rtx4090-regularization-completed-v1/summary.json) |
| H-M009 | development / 1.000 | 标签平滑 0.05 | 96.160% | 2500 | RTX 4090；默认运行时；prefix；2 轮 | [摘要](../../reports/rtx4090-regularization-completed-v1/summary.json) |
| H-M010 | development / 1.000 | warmup 5% | 96.320% | 2500 | RTX 4090；默认运行时；prefix；2 轮 | [摘要](../../reports/rtx4090-regularization-completed-v1/summary.json) |
| H-M011 | development / 1.000 | warmup 5% + 标签平滑 0.05 | 96.440% | 2500 | RTX 4090；默认运行时；prefix；2 轮 | [摘要](../../reports/rtx4090-combined-completed-v1/summary.json) |
| H-M012 | development / 1.000 | warmup + 平滑 + R-Drop α=1 | 96.080% | 2500 | RTX 4090；prefix；2 轮；双前向 | [摘要](../../reports/rtx4090-rdrop-completed-v1/summary.json) |
| H-M013 | development / 1.000 | 4090 两轮基线 | 96.480% | 2500 | RTX 4090；默认运行时；prefix；2 轮 | [摘要](../../reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/cuda-saved-validation.json) |
| H-M014 | development / 1.000 | warmup 5% + 标签平滑 0.05 | 96.480% | 2500 | RTX 4090；默认运行时；prefix；2 轮 | [摘要](../../reports/laya-rtx4090-ce-combined-s43-epochs2-fp32-v1/cuda-saved-validation.json) |
| H-M015 | development / 1.000 | 45 条重采样：同期控制 | 95.960% | 2500 | 4090；默认运行时；warmup + 平滑；2 轮 | [摘要](../../reports/laya-rtx4090-ce-resampling-control-pool45-s43-epochs2-fp32-v2/cuda-saved-validation.json) |
| H-M016 | development / 1.000 | 45 条重采样：增强 | 96.160% | 2500 | 4090；默认运行时；warmup + 平滑；2 轮 | [摘要](../../reports/laya-rtx4090-ce-resampling-resampled-pool45-s43-epochs2-fp32-v2/cuda-saved-validation.json) |
| H-M017 | development / 1.000 | 45 条重采样 strict：同期控制 | 96.320% | 2500 | 4090；strict FP32；warmup + 平滑；2 轮 | [摘要](../../reports/laya-strict-pair-result-check-20261013-v1/summary.json) |
| H-M018 | development / 1.000 | 45 条重采样 strict：增强 | 96.400% | 2500 | 4090；strict FP32；warmup + 平滑；2 轮 | [摘要](../../reports/laya-strict-pair-result-check-20261013-v1/summary.json) |
| H-M019 | development / 1.000 | 语义 A：原文重复 | 96.520% | 2500 | strict FP32；30 来源；5% 曝光；2 轮 | [摘要](../../reports/semantic30-abc-completed-20261013-v1/summary.json) |
| H-M020 | development / 1.000 | 语义 B：普通改写 | 96.320% | 2500 | strict FP32；30 来源；5% 曝光；2 轮 | [摘要](../../reports/semantic30-abc-completed-20261013-v1/summary.json) |
| H-M021 | development / 1.000 | 语义 C：语义改写 | 96.440% | 2500 | strict FP32；30 来源；5% 曝光；2 轮 | [摘要](../../reports/semantic30-abc-completed-20261013-v1/summary.json) |
| H-M022 | development / 1.000 | plain：首尾窗口 | 96.080% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-baseline-s42-fp32-v1/analysis.json) |
| H-M023 | development / 1.000 | plain：前缀 + 温度/偏置 | 95.800% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-baseline-s42-fp32-v1/analysis.json) |
| H-M024 | development / 1.000 | plain：首尾 + 温度/偏置 | 96.040% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-baseline-s42-fp32-v1/analysis.json) |
| H-M025 | development / 1.000 | plain：首尾窗口 | 96.840% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-baseline-s43-fp32-v1/analysis.json) |
| H-M026 | development / 1.000 | plain：前缀 + 温度/偏置 | 96.520% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-baseline-s43-fp32-v1/analysis.json) |
| H-M027 | development / 1.000 | plain：首尾 + 温度/偏置 | 96.840% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-baseline-s43-fp32-v1/analysis.json) |
| H-M028 | development / 1.000 | warmup + 平滑：首尾窗口 | 96.680% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-combined-s42-fp32-v1/analysis.json) |
| H-M029 | development / 1.000 | warmup + 平滑：前缀 + 温度/偏置 | 96.440% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-combined-s42-fp32-v1/analysis.json) |
| H-M030 | development / 1.000 | warmup + 平滑：首尾 + 温度/偏置 | 96.720% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-combined-s42-fp32-v1/analysis.json) |
| H-M031 | development / 1.000 | warmup + 平滑：首尾窗口 | 96.640% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-combined-s43-fp32-v1/analysis.json) |
| H-M032 | development / 1.000 | warmup + 平滑：前缀 + 温度/偏置 | 96.400% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-combined-s43-fp32-v1/analysis.json) |
| H-M033 | development / 1.000 | warmup + 平滑：首尾 + 温度/偏置 | 96.560% | 2500 | 固定旧权重；仅推理/校准；512 token | [摘要](../../reports/laya-posthoc-combined-s43-fp32-v1/analysis.json) |
| H-M034 | development / 1.000 | plain：head/head_tail 门控等权 | 96.160% | 2500 | 固定权重；短文 head，长文两窗口概率均值 | [摘要](../../reports/laya-probability-fusion-s43-s42-20261014-v1/summary.json) |
| H-M035 | development / 1.000 | plain：head/head_tail 门控等权 | 96.800% | 2500 | 固定权重；短文 head，长文两窗口概率均值 | [摘要](../../reports/laya-probability-fusion-s43-s42-20261014-v1/summary.json) |
| H-M036 | development / 1.000 | warmup + 平滑：head/head_tail 门控等权 | 96.680% | 2500 | 固定权重；短文 head，长文两窗口概率均值 | [摘要](../../reports/laya-probability-fusion-s43-s42-20261014-v1/summary.json) |
| H-M037 | development / 1.000 | warmup + 平滑：head/head_tail 门控等权 | 96.720% | 2500 | 固定权重；短文 head，长文两窗口概率均值 | [摘要](../../reports/laya-probability-fusion-s43-s42-20261014-v1/summary.json) |
| H-M038 | development / 1.000 | 两模型融合：冻结主方案 | 96.960% | 2500 | 固定旧权重；概率融合 | [摘要](../../reports/laya-probability-fusion-s43-s42-20261014-v1/summary.json) |
| H-M039 | development / 1.000 | 两模型融合：固定 50:50 辅助对照 | 97.000% | 2500 | 固定旧权重；概率融合 | [摘要](../../reports/laya-probability-fusion-s43-s42-20261014-v1/summary.json) |
| H-M040 | development / 1.000 | 历史八视角等权融合：复核值 | 96.880% | 2500 | 固定旧权重；概率融合 | [摘要](../../reports/laya-probability-fusion-s43-s42-20261014-v1/summary.json) |
| H-M041 | development / 1.000 | 八视角无截距 OLS：诊断 | 96.680% | 2500 | 固定旧权重；概率融合 | [摘要](../../reports/laya-probability-fusion-s43-s42-20261014-v1/summary.json) |
| H-M042 | development / 1.000 | plain：选项反序 | 96.560% | 2500 | 固定旧权重；prefix；无训练 | [摘要](../../reports/laya-decision-study-s43-s42-strict-20261013-v1/summary.json) |
| H-M043 | development / 1.000 | plain：N1 候选重排探针 | 96.480% | 2500 | 固定旧权重；prefix；无训练 | [摘要](../../reports/laya-decision-study-s43-s42-strict-20261013-v1/summary.json) |
| H-M044 | development / 1.000 | warmup + 平滑：选项反序 | 96.320% | 2500 | 固定旧权重；prefix；无训练 | [摘要](../../reports/laya-decision-study-s43-s42-strict-20261013-v1/summary.json) |
| H-M045 | development / 1.000 | warmup + 平滑：N1 候选重排探针 | 96.440% | 2500 | 固定旧权重；prefix；无训练 | [摘要](../../reports/laya-decision-study-s43-s42-strict-20261013-v1/summary.json) |
| H-M046 | development / 1.000 | 前缀训练控制：前缀推理 | 96.360% | 2500 | strict FP32；2 轮；固定训练预算 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M047 | development / 1.000 | 前缀训练控制：首尾推理（主比较） | 96.640% | 2500 | strict FP32；2 轮；固定训练预算 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M048 | development / 1.000 | 前缀训练控制：选项反序诊断 | 96.320% | 2500 | strict FP32；2 轮；固定训练预算 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M049 | development / 1.000 | 前缀训练控制：门控等权诊断 | 96.720% | 2500 | strict FP32；2 轮；固定训练预算 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M050 | development / 1.000 | 前缀训练控制：N1 候选特征探针 | 96.360% | 2500 | strict FP32；2 轮；固定训练预算 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M051 | development / 1.000 | 随机窗口训练：前缀推理 | 96.240% | 2500 | strict FP32；2 轮；固定训练预算 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M052 | development / 1.000 | 随机窗口训练：首尾推理（主比较） | 96.440% | 2500 | strict FP32；2 轮；固定训练预算 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M053 | development / 1.000 | 随机窗口训练：选项反序诊断 | 96.280% | 2500 | strict FP32；2 轮；固定训练预算 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M054 | development / 1.000 | 随机窗口训练：门控等权诊断 | 96.560% | 2500 | strict FP32；2 轮；固定训练预算 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M055 | development / 1.000 | 随机窗口训练：N1 候选特征探针 | 96.240% | 2500 | strict FP32；2 轮；固定训练预算 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M056 | development / 1.000 | N8：最终轮控制 | 96.640% | 2500 | 前缀训练权重；首尾推理；strict FP32；无新更新 | [摘要](../../reports/laya-checkpoint-average-prefix-s42-strict-20261014-v1/summary.json) |
| H-M057 | development / 1.000 | N8：epoch 1/2 参数平均 50:50 | 96.640% | 2500 | 前缀训练权重；首尾推理；strict FP32；无新更新 | [摘要](../../reports/laya-checkpoint-average-prefix-s42-strict-20261014-v1/summary.json) |
| H-M058 | development / 1.000 | L1：编码器输出线性探针 | 96.680% | 2500 | 冻结编码器；训练线性读出；首尾输入 | [摘要](../../reports/laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1/summary.json) |
| H-M059 | development / 1.000 | L1：head 后线性探针 | 96.800% | 2500 | 冻结编码器；训练线性读出；首尾输入 | [摘要](../../reports/laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1/summary.json) |
| H-M060 | development / 1.000 | L2：重新初始化原始头 | 96.520% | 2500 | 4090；strict CUDA FP32；首尾训练/推理；2 轮 | [摘要](../../reports/laya-head-control-analysis-20261015-v1/summary.json) |
| H-M061 | development / 1.000 | L2：共享线性头 | 96.720% | 2500 | 4090；strict CUDA FP32；首尾训练/推理；2 轮 | [摘要](../../reports/laya-head-control-analysis-20261015-v1/summary.json) |
| H-M062 | development / 1.000 | L2：重新初始化原始头 | 96.680% | 2500 | 4090；strict CUDA FP32；首尾训练/推理；2 轮 | [摘要](../../reports/laya-head-control-analysis-20261015-v1/summary.json) |
| H-M063 | development / 1.000 | L2：共享线性头 | 96.480% | 2500 | 4090；strict CUDA FP32；首尾训练/推理；2 轮 | [摘要](../../reports/laya-head-control-analysis-20261015-v1/summary.json) |
| H-M064 | development / 1.000 | L2：重新初始化原始头 | 96.360% | 2500 | 4090；strict CUDA FP32；首尾训练/推理；2 轮 | [摘要](../../reports/laya-head-control-analysis-20261015-v1/summary.json) |
| H-M065 | development / 1.000 | L2：共享线性头 | 96.280% | 2500 | 4090；strict CUDA FP32；首尾训练/推理；2 轮 | [摘要](../../reports/laya-head-control-analysis-20261015-v1/summary.json) |
| H-M066 | development / 1.000 | Modal：CE 同期控制 | 96.240% | 2500 | L40S；strict CUDA FP32；head/prefix；2 轮 | [摘要](../../reports/laya-robust-loss-analysis-20261016-v1/summary.json) |
| H-M067 | development / 1.000 | Modal：0.5 CE + 0.5 GCE (q=0.7) | 96.400% | 2500 | L40S；strict CUDA FP32；head/prefix；2 轮 | [摘要](../../reports/laya-robust-loss-analysis-20261016-v1/summary.json) |
| H-M068 | development / 1.000 | Modal：CE 同期控制 | 96.160% | 2500 | L40S；strict CUDA FP32；head/prefix；2 轮 | [摘要](../../reports/laya-robust-loss-analysis-20261016-v1/summary.json) |
| H-M069 | development / 1.000 | Modal：0.5 CE + 0.5 GCE (q=0.7) | 96.160% | 2500 | L40S；strict CUDA FP32；head/prefix；2 轮 | [摘要](../../reports/laya-robust-loss-analysis-20261016-v1/summary.json) |
| H-M070 | development / 1.000 | Modal：CE 同期控制 | 96.560% | 2500 | L40S；strict CUDA FP32；head/prefix；2 轮 | [摘要](../../reports/laya-robust-loss-analysis-20261016-v1/summary.json) |
| H-M071 | development / 1.000 | Modal：0.5 CE + 0.5 GCE (q=0.7) | 96.640% | 2500 | L40S；strict CUDA FP32；head/prefix；2 轮 | [摘要](../../reports/laya-robust-loss-analysis-20261016-v1/summary.json) |
| H-M072 | official_test / 1.000 | 正式测试：plain-s43 首尾单模型 | 96.244% | 25000 | 冻结后一次正式测试；25000 条 | [摘要](../../reports/laya-frozen-official-test-20261014-v1/summary.json) |
| H-M073 | official_test / 1.000 | 正式测试：冻结主融合 | 96.344% | 25000 | 冻结后一次正式测试；25000 条 | [摘要](../../reports/laya-frozen-official-test-20261014-v1/summary.json) |
| H-M074 | development_selective / 1.000 | 首尾单模型 | 96.840% | 2500 | 按开发置信度保留前 100% | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M075 | development_selective / 1.000 | 冻结主融合 | 96.960% | 2500 | 按开发置信度保留前 100% | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M076 | development_selective / 0.900 | 首尾单模型 | 98.844% | 2250 | 按开发置信度保留前 90% | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M077 | development_selective / 0.900 | 冻结主融合 | 98.844% | 2250 | 按开发置信度保留前 90% | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M078 | development_selective / 0.800 | 首尾单模型 | 99.500% | 2000 | 按开发置信度保留前 80% | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M079 | development_selective / 0.800 | 冻结主融合 | 99.550% | 2000 | 按开发置信度保留前 80% | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M080 | development_selective / 0.700 | 首尾单模型 | 99.600% | 1750 | 按开发置信度保留前 70% | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M081 | development_selective / 0.700 | 冻结主融合 | 99.714% | 1750 | 按开发置信度保留前 70% | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M082 | development_selective / 0.500 | 首尾单模型 | 99.760% | 1250 | 按开发置信度保留前 50% | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M083 | development_selective / 0.500 | 冻结主融合 | 99.920% | 1250 | 按开发置信度保留前 50% | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M084 | development_selective / 1.000 | 首尾单模型 | 96.840% | 2500 | 校准目标 100%；阈值 0.5000000000 | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M085 | development_selective / 1.000 | 冻结主融合 | 96.960% | 2500 | 校准目标 100%；阈值 0.5000000000 | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M086 | development_selective / 0.886 | 首尾单模型 | 98.962% | 2216 | 校准目标 90%；阈值 0.9753408463 | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M087 | development_selective / 0.907 | 冻结主融合 | 98.809% | 2267 | 校准目标 90%；阈值 0.9703539312 | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M088 | development_selective / 0.811 | 首尾单模型 | 99.408% | 2027 | 校准目标 80%；阈值 0.9890679887 | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M089 | development_selective / 0.816 | 冻结主融合 | 99.510% | 2041 | 校准目标 80%；阈值 0.9856013400 | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M090 | development_selective / 0.685 | 首尾单模型 | 99.650% | 1712 | 校准目标 70%；阈值 0.9928849556 | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M091 | development_selective / 0.691 | 冻结主融合 | 99.711% | 1728 | 校准目标 70%；阈值 0.9896130320 | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M092 | development_selective / 0.500 | 首尾单模型 | 99.760% | 1249 | 校准目标 50%；阈值 0.9943184650 | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M093 | development_selective / 0.497 | 冻结主融合 | 99.919% | 1242 | 校准目标 50%；阈值 0.9910186154 | [摘要](../../reports/laya-risk-coverage-20261015-v1/summary.json) |
| H-M094 | development_common2100 / 1.000 | ModernBERT-base：公共 2100 条 | 94.667% | 2100 | 开发集抽出 400 条统一温度拟合，剩余 2100 评估 | [摘要](../../reports/model-comparison/all_metrics.json) |
| H-M095 | development_common2100 / 1.000 | Laya-CE full：公共 2100 条 | 95.762% | 2100 | 开发集抽出 400 条统一温度拟合，剩余 2100 评估 | [摘要](../../reports/model-comparison/all_metrics.json) |
| H-M096 | development_common2100 / 1.000 | Laya-RLCD pilot：公共 2100 条 | 93.667% | 2100 | 开发集抽出 400 条统一温度拟合，剩余 2100 评估 | [摘要](../../reports/model-comparison/all_metrics.json) |
| H-M097 | development_common2100 / 1.000 | Laya-CE pilot：公共 2100 条 | 93.619% | 2100 | 开发集抽出 400 条统一温度拟合，剩余 2100 评估 | [摘要](../../reports/model-comparison/all_metrics.json) |
| H-M098 | development_truncated407 / 1.000 | baseline-s43：EXP-2 滑窗宽度 128 | 96.069% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-study-s43-s42-strict-20261013-v1/summary.json) |
| H-M099 | development_truncated407 / 1.000 | baseline-s43：EXP-2 滑窗宽度 256 | 95.823% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-study-s43-s42-strict-20261013-v1/summary.json) |
| H-M100 | development_truncated407 / 1.000 | baseline-s43：EXP-2 滑窗宽度 466 | 97.052% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-study-s43-s42-strict-20261013-v1/summary.json) |
| H-M101 | development_truncated407 / 1.000 | combined-s42：EXP-2 滑窗宽度 128 | 94.349% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-study-s43-s42-strict-20261013-v1/summary.json) |
| H-M102 | development_truncated407 / 1.000 | combined-s42：EXP-2 滑窗宽度 256 | 96.069% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-study-s43-s42-strict-20261013-v1/summary.json) |
| H-M103 | development_truncated407 / 1.000 | combined-s42：EXP-2 滑窗宽度 466 | 96.314% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-study-s43-s42-strict-20261013-v1/summary.json) |
| H-M104 | development_truncated407 / 1.000 | prefix-trained：EXP-2 滑窗宽度 128 | 96.314% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M105 | development_truncated407 / 1.000 | prefix-trained：EXP-2 滑窗宽度 256 | 96.560% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M106 | development_truncated407 / 1.000 | prefix-trained：EXP-2 滑窗宽度 466 | 97.297% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M107 | development_truncated407 / 1.000 | random-window-trained：EXP-2 滑窗宽度 128 | 95.577% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M108 | development_truncated407 / 1.000 | random-window-trained：EXP-2 滑窗宽度 256 | 96.314% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M109 | development_truncated407 / 1.000 | random-window-trained：EXP-2 滑窗宽度 466 | 97.789% | 407 | 仅 407 条截断评论；predict_long；步长=窗口一半 | [摘要](../../reports/laya-decision-execution-20261014-v1-final-study/summary.json) |
| H-M110 | development / 1.000 | Mac 一轮 CE：mps-fp32 | 95.760% | 2500 | 同一权重；后端核对 | [摘要](../../reports/laya-backend-comparison/mps-fp32-eval.json) |
| H-M111 | development / 1.000 | Mac 一轮 CE：mlx-fp32 | 95.760% | 2500 | 同一权重；后端核对 | [摘要](../../reports/laya-backend-comparison/mlx-fp32-eval.json) |
| H-M112 | development / 1.000 | Mac 一轮 CE：mlx-fp16 | 95.760% | 2500 | 同一权重；后端核对 | [摘要](../../reports/laya-backend-comparison/mlx-fp16-eval.json) |
| H-M113 | development / 1.000 | Mac 两轮 原 LR：MLX FP16 | 96.960% | 2500 | 同权重 MLX FP16，batch 8 | [摘要](../../reports/laya-mac-ce-comparison-v1/summary.json) |
| H-M114 | development / 1.000 | Mac 两轮 LR ×0.6：MLX FP16 | 96.120% | 2500 | 同权重 MLX FP16，batch 8 | [摘要](../../reports/laya-mac-ce-comparison-v1/summary.json) |
| H-M115 | development / 1.000 | 首尾单模型：CUDA FP32 | 96.840% | 2500 | 已有权重完整开发集工程复核 | [摘要](../../reports/laya-fusion-inference-cuda-fp32-20261014-v1/summary.json) |
| H-M116 | development / 1.000 | 主融合：CUDA FP32 | 96.960% | 2500 | 已有权重完整开发集工程复核 | [摘要](../../reports/laya-fusion-inference-cuda-fp32-20261014-v1/summary.json) |
| H-M117 | development / 1.000 | 50:50 辅助融合：CUDA FP32 | 97.000% | 2500 | 已有权重完整开发集工程复核 | [摘要](../../reports/laya-fusion-inference-cuda-fp32-20261014-v1/summary.json) |
| H-M118 | development / 1.000 | 首尾单模型：MLX FP16 | 96.840% | 2500 | 已有权重完整开发集工程复核 | [摘要](../../reports/laya-fusion-inference-mlx-fp16-20261014-v1/summary.json) |
| H-M119 | development / 1.000 | 主融合：MLX FP16 | 96.960% | 2500 | 已有权重完整开发集工程复核 | [摘要](../../reports/laya-fusion-inference-mlx-fp16-20261014-v1/summary.json) |
| H-M120 | development / 1.000 | 50:50 辅助融合：MLX FP16 | 97.000% | 2500 | 已有权重完整开发集工程复核 | [摘要](../../reports/laya-fusion-inference-mlx-fp16-20261014-v1/summary.json) |
| H-M121 | development / 1.000 | 首尾单模型：MLX precise FP32 | 96.840% | 2500 | 已有权重完整开发集工程复核 | [摘要](../../reports/laya-fusion-inference-mlx-fp32-precise-20261014-v1/summary.json) |
| H-M122 | development / 1.000 | 主融合：MLX precise FP32 | 96.960% | 2500 | 已有权重完整开发集工程复核 | [摘要](../../reports/laya-fusion-inference-mlx-fp32-precise-20261014-v1/summary.json) |
| H-M123 | development / 1.000 | 50:50 辅助融合：MLX precise FP32 | 97.000% | 2500 | 已有权重完整开发集工程复核 | [摘要](../../reports/laya-fusion-inference-mlx-fp32-precise-20261014-v1/summary.json) |
| N-001 | development / 1.000 | Laya 512 token | 96.320% | 2500 | 同 seed42、原标签、两轮 FP32、首尾、最终 epoch2 | [摘要](../../reports/laya-context-pair-s42-modal-20261016-v1/evaluation/summary.json) |
| N-002 | development / 1.000 | Laya 2048 token | 96.320% | 2500 | 同 seed42、原标签、两轮 FP32、首尾、最终 epoch2 | [摘要](../../reports/laya-context-pair-s42-modal-20261016-v1/evaluation/summary.json) |
| N-003 | development / 1.000 | 用户修订 v2 | 96.400% | 2500 | 61 翻标+3 删除；原始 512 对照；最终 epoch2 | [摘要](../../reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation/summary.json) |
| N-004 | development / 1.000 | 固定长度路由 | 96.560% | 2500 | 原 review tokens<=466 走 512，长文走 2048 | [摘要](../../reports/laya-length-router-20261017-v1/summary.json) |
| N-005 | development / 1.000 | 路由 A | 96.560% | 2500 | 固定旧模型与长文分支，400 校准后冻结 | [摘要](../../reports/laya-long-fusion-bc-20261017-v1/summary.json) |
| N-006 | development / 1.000 | 512/2048 融合 B | 96.520% | 2500 | 固定旧模型与长文分支，400 校准后冻结 | [摘要](../../reports/laya-long-fusion-bc-20261017-v1/summary.json) |
| N-007 | development / 1.000 | 旧主方案+长文 C | 96.960% | 2500 | 固定旧模型与长文分支，400 校准后冻结 | [摘要](../../reports/laya-long-fusion-bc-20261017-v1/summary.json) |
| N-008 | development / 1.000 | 辅助方案+长文 C_aux | 97.000% | 2500 | 固定旧模型与长文分支，400 校准后冻结 | [摘要](../../reports/laya-long-fusion-bc-20261017-v1/summary.json) |
| N-009 | development / 1.000 | Laya 原主方案 | 96.960% | 2500 | DeBERTa CUDA BF16；原 Laya 主概率；固定 50:50 | [摘要](../../reports/deberta-laya-final-analysis-20261018-v1/summary.json) |
| N-010 | development / 1.000 | DeBERTa-v3-large | 97.040% | 2500 | DeBERTa CUDA BF16；原 Laya 主概率；固定 50:50 | [摘要](../../reports/deberta-laya-final-analysis-20261018-v1/summary.json) |
| N-011 | development / 1.000 | 固定异构融合 | 97.280% | 2500 | DeBERTa CUDA BF16；原 Laya 主概率；固定 50:50 | [摘要](../../reports/deberta-laya-final-analysis-20261018-v1/summary.json) |
| N-012 | official_test / 1.000 | 旧 Laya 主方案 | 96.344% | 25000 | 完整官方 25k；CPU FP32 DeBERTa 与旧 CUDA Laya 缓存 | [摘要](../../reports/deberta-laya-final-cpu-evaluation-20261018-v1/summary.json) |
| N-013 | official_test / 1.000 | DeBERTa CPU | 96.932% | 25000 | 完整官方 25k；CPU FP32 DeBERTa 与旧 CUDA Laya 缓存 | [摘要](../../reports/deberta-laya-final-cpu-evaluation-20261018-v1/summary.json) |
| N-014 | official_test / 1.000 | 固定异构融合 | 97.028% | 25000 | 完整官方 25k；CPU FP32 DeBERTa 与旧 CUDA Laya 缓存 | [摘要](../../reports/deberta-laya-final-cpu-evaluation-20261018-v1/summary.json) |
| N-015 | development / 1.000 | TF-IDF 单分支 | 91.120% | 2500 | 22100 全文 fit；400 校准温度与权重；CPU | [摘要](../../reports/bow-nb-complementarity-cpu-20261018-v1/summary.json) |
| N-016 | development / 1.000 | NB-ngram 单分支 | 92.240% | 2500 | 22100 全文 fit；400 校准温度与权重；CPU | [摘要](../../reports/bow-nb-complementarity-cpu-20261018-v1/summary.json) |
| N-017 | development / 1.000 | TF-IDF 融合 | 97.320% | 2500 | 22100 全文 fit；400 校准温度与权重；CPU | [摘要](../../reports/bow-nb-complementarity-cpu-20261018-v1/summary.json) |
| N-018 | development / 1.000 | NB 融合 | 97.280% | 2500 | 22100 全文 fit；400 校准温度与权重；CPU | [摘要](../../reports/bow-nb-complementarity-cpu-20261018-v1/summary.json) |
| N-019 | development / 1.000 | 校准概率权重融合 | 97.160% | 2500 | 400 校准最小 NLL；原温度保持 | [摘要](../../reports/laya-deberta-calibration-weight-cpu-20261018-v1/summary.json) |
| N-020 | development / 1.000 | 正则化 logit 融合 | 97.400% | 2500 | 校准冻结；MLP 基于两标记缓存；无编码器更新 | [摘要](../../reports/final-cpu-alternatives-20261018-v2/summary.json) |
| N-021 | development / 1.000 | 长文辅助分支 | 97.280% | 2500 | 校准冻结；MLP 基于两标记缓存；无编码器更新 | [摘要](../../reports/final-cpu-alternatives-20261018-v2/summary.json) |
| N-022 | development / 1.000 | 普通 Dropout 辅助分支 | 97.240% | 2500 | 校准冻结；MLP 基于两标记缓存；无编码器更新 | [摘要](../../reports/final-cpu-alternatives-20261018-v2/summary.json) |
| N-023 | development / 1.000 | 四次 Dropout 辅助分支 | 97.240% | 2500 | 校准冻结；MLP 基于两标记缓存；无编码器更新 | [摘要](../../reports/final-cpu-alternatives-20261018-v2/summary.json) |
| N-024 | development / 1.000 | mlp_dropout1 独立分支 | 96.840% | 2500 | 校准冻结；MLP 基于两标记缓存；无编码器更新 | [摘要](../../reports/final-cpu-alternatives-20261018-v2/summary.json) |
| N-025 | development / 1.000 | mlp_dropout4 独立分支 | 96.840% | 2500 | 校准冻结；MLP 基于两标记缓存；无编码器更新 | [摘要](../../reports/final-cpu-alternatives-20261018-v2/summary.json) |
| N-026 | development / 1.000 | 均值分支 | 96.840% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-027 | development / 1.000 | 稠密查询分支 | 96.840% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-028 | development / 1.000 | Top32 分支 | 96.800% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-029 | development / 1.000 | 均值融合 | 97.280% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-030 | development / 1.000 | 稠密查询融合 | 97.280% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-031 | development / 1.000 | Top32 融合 | 97.240% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-032 | development / 1.000 | mean 单种子 s42 | 96.760% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-033 | development / 1.000 | mean 单种子 s43 | 96.840% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-034 | development / 1.000 | mean 单种子 s44 | 96.880% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-035 | development / 1.000 | dense 单种子 s42 | 96.760% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-036 | development / 1.000 | dense 单种子 s43 | 96.840% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-037 | development / 1.000 | dense 单种子 s44 | 96.880% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-038 | development / 1.000 | top32 单种子 s42 | 96.720% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-039 | development / 1.000 | top32 单种子 s43 | 96.880% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
| N-040 | development / 1.000 | top32 单种子 s44 | 96.920% | 2500 | 冻结 s43 编码器；FP32 全 token；三个头部种子各 5 轮 | [摘要](../../reports/laya-token-query-mps-fp32-20261018-v1/summary.json) |
