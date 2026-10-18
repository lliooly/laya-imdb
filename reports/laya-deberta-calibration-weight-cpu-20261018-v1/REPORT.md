# Laya / DeBERTa：校准集单系数融合实验

本地 CPU，仅复用冻结缓存。400 条校准拟合一个系数并封存，独立 evaluate 进程才读取 2500 条开发预测；不读取评论正文、官方测试文件或分布，不进行训练与新模型推理。

固定 p=α·p_Laya+(1−α)·p_DeBERTa，α∈[0,1]，只最小化校准 NLL，无正则、偏置、新温度、长度门控或词袋分支。原 Laya 内部融合和 DeBERTa 温度保持不变。DeBERTa 使用产生开发 97.28% 的 CUDA BF16 缓存。

**冻结权重：Laya 21.9419%，DeBERTa 78.0581%。**

| 方案 | 正确 / 2500 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| frozen_50_50 | 2432 | 97.28% | 0.088768 | 0.916% |
| calibration_nll_weight | 2429 | 97.16% | 0.088108 | 0.949% |

相对原 50:50：修复 8 条，退化 11 条，净变化 -3 条；Accuracy 差 -0.120 pp，条件配对 bootstrap 95% 区间 [-0.480, +0.200] pp；精确 McNemar p=0.647606。

## 校准目标与解释

校准 NLL：50:50 为 0.05543934，拟合后为 0.05413448。这只说明在该校准集的预定 NLL 目标改善，不保证开发 Accuracy 增加。

全部参数在开发评估前封存，无开发权重扫描、事后择优或追加模型。20,000 次类别分层配对 bootstrap 固定已训练模型和系数，seed=20261018；区间不包含校准估计、训练种子变化或反复开发使用的选择偏差，属于探索性结果。

## 验证与产物

原融合逐样本概率与 2432/2500 成绩完全复现；输入 ID/标签对齐、来源/源码哈希和冻结契约核验通过。独立 SciPy 标量优化核对校准系数，sklearn 独立复算开发 Accuracy/NLL/AUROC。

protocol.json / frozen.json 保存配置、来源 SHA、系数和冻结时间；calibration-predictions.json 与 development-predictions.json 保存对应预测；fit/evaluate-access-audit.json 保存读路径；verification.json 保存独立复算。训练阶段拒绝开发输入，两个进程均禁止官方测试路径。

入口：scripts/run_laya_deberta_calibration_weight.py，按 fit、evaluate 顺序分别运行。使用 environments/bow-cpu 的锁定环境，已有报告目录时拒绝覆盖。默认部署未切换，未据此重新打开官方测试。
