# 全序列情感查询与稀疏选择：实验结果

本轮已完成本机 Apple GPU 的真实训练与开发集评估，未获得 Accuracy 提升。稠密查询与均值池化对照的三种子集成均为 **96.84%**，且开发集 2500 条分类结果逐条一致；Top-32 为 **96.80%**。加入原 97.28% 神经融合后，前两组保持 97.28%，Top-32 降至 97.24%。本轮结果未超过此前 97.40% 的探索性开发成绩。

## 实验实际覆盖的范围

使用已经微调完成的 baseline-s43 ModernBERT 编码器，冻结其参数，重新提取全部有效 token 的 FP32 表征。原始输入、头尾截断、候选顺序和 22100/400/2500 划分不变。三种决策头的区别仅为聚合权重：

- 均值池化：有效 token 均匀加权，作为结构对照。
- 稠密查询：两个情感查询读取完整有效 token，学习 softmax 注意力。
- Top-32 查询：每个查询只对得分最高的至多 32 个有效 token 进行 softmax 聚合。

每个头 265089 个参数，种子 42/43/44，每种子内初始权重逐张量相等。各头固定训练 5 轮，使用相同样本顺序、batch 32、AdamW lr=3e-4、weight decay=.01、dropout=.1、梯度裁剪1。共 45 个训练轮次、31095 次更新、994500 次样本暴露；九个头训练与校准预测合计约 488 秒，不含编码器提取。

这轮检验的是冻结编码器上的全序列聚合能力。没有更新编码器，没有进行领域继续预训练，也没有复现 DeepSeek DSA 或 DSpark。缓存仅包含输入保留的 token，无法恢复截断掉的内容。

## 结果与对照

| 方法 | seed 42 | seed 43 | seed 44 | 三种子概率集成 | 加入原融合后 |
|---|---:|---:|---:|---:|---:|
| 均值池化 | 96.76% | 96.84% | 96.88% | 96.84% | 97.28% |
| 稠密查询 | 96.76% | 96.84% | 96.88% | 96.84% | 97.28% |
| Top-32 | 96.72% | 96.88% | 96.92% | 96.80% | 97.24% |

没有根据开发成绩选择 seed 44。三种子概率集成与单种子准确率平均是不同的运算，Top-32 集成低于部分单种子是实际观测结果。

稠密查询与均值池化的概率不同，但分类结果相同。稠密分支 NLL 为 0.104309，均值组为 0.104016，没有观察到额外优势。两者分别相对原 s43 头修复 2 条、增加 2 条错误，净增为 0。Top-32 相对均值组没有修复样本，增加 1 条错误。

各单种子温度只在固定校准 400 条上拟合，随后三种子概率取均值。新增分支融合权重以校准 NLL 在 [0,.25] 内确定，得到均值组 0、稠密组 0、Top-32 0.0036796866783298885。全部模型、温度和权重均在读取本轮开发数据前冻结。

Top-32 融合相对原神经融合修复 0 条、新错 1 条，净变化 -0.04 pp；条件配对分层 bootstrap 95% 区间为 [-0.12,0] pp，精确 McNemar p=1，三候选 Holm p=1。其融合 NLL 从 0.088767765 微降至 0.088764968，不能据此称为 Accuracy 提升。各组均未达到预先声明的继续条件。

## 核验与复现

训练、校准、开发共 25000 条的输入摘要与历史记录一致；标记向量差值的历史核对最大误差为 0。九个头保存重载后的校准预测完全一致。独立 NumPy FP64 前向与 MPS 输出最大差值约 1.21e-6；独立温度、融合最优解、sklearn 指标、精确检验、Holm 修正、训练暴露次数及访问审计均通过。5 项单元测试通过，包括 padding 隔离、稀疏掩码、有效梯度、共同初始化及开发/测试访问阻断。

历史开发集与校准集已反复用于研究，本轮结果不代表新的无偏盲评。本轮没有读取官方测试数据或预测，没有切换默认推理模型。

配置：`configs/laya-token-query-mps-fp32-20261018-v1.json`。

入口：`scripts/run_token_query_study.py`，按 `extract`、`train`、`develop`、`evaluate` 分进程执行。前三阶段使用主项目 uv.lock、FP32 MPS；评价与独立验证使用 `environments/bow-cpu/uv.lock`。在仓库根目录运行：

```sh
uv run --locked --offline --no-sync python scripts/run_token_query_study.py extract
uv run --locked --offline --no-sync python scripts/run_token_query_study.py train
uv run --locked --offline --no-sync python scripts/run_token_query_study.py develop
uv run --project environments/bow-cpu --locked --offline --no-sync python scripts/run_token_query_study.py evaluate
uv run --project environments/bow-cpu --locked --offline --no-sync python scripts/verify_token_query_study.py
```

已存在的结果目录会拒绝覆盖。实际复跑需使用新的 run_id 并同步独立配置与目录；MPS 阶段需在可以访问本机 GPU 的终端执行。

完整证据保存在 `reports/laya-token-query-mps-fp32-20261018-v1/`，包括 `REPORT.md`、`summary.json`、`heads-frozen.json`、`independent-verification.json`、源码快照及逐阶段访问审计。FP32 token 缓存与权重位于同名 artifacts 目录，仅本地保存。

## 后续判断

本轮没有给继续扩展冻结编码器上的头部搜索提供 Accuracy 证据，可以按当前资源目标准备收尾。端到端查询头适配、影评领域继续预训练属于尚未完成的独立研究方向；本轮结果不能判定它们无效，也不足以承诺它们能超过现有融合。
