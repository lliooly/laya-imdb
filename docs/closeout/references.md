# 参考资料、来源与证据等级

更新：2026-10-18。保留[早期选型清单](../guides/references.md)，本页补齐实际采用与审计过的来源；不是最新 SOTA 排行榜。未下载或转载整篇付费论文。

## 数据、模型与基础方法

| 来源 | 项目用途与证据边界 |
|---|---|
| [Maas et al., ACL 2011](https://aclanthology.org/P11-1015/)、[IMDB 官方数据](https://ai.stanford.edu/~amaas/data/sentiment/) | 数据来源；训练、开发、校准和测试协议见本地 manifest |
| [ModernBERT](https://arxiv.org/abs/2412.13663)、[作者说明](https://huggingface.co/blog/modernbert) | Laya 编码器；局部/全局交替注意力。项目实际配置和权重锁优先于模型家族概述 |
| [Laya 代码](https://github.com/NandhaKishorM/laya)、[模型](https://huggingface.co/convaiinnovations/laya) | 上游 typed decision 架构与训练器；laya==0.4.1，具体 revision 见 configs/laya_model_lock.json |
| [DeBERTaV3](https://arxiv.org/abs/2111.09543)、[large 模型卡](https://huggingface.co/microsoft/deberta-v3-large) | 异构训练；项目开发/测试成绩以本地冻结报告为准 |
| [Wang & Manning, ACL 2012](https://aclanthology.org/P12-2018/) | NB 加权 n-gram 强基线思路；本项目实现为 NB 特征+LR，不能称完整复现其他组合模型 |
| [层次注意力文档分类](https://aclanthology.org/N16-1174/) | 文档证据聚合的相关思路；本项目两个查询头是独立受控实现，不声称复现该网络 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 精确注意力的计算效率；换后端不等于相同配置下必然提升 Accuracy |
| [领域/任务继续预训练](https://arxiv.org/abs/2004.10964) | 后续方向；当前项目没有完成该实验 |

## 高分文献与纠错

2019 年[文档向量余弦相似度论文](https://aclanthology.org/P19-2057/)的原 97.42% 组合结果已有正式纠错：Bingyu 与 Arefyev 的[2022 复核](https://aclanthology.org/2022.insights-1.17/)指出拼接了同类的不同文档表示，纠正后的结果为 93.68%。因此不能只用“用了无标签/测试文本做表示学习”解释旧高分，更不能把 97.42% 当作当前严格协议的有效目标。

[BERT+BiLSTM 论文](https://arxiv.org/abs/2502.20682)及[作者公开仓库](https://github.com/gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis)属于另一种证据：本项目审计了固定提交 c4cbd9a179a9526ee952d3e58247d3f1466ff0c6，发现公开代码的分区和后处理问题，未支持单时间步退化假设。缺少绑定论文 97.67% 的原运行产物，不能据此宣布论文数字无效。[本地完整审计](../../reports/bert-bilstm-literature-audit-20261018-v1/REPORT.md)保留源码身份与静态推导。

## 前沿机制讨论

- [DSpark](https://arxiv.org/abs/2607.05147)：生成式模型的推测解码。没有直接移植到本项目一次前向的二分类流程。
- [DeepSeek-V3.2 / DSA](https://arxiv.org/abs/2512.02556)：学习式稀疏注意力及训练适配。本项目 Top32 只作用于两个分类查询读取 token 的头，不是 DSA 复现。
- LoRA、伪标签自训练、第三模型、领域预训练与端到端查询头：保留研究方向，不写成已完成或保证收益。

## 原始方案附件

用户提供的早期方案讨论原文保存在[只读来源说明](../archive/source-notes/README.md)。其中的伪标签纯度、测试增益、第三模型可修复比例及部分文献解释属于原提案断言，不能直接作为实验事实。所有定量结论优先采用本地冻结报告和上面的正式原文来源。

本页记录链接与项目中的使用状态；完整第三方许可证边界见[第三方声明](../../THIRD_PARTY_NOTICES.md)。

## 同期补充分析材料

另有[Accuracy 空间研究建议](../../reports/accuracy-improvement-research-20261018-v1/REPORT.md)，保留为独立分析资料，未新增训练。其标签噪声讨论、预期收益和“上限”属于推断，不作为新增实测结果或 Accuracy 数学界限；标签错误率本身不能直接推出模型对原评测标签的最高可达准确率。已归档实验结论仍以研究台账和冻结摘要为准。
