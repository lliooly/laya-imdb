# 原始讨论资料

`2026-10-18-user-provided-improvement-proposal.txt`为用户在会话中提供的原始方案附件，按原字节保存，供回溯当时的研究动机。原文不是核验报告，保留它不表示认可其中每项推断。

后续已经核实或修正：

- dv-cosine 旧 97.42% 有正式文献纠错至 93.68%，不能只归因于 transductive 文本使用。
- 高置信度不保证伪标签 99.7% 纯度；拒答覆盖率成绩不能推导无标签集真实准确率。
- 466 token 阈值属于 Laya 模板预算；DeBERTa 使用不同分词器和 510 评论 token 预算。
- 新融合相对旧 Laya 净+171，与相对 DeBERTa 净+24 是不同配对，不能共享显著性。
- 94 是融合相对旧 Laya 的新增错误，不是已有证据保证第三模型能救回的集合。
- 当前词袋、长上下文、校准权重和查询头的真实结果见[研究台账](../../closeout/research-ledger.md)。

原文附件：[原始文本](2026-10-18-user-provided-improvement-proposal.txt)。正式引用与证据边界见[参考资料](../../closeout/references.md)。
