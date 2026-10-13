# 选型依据与阅读清单

核对日期：2026-10-09。本轮只采用官方文档、模型卡和论文原文入口。此前聊天中的前沿候选不自动当作已经复现的事实。

| 来源 | 核对到的事实 | 项目中的用途与限制 |
|---|---|---|
| [Maas et al., Learning Word Vectors for Sentiment Analysis, ACL 2011](https://aclanthology.org/P11-1015/)；[官方数据](https://ai.stanford.edu/~amaas/data/sentiment/) | 25k训练、25k测试的二分类数据，另有无标签数据 | 数据来源；引用原论文，保留官方测试划分 |
| [ModernBERT论文](https://arxiv.org/abs/2412.13663)；[作者模型卡](https://huggingface.co/answerdotai/ModernBERT-base) | 2024年提出，base 149M参数，原生8192上下文；使用RoPE及交替局部/全局注意力 | 主编码器候选；不是2026年新算法；作者GPU结果不能替代Mac实测 |
| [BERT模型卡](https://huggingface.co/google-bert/bert-base-uncased)；[MLX官方BERT示例](https://github.com/ml-explore/mlx-examples/tree/main/bert) | BERT-base 110M参数；有MLX示例入口 | 回退方案与适配参考；仍需检查权重映射、掩码和梯度 |
| [DistilBERT模型卡](https://huggingface.co/distilbert/distilbert-base-uncased) | 轻量预训练BERT家族模型 | 学生候选；是否比自主截层学生适合由实现成本和实测决定 |
| [AdaptBPE, 2026](https://arxiv.org/abs/2601.21665) | 根据适配语料替换低效token，保持目标词表大小；arXiv页面注明EACL 2026 | 领域分词研究候选；未在本项目复现，不能宣称影评准确率收益 |
| [Hinton et al., Distilling the Knowledge in a Neural Network, 2015](https://arxiv.org/abs/1503.02531) | 教师软目标的知识蒸馏方法 | 比较同一学生普通微调与蒸馏，不能只比较教师和学生 |
| [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index) | Rust分词实现及Python接口 | 今天用原版建立参照；后续自主Rust流水线需证明token ID一致性 |
| [MLX安装](https://ml-explore.github.io/mlx/build/html/install.html)；[神经网络API](https://ml-explore.github.io/mlx/build/html/python/nn.html) | Apple Silicon/Metal支持，文档也列出CUDA安装途径 | 本机先验证Metal；服务器CUDA及跨后端权重一致性另做验证 |

今日决策：优先原配tokenizer与ModernBERT候选，BERT保障交付；AdaptBPE和蒸馏有明确对照后再做。
暂不将Gigatoken、Wordchipper、LangMAP、JEPA或强化学习列入Day1实现，以免引入尚未核实的兼容性与额外工作。
本清单是选型材料，并非全面综述或“当前最优”排行榜。正式报告引用前补读采用方法的全文、检查具体实验协议与许可证。
