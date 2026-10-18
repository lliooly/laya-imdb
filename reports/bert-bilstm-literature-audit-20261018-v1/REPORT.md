# BERT + BiLSTM 97.67%：公开实现审计

结论：**作者公开实现不满足官方 25k/25k 同协议复现要求；目前不能据此接受或否定论文 97.67% 的实际成绩。** 查到的数据划分差异明确；“BiLSTM 退化成单个时间步”没有得到这份代码支持。未训练模型，未读取本项目 IMDb 正文或预测，未获取官方测试分布。

## 来源与可追溯性

审计的是作者公开仓库 `gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis`，固定提交 `c4cbd9a179a9526ee952d3e58247d3f1466ff0c6`（2022-06-24T17:19:09Z），共三个 .py/.md 文件。源码原字节、Git blob SHA、文件 SHA256 及下载地址保存在 artifacts/bert-bilstm-literature-audit-20261018-v1/source-provenance.json。未执行第三方代码，只做 AST 静态检查和自写的合成来源计数。

论文为 [arXiv:2502.20682v1](https://arxiv.org/html/2502.20682v1)，版本日期 2025-02-28。公开代码最后提交为 2022 年，未发现绑定论文该成绩的精确运行配置、权重、预测或环境锁。因此以下判断适用于已审计公开提交，不自动等于论文实际运行情况。

## 1. [必须修复] 拼接官方分区后重新切分

源码分别加载并打乱官方 train 和 test，再按 train 在前、test 在后的顺序拼接；没有先对拼接后的全集做全局混洗。随后先取前 90% 作训练/验证池，再从该池取前 90% 作训练。[加载与拼接](https://github.com/gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis/blob/c4cbd9a179a9526ee952d3e58247d3f1466ff0c6/BERT%2BBiLSTM-SA.py#L37-L64)、[二次切分](https://github.com/gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis/blob/c4cbd9a179a9526ee952d3e58247d3f1466ff0c6/BERT%2BBiLSTM-SA.py#L171-L188)。

只按这段代码的切片规则，用 50,000 个虚拟来源标记推导：

| 新分区 | 总条数 | 来自官方 train | 来自官方 test |
|---|---:|---:|---:|
| 梯度训练 | 40,500 | 25,000 | 15,500 |
| 验证 | 4,500 | 0 | 4,500 |
| 最终评估 | 5,000 | 0 | 5,000 |

即官方测试集中 62% 的样本及其标签进入训练、18% 用作验证，剩余 20% 才作最终评估。此比例来自合成来源推导，不是读取真实测试内容或标签分布。新评估子集与新梯度子集在索引上分开，不能把它描述为“同一条样本必然同时训练和评估”；但它显然不再是官方固定 25k/25k 任务，也与论文表 I 的 25k/25k 数量不一致。

**同部电影泄漏尚未量化。** 官方分区内重切可能改变电影级隔离，但仓库没有保留电影 ID 与划分清单；本轮不下载数据、不作实际电影交集统计。不能把 97.67% 的来源归因为已经证实的电影泄漏，更不能凭此估计它在正确协议下会降到多少。

建议：从原始官方 train 单独划分验证集，原 test 完整保留；持久化 ID/哈希，再重新训练并完整评估。

## 2. [必须修复] 预测和标签使用不同的 shuffle 遍历

测试 Dataset 包含 shuffle，预测结束后，extract_labels 又重新遍历 Dataset，并按位置匹配预测。[测试构造](https://github.com/gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis/blob/c4cbd9a179a9526ee952d3e58247d3f1466ff0c6/BERT%2BBiLSTM-SA.py#L175-L178)、[预测与提取](https://github.com/gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis/blob/c4cbd9a179a9526ee952d3e58247d3f1466ff0c6/BERT%2BBiLSTM-SA.py#L263-L297)。[TensorFlow 文档](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle)说明默认在迭代间重新洗牌；因此两次遍历没有固定的逐样本对应保证。

该问题影响后处理的预测—标签配对。**不能据此声称 model.evaluate 的准确率被抬高**：model.evaluate 在同一次迭代中取得输入及标签，不依赖 extract_labels。源码中的分类报告调用还处于注释状态。

建议：评估和预测 Dataset 不打乱；输出 sample_id，按 ID 配对，或一次遍历同时收集预测与标签。

## 3. [必须修复] 标签提取在第一批后提前返回

[第 312 行](https://github.com/gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis/blob/c4cbd9a179a9526ee952d3e58247d3f1466ff0c6/BERT%2BBiLSTM-SA.py#L270-L316)的 return 位于外层批次循环内部。AST 检查确认这一缩进关系。在此代码导出的 5,000 条评估数据上，第一批为 32 条，函数便返回，后续 4,968 条不进入该后处理标签向量。

这同样不改变独立调用的 model.evaluate；只说明不能用这个提取函数验证全量逐样本统计。建议循环完成后统一返回，并断言预测数、标签数和唯一 ID 数等于整个评估集。

## 4. 序列维度：当前代码不支持“退化”的判断

公开实现调用的是 BERT 内部主层，读取第一个返回值，再接 Dropout 和 BiLSTM。按 [Hugging Face 的 BERT 返回约定](https://huggingface.co/docs/transformers/v4.20.1/en/model_doc/bert#transformers.TFBertModel)，这是 token 序列表示，预期形状 [batch,256,768]；BiLSTM 接收三维张量符合 [Keras 约定](https://keras.io/api/layers/recurrent_layers/lstm/)。[对应源码](https://github.com/gnkhata1/Finetuning-BERT-on-Movie-Reviews-Sentiment-Analysis/blob/c4cbd9a179a9526ee952d3e58247d3f1466ff0c6/BERT%2BBiLSTM-SA.py#L209-L220)。

论文文字描述称使用 [CLS]，与公开实现的序列读取存在差异，但不能据文字描述反过来认定代码把序列压为一个时间步。本轮未安装 TensorFlow、未运行原模型，形状判断为静态 API 推导，不冒充运行时检查。

另有需要运行时核验的点：attention_mask 仅显式传入 BERT，未显式传给后续 BiLSTM；需检查原环境的 Keras mask 是否传播，不能将潜在 padding 问题直接写成已经量化的准确率缺陷。

## 5. 训练配方也没有对齐论文的二分类设置

| 项目 | 论文 IV-C 二分类描述 | 公开实现 |
|---|---|---|
| 最大长度 | 128 | 256 |
| 学习率 | 3e-5 | 1e-4（含 decay=1e-5） |
| epochs | 10 | 4，且 Dataset repeat(2) |
| 梯度训练样本 | 表 I 为 25,000 | 切片推导为 40,500 |
| 冻结 | 文字称初始 BERT 层冻结 | 按 model.layers[2] 位置冻结，未记录实际层名 |

公开有限 Dataset 每 epoch 曝光 81,000 条，四轮合计 324,000 次，属于自写计数推导，不是实际运行日志。冻结语句采用层索引；通常对应内部 BERT 主层，但原依赖环境未锁定，不能从该索引直接声称只冻结了若干底层。上述差异意味着即使直接跑通仓库，也不能称为论文精确复现，更不能孤立归因于 BiLSTM 头。

## 6. 对本项目结论的修正

- 历史“71”指 71 条完整开发集方案记录，不等于 71 次独立训练或 71 组同构调参实验；同权重不同输入、后端复核等不能当独立机制证据。见本地 all-experiments-metrics-20261016-v1/REPORT.md。
- 68 条是当前 Laya+DeBERTa 50:50 融合的全部开发错误；47 条是此前明确三条流水线共同错误的不同集合。
- 正式测试净 +171 条及 p≈5.54e-20 比较的是新融合与旧 Laya 主方案。相对 DeBERTa 单模型仅净 +24 条，不能借用前一个 p 值宣称超过 DeBERTa 的显著性。
- 已尝试多个有代表性的方向，当前未发现更多稳定收益；不能表述为主流假设已被穷尽或证明只能有两个后续方向。

## 可用于研究报告的表述

> 对 BERT+BiLSTM 公开实现的静态审计发现，其合并官方训练与测试分区后二次划分的流程，与官方 25k/25k 基准及论文所列分区数量不一致；后处理还存在标签顺序对齐和批次覆盖问题。公开代码读取完整 token 序列，未支持单时间步退化假设。由于缺少绑定报告成绩的运行产物，本项目将 97.67% 记为尚未完成同协议复现的文献结果，不将其作为可直接比较的准确率结论，亦不据此宣布论文成绩无效。

此段不替代对 2019 年 dv-cosine 的独立纠错文献：两者证据类型不同，不能把“已有正式复核纠正”与“公开实现协议不一致”混为一谈。

## 复核与产物

运行 `uv run --locked --offline --no-sync python scripts/audit_bert_bilstm_literature.py`（已有目录时拒绝覆盖）。检查包含源码 SHA、AST 加载/切分结构、合成来源计数、return 缩进、公开超参数和 token 序列取值表达式。不执行下载的 Python，不导入 TensorFlow，不加载或下载模型，不访问本项目 IMDb 正文、官方测试预测或分布。不生成任何新的 accuracy 成绩。
