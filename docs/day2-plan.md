# Day2：预训练模型验证与训练预算

执行日期：2026-10-09（按阶段推进，不要求等到日历10/10）。延续已确认的 Day1 设计。

## 交付与顺序

1. 下载 Day1 锁定revision的 ModernBERT-base 原始预训练权重，记录 SHA-256。
2. 实现 MLX 编码器：embedding norm、首层无attention norm、交替局部/全局注意力、非交错RoPE、GeGLU、最终norm。
3. 使用自主 masked-mean pooling + 新线性二分类头。该头与 Hugging Face 内建序列分类头不同，需明确报告。
4. 对照 Transformers 4.57.6 的 CPU eager float32 实现；禁用 torch.compile，避免额外编译和平台差异。
5. 通过数值关卡后，探测长度256/512、batch4/8、float32/bfloat16，记录训练步耗时、MLX峰值内存、epoch估算。
6. 以256条训练评论做32步全量微调，128条固定验证评论作诊断，记录训练前后指标、编码器真实参数更新和保存重载一致性。
7. 整理本地训练建议与服务器决策，接续Day3完整基线。

不访问官方测试集，不根据短程验证的好坏增加参数搜索，不宣称得到最终准确率。

## 精度问题与处理依据

M5上的MLX默认可使用降低精度的float32矩阵运算路径；`float32`张量类型本身不足以说明计算精度。
按 [MLX官方精度文档](https://ml-explore.github.io/mlx/build/html/usage/precision.html)，在导入MLX前设置 `MLX_ENABLE_TF32=0`。
`day2_runtime.py` 提供该默认值；数值验证拒绝TF32开启状态。初始正式基线使用这一策略。

第一轮默认精度下出现明显前向与梯度偏差，未直接进入训练。关闭TF32后，大部分误差显著下降。
极短16-token截断在预训练深层出现很大的中间激活，float32舍入的差异可被放大；逐元素allclose在少量位置仍失败。
float64仲裁显示PyTorch float32自身也有此类误差，因此采用两层证据，而非声称跨框架逐位相同：

- 随机3层小模型：严格逐元素检查有效token、logits、loss和**所有参数梯度**；包含局部窗口及padding。
- 原始22层模型：长度16/160/256/512前向；hidden相对L2误差≤3e-4；logits仍用atol/rtol各3e-4的逐元素标准。
- 原始模型候选长度256反向：检查embedding、首层QKV、末层MLP、final norm及分类头，梯度相对L2误差≤1e-3；loss保持严格逐元素检查。
- padding扩展和屏蔽token ID扰动不影响分类输出。

所有最大绝对误差、相对误差及逐元素判定同时保留。标准是在定位TF32及激活敏感性后调整的工程验收标准，不是事前注册的实验假设。
长度16的原始梯度比较仍有约0.6%相对误差，记录于 `reports/day2_short_sequence_diagnostic.json`；本阶段不能宣称所有长度的反向等价。
训练候选从256起步；正式短评论仍允许右padding到256，不能把16-token截断诊断等同于自然短评论性能。

## 测速协议

每组独立进程，重载相同预训练权重和初始化种子；全部编码器参数参与反向。
真实训练评论预先分词，固定padding到256或512，排除下载、模型加载和tokenizer。
每组2步预热、5步计时；计时包含前向、loss、反向、梯度裁剪、AdamW与 `mx.eval` 同步。
固定padding给出这套实现的训练工作量估算，动态padding或更高效局部attention实现可能更快。
当前局部注意力用稠密mask表达，未实现原论文FlashAttention的稀疏/去padding优化；不能将论文速度结论直接用于此实现。

`mlx_peak_gib` 是MLX分配器峰值，不是系统总内存或所有进程RSS。epoch估算按22,500条、完整batch数向上取整外推。
估算不包含验证、检查点I/O、长期热降频、其他程序负载与冷启动，应留额外预算。
bfloat16探测采用低精度参数和优化器状态，有限梯度不意味着更新充分或收敛可靠；不能将其当作成熟混合精度训练。

## 短程训练协议

固定seed42，训练256条、验证128条，类别均衡；仅用train/validation，记录具体样本ID。
float32、TF32关闭、batch8、长度256、学习率2e-5、weight decay0.01、梯度norm裁剪1.0，32步（一次256条遍历）。
本次使用固定学习率，不执行最终训练的warmup、早停或checkpoint择优，避免把流程探测包装成完整实验。
分类头随机初始化，全部编码器可训练；记录末层MLP权重变化确认不是只训练分类头。
保存的是可加载的模型参数及运行元数据，**没有优化器/调度器状态，不作为精确续训检查点**。
报告训练前后Accuracy、Macro-F1、loss及混淆矩阵；这些小样本结果仅用于检查学习链路。

## 来源与后续约束

模型结构与权重命名参考 [Transformers ModernBERT源码](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/modernbert/modeling_modernbert.py)，Apache-2.0；代码文件保留来源说明。
论文与模型来源沿用 `docs/references.md`。预训练MLM头未用于情感分类；自主分类头需通过IMDB标签训练。
实现当前仅支持锁定base配置；没有承诺任意ModernBERT变体、Linux/CUDA或跨设备加载一致性。
