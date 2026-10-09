# Day3：完整基线与 Rust 分词流水线

延续已确认的两周方案。阶段编号代表工作进度，实际执行日期为 2026-10-09。

## 训练协议

固定 ModernBERT-base 原配 tokenizer、Day1 划分及 seed 42。训练 22,500 条，验证 2,500 条，官方测试集不参与训练、分词缓存、模型选择和错误分析。

- 全编码器微调，masked mean pooling + 自主二分类线性头。
- float32，导入 MLX 前关闭 TF32；长度 256，右侧截断及固定右 padding，batch 8。
- 3 个 epoch，每个 epoch 2,813 步，最后一批 4 条，不丢样本；共 8,439 步。
- AdamW，峰值学习率 2e-5，weight decay 0.01，梯度范数裁剪 1.0。
- 前 844 步线性预热，然后线性衰减到最后一步的 0。
- **显式开启 Adam 偏差修正**。Day2 流程探测沿用 MLX 默认关闭值，Day3 是正式完整基线，不能将两者的学习曲线作为只改变训练样本数的消融实验。
- 每轮遍历后在完整验证集评估，按 Macro-F1 选择检查点，相同时选择较低 loss。连续两轮无改善早停，最多训练三轮。
- 固定抽取 2,048 条类别均衡训练评论，每轮用 eval 模式评估。这是训练集监测子集，不是完整训练集指标。

训练在线平均 loss 来自同一轮中不断变化的模型，与轮末固定模型的验证 loss 口径不同。报告另列训练监测子集与验证集的轮末指标，辅助观察过拟合；单次划分、单随机种子不能证明泛化稳定。

本配置只有三轮上限，patience=2最早在第三轮末达到阈值，因此不会比预设三轮更早结束。本轮主要通过验证检查点择优控制最终模型选择；没有声称早停减少了训练预算。

## 保存、恢复和推理

每 500 步及每轮结束保存模型、AdamW 状态、MLX 随机数状态、已执行步数、下一批位置、累计在线 loss、历史指标和最佳轮次。每轮使用 `seed + epoch` 的确定性排列，恢复时从保存的批位置继续。学习率由总步数和当前步数重建。

恢复检查配置、数据清单、模型锁文件、相关源码、运行库版本及检查点文件 SHA-256。一代完整写入后才切换 `latest.json`，再删除上一代；中断时从最近已提交的一代恢复，最多重算未保存的步骤。这里的原子发布保证进程中断情况下指针不会指向写了一半的文件；未声称硬盘断电情况下的完整事务保证。

`validate_resume.py` 用含 dropout 的合成网络检查恢复后的下一步 loss、参数和优化器状态与连续执行完全一致，并检查配置错配被拒绝。它验证恢复机制，不等同于已经在完整 ModernBERT 训练中人为中断并重启。

各个被选中的 epoch 权重另行保存。训练完成后重载最佳权重，重新评估完整验证集核对 Macro-F1；推理读取 `best.json` 并验证权重哈希。正式测试留到方案冻结之后。

## Rust 分词流水线

Rust CLI 调用固定 `tokenizers = 0.22.2` 与原配 tokenizer JSON，处理流式 JSONL。Python包装器以每批256条运行CLI，并设置2个Rayon工作线程。只读取 `text` 作为分词输入，`id` 和 `label` 仅作对齐与输出元数据。本机编译与静态检查使用Rust 1.95.0。

输出 token ID、attention mask 和 Unicode 码点位置；采用 `encode_batch_char_offsets`，避免 Rust 字节位置与 Python 字符位置混用。特殊 token 和 padding 的位置保留库的原始约定。此处不重新训练词表、不增删 token、不改变 embedding。

Python 包装器只接受冻结的 train/validation 和长度 256/512。缓存键包括数据、tokenizer、二进制、源码哈希、版本和分词配置；先生成临时缓存，逐条与 Python bindings 比较 ID、mask、offset、label、id，再发布带哈希的 JSONL、NumPy 数组和元数据。再次读取会检查哈希。

Day3 正在训练的基线继续使用启动时的 Python bindings，避免中途变更输入路径。Rust 缓存可供后续阶段接入；本阶段的全量一致性核对为接入提供证据。

Python bindings 本身也调用 Hugging Face 的 Rust 库。两端一致验证的是工程集成，而非两套独立算法相互证明；CLI 含进程启动和 JSONL I/O，不能仅凭语言名称宣称更快。本阶段记录缓存生成耗时，不作为端到端预测加速结论。

## 验收与后续

交付完整训练结果、逐轮验证预测、最佳权重、可恢复状态、手动推理入口、Rust 工程及 Cargo.lock、全量分词核对记录和阶段报告。后续先分析截断与错误分布，再决定长度对照和领域 tokenizer 适配；不由单个高置信度错误直接修改正式测试或追加无边界搜索。

参考：[MLX AdamW 文档](https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdamW.html)、[MLX 精度说明](https://ml-explore.github.io/mlx/build/html/usage/precision.html)、[Tokenizers 0.22.2 字符位置接口](https://docs.rs/tokenizers/0.22.2/tokenizers/tokenizer/struct.TokenizerImpl.html#method.encode_batch_char_offsets)。
