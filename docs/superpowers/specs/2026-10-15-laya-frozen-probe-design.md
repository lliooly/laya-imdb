# L1 冻结表示线性探针协议

日期：2026-10-15。依据用户已批准的阶段二诊断 §5 和本轮“冻结探针”请求执行。无 worktree，不切分支，不提交 Git，不使用视觉伴侣。

## 1. 问题与方案

问题是固定微调权重的编码器输出能否被线性读出，不是训练新编码器，也不是对头部结构作因果消融。已有候选包括直接使用 scorer 输入、提取原始编码器输出、重训不同头部。scorer 输入位于 head 之后，只适合作为同协议诊断参照；重训头部属于 L2。本次主方案选原始编码器输出，并在同一次 forward 中取得 head 后表示作为诊断，避免额外编码器推理。

仅使用已选定的 plain seed43 权重 `2f8c6635beb15204f247ed3370088054c6d91e4cd6cdcae0ff73e6e5d964f338`，与既有 `baseline-s43/head_tail` 的 96.84% 开发集结果比较。不按探针的开发成绩选择其他 checkpoint。

## 2. 固定输入与提取位置

- 输入沿用原问题和 negative/positive 顺序，512 总 token、192 head 上限、466 评论 token。超过预算直接选原始 token 的首 233 和尾 233，不解码重编码。
- 提取主特征：`model.encoder(...).last_hidden_state`，在加入 `type_emb` 和进入两层 Transformer head **之前**；按实际 `marker_pos` 取 positive `[MASK]` 表示减 negative `[MASK]` 表示。
- 诊断特征：`model.scorer` 输入中的两个候选表示之差，已经过类型嵌入与 Transformer head。它不被称为原始编码器表示。
- 两者均为一条样本一个 hidden_size 维向量，不使用全文池化、层组合、额外视图或非线性读出。槽位顺序固定，候选差不证明模型已对选项反序不敏感。
- 使用 FP32、eval、inference_mode；全部原模型参数 `requires_grad=False`，没有原模型 backward 或 optimizer。MPS 为本次设备，不隐式回退 CPU；线性优化单独使用 CPU float64。

## 3. 分区、拟合和冻结顺序

沿用 baseline-s43 的原 manifest：22100 梯度训练样本、400 校准样本、2500 开发样本，ID 唯一、互斥。原始文件 SHA、问题和 tokenizer 语义保持一致，测试集正文与预测不读取。

顺序为：核验来源 → 在全部 400 校准样本上提取并验证现有全模型 logits → 提取 22100 梯度训练样本 → 拟合两组探针并校准选择 → 保存冻结参数与哈希 → 才读取开发集正文、提取并评估一次。校准特征先提取不意味着用它拟合标准化参数。

每组探针仅在 22100 条上估计逐维均值、总体标准差；小于 1e-6 的标准差固定为 1。以标准化差向量拟合带截距的二分类 logistic regression：

```text
z = ((h_positive − h_negative − train_mean) / train_scale) @ w + b
objective = mean(logaddexp(0, z) − y*z) + lambda/2 * sum(w*w)
```

截距不正则化；零初始化；CPU float64 全量 L-BFGS、strong-Wolfe、最多 400 次迭代，记录迭代数、最终目标和最大绝对梯度，超过 1e-5 不作成功拟合。固定 lambda 网格为 `[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]`，两种提取位置使用相同协议，各自选择。

校准选择依次为 Accuracy 最高、NLL 最低、lambda 较大；NLL 在 1e-12 内视为并列。最终概率直接为 sigmoid(z)，不再拟合温度或开发集阈值，`z > 0` 预测 positive，否则 negative。不将校准集并回梯度训练分区重拟合。

## 4. 准入与防错

- 开始和结束核验所有 checkpoint 资产；原 tokenizer 可能持久化 padding/truncation，但语义不能变化。加载前拒绝会触发上游 tokenizer 自动修写的配置。
- 校准集每条比较原缓存的输入 SHA、标签、截断标记和 logits。MPS 对 CUDA FP32 的预设绝对容差为 logits 1e-3、概率 1e-4，所有预测类别一致。失败即停止，不放宽容差，也不读取开发集正文。
- 所有参数、隐藏状态、特征、logits、概率、拟合目标与梯度必须有限。marker 必须恰为两个真实 `[MASK]` token，编码器 hook 与 scorer hook 每批各触发一次。
- 使用真实微型模型验证提取点和标签不进入输入；验证线性优化、分区隔离和拒绝覆盖。原全模型对照与两组探针使用同一批新提取输入，另报告与历史缓存的一致性。
- 新 run_id 与新目录，拒绝覆盖历史输出。特征缓存只保存差向量、ID、标签、截断信息、输入哈希和 logits，不保存正文或整个 token 隐状态。

## 5. 结果与解释

报告全体、截断与未截断的计数、Accuracy、NLL、Wilson 95% 区间；固定参数下准确率差使用按类别分层的配对 bootstrap 10000 轮、seed 20261015，并给出名义精确 McNemar。两种方案均覆盖同一批全部样本，此处不使用风险—覆盖分析中的独立保留集合。

主比较是编码器探针与同批全模型，head 后探针为诊断参照。区间不覆盖训练 seed、校准选参、标签偏差或开发集历史复用，分层区间不作同时保证。不以某个开发集分数自动触发 L2，不把较差的线性探针解释成“加结构必然有效”，也不把接近解释成“所有结构都无效”。

报告实际特征 forward 数、耗时、缓存大小、设备、拟合用时与收敛状态；25000 条样本是计划量，不预写成完成结果。冻结微调编码器已与原头共同适配，L1 不是同初始化重训头部的因果替代。

方法参考：[Alain 与 Bengio，Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644)。
