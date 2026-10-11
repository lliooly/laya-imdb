# Repository instructions

- 不使用视觉伴侣。不创建 worktree，除非用户明确要求；如需 dev 分支，先提醒用户。
- 所有命令在本仓库根目录运行；使用 uv 和锁定依赖，不提交本地环境、权重、原始数据或缓存。
- 当前主线为 Laya-CE：PyTorch 训练、Mac MLX FP16 推理。RTX 6000D 的 FP32 小规模训练/重载/Mac 核对已通过；完整参照成绩另行报告。AMP FP16 梯度溢出未通过准入；CUDA 入口必须保留非有限值保护，不将规划写成已有结果。
- 已确认路线：先优化 Laya-CE，再将有效方法迁移到 ModernBERT-base 和 Laya-RLCD，重新训练并做参照/改进对照。以开发验证 Accuracy 选择最终方案，保留其他实验用于论文比较；98% 为研究目标，不作为迁移硬性门槛，不根据官方测试结果调参或换模型。
- 保留固定数据划分和测试集封存规则。修改实验配置使用新 run_id，不覆盖历史结果。
- reports 中历史 manifest、哈希、诊断和 reproduction 快照用于审计，不直接改写；源码变化后按相应训练协议重新验证。
- 本仓库根 LICENSE 为 MIT；保留第三方来源声明及 docs/THIRD_PARTY_APACHE_LICENSE.txt。
