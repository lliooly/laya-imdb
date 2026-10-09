# Repository instructions

- 不使用视觉伴侣。不创建 worktree，除非用户明确要求；如需 dev 分支，先提醒用户。
- 所有命令在本仓库根目录运行；使用 uv 和锁定依赖，不提交本地环境、权重、原始数据或缓存。
- 当前主线为 Laya-CE：PyTorch 训练、Mac MLX FP16 推理。CUDA 适配未验证，不将规划写成已有结果。
- 保留固定数据划分和测试集封存规则。修改实验配置使用新 run_id，不覆盖历史结果。
- reports 中历史 manifest、哈希、诊断和 reproduction 快照用于审计，不直接改写；源码变化后按相应训练协议重新验证。
- 本仓库根 LICENSE 为 MIT；保留第三方来源声明及 docs/THIRD_PARTY_APACHE_LICENSE.txt。
