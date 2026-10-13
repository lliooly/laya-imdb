# R-Drop 4090 准入与启动记录

2026-10-11。目标：在已完成的 warmup 5% + 标签平滑 0.05 基线上，仅增加 R-Drop alpha=1。两轮、FP32、正式 micro-batch 8、grad_accum 1、同一冻结划分与种子 42，从原始预训练权重开始，不续训组合终点。当前 CUDA 参照 Accuracy 96.44%；本轮尚无正式 Accuracy，历史整体最佳仍为 96.96%。

本地与远端 39 项 CPU 测试均通过，包括两支梯度、padding、零 alpha 的随机数与参数逐值一致、真实上游循环更新数、异常恢复、非有限保护和不支持 RLCD 的准入拒绝。两份新配置本地校验通过。

## CUDA 冒烟

- 正式 micro-batch 8、grad_accum 1；128 条源训练，16 条校准，32 条验证，两轮。
- 28 次 optimizer 更新，warmup 2 次；28 个成对批次、56 次前向，调度与梯度窗口计数匹配。
- 平均对称 KL 为 0.00028463，说明两支存在随机差异；不是泛化收益证据。
- peak allocated 7.77 GiB，reserved 8.21 GiB；含审计耗时 41.32 秒。
- loss、梯度、参数均通过非有限检查。保存 FP16 表示与重载权重完全一致，32 条预测一致，FP32 与保存 FP16 logits 最大差 0.00526。
- 冒烟输出见 `reports/laya-rtx4090-smoke-ce-warmup005-ls005-rdrop1-fp32-v1/`。

## 正式任务

2026-10-11 15:30:45（北京时间）在租用平台启动，runner PID 1739，run_id 为 `laya-rtx4090-ce-epochs2-warmup005-ls005-rdrop1-l512-s42-fp32-v1`。15:33 已确认第一轮执行到 64 步，GPU 利用率 100%，无报错。总 optimizer 更新 5,526，两次前向共享相同编码输入和选项顺序。后台串行执行训练和完整保存权重开发验证；训练后要求 5,526 个成对批次、11,052 次前向，失败停止后续阶段。

预计 85–90 分钟、约 1.8 元，训练硬超时 120 分钟，验证硬超时 5 分钟，启动前有余额时间门槛。启动前余额估算 4.44 元，按用户 13:26 左右约 7 元及 1.24 元/小时计算，包含空闲。任务不会自动追加其他 alpha 搜索，不提供自动关机。

平台实时状态为 `reports/rtx4090-rdrop-preparation-v1/batch_status.json`；本地 `startup_batch_status.json` 是启动快照，不能当作实时状态。测试日志、冒烟审计、来源 manifest、预算快照均已取回。启动证据归档 SHA-256：`bee703f90ed37c03789715cceae7683aa925e449d2e42b9a62e0ce79b630d940`。

代码与配置同步归档 SHA-256：`e1caa279157169191dce0e6877d862ae0aa7052a6cc05ce14c439fff2b976279`；平台替换源码前已备份到 `artifacts/source-backups/ce-rdrop-20261011/` 并记录原哈希。`transfer_manifest.json` 列出逐文件校验和。历史 run 的 manifest 和证据没有改写；Mac MLX 新权重核对尚未进行，官方测试集未访问。
