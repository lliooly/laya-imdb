# 512 / 2048 上下文正式训练对照

状态：完整模型 CUDA 精确恢复及独立 CPU 重载准入已通过，512 第一轮正式训练于 2026-10-16 21:27:51（北京时间）启动；尚无正式准确率结果。

提交时间：2026-10-16T21:25:58.360487+08:00；绝对停止期限：2026-10-17T01:10:58.360487+08:00。

[Modal 作业](https://modal.com/apps/linzhouping88/main/ap-vG0zfo4UVRvJgnHuPkorPt)，协调调用 `fc-01M52E834JCWWYSK1YEDZ9ZCQ6`。

- 单种子 seed 42，长度 512 / 2048，各 2 epochs；固定 22100 梯度训练、400 校准、2500 开发划分。
- 两边从冻结发布版 Laya 相同 CE 权重开始；ModernBERT-large 编码器，原决策头；FP32、AMP / TF32 关闭，首尾输入规则训练和推理一致。
- 第 2 epoch 固定为最终模型；两模型冻结后统一校准与评估，官方测试集不读取。
- 已通过 5 项本地检查：原 512 CE 循环精确等价、双长度断点恢复、样本 / 选项顺序配对、绝对期限停止、封存篡改检测。完整 2048 模型的 CUDA 精确恢复准入已通过；独立 CPU logits 最大差异 7.3909759521484375e-06。
- 每轮权重和完整优化器 / 调度器 / RNG 存入持久化 Volume，经独立进程严格重载；本机后台接收并核对整文件 SHA 与 CPU logits。云端收据与本机收据分开保存。
- 用户确认余额 15 美元，保守扣除先前短跑预留 1 美元；本轮新增预留最多 12 美元，剩余安全余量 2 美元。旧累计保守账本不重置，预留不代表账单实际支出。
- 单 L40S 串行、无自动重试；从提交起绝对 13500 秒硬停止。暂估 2～3.5 小时，实际取决于分配、真实批次速度和校验时间。

新协议与代码哈希：`configs/laya-context-pair-s42-modal-20261016-v1.json`。设计文档：`docs/superpowers/specs/2026-10-16-laya-context-pair-design.md`。

## 结果判读

最终同时报告整体和共同截断组（原 review tokens >466）的 Accuracy / Wilson 区间、Macro-F1、AUROC、原始 / 校准 NLL / ECE / Brier、配对修复退化、精确 McNemar 与配对 bootstrap 区间。单种子不足以说明训练方差；开发集反复使用带有选择乐观性，不做未见测试成绩承诺。

正式评估完成后，带指标的报告位于 `evaluation/REPORT.md`，机器结果位于 `evaluation/summary.json`。本机交付凭据为 `local-delivery.json`；只有该凭据明确写入 `verified_local` 才算权重已回传验证。
