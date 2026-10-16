# Modal 免费额度训练与本地权重交付

**2026-10-16 收尾：六份最终模型、冻结与开发评估包均已本地严格验证，配对分析完成。** CE / mixed 三种子均值为 96.32% / 96.40%，差 +0.08 pp，条件样本区间 [−0.174, +0.347] pp，未达到既定继续门槛；详见 [完整分析](../../reports/laya-robust-loss-analysis-20261016-v1/REPORT.md)。补充五项评估独立预留最多 1 美元，累计保守预算上界 25 美元，仍低于用户原确认的 30 美元；所有本次 App 已停止、任务数为 0。第一轮模型及 AdamW / RNG 恢复包继续本机后台接收，不能称全部逐轮备份完成。新入口只完成已冻结模型评估，不进行重训；[收尾执行记录](../../reports/laya-robust-loss-modal-evaluation-20261016-v1/REPORT.md)。下文为各阶段历史记录，不能代替最新状态。

**2026-10-16 下午核查：六个正式模型已训练并冻结，开发评估完成 1 / 6，触及预算窗口后停止。** App 于北京时间 14:19 停止、任务数为 0，后续未启动 GPU。六份最终权重和冻结包的云端收据已核验；新增最终模型正在回传，本地交付完成数以 [恢复报告](../../reports/laya-robust-loss-modal-recovery-20261016-v1/REPORT.md) 为准。CE seed42 开发准确率为 96.24%，不能据此判断损失干预收益。剩余五项仅需评估现有冻结模型，不需要重训；补齐后才能进行三种子配对分析。云端状态的保守累计费用上界约 23.79 美元，账本为清理开销保留完整 24 美元预留，均不是实际账单。

**2026-10-15 23:42 后的核查：后台任务中断，尚未全部完成。** 平台抢占后，原编排被一次性防重复标记拦截；正式冻结 1 / 6，mixed seed42 可从 epoch-001 恢复，开发评估 0 / 6。旧入口不能直接重新提交整批任务。CE seed42 的完整最终权重已取回并通过本机独立严格加载，其余包未声明本地交付；详见 [中断核查与接收报告](../../reports/laya-modal-interruption-audit-20261015-v1/REPORT.md)。下文的提交命令和预计时间是原启动记录，不能视为当前成功状态或恢复入口。

2026-10-15：用户授权使用已登录 Modal 账户，确认剩余赠送额度为 30 美元。沿用已批准的 [损失对照与交付设计](../superpowers/specs/2026-10-15-laya-robust-loss-and-checkpoint-delivery-design.md)，仅更换计算及交付平台。旧 SSH 协议和历史源码绑定保持原样。

**最新用户指示优先：云端先完成训练，用户唤醒后再回传分析。** 当前后台协议为 `laya-robust-loss-modal-deferred-20261015-v2`，使用独立持久化 Volume、逐 epoch 云端 SHA / 独立加载和明确标为未本地交付的云端收据。以 `uv run --locked --project environments/laya-modal modal run --detach scripts/modal_laya_deferred.py` 提交。无需本机或助手持续在线；不再使用下文同步交付入口启动本轮。详见 [后台报告](../../reports/laya-robust-loss-modal-deferred-20261015-v2/REPORT.md)。下文保留同步 v3 的机制与历史预算解释，不能反向覆盖旧证据。

## 执行方案

- 当前协议：`laya-robust-loss-modal-20261015-v3`；所有新模型 ID 均包含 `modal`。v1 的轮询超时类型错误、v2 的控制器日志目录冲突均在正式训练前停止，记录和输入快照保留。
- L40S 单卡串行，4 CPU、32 GiB 内存，自动重试为零、闲置窗口 2 秒。
- 使用原 CUDA 环境的 `uv.lock` 构建 Python 3.12 镜像，固定 PyTorch 2.14.1+cu126、Laya 0.4.1；FP32、AMP / TF32 关闭，训练子进程禁止联网。
- 持久化 v1 文件系统 Volume：`end-imdb-laya-robust-20261015-v3`（名称中的 v3 是实验版本），代码及产物目录为 `/End-IMDB`。只上传源文件清单、原训练 / 开发划分和发布版模型资产；不上传官方测试或本机凭据。
- 训练臂、种子、损失、输入、更新次数、校准与统一开发门禁完全沿用原设计。

直接把整轮输出留在临时容器不满足交付要求；仅存入 Volume 也不算本地交付。本方案同时保留持久化 Volume 和本地实文件，用已有严格校验器发放继续训练的收据。

## 权重回传门禁

每个 epoch 写完整 FP32 权重与审计，epoch-001 另存优化器 / 调度器 / 随机状态。父控制进程显式 `Volume.commit()` 后通知本地接收器。本地逐文件核验大小与 SHA、写入 `.partial` 后原子改名，再用本仓库 `.venv/bin/python` 在独立 CPU 进程严格加载模型和恢复文件。

本地文件和验证收据持久化完成后，才能通过 Modal Dict 传回 ACK。云端控制器把 ACK 落到 Volume，训练子进程才可以继续。Dict 只承载控制信息，不承载权重，也不能代替本地收据。训练日志打开期间不调用 `Volume.reload()`；只在新容器启动、尚未打开日志时刷新。

每项传输最多尝试 3 次；错误前缀会重新下载，不能仅凭 `.partial` 长度恢复。传输失败、加载失败、额度不足或超时均停止后续训练；已提交的 Volume 权重不自动删除。每个阶段的本地路径为 `checkpoints/<run_id>/epoch-001`、`epoch-002`、`frozen`、`evaluated`，收据保存在 `delivery/receipts/`。

## 预算边界

[Modal 定价](https://modal.com/pricing) 中 Starter 的每月 30 美元是赠送额度，超出可能按量计费；不能称作无限免费 GPU。当前 CLI 1.2.6 不能读取 Starter 剩余额度；30 美元来自用户本次确认，不是程序查询结果，也不证明账户层已经设置硬预算。

本次程序将计算预算限制为 24 美元，预占 1 美元镜像准备费用，GPU 计算按高于公开 L40S + CPU + 内存合计的 3 美元 / 小时保守记账，另外保留 6 美元余量。提交任务前先预留其最大运行时间对应的费用；成功后按从提交到返回的墙钟时间加 30 秒记账，异常则保留全部预留，不盲目返还。预算账本在 `delivery/modal-budget.json`，跨 v1 / v2 / v3 共用总预算，前两次停止任务仍各占用 2.75 美元预留；存在未决任务时禁止重复启动。

云端每个任务和本地控制器均有运行时限；准入 / 单模型 / 评估分别有最长时限。程序不调用其他 GPU 类型、不并行扩大 GPU 数量、不自动追加种子。该记账是保守估计，最终账单以 Modal 为准；若账户还有其他任务或调整费率，需重新核实可用赠送额度。程序不修改整个账户的既有预算。

本地继续要求至少 100 GiB 空闲。Volume 的 `statfs` 容量是占位值，不能冒充已验证 100 GiB 云端空闲；本轮改以持久化 Volume 保存产物，临时容器不承担完整交付存储。Volume 有独立存储与流量计价，不能将持久化自动等同于永久免费；本轮在公开免费 allowance 内预估，并保留预算余量。

## 命令

从仓库根目录，使用隔离的 Python 3.12 / Modal 1.2.6 控制环境，复用用户原有登录。全局 CLI 的 Python 3.9 与云端序列化函数不兼容，不修改其安装：

```bash
uv run --locked --project environments/laya-modal modal run scripts/modal_laya_train.py --phase runtime
uv run --locked --offline --no-sync python scripts/laya_modal_runtime.py --prepare
uv run --locked --project environments/laya-modal modal run scripts/modal_laya_train.py --phase all
```

`runtime` 仅构建 / 检查 CPU 镜像，不做 GPU 准入。`all` 依次执行四次短跑准入、六次正式训练及共同冻结后的开发评估。每个短跑都交付两个 epoch 权重及最终冻结证据，第二次重复还测试真实 CUDA 从 epoch-001 恢复。准入失败不会启动正式模型。

可用 `--phase admission` / `formal` / `evaluate` 分段执行。已经本地交付的最终模型会重新检查收据后跳过；中途失败的任务不会自动重训，应先核对已保留的 Volume 文件、账本和本地收据，再以新版本适配显式恢复。不能删除预算账本绕过失败任务预留。

## 完成状态

分别报告镜像核验、CUDA 准入、正式训练、权重交付及开发分析。CPU 文件传输测试不能写作真实 Modal 回传准入；容器完成不能写作本地权重交付完成。实际进展见同协议报告目录及 Modal App 链接。
