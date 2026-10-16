# Laya 鲁棒损失对照与权重交付

本入口执行 [已批准设计](../superpowers/specs/2026-10-15-laya-robust-loss-and-checkpoint-delivery-design.md)。当前仅完成本地验证；没有连接旧服务器或获得新的 GPU 资源。所有命令从仓库根目录执行。

## 1. 实现与固定条件

`scripts/laya_robust_training.py` 提供 CE 与 `0.5 CE + 0.5 GCE(q=0.7)`，不修改数据、标签、选项规则或模型结构。`run_laya_robust_loss.py` 是单模型训练 / 恢复 / 独立重载 / 开发入口；`run_laya_robust_study.py` 固定三种子六模型及共同冻结门槛。

新循环只支持已批准的 2 epochs、FP32、无 AMP、全模型训练、零标签平滑和累积 1；不为其他配方提供隐含兼容。CE 路径与旧循环通过小模型逐更新回归；CUDA 是否逐位一致必须在新主机准入时验证。

协议准备命令只绑定代码与数据，不训练：

```bash
uv run --locked --offline --no-sync python scripts/prepare_laya_robust_study.py
```

准备后的协议若源码变化会拒绝覆盖。需要修改时使用新版本、study_id / run_id；不能在已有正式结果下刷新 SHA。旧 L2 的源码、配置和结果不变。

## 2. 本地接收端

下一台服务器可用后，先配置可用的 OpenSSH 认证和已核验 host key。接收器要求能执行远端 Python 标准库程序的 SSH 连接；交互式实例菜单不算已通过兼容性检查。凭据不写进配置、仓库或日志。

本地接收器命令使用 `scripts/laya_checkpoint_delivery.py`，必须指定：

| 参数 | 含义 |
| --- | --- |
| `--local-root` | 本地仓库绝对路径 |
| `--host` / `--port` | 新服务器主机或 OpenSSH 配置别名及端口 |
| `--remote-root` | 远端仓库绝对路径 |
| `--identity-file` | 可选，已配置的 SSH 私钥路径 |
| `--watch-seconds` | 接收器持续运行的时长；实际执行前按带宽及租用预算设置 |

默认要求接收前本地至少 100 GiB 空闲。接收器可先于训练启动；远端尚未生成请求时等待，认证和网络错误不会伪装成“没有请求”。必须先通过实际 SSH / 大文件传输，再启动昂贵训练。

`--source-root` 仅用于本地文件测试，其收据标记为 `local-test`，正式训练入口会拒绝。不能用服务器上的第二个目录假装已经回传本地。

## 3. CUDA 侧执行阶段

在新服务器按 `environments/laya-cu126/uv.lock` 安装锁定环境。以下命令要求该环境已就绪，不自动安装或升级依赖：

```bash
uv run --project environments/laya-cu126 --locked --offline --no-sync python scripts/run_laya_robust_study.py --phase admission
uv run --project environments/laya-cu126 --locked --offline --no-sync python scripts/run_laya_robust_study.py --phase formal
uv run --project environments/laya-cu126 --locked --offline --no-sync python scripts/run_laya_robust_study.py --phase evaluate
```

三个阶段分开执行，每个完成后核验状态和本地收据再进入下一阶段。正式入口自身也检查准入，绕开 study 编排直接调用单模型入口不能跳过它。

准入每个损失臂重复两次 32-update 真实训练。第二次在第一 epoch 已交付后主动退出，再以独立进程恢复，比较连续训练与恢复训练的逐更新记录和最终权重。四个准入模型也全部逐 epoch 回传，**其权重与优化器额外占约 25.1 GiB**，不包含在正式六模型约 37.6 GiB 中。两端实际存储和传输预算按总计约 62.7 GiB 再加公共资产、审计及空间余量计算。

没有当前协议下的 CUDA 重现性、真实 SSH 交付和磁盘准入证明，正式阶段不会开始。六模型全部冻结并获得本地最终收据后，才允许读取开发集。开发预测和可复算指标另行封装为 `evaluated` 包回传；不重新读取官方测试。

## 4. 交付与故障恢复

每个 epoch 先保存完整权重。第一 epoch 还保存优化器、调度器、随机状态、选项顺序 RNG 和训练游标。封装时复制输入 / 更新日志和源码快照，确保后续日志追加不改变已封存清单。

本地逐文件续传到 `.partial`，续传前核验已有前缀；完整 SHA 通过才原子改名。随后在独立 CPU 进程 strict load，核验有限 FP32 状态、恢复状态形状及训练身份；落盘收据后才回传 ACK。重新计算的温度 / 指标允许至多 `1e-12` 的跨平台浮点误差；模型状态 SHA、保存的校准参考 logits 和独立 CUDA 重载 logits 仍要求完全相等。

交付失败最多尝试 3 次。任一断线、缺失、校验失败、加载失败或接收端超时都停止后续更新。状态为 `blocked_delivery` 时，先恢复接收器，核验本地收据并重发 ACK，再显式恢复同一 run。ACK 丢失不要求重训。

单模型恢复使用 `run_laya_robust_loss.py --mode train --arm ce或mixed --seed 对应种子 --resume`；准入 run 额外保留原 `--repeat`。也可以对 study 指定 `--resume-run`。身份、数据、协议、模型或恢复前缀不匹配时拒绝恢复，不覆盖旧日志。未完整封存的中途状态不得冒充有效检查点；第一 epoch 之前或中途崩溃仍可能丢失这一段进度。

第二 epoch 已保存时恢复只补交付和校准，不重复训练。已封存的最终冻结包只重发收据。若进程在文件封装完成前崩溃，需保留原目录核验，不能通过删除清单或修改 SHA 强行继续。

系统不自动删除远端文件或释放实例。分别记录训练、交付和分析状态；结果报告完成不代表权重已交付。全部模型、校准和开发预测实际在本地后，再进行配对统计分析；尚无新训练结果时不报告任何准确率收益。
