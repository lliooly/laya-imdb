# 2026-10-19 补实验远端执行记录

## 授权与实例

用户已授权补实验，明确同意将约 1.8GB IMDB 训练/开发数据、预训练模型与实验源码上传至本次算家云实例。总预算上限 300 元，实例单价 1.24 元/小时。未上传官方测试数据、论文正文、Git 目录或凭据。密码只在 SSH 的交互认证中使用，不写入项目文件。

- SSH 网关：`xn-b.suanjiayun.com:2020`；进入菜单中的 `PxqE7xkQ`。
- GPU：RTX 4090 24GB；驱动 550.107.02；Ubuntu 24.04.3。
- 远端项目：`/root/sj-tmp/imdb-paper-20261019`；数据盘约 50GB。
- SFTP 的实例根映射：`PxqE7xkQ-默认项目-默认的单位/`。
- 本机 SSH 控制套接字：`/private/tmp/imdb-paper-ssh-20261019/control`。网关交互 shell 与 SFTP 可用；普通 SSH exec 不直接进入实例。部分 SFTP 客户端完成传输后未自动退出，需先核验文件再关闭挂起客户端。

## 环境与传输

`artifacts/paper-gpu-supplement-20261019-v1/upload-manifest.json` 记录 186 个允许上传的文件。远端 `upload-verification.json` 已逐项核验文件大小与 SHA-256。

保留 `environments/laya-cu126/uv.lock` 和 `environments/deberta-cu126/uv.lock`。原 PyPI 文件通道下载缓慢，改用锁文件导出的带哈希 requirements，通过清华镜像下载；PyTorch 仍使用锁中的官方 wheel URL。`uv pip sync --require-hashes` 验证原哈希，环境准入脚本还需逐项比较安装版本并执行有限 BF16 CUDA 前向/反向。Python 统一使用独立安装的 3.12.15，以提供首次 Triton 编译所需头文件。

大文件最终采用分块下载，16 个 GPU wheel 均对原锁 SHA-256 复核。派生 requirements 将本地 wheel 绑定到对应单一原始哈希。两环境分别有 45、46 个锁定依赖逐项一致，BF16 CUDA 有限前向与反向通过；证据已回传至 `reports/paper-gpu-supplement-20261019-v1/environment-admission.json`。安装后的可重建下载缓存已清理，训练环境及源模型保留。

## 固定执行入口

- `scripts/paper_gpu_supplement.py`：显式 seed 的 DeBERTa 初始化、训练与独立进程重载；Laya 复用原上游训练循环和原优化适配器，保留 FP32 训练、FP16 保存、FP32 重载的历史流程。
- `scripts/paper_gpu_environment.py`：锁定版本与 CUDA 准入。
- `scripts/paper_gpu_orchestrate.py`：先进行种子/短程/严格重载准入与费用估算，再串行训练九个固定模型；失败停止，不自动改批量或重试选种子。
- `scripts/paper_gpu_orchestrate_v2.py`：首次正式训练前的运行时限修订，最大 72 小时。用户已明确确认；科学入口和种子不变。72 小时计算费上限 89.28 元，原 300 元总预算不变。旧 48 小时排程因冷启动四步外推约 50.25 小时而停止，旧状态与源码保留。
- `scripts/paper_gpu_evaluate.py`：校准、九模型共同冻结、开发评估与汇总；另提供相同实例上实际三模型前向测速。
- `artifacts/paper-gpu-supplement-20261019-v1/runner-manifest.json`：运行入口的 SHA-256；正式启动后不改写绑定代码。

三组 baseline/combined/DeBERTa 种子分别为 `(43,42,42)`、`(143,142,142)`、`(243,242,242)`。各训练两轮，固定第二轮。梯度 22100 条、校准 400 条，全部模型及校准参数冻结后才评估同一 2500 条开发记录。不会选择最好种子，不调用官方测试。

当前远端正式排程状态在 `reports/paper-gpu-supplement-20261019-v1/schedule-status-v2.json`；各阶段日志在同目录，各模型证据在 `reports/paper-gpu-supplement-20261019-v1-<kind>-s<seed>/`。正式 DeBERTa seed42 已开始，前三次更新的实际 ID、损失、梯度范数、学习率和样本数与准入相同。不要将短程准入写成已完成九次训练。

## 磁盘与回传

逐个模型运行。新任务前检查磁盘，DeBERTa 至少留 12GiB、Laya 至少留 5GiB。先回传最终模型、预测、清单和日志；逐文件核对大小与 SHA-256 后才可清理远端已归档文件。已有本地历史资产保持原样；不凭一次传输命令的退出状态声称回传成功。

三项短程训练和独立进程重载均通过。准入完整检查点和报告回传到新建隔离目录 `artifacts/paper-gpu-supplement-20261019-v1/received-admission-v1/`，全部文件通过本地 SHA-256 检查后才可清理对应远端副本。

用户已明确同意将三个 DeBERTa 第一轮完整恢复包（合计约 15.7GB）迁移到空闲系统盘。远端 `archive_epoch1.py` 已启动，归档根目录为 `/root/imdb-paper-epoch1-archive-20261019`。它等待每个第一轮恢复包封存，逐文件核对源文件与归档文件的大小、SHA-256，并核对清单哈希；归档完成且生成回执后，才删除数据盘上对应的重复副本。第二轮最终模型不动。状态在 `reports/paper-gpu-supplement-20261019-v1/epoch1-storage-status.json`；启动核验时为 `waiting_for_sealed_epoch1`，尚无已迁移包。系统盘归档仍在租用实例上，不能视为已回传本机。

`scripts/paper_gpu_receive.py` 已作为当前排程的一次性接收进程启动，最长等待 72 小时，目标为全新目录 `artifacts/paper-gpu-supplement-20261019-v1/received-formal-v1/`。它核验九份最终权重与报告，不删除远端文件。状态在该目录的 `receiver-status.json`。连接元数据在被 Git 忽略的 `remote-access.json`，不含密码。若本机连接中断，远端 nohup 排程仍可继续；接收依赖恢复可用的认证连接。

远端 `benchmark_after_schedule.py` 在本轮九次训练/开发评估完成后只执行一次真实 GPU 测速，受同一 72 小时时限限制。仅将原锁环境的 Laya 模块复制到独立模块目录，以便在 DeBERTa 环境运行三模型；不修改任何训练环境。`benchmark-environment.json` 记录复制文件 SHA，测速使用第一组新训练权重。`postprocessing-status.json` 记录完成或失败。模型加载统计包含完整性核验，实际前向吞吐不包含加载。

正式训练与结果仍在运行，完成状态以远端报告和本地接收回执为准。独立新评估数据仍未补齐，开发集多种子结果只支持探索性稳定性分析。脚本完成不等于平台停止计费，需在结果和回传核验完成后到平台停止实例。

## 2026-10-19 22:23 状态核验

远端排程持续运行约 6.8 小时。已完成 DeBERTa 42/142、baseline 43/143、combined 42 的训练、独立重载和校准；第六次 combined 142 正在第二轮。实测 DeBERTa 每次约 2.01 小时，Laya 每次约 41.1 分钟，早期短程冷启动外推的 30–50 小时估计偏保守。两份 DeBERTa 第一轮恢复包已逐文件核验归档，系统盘剩余约 21GiB，数据盘约 22GiB。

原本机 SSH 控制连接及接收进程已退出，远端 nohup 排程、归档与测速等待进程均正常。已重新认证连接；接收改为独立进程会话，使用全新隔离目录 `artifacts/paper-gpu-supplement-20261019-v1/received-formal-v2/`，原 v1 目录及日志保留。接收程序还明确等待独立测速完成回执后再下载最终报告，避免训练完成与测速完成之间的时间差造成误判。此次修订仅涉及本机接收，不改动已绑定的科学训练、评估与远端排程源码。

## 2026-10-20 完成后平台关机授权

用户明确表示睡后无法手动关机，授权代为停机。已更新当前对话的 `imdb` 自动检查，取代先前“仅提醒用户停机”的安排。九次训练、评估、测速全部成功，九份最终模型、报告及系统盘三份第一轮完整恢复包均回传本机并逐文件核验后，才通过算家云控制台对本轮实例 `PxqE7xkQ` 执行“立即关机”。不执行释放或重置实例，不以 SSH `shutdown`、`poweroff` 或 GPU 空闲替代平台关机。必须确认页面显示已关机、不再运行或计费，留存证据后才报告停机成功并结束自动检查。

已读取用户登录的 Chrome 控制台 `https://suanjiayun.com/container/#/instanceMine`：唯一实例为本轮单卡 RTX 4090，当前运行中、按量计费，具有“立即关机”入口。检查阶段未点击关机。浏览器标签页 ID `1768368672`、browser `1`；后续每次操作前需读取最新页面。登录失效、验证码或无法核实停机时应报告具体阻碍，不能假称已停费。

00:56 自动检查：第八次 baseline 243 正常进行第二轮，GPU 满载，系统盘和数据盘空闲分别约 51%、37%，无需清理。为缩短训练完成后的等待，三份已经封存的第一轮完整恢复包开始并行回传；一次性独立进程 `artifacts/paper-gpu-supplement-20261019-v1/receive_epoch1_archive.py` 使用现有受验证的 SFTP 接收器，不改训练源码或远端文件。目标为全新隔离目录 `artifacts/paper-gpu-supplement-20261019-v1/received-epoch1-v1/`，状态在该目录 `receiver-status.json`。只有三份清单、权重、优化器/RNG 状态及迁移回执全部核验后，才生成顶层 `local-delivery.json`；保留远端原件。后续自动检查先读取此状态，不要重复启动回传。

02:28 自动检查：九次训练、重载、校准、统一冻结后的九次开发评估、汇总及真实 GPU 测速全部成功。正式排程耗时 10.4216 小时，对应约 12.9228 元（不含环境准备、测速与传输等待）。三份第一轮恢复包已回传、逐文件核验，合计 15,663,540,269 字节，顶层交付回执在 `received-epoch1-v1/local-delivery.json`。SSH 控制连接曾断开，已按既有授权重新认证并恢复同一接收进程，新的交互会话 ID 51947，加入 keepalive；未重复训练或重复启动接收。

`received-formal-v2` 正在下载最终模型，首份权重已实际增长，尚未达到关机条件。远端新增只读快照清单 `reports/paper-gpu-supplement-20261019-v1/final-report-files.sha256.json`，包含 129 份报告文件，清单 SHA-256 为 `603c9e30b966c977431016e8ba11e08c1e0c58d501407650355dbf9723963d6b`。接收完成后须绑定此清单哈希，对每份本机报告再检查字节数与 SHA-256，生成独立核验回执，然后才能平台关机。当前结果已整理到 `reports/paper-gpu-supplement-20261019-v1/REPORT.md`，原启动记录保留为 `STARTUP-REPORT.md`。三组融合和 DeBERTa 的准确率均值同为 97.08%，差值为 +0.08、+0.08、−0.16 个百分点；不支持融合稳定提高准确率的结论。最终通知须等交付和平台停机都已核实。


02:58 最终交付：九份最终模型和三份完整恢复包在本机再次逐文件重算 SHA-256，75 个模型文件共 25,962,070,425 字节全部一致；129 份报告按已绑定远端清单逐文件核验全部一致。核验回执 pre-shutdown-verification.json 保留关机前状态，不改写。通过已登录平台对 PxqE7xkQ 点击“立即关机”并确认；页面刷新后显示“已关机”“已暂停计费”，运行中数量为 0，实例未释放。截图与页面文本保存在 reports/paper-gpu-supplement-20261019-v1/platform-stopped-20261020.{png,txt}，完成回执为同目录 completion.json。自动检查 imdb 已通过工具暂停（PAUSED），不再继续轮询。所有本机实验资产、远端磁盘原件及本对话保留。
