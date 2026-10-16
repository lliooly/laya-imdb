# Modal 五项固定评估收尾

2026-10-16。用户在得知六模型训练完成、开发评估尚余五项后，要求「继续推进」。本次完成原科学协议中的五项开发评估、结果本地交付与配对分析；未更改训练、模型、校准温度、数据划分或官方测试访问规则。

## 结果

六模型全部评估完成，六份最终模型、冻结及评估包均在本地通过原独立 CPU strict-load 校验。完整分析见 [鲁棒损失三种子报告](../laya-robust-loss-analysis-20261016-v1/REPORT.md)：CE 均值 96.32%，mixed 96.40%，三种子差 +0.16 / 0.00 / +0.08 pp，均值 +0.08 pp，10,000 次分层配对 bootstrap 的条件样本 95% 区间 [−0.174, +0.347] pp。未达到预设 +0.20 pp 均值继续门槛，不追加种子或 q / 比例搜索。

第一轮权重与 AdamW / RNG 恢复包正在本机后台接收，完整报告生成时 1 / 6；实时进度在恢复报告目录的 `local-first-epoch-delivery.json`，不能将最终模型全部到手说成所有逐轮恢复包交付完成。原云端 Volume 保留。

## 执行与防重复

入口为 `scripts/modal_laya_evaluation_completion.py`，纯预算 / 范围检查为 `scripts/laya_evaluation_completion.py`。执行配置 `configs/laya-robust-loss-modal-evaluation-20261016-v1.json` 绑定四项操作源码 / 测试 / 设计文件，科学协议仍是原 deferred-v2，122 项原科学绑定未改，旧恢复层四项绑定未改。

新 GPU 函数在单 L40S 上串行调用原 worker 的本地实现；五项任务固定 `evaluate=True`，实际命令都仅为 `evaluate`，没有重新训练、优化器更新、短跑准入或额外种子。读取六模型共同冻结门禁和各自云端最终包收据后才能运行。既有封包可跳过，中断残留沿用原隔离保留逻辑。

首次提交在 GPU 启动前遇到未 hydration 的旧 Modal Function 对象序列化错误，App `ap-b5FaoZggSdGHv3LrgE8FjN` 于 15:45:57 停止，没有预算预留、没有 GPU 任务。修复方式是在云端函数内部导入旧 worker，再执行本地实现；并未复制或修改冻结训练源码。

成功提交的 App 为 [ap-GulvmjvLvhJeZEMLpeoSrO](https://modal.com/apps/linzhouping88/main/ap-GulvmjvLvhJeZEMLpeoSrO)，调用 `fc-01M51TVWD45B9S8TDB3J36M3ZP`，提交时间约北京时间 15:47:15，调用终态 SUCCESS。App 于 **15:56:06 停止，任务数为 0**；调用图中没有额外 GPU 子调用。旧恢复 App 同时保持 stopped / 0 tasks。

## 预算

用户此前确认赠送额度 30 美元。原 24 美元是执行层保留余量后的保守计算上限；本次从原余量独立预留最多 1 美元，累计保守上限 25 美元，留下 5 美元缓冲。旧预算记录保留，未决预留或相同收尾 ID 再次提交均拒绝；不修改账户级预算，也不声称查询到了实时剩余额度。

GPU 函数 timeout=1020 秒、startup_timeout=120 秒；从本地提交起保留 1080 秒绝对截止时间，worker 留出 60 秒停止缓冲。零应用重试、单并发、闲置 2 秒；没有另启 CPU 编排。即使同一输入平台重入，也不能刷新截止时间。

已核对调用成功、App 停止和零任务。共享账本将本次记录标记为 `completed_confirmed`，为启动 / 清理开销保留完整 1 美元，累计保守账本仍为 25 美元。这是上界与预留，**不是实际费用或赠送余额**；云端终态另保留按每小时 3 美元、墙钟时间加缓冲计算的上界。公开单价参照 [Modal 定价](https://modal.com/pricing)，实际账单以 Usage & Billing 为准。

## 本地接收与分析

已有六份最终模型本地严格加载成功，才允许启动本次收尾。评估完成后接收原封包，以大小与 SHA 校验，独立 CPU 再次 strict load 最终模型，核对冻结 state SHA、2,500 个开发 ID、校准温度并从 logits 重算全部保存指标。六份评估的云端 ACK / 收据共 12 份另行下载并验证，云端与本地收据各自保留原 schema，没有用本地 ACK 覆盖云端训练控制记录。

分析共验证 18 份本地最终 / 冻结 / 评估收据和对应文件，六模型样本行 / 输入 SHA 完全相同。主分析按原定 mixed−CE 方向、三种子配对及截断分组执行；保存标准差、范围、留一结果、精确 McNemar、三种子 sign-flip 和条件样本区间，不将重复开发样本当作 7,500 条独立观察。分析脚本 SHA 与测试 SHA 记入新分析 `summary.json`。

新增预算测试 4 项、分析测试 3 项、原 Modal 传输及恢复测试 16 项通过，合计 23 项；CLI 导入、语法检查通过。图表已实际渲染检查，保留 PNG / SVG。

证据：`submission.json`、`cloud-completion.json`、`cloud-state.json`、`call-graph.json`、`apps-terminal.json`、`cloud-evaluation-receipts/verification.json`、`local-evaluations-delivery.json`、`local-evaluations-download.log`、`analysis-launch.log`。完整分析的结果与限制不依赖本执行记录的文字推断。
