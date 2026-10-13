# 严格确定性配对训练：准入通过、完整训练运行中

2026-10-13。用户批准接入已验证的严格协议、使用 4090 测试与训练，并审阅现有 dev 分支。工作区沿用 dev/semantic-pair-pilot；未创建分支或 worktree。

## 通过的验证

- 本地相关训练测试 32 项通过：运行时 4、重采样 3、短程追踪 3、设备准入 5、逐轮检查点 4、优化适配 13。配置检查与 git diff --check 通过。
- 远端重新运行运行时／重采样共 7 项测试通过。
- 新正式入口两次独立进程，各用完整训练／调度预算执行 16 步后主动停止；16 条输入、RNG、logits、loss、梯度、学习率及更新后模型记录完全一致，205 个参数的首步梯度记录一致。与已验证 strict-control-1 的同项记录也完全一致；本地重新计算再次通过。短程诊断的训练入口失败状态属于预期 DiagnosticStop，不是训练数值失败。
- 两组 GPU 冒烟各 128 实际梯度样本、两轮 32 更新均通过。梯度窗口有限、调度与更新预算相符、两轮检查点保留且 SHA 核验、保存 FP16 表示与重载一致、32 条冒烟验证预测重载后保持一致。
- 下载准入证据 22 个文件，逐文件及整包 SHA 校验通过。压缩包 SHA：6a35dd4dd0e4aed68efa045c8de8068253572b3d06c25f50d24acca78bd66664。见 summary.json；原始文件归回各独立诊断目录，包保存在 artifacts/strict-pair-admission-backup-20261013-v1。

这是短程一致及保存重载准入，不代表完整两轮重复已被证明，也不代表准确率收益。

## 正在执行

最新观测：北京时间 2026-10-13 14:03:47，流水线 running，完整控制组第一轮日志已记录 step 816。控制组共 5,526 更新；随后自动执行重采样组，并导出两组逐轮及最终完整开发预测。见 launch-observation.json。该记录是当时快照，后续状态以远端流水线 status.json 为准。

实际正式 manifest：22,100 梯度来源、400 校准、2,500 开发验证；runtime_profile=strict，CUBLAS workspace :4096:8，deterministic_algorithms=True、warn_only=False、cudnn.deterministic=True、benchmark=False。新入口与运行时适配器 SHA 已与本地准备快照绑定。FP32、有限值保护、45 条候选、seed43、两轮和其余超参数保持既定协议，官方测试封存。

流水线 PID 123，当前控制进程 PID 595，输出目录 artifacts/resampling45-seed43-strict-20261013-v1。新正式 run_id 为 laya-rtx4090-ce-resampling-{control,resampled}-pool45-s43-epochs2-strict-20261013-v1。逐轮及最终权重留在各自 checkpoints/run_id，运行结束后再完成全量结果和权重备份核验。本报告中的备份声明仅覆盖已下载的准入证据。

流水线最长 7,200 秒，失败停止后续阶段并保留已有材料。预算记录在 preparation.json：用户授权“钱管够”，没有精确余额；此前单价 1.24 元/小时未经本日重新核验，按该单价两小时估算 2.48 元。

## dev 分支审阅

审阅及 9 项语义工具测试完成，90 输入／三组曝光预算重新独立核验通过。优势是分开控制来源重复、普通改写和语义改写，并具备来源／哈希／审核审计；当前固定权重诊断没有修正原文错题，人工审核与实际训练消费适配仍待完成，30 个来源在增强计划中各额外曝光 63–87 次。完整分析见 reports/dev-semantic-pair-review-20261013-v1/REPORT.md。语义试制保持独立，不注入本次配对。
