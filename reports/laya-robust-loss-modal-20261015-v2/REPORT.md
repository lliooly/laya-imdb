# Modal v2 调度停止记录

2026-10-15。**未进入正式六模型训练，没有完成优化器更新或产生新准确率。**

[本次 App](https://modal.com/apps/linzhouping88/main/ap-L9tdAzu91W6MHpxHzcFIsi) 的输入快照已上传。科学训练 runner 按设计拒绝已存在的实验报告目录：控制器提前把日志放入该目录，触发 `FileExistsError`，因此在模型训练前停止。这是执行适配错误，不是损失候选的实验结果。

控制器日志已在后续 v3 移至协议级独立目录。v2 源快照保存在本地 `artifacts/laya-modal-v2-source-before-controller-log-fix/` 和 v2 Volume，原配置 / SHA 保留。Modal 调用图已核验该次任务终态为 `FAILURE`，前次 v1 为 `TERMINATED`；不在后台继续运行。

本地账本仍保守保留该次 2.75 美元最大预留，与 v1 共用原 24 美元计算上限，未重新获得一份预算。这不代表实际已收费 2.75 美元。后续执行与真实权重交付以 v3 报告为准。
