# DeBERTa 异构互补验证：启动与交付记录

用户已批准 `docs/superpowers/specs/2026-10-17-deberta-heterogeneous-complementarity-design.md`。新运行采用原始 22100 梯度训练、400 校准、2500 开发分区，一个 seed42、固定两轮、官方 DeBERTa-v3-large 编码器和标准二分类头。

启动阶段状态：本地 5 项必要检查通过；官方权重完整 SHA 和 FP32 加载通过，模型 435063810 参数；39 份源码和输入绑定 SHA。此记录本身不表示 GPU 已通过准入或正式训练完成，后续状态以 `submission.json`、`admission.json`、`cloud-state.json` 与最终交付凭据为准。

官方 revision：`64a8c8eab3e352a784c658aef62be1662607476f`。官方权重文件 SHA256：`dd5b5d93e2db101aaf281df0ea1216c07ad73620ff59c5b42dccac4bf2eef5b5`。标准分类头和池化头为 seed42 初始化，编码器无意外缺失权重。

使用官方慢速 SentencePiece，保留字节回退；草稿 fast 转换告警及输入保留在 v1 fast-draft 目录，正式缓存不混用。训练缓存 22100 条，2687 条在 DeBERTa 510 review-token 预算下截断；这是训练集统计，不能与 Laya 开发集 407 条直接比较。

GPU 准入包含正式更新函数的代表与最长输入、完整保存、独立 CUDA 恢复等价、CPU 严格重载和小文件本机回传。根据实测吞吐保守估算，预计两轮、评估和保存无法落入剩余总窗口即停止，不减少轮数、改精度或延长预算。

本轮追加总上限 5 美元，GPU资源窗口 90 分钟，余量包含 CPU 构建、协调、传输及 v1 构建失败。v1 在镜像函数导入时失败，未启动 GPU；修复后使用 v2 新运行，旧代码/配置快照保留。预算是本项目已知活动的保守预留，不是实际账单余额。

[Modal 应用](https://modal.com/apps/linzhouping88/main/ap-oTP9Yj43WfKZavN9Ig51An)。本机接收器已经启动，只执行下载、SHA 校验和 CPU 重载，不创建 GPU；中断可重新接收已保存分块。本地权重是否完整到达，只以 `local-delivery.json` 明确的 `verified_local` 为准。

若正式训练通过准入，逐轮交付 `checkpoints/deberta-heterogeneous-s42-modal-20261017-v2/epoch-001/`、`epoch-002/` 的完整 FP32 权重及优化器/调度器/RNG；最终评估单独封存，比较冻结主融合与固定 50:50 概率平均，不扫描系数。官方测试不上传、不读取；默认部署保持。
