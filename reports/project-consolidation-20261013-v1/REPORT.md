# 主线整合与文档整理

2026-10-13按用户授权整合所有本地分支和已有未提交成果。使用当前仓库，不创建worktree。旧本地main为cc4aef1，dev/semantic-pair-pilot为99912ef；远端origin/main在本次整理时为547a688。正式提交与合并信息见git-verification.json。

建立统一文档索引、43项实验总表、当前状态和后续路线。规范11个文档路径，保留旧路径兼容跳转；移出当前scripts/configs的7个过期入口或配置，原文件SHA不变保存在archive/legacy-code。日期实验设计和报告目录保持原路径。完整迁移表见relocations.json。

整理前为1062个已有报告文件保存SHA；整理后检查原文件未修改、未删除。全部数据、权重、校准/开发划分与reproduction保持；当前默认权重不变。新报告只提交必要REPORT.md和summary.json，逐条大型诊断、训练正文和权重仍在本地；这不是删除原始材料。

保留历史负结果用于课程比较；不把未通过准入、训练内诊断或计划当作已取得Accuracy。最高历史开发成绩仍为Mac两轮96.96%，最新语义增强未超同期原文参照。下一阶段优先权重备份、首尾输入一致训练/推理和公平模型迁移。本次不新启动GPU训练，不推送origin。

验证结果见verification.json；若某项设备核对未运行，不能由CPU测试替代。
