# 实验思路与结果总表

> 历史记录：以下保留当时的训练安排或实验台账。2026-10-18最新结果、默认部署差异与收尾状态见[收尾总览](../closeout/README.md)。

更新：2026-10-13。E01–E37 为已做过的模型、部署、诊断或审核工作，E38–E43 为尚未完成的方向。准确率来自固定 2500 条开发集，不能当作官方测试成绩；训练内预测一致率、机械准入与合成诊断另行标明。不同平台、seed、运行时、训练或推理预算不可直接解释为单因素收益。

| 编号 | 思路／实验 | 协议与比较范围 | 结果 | 当前决策 | 证据 |
| --- | --- | --- | --- | --- | --- |
| E01 | ModernBERT-base 历史全量基线 | MLX；256 token；3 轮；s42 | 94.64%；第 1/2/3 轮 94.16/94.48/94.64% | 保留基线；与 Laya 预算不同 | [来源](../../reports/model-comparison/REPORT.md) |
| E02 | Laya-RLCD pilot | MPS；512 token；1 轮；小训练预算 | 93.72%；公共 2100 条与 CE 配对 p=1 | 保留损失对照；不是全量 RLCD | [来源](../../reports/model-comparison/REPORT.md) |
| E03 | Laya-CE pilot | 与 RLCD pilot 同预算 | 93.52%；概率指标部分优于 RLCD | 保留同预算对照 | [来源](../../reports/model-comparison/REPORT.md) |
| E04 | Laya-CE Mac 全量一轮 | MPS FP32；512 token；s42 | 95.76%；106 错 | 历史全量参照 | [来源](../../reports/laya-mac-ce-comparison-v1/REPORT.md) |
| E05 | 同权重 MLX FP32/FP16 推理 | 固定一轮 CE 权重；2500 条 | 均 95.76%；标签全一致；FP16 端到端提速 2.52–3.68 倍 | 保留 Mac 部署路线；非训练加速证据 | [来源](../../reports/laya-backend-comparison/REPORT.md) |
| E06 | CUDA FP32 数值准入与测速 | RTX 6000D；小规模训练/重载/Mac 核对 | 通过 FP32 准入；训练 FP16 梯度溢出 | 保留 FP32 与有限值保护 | [来源](../../reports/cuda-preparation-v1/REPORT.md) |
| E07 | 完整 CUDA 一轮参照 | RTX 6000D FP32；512 token；s42 | 96.28%；与 Mac 推理标签一致 | 保留跨设备参照，非干净方法消融 | [来源](../../reports/laya-cuda-reference-ce-l512-s42-fp32-v1/REPORT.md) |
| E08 | Mac 两轮原学习率 | MPS FP32；512 token；s42 | 96.96%；76 错；较一轮修正 42/退化 12 | 最高历史开发成绩；单种子 | [来源](../../reports/laya-mac-ce-comparison-v1/REPORT.md) |
| E09 | Mac 两轮学习率 ×0.6 | 与 E08 同协议，仅降低初始 LR | 96.12%；较 E08 修正 10/退化 31 | 停止沿此幅度继续搜索 | [来源](../../reports/laya-mac-ce-comparison-v1/REPORT.md) |
| E10 | Mac 两轮模型 MLX FP16 核对 | E08/E09 固定权重；2500 条 | 标签全一致；个别概率差超过 1% | 分类可用；严格概率核对未通过 | [来源](../../reports/laya-mac-ce-comparison-v1/REPORT.md) |
| E11 | RTX 4090 两轮基线 | FP32；512 token；s42 | 95.88%；103 错 | CUDA 同期参照 | [来源](../../reports/rtx4090-combined-completed-v1/REPORT.md) |
| E12 | 标签平滑 0.05 | 与 E11 同协议 | 96.16%；未校准概率更好；校准后 NLL 非全面改善 | 保留概率质量证据 | [来源](../../reports/rtx4090-regularization-completed-v1/REPORT.md) |
| E13 | warmup 5% | 与 E11 同协议 | 96.32%；92 错 | 保留调度候选 | [来源](../../reports/rtx4090-regularization-completed-v1/REPORT.md) |
| E14 | warmup＋标签平滑 | 4090 FP32；两轮；s42 | 96.44%；89 错 | s42 有收益，等待跨 seed 解释 | [来源](../../reports/rtx4090-combined-completed-v1/REPORT.md) |
| E15 | R-Drop alpha=1 | 基于 E14；双前向；s42 | 96.08%；5526 更新/11052 前向；低于组合 | 停止默认使用，不追加 alpha 搜索 | [来源](../../reports/rtx4090-rdrop-completed-v1/REPORT.md) |
| E16 | 基线 seed43 复验 | 4090 FP32；两轮；默认运行时 | 96.48%；第 1 轮 94.76% | 保留历史基线；不能混同 strict | [来源](../../reports/rtx4090-seed43-completed-v1/REPORT.md) |
| E17 | warmup＋平滑 seed43 复验 | 与 E16 同协议 | 96.48%；第 1 轮 95.24%；对基线净 0/p=1 | 准确率增益未跨 seed 复现 | [来源](../../reports/rtx4090-seed43-completed-v1/REPORT.md) |
| E18 | 固定权重首尾窗口：基线 s42 | 不训练；466 评论 token=头233＋尾233 | 95.88→96.08%；修正 7/退化 2 | 保留输入策略候选 | [来源](../../reports/laya-posthoc-consolidated-v2/REPORT.md) |
| E19 | 固定权重首尾窗口：基线 s43 | 同 E18；s43 权重 | 96.48→96.84%；修正 12/退化 3 | CUDA 固定权重诊断最高，非新训练结果 | [来源](../../reports/laya-posthoc-consolidated-v2/REPORT.md) |
| E20 | 固定权重首尾窗口：组合 s42 | 同 E18；组合权重 | 96.44→96.68%；修正 9/退化 3 | 保留 | [来源](../../reports/laya-posthoc-consolidated-v2/REPORT.md) |
| E21 | 固定权重首尾窗口：组合 s43 | 同 E18；组合权重 | 96.48→96.64%；修正 8/退化 4 | 四组均正向；仅两个种子，仍属探索 | [来源](../../reports/laya-posthoc-consolidated-v2/REPORT.md) |
| E22 | 前缀温度＋偏置校准 | 仅原 400 校准样本拟合；四组固定权重 | 基线 s42/s43=95.80/96.52%；组合=96.44/96.40% | Accuracy 不一致，不设默认偏置 | [来源](../../reports/laya-posthoc-consolidated-v2/REPORT.md) |
| E23 | 首尾＋温度/偏置校准 | 同 E22；首尾输入 | 基线 s42/s43=96.04/96.84%；组合=96.72/96.56% | 作为概率诊断，不按开发结果挑偏置 | [来源](../../reports/laya-posthoc-consolidated-v2/REPORT.md) |
| E24 | 训练内难例筛选与审阅 | 梯度分区隔离；223 条审阅，145 条分歧全覆盖 | 二次冻结 48 条；负17/正31；非折外难度 | 保留审阅链路；不是增强收益 | [来源](../../reports/laya-train-hardcase-review-v2/REPORT.md) |
| E25 | 边界复核后 45 条冻结池 | 原 48 条再隔离 3 条；原标签不变 | 形成 45 条执行池；不修改训练/校准分区 | 后续用固定45池；48池旧配置归档 | [来源](../../reports/laya-resampling-45-preparation-v2/REPORT.md) |
| E26 | 45 条重采样默认运行时 | s43；两轮；同期控制/增强 | 95.96→96.16%；修正30/退化25，净+5 | 不足以确认稳定；控制未复现历史96.48% | [来源](../../reports/laya-resampling45-seed43-completed-v1/REPORT.md) |
| E27 | 历史控制漂移核查 | 模型/分区/输入哈希与超参数核对 | 历史96.48 vs 本次95.96%，差13条 | 不能简单归因于采样适配器 | [来源](../../reports/laya-resampling45-control-drift-audit-v2/REPORT.md) |
| E28 | 四次短程重复性诊断 | 原路径/控制路径各两次；每次16更新 | 默认路径首步梯度已不同，前向/RNG匹配 | 定位反向边界；不是完整训练 | [来源](../../reports/laya-short-drift-completed-20261012-v1/REPORT.md) |
| E29 | 关闭重计算／强制 math 注意力 | 独立16步重复矩阵 | 仍不逐位一致；首步嵌入梯度不同 | 停止把这两项作为修复手段 | [来源](../../reports/laya-determinism-completed-20261012-v1/REPORT.md) |
| E30 | strict 运行时协议 | CUBLAS workspace＋确定性算法；保留重计算 | 原路径/控制路径16步逐位一致 | 保留准入；不声称全两轮逐位复现 | [来源](../../reports/laya-determinism-completed-20261012-v1/REPORT.md) |
| E31 | 独立嵌入层合成复验 | compiled/compiled-strict/eager，四次反向 | compiled 梯度不一致；strict/eager 一致 | 支持算子诊断；非真实首批张量重放 | [来源](../../reports/laya-determinism-completed-20261012-v1/REPORT.md) |
| E32 | strict 45 条重采样复验 | s43；两轮；两组各5526更新 | 96.32→96.40%；修正27/退化25；净+2；p=.8899 | 暂停重采样加码；无稳定收益 | [来源](../../reports/laya-strict-pair-result-check-20261013-v1/REPORT.md) |
| E33 | 30 组语义成对试制与固定模型探测 | 训练内来源；原文/B普通/C语义；未训练 | 原标签一致24/21/23；B退化3/C退化1 | 仅输入敏感性诊断；非泛化准确率 | [来源](../../reports/semantic-pair-pilot-20261013-v2/REPORT.md) |
| E34 | 真实人工审核与机械去重 | 用户接受30组；三版本SHA绑定 | 词五元组近重复0匹配；CUDA机械6测试通过 | 审核已更新；旧 pending 表保持历史状态 | [来源](../../reports/semantic30-abc-launch-20261013-v1/REPORT.md) |
| E35 | 30 来源 A：原文重复 | strict s42；5%总曝光；两轮 | 96.52%；2413/2500；NLL .133592 | 本轮同期参照；并非无替换基线 | [来源](../../reports/semantic30-abc-completed-20261013-v1/REPORT.md) |
| E36 | 30 来源 B：普通改写 | 与 E35 同曝光槽/标签/预算 | 96.32%；相对A修正11/退化16；净−5 | 无收益；不继续按小池重放加码 | [来源](../../reports/semantic30-abc-completed-20261013-v1/REPORT.md) |
| E37 | 30 来源 C：语义改写 | 与 E35 同协议；平均额外曝光73.67次/源 | 96.44%；对A净−2/p=.8388；对B净+3/p=.7011 | 未超A；不自动追加s43，不否定所有语义增强 | [来源](../../reports/semantic30-abc-completed-20261013-v1/REPORT.md) |
| E38 | 首尾策略一致训练＋推理 | 未来：先同权重诊断，再strict单因素训练 | 未运行；保留512总预算和原冻结分区 | 下一优先级；输入哈希及Mac核对必需 | [来源](../../docs/superpowers/specs/2026-10-11-laya-posthoc-headtail-design.md) |
| E39 | 更长输入／分块／集成 | 不同计算预算的未来独立实验 | 未运行；不能靠当前截断错题承诺98% | 暂缓；不与首尾单因素混做 | [来源](../../docs/archive/roadmaps/2026-10-09-research-roadmap.md) |
| E40 | 扩大语义池／一致性训练 | 原目标至多240来源；新审核与新协议 | 未运行；本轮30源不等同正式池 | 暂缓，需独立失败分析后再设计 | [来源](../../docs/superpowers/specs/2026-10-13-semantic-pair-pilot-proposal.md) |
| E41 | ModernBERT-base 公平迁移对照 | 未来：先512-token CE参照，再迁移有效输入策略 | 未运行；历史256-token结果预算不同 | Laya筛选收尾后执行 | [来源](../../docs/experiments/improvement-roadmap.md) |
| E42 | Laya-RLCD 全量迁移对照 | 未来：与CE同数据/预算/输入策略 | 未运行；现有93.72%只是pilot | 保留课程方法对照，不将pilot充作全量 | [来源](../../docs/experiments/improvement-roadmap.md) |
| E43 | 官方测试与课程交付 | 冻结最终方案后一次正式测试；Word/视频/源码 | 官方测试正文封存；独立应用数据尚无 | 最终阶段；98%仍是研究目标 | [来源](../../docs/experiments/current-status.md) |

机器可读版本：[experiment-registry.json](experiment-registry.json)。当前候选、备份边界与下一步见 [当前状态](current-status.md) 和 [改进路线](improvement-roadmap.md)。
