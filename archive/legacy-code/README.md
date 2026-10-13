# 已停用的实验入口

这些原文件按 SHA 不变归档，不属于当前可执行工作流。重新实验应采用当前训练适配、strict准入、新run_id和新配置；旧驱动绑定历史任务，直接运行会冲突或使用过期协议。

| 原入口 | 停用原因 |
| --- | --- |
| scripts/run_laya_resampling_pair.py | 默认运行时配对已由strict配对替代 |
| scripts/run_seed43_pair.py | seed43原复验已完成；固定日期/配置的一次性驱动 |
| scripts/ghostty_ssh_bridge.py | 当时连接故障使用的机器专用桥，不作为仓库公共训练接口 |
| configs/*resampling*-s43-epochs2-fp32.json（无pool45后缀） | 48条池的旧计划，正式实验改为冻结45条；未作为新默认入口 |

保留归档用于理解原实验过程。被核心模块或测试依赖的R-Drop、重采样、语义适配和历史ModernBERT组件仍在scripts中；未因某次负结果删除其实现或测试。
