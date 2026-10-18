# 隔离依赖环境

每个子目录保留自己的 pyproject.toml 和 uv.lock。[复现说明](../docs/closeout/reproduction.md)列出用途与平台限制。虚拟环境不进入源码包，不将不同 NumPy 版本或 CUDA 构建无记录地混装。

bow-cpu 用于统计词袋和独立 SciPy/sklearn 核验；deberta-prep 用于模型准备；deberta-cu126 和 laya-cu126 面向对应 Linux CUDA 环境；laya-modal 用于既有云端训练/交付流程。历史云端预算记录不授权本次启动计算。
