# L2 的 CUDA 12.6 隔离环境

用途：适配本轮已核验的 RTX 4090 / 550.107.02 驱动，保留 PyTorch 2.14.1 的主版本。原项目 `uv.lock` 不改写；该目录的 Linux x86_64 锁单独版本化。

与原锁共有的软件包中，只有 `torch` 的 CUDA 构建标识、`cuda-toolkit` 和 `cuda-bindings` 版本不同。CUDA 配套包随 CUDA 12.6 构建重新解析；Laya、Transformers、NumPy、safetensors、tokenizers 和 huggingface-hub 的版本与原锁一致。完整差异与来源见 [服务器准备核验](../../reports/laya-head-control-remote-readiness-20261015-v1/REPORT.md)。

该锁已经在用户提供的服务器安装，45 个依赖版本逐项核验通过；四次 v2 CUDA 准入和独立重载核验已通过，正式六次训练已启动。当前证据见 [执行记录](../../reports/laya-head-control-launch-20261015-v1/REPORT.md)。不要在 Mac 上安装这份 Linux 环境，也不要将其基础矩阵运算测试当成训练通过。

服务器装好 uv 后，从项目根目录执行：

```bash
UV_PROJECT_ENVIRONMENT="$PWD/.venv-laya-cu126" \
  uv sync --project environments/laya-cu126 --locked --python 3.12
```

执行 L2 入口时沿用同一 `UV_PROJECT_ENVIRONMENT` 和 `--project environments/laya-cu126 --locked`，记录实际 Python、Torch CUDA 构建、驱动与这份锁的 SHA。训练前仍须完成输入配对、有限梯度、短程独立进程复验和 FP32 保存重载核验；本轮已获用户明确训练授权，由实验编排器在全部准入通过后启动固定六次训练。首次准入缺 Python.h；现通过隔离解包的 Python 3.12 开发头文件和 CPATH 补齐，未升级系统运行时。
