# L2 服务器连接与隔离环境准备

日期：2026-10-15。用户提供 SSH 访问并授权在需要时连接；已成功登录并执行只读核验。密码未写入仓库、命令文件或连接日志。本次没有安装服务器依赖、传输模型权重或启动训练。

## 1. 实际服务器状态

| 项目 | 连接时核验结果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090，24,564 MiB |
| 驱动 / Driver API | 550.107.02 / 12.4 |
| 负载快照 | 0% GPU 利用率，2 MiB 已用显存，没有计算进程 |
| Python | 3.13.5，不满足项目固定的 Python 3.12 要求 |
| 镜像 PyTorch | 2.8.0+cu128，CUDA 12.8 |
| 已有环境小检查 | `cuda.is_available=True`；CUDA FP32 的 16×16 全 1 矩阵相乘得到 16 |
| 数据盘空闲 | 接口快照为 50.0 GiB |
| 项目与工具 | 检查的目录未找到原项目，未找到 uv、Laya、Transformers 或 safetensors |

该检查只证明镜像自带环境能执行一次小矩阵运算，不能替代 L2 所需的锁定环境、完整模型 forward、梯度、确定性或保存重载准入。磁盘与负载均为此次快照，不作后续空闲保证。

## 2. 为什么不直接沿用镜像或原 Linux 锁

镜像是 Python 3.13 / PyTorch 2.8，项目为 Python 3.12 / PyTorch 2.14.1，不能把镜像环境当作已经满足原协议。

原锁的 Linux 构建使用 CUDA 13 配套包，而 CUDA 13 的常规最低驱动为 580；当前驱动是 550。CUDA 12.x 与 CUDA 13.x 的兼容边界不能混用，见 [NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。本轮没有升级宿主驱动、重启或重建实例。

已从 [PyTorch 官方 CUDA 12.6 索引](https://download.pytorch.org/whl/cu126/torch/) 找到 `torch-2.14.1+cu126` 的 CPython 3.12 / Linux x86_64 wheel，并取得其索引声明的 SHA；这提供了保留 PyTorch 主版本的适配路径。仅查到 wheel 和解析依赖，不代表下载后已核验 SHA 或执行成功。

## 3. 已准备的独立依赖锁

新增 [环境配置](../../environments/laya-cu126/pyproject.toml)、[独立锁](../../environments/laya-cu126/uv.lock) 与 [使用说明](../../environments/laya-cu126/README.md)。没有修改根目录的 `pyproject.toml` 或 `uv.lock`。

使用 uv 0.12.15 为 Linux x86_64 解析出 46 个包。与原锁共有的软件包版本差异为：

| 软件包 | 原锁 | 本轮隔离锁 |
| --- | --- | --- |
| torch | 2.14.1 | 2.14.1+cu126 |
| cuda-toolkit | 13.0.3.0 | 12.6.3 |
| cuda-bindings | 13.4.3 | 12.9.9 |

CUDA 对应的 NVIDIA 包随构建切换。所有共有的非 CUDA 软件包版本保持一致，包括 Laya 0.4.1、Transformers 4.57.6、NumPy 2.5.3、safetensors 0.8.0、tokenizers 0.22.2、huggingface-hub 0.36.2。两种头的六次运行必须采用同一份环境锁与同一实际设备；不能混用原 CUDA 13 构建或镜像的 PyTorch 2.8。

这项环境适配不更改 [L2 设计](../../docs/superpowers/specs/2026-10-15-laya-head-control-design.md) 中的头部、初始化、输入、训练预算、seed 或选模规则。执行时应在新运行的 manifest 中明确写入独立锁的 SHA 与 CUDA 构建，不追改原报告的运行环境。

## 4. 网络与后续准备

服务器可访问 PyPI 的 uv 元数据和原锁的 PyTorch wheel URL，两者 HEAD 返回 200。访问锁定 Hugging Face revision 的配置请求发生一次 ConnectionError；没有据此判断所有模型下载路径均不可用。模型可通过另行核验的下载路径或本地已有文件同步，最终必须匹配原模型锁的 SHA。

接下来依次需要安装隔离 Python 3.12 环境、同步并核验源资产、实现 L2 对照入口、完成 CPU 检查，再做两种头的 CUDA 短程训练、独立进程复验和 FP32 重载准入。上述条件通过前不启动六次正式训练，不读取官方测试正文或预测。

本轮连接核验完成后已关闭连接，没有驻留的安装或训练任务。完整结构化快照、环境差异及锁文件 SHA 见 [summary.json](summary.json)。
