# 依赖环境

使用 Python 3.12 和 uv。每个子项目保留独立的 `pyproject.toml` 与 `uv.lock`；虚拟环境、模型和缓存不进入 Git。以下命令均从仓库根目录执行。

| 环境 | 安装 | 用途 |
| --- | --- | --- |
| 根环境 | `uv sync --locked --extra laya --extra laya-mlx` | Apple Silicon 上 Laya 训练和 MLX 推理 |
| 根环境 + 图表 | 上述命令增加 `--extra analysis` | Matplotlib 图表构建，保留模型依赖 |
| `bow-cpu` | `uv sync --project environments/bow-cpu --locked` | 词袋、SciPy / scikit-learn 与独立统计核验 |
| `deberta-prep` | `uv sync --project environments/deberta-prep --locked` | DeBERTa 模型准备和分词 |
| `laya-cu126` | `uv sync --project environments/laya-cu126 --locked` | Linux CUDA 12.6 Laya 训练与头部对照 |
| `deberta-cu126` | `uv sync --project environments/deberta-cu126 --locked` | Linux CUDA DeBERTa 训练 |
| `laya-modal` | `uv sync --project environments/laya-modal --locked` | 历史 Modal 调度与权重接收 |

CUDA 环境仅在匹配的 Linux / GPU 平台安装。DeBERTa BF16 与 Laya FP32 使用各自的准入协议；不要把一条路线的数值结论套到另一条路线。

已有环境运行时使用 `uv run --locked --offline --no-sync`；独立环境在 `uv run` 后增加 `--project environments/NAME`。`uv sync` 默认会移除本次未声明的可选依赖，日常主环境应同时保留需要的 extras。完整数据、模型和训练前提见 [复现指南](../docs/closeout/reproduction.md)。

Word、TeX 和 PDF 构建还需文档工具与系统程序，见 [报告说明](../paper/README.md)；根 Python 锁不包含整套排版工具链。
