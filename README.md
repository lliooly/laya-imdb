# Laya IMDB

基于 Laya 的 IMDB 电影评论情感分类项目：使用 **PyTorch 微调**，在 **Apple Silicon Mac 上通过 MLX FP16 推理**，研究分类准确率、概率可信度与部署成本之间的关系。

这是 NLP 课程大作业的研究与工程仓库。当前已完成 Mac 微调和同权重推理后端对照；下一阶段计划迁移到 CUDA 训练，优化训练方法与长评论处理，并交付最高准确率版和高效单模型版。

> 官方测试集尚未正式评估。98% 是后续研究目标，不是已取得的成绩。CUDA、混合精度训练和最终集成方案尚未完成验证。

## 当前结果

以下均来自固定的 **2,500 条开发验证集**，不是官方测试集成绩。

| 模型 / 推理后端 | Accuracy | 说明 |
|---|---:|---|
| ModernBERT-base / MLX | 94.64% | 历史完整基线，256 token，3 轮训练 |
| Laya-CE / PyTorch MPS FP32 | 95.76% | 512 token 总预算，1 轮训练 |
| 同一 Laya-CE / MLX FP16 | 95.76% | 与 MPS FP32 的验证预测标签全部一致 |

Laya 约 4.21 亿参数，ModernBERT-base 约 1.49 亿参数。两套训练配置的规模、文本预算、训练与校准设置不同，不能将准确率差异单独归因于 Decision Head。

同权重后端实验中，MLX FP16 的端到端 P50 延迟相对 MPS FP32 改善约 2.52–3.68 倍，具体取决于评论长度与 batch。这同时涉及框架和精度变化，不能推断为训练加速或相对 ModernBERT 的加速。

- [模型比较与证据边界](reports/model-comparison/REPORT.md)
- [同权重推理后端对照](reports/laya-backend-comparison/REPORT.md)
- [新路线与 14 天计划（讨论稿）](docs/laya-revised-roadmap.md)

## 环境与依赖

- Python 3.12；依赖由 `uv` 管理，版本锁定在 `uv.lock`。
- Mac 推理：Apple Silicon 与 Metal，使用 `laya-mlx` extra。
- 当前已验证的 Laya 训练：PyTorch / MPS，使用 `laya` extra。
- CUDA 训练：计划使用 RTX 4090，适配与实测尚待完成。
- Rust tokenizer 是可选工程工具；仅复现相关实验时需要 Rust/Cargo。

克隆仓库后进入项目根目录，以下命令均在根目录执行：

```bash
git clone https://github.com/lliooly/laya-imdb.git
cd laya-imdb
uv sync --locked --extra laya-mlx
```

克隆不包含数据、预训练缓存或微调权重。首次安装和下载需要网络，准备完成后可以离线运行。

## Mac 本地推理

先将已微调的完整 checkpoint 放入 `checkpoints/laya-l512-s42-ce/`。所需结构与服务器权重衔接见 [推理说明](docs/inference.md)。默认入口不会在缺失权重时下载或冒充已微调模型。

```bash
uv run --offline --locked --extra laya-mlx python scripts/predict.py \
  'The acting was excellent, and I loved the ending.'
```

输出包括标签、校准前后概率、输入长度和截断信息。指定其他本地 checkpoint：

```bash
uv run --offline --locked --extra laya-mlx python scripts/predict.py \
  'A disappointing movie with a weak script.' \
  --checkpoint checkpoints/my-run
```

完整权重需要包含 `model.safetensors`、`rl_agent_config.json`、`encoder/config.json` 和 `tokenizer/`。权重需从已有本地实验迁移或自行训练，仓库没有公开权重下载地址。

## 数据准备与当前训练入口

数据来自 [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)。固定划分：训练 22,500、开发验证 2,500、官方测试 25,000；标签 0=负面、1=正面。

```bash
uv sync --locked --extra laya
uv run --locked --extra laya python scripts/prepare_data.py
uv run --locked --extra laya python scripts/verify_data.py
uv run --locked --extra laya python scripts/prepare_laya.py
```

`prepare_data.py` 会重建派生数据和划分清单；已有实验环境先运行 `verify_data.py`，不要无故重建冻结数据。下载与模型 revision、哈希分别记录在 `reports/split_manifest.json` 和 `configs/laya_model_lock.json`。

当前训练配置默认使用 **MPS**，并从原训练划分中保留 400 条做校准，其余 22,100 条用于梯度更新：

```bash
uv run --offline --locked --extra laya python scripts/train_laya.py --check-only
uv run --offline --locked --extra laya python scripts/train_laya.py \
  --config configs/laya-full-ce.json
```

已有运行目录禁止覆盖。新实验先复制配置并修改 `run_id`。当前 Laya 训练入口没有精确断点恢复，不接受基线脚本的 `--resume` 参数。新 CUDA 环境应先检查依赖、显存、保存重载和 Mac 预测一致性，再正式训练。

详细说明见 [Laya 训练协议](docs/laya-training.md)。ModernBERT、Rust 分词、数值核对和历史实验的命令保留在 [历史工程流程](docs/historical-workflows.md)。

## 评估与复现原则

- 同清理文本按组划分，保留否定、大小写与标点；模型只读取评论文本，不读取带评分的文件名。
- 官方测试集仅做过完整性审计；正式评估前不用于调参、选模型或拟合校准参数。
- 校准样本不参与梯度训练；开发验证集已参与模型开发，成绩存在选择偏差。
- 原始数据存在的训练/测试重复会记录，不静默删除，不声称绝对无泄漏。
- 保存配置、数据/权重哈希、依赖版本与逐样本预测；历史快照和诊断用于审计，不能因失败而删除。

运行现有 CPU 单元测试，无需启动训练或读取官方测试集：

```bash
uv run --offline --locked python -m unittest discover -s tests -v
```

GPU 数值核对与性能验证需要可访问 Metal/CUDA 的设备；普通单元测试通过不代表 GPU 迁移已验证。

## 仓库布局

```text
configs/         冻结配置与模型/tokenizer 锁文件
docs/            方法、训练与推理协议、开发路线、历史复现说明
scripts/         数据准备、训练、推理、评估与分析入口
tests/           数据划分、训练工具和统计指标测试
rust-tokenizer/  可选 Rust CLI 源码及 Cargo.lock
reports/         已有实验指标、预测、图表与冻结复现快照
```

`data/`、`checkpoints/`、`artifacts/`、`.cache/`、`.venv/`、Rust 构建目录、报告中的训练数据副本、ZIP 和视频仅保留在本地，由 `.gitignore` 排除。`reports/model-comparison/reproduction/` 是分析时的冻结源码快照，与当前入口重复但用于核验历史结果，保留其原始内容与校验关系。

新增实验使用独立 `run_id`，提交源码、配置和必要结果，避免将大规模原始输出、数据和权重提交到 Git。现有历史脚本仍被复现链路引用，保留其名称和接口。

## 课程交付

最终提交 Word 报告（题目、摘要、引言、方法、实验、结论、参考文献）和源码压缩包；包内包含训练、测试视频，每个视频不超过 5 分钟。GitHub 仓库用于版本管理，不能替代课程要求的最终提交文件。

## 许可证与致谢

本项目原创代码使用 [MIT License](LICENSE)。第三方代码及其衍生部分、依赖、预训练权重与 IMDB 数据遵循各自原有条款，不因仓库使用 MIT 而更改许可证。ModernBERT MLX 实现中的来源声明与 Apache-2.0 文本保留，详见 [第三方说明](THIRD_PARTY_NOTICES.md)。

模型与方法来源见 [参考文献](docs/references.md)。
