# 历史工程流程与复现命令

本页保留首次版本管理前的工程说明。当前入口与路线以根目录 README 和新版路线为准；历史结果、限制与命令用于追溯。

# End-IMDB：Apple Silicon 电影评论情感分类

目标：提高分类准确率、降低预测成本，并通过固定验证划分和受控对照检验泛化。
Mac 默认推理使用 Laya-CE + MLX FP16；Laya 微调保留官方 PyTorch 训练器，当前支持的本机流程使用 MPS，未来服务器训练计划使用 CUDA。ModernBERT MLX 基线与 Rust tokenizer 流水线继续保留。

## 默认推理：MLX FP16

```bash
cd /path/to/laya-imdb
uv sync --locked --extra laya-mlx
uv run --offline --locked --extra laya-mlx python scripts/predict.py \
  'The acting was excellent, and I loved the ending.'
```

读取已完成微调的 `checkpoints/laya-l512-s42-ce`，返回情感标签、校准前后概率与截断信息。Mac 推理不需要 PyTorch；首次安装依赖联网，随后离线执行。未来 CUDA 微调权重可通过 `--checkpoint` 指定完整本地目录。

安装、基线兼容与服务器训练衔接见 [推理说明](inference.md)。训练命令仍使用下方 PyTorch 入口。

## Laya 全量 CE 训练（当前入口）

开源 Laya 英文权重、分词器和依赖已缓存，模型 revision 与 SHA256 见 `configs/laya_model_lock.json`。训练仅访问本地文件，并开启 Hugging Face 离线模式及 Python 网络连接拦截。

插电、打开录屏后运行：

```bash
cd /path/to/laya-imdb
caffeinate -i uv run --offline --locked --extra laya python -u scripts/train_laya.py \
  --config configs/laya-full-ce.json
```

已完成 RLCD 与 soft-CE 先导对照，用户选择 CE 作为后续训练方法。两轮在同一 256 条验证样本上准确率均为 93.75%，CE 的 NLL 和 Brier 更低，校准温度没有触及上限；这是一轮小规模实验的观察，尚不能证明跨种子优势。

当前配置从原始 Laya 权重开始，训练 1 轮：固定训练集 22,500 条中留出 400 条校准、22,100 条梯度训练，独立完整验证 2,500 条。使用 CE、选项顺序增强、512 token 总预算和梯度检查点，微调整个编码器与决策头。正式测试集继续封存。

此全量训练已由用户完成，训练结束的完整验证准确率为 **95.76%**；此前 ModernBERT-base 基线在同一验证集上为 **94.64%**，相差 **1.12 个百分点**。这是两套当前配置的比较，模型大小、输入预算、训练轮数及校准设置不同，不能把差异单独归因于决策头。复现时应复制配置并更换 `run_id`，已有输出目录会触发覆盖保护。

已整理 [ModernBERT / Laya 实验分析](../../reports/model-comparison/REPORT.md) （源码仓库不包含报告 ZIP）：包含全量基线、全量 CE、RLCD pilot 与 CE pilot 的对齐预测、统一概率指标、独立校准划分、配对检验与曲线。RLCD 尚无全量训练，报告明确保留该限制。

日志自动写入 `reports/laya-l512-s42-ce/console.log`，指标与状态也在该目录；权重写入 `checkpoints/laya-l512-s42-ce/`。首次加载后会先验证原始模型，再开始输出训练 step。保持电脑插电开盖；`caffeinate -i` 防止空闲休眠。当前 Laya 入口不支持精确断点续训。

完整说明、对照实验及全量配置见 [Laya 训练说明](training.md)。已有 ModernBERT 基线保留作对照。下方 Day1–Day3 命令仍可复现历史基线。

## Laya 同权重后端对照

已增加 `laya-mlx==0.3.0` 可选依赖，使用现有全量 CE 微调权重比较 MPS FP32、MLX FP32 与 MLX FP16。方案见 [后端对照设计](../archive/designs/laya-backend-design.md)，结果见 [推理对照报告](../../reports/laya-backend-comparison/REPORT.md)。这是推理实验，不改变现有 PyTorch 训练入口。

```bash
uv sync --locked --extra laya --extra metal --extra laya-mlx
uv run --offline --locked --extra laya --extra metal --extra laya-mlx python -u scripts/compare_laya_backends.py
```

验证只访问开发验证集。脚本逐条检查输入一致性，保存全部预测概率；测速按评论长度和 batch 分组。MLX 禁用 TF32，限制缓存为 2 GiB。首次安装额外依赖需要网络，之后本实验离线运行。

## uv 环境与运行

Python依赖统一用uv管理：`pyproject.toml`声明依赖，`uv.lock`锁定版本，`.python-version`固定本次验证使用的Python 3.12.14；`.venv`由uv自动维护。Rust部分继续使用Cargo和Cargo.lock。

```bash
cd /path/to/laya-imdb
uv sync --locked --extra metal
uv run --locked --extra metal python --version
```

日常运行统一使用 `uv run --locked --extra metal python ...`，无需手动激活环境。`--locked`要求依赖声明与锁文件一致，不自动修改锁文件。新增或移除依赖使用 `uv add`、`uv remove`；实验期间避免随意升级已锁定版本。跨框架核对时额外选择 `--extra reference`，安装PyTorch与Transformers。

用法依据：[uv项目管理](https://docs.astral.sh/uv/guides/projects/)、[锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)。

## 当前模型来源

当前基线采用[answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)的预训练编码器结构、权重和原配tokenizer，具体revision和文件哈希见 `configs/model_lock.json`。

我们参考公开结构和参数命名，在 `scripts/modernbert_mlx.py` 实现MLX编码器，并通过前向、梯度和padding核对；在编码器上添加自主实现的masked mean pooling与随机初始化的二分类线性头，再用IMDB标签微调整个模型。

这是现有预训练模型的自主MLX实现与任务微调，尚不是从零设计的新编码器架构。现阶段的自主工作包括数据划分、训练与评估、精度核对、恢复状态、Rust分词缓存及错误分析；领域分词适配与蒸馏属于后续实验。

## 自己训练与录屏

已提供 `configs/day3-recording.json`，与完整基线参数相同，使用独立run_id保存结果。开始录屏后执行：

```bash
caffeinate -i uv run --locked --extra metal python -u scripts/train_baseline.py \
  --config configs/day3-recording.json
```

本机完整三轮训练约65分钟，每100步输出进度、每500步保存恢复点。再次运行同一目录时，需要从最近的恢复点继续：

```bash
caffeinate -i uv run --locked --extra metal python -u scripts/train_baseline.py \
  --config configs/day3-recording.json --resume
```

训练完成后检查自己的模型：

```bash
uv run --locked --extra metal python scripts/predict.py \
  'The acting was excellent, and I loved the ending.' \
  --checkpoint checkpoints/modernbert-l256-s42-recording
```

该运行的权重在 `checkpoints/modernbert-l256-s42-recording/`，进度和逐轮指标在 `reports/modernbert-l256-s42-recording/`。原有基线结果保留在原目录。

作业的训练视频不超过5分钟；可录制真实完整过程，再剪辑或加速，保留启动、训练进度、验证指标和完成结果。视频时长不要求缩短正式训练。

## Day1 产出

- [实际结果与 Day2 接续](../../reports/day1.md)：本机环境、数据审计、分词长度及截断比例。
- [技术方案与实验协议](../archive/plans/day-01-design.md)：候选模型、最小实验矩阵、防泄漏和两周安排。
- [论文与官方依据](references.md)：采用理由、来源及适用限制。
- `reports/data_audit.json`、`reports/split_manifest.json`：数据统计、每条样本ID及文件哈希。
- `configs/tokenizer_lock.json`、`uv.lock`：原配tokenizer commit/哈希与Python依赖锁定。
- `reports/environment.json`：MLX Metal前向、梯度、参数更新与保存加载的合成验证记录。

Day1未训练 IMDB 分类器；Day2已完成下面的小规模训练和预算估算。
Day3已建立独立Rust CLI及带哈希的分词缓存；本轮基线沿用原Python接口，供后续受控接入。

## Day2 产出与复现

[Day2实际结果](../../reports/day2.md) 包含数值核对、八组训练测速、32步真实评论微调与服务器判断。
[实现与验证约定](../archive/plans/day-02-plan.md) 记录精度问题、失败诊断、工程验收标准及适用限制。

```bash
uv sync --locked --extra metal --extra reference
uv run --locked --extra metal python scripts/verify_data.py
uv run --locked --extra metal python scripts/download_model.py
uv run --locked --extra metal --extra reference python scripts/validate_modernbert.py
uv run --locked --extra metal --extra reference python scripts/debug_parity.py
uv run --locked --extra metal python scripts/benchmark_day2.py
uv run --locked --extra metal python scripts/pilot_day2.py
uv run --locked --extra metal python scripts/summarize_day2.py
uv run --locked --extra metal python scripts/predict.py 'The acting was excellent, and I loved the ending.' --checkpoint checkpoints/day2-pilot
```

首次额外下载约600MB ModernBERT原始预训练权重，版本与Day1 tokenizer一致，文件哈希见 `configs/model_lock.json`。
`reference`额外依赖仅用于跨框架核对；Day2训练本身使用MLX。加入Transformers后Tokenizers依赖锁定为0.22.2，Day1历史结果保留原版本记录；已检查原有示例token ID保持一致。
Day2脚本在导入MLX前默认 `MLX_ENABLE_TF32=0`；初始基线不使用M5默认降低精度的float32矩阵路径。数值验证会拒绝TF32开启，训练脚本会拒绝未通过或源码/配置已变更的核对记录。
调整模型实现或配置后，重新执行验证命令。运行需可访问Metal；不要并行启动GPU测速，以免互相干扰。

`checkpoints/day2-pilot/` 是256条训练评论的流程探测权重，可以手动推理，但不属于最终分类模型，也不包含精确续训需要的优化器状态。
推理脚本现在默认读取全量 Laya-CE；历史基线使用 `--model modernbert` 或显式指定 Day2/Day3 checkpoint。
Day2没有执行完整数据集训练、正式测试或证明泛化。原始失败检查保留于 `reports/day2_*diagnostic.json`。

## Day3 产出与复现

[Day3实际结果](../../reports/day3.md) 汇总完整基线、逐轮指标、最佳权重和验证集长度分层。
[训练与恢复协议](../archive/plans/day-03-plan.md) 说明学习率、择优、续训和Rust一致性检查。

```bash
uv sync --locked --extra metal --extra reference
uv run --locked --extra metal python scripts/validate_resume.py
caffeinate -i uv run --locked --extra metal python -u scripts/train_baseline.py
uv run --locked --extra metal python scripts/analyze_baseline_errors.py
uv run --locked --extra metal python scripts/benchmark_inference.py
uv run --locked --extra metal python scripts/predict.py 'The acting was excellent, and I loved the ending.' --model modernbert
```

`caffeinate`防止训练期间系统空闲睡眠。发生中断时用同一配置恢复：

```bash
caffeinate -i uv run --locked --extra metal python -u scripts/train_baseline.py --resume
```

已有运行目录时禁止直接覆盖；另做对照需复制配置并修改 `run_id`。
恢复要求训练相关源码、配置、数据清单和运行版本一致。每500步保存一次完整状态，最多重算上次保存后的步骤。
`checkpoints/modernbert-l256-s42/best.json` 指向验证择优权重，`latest.json` 指向续训状态；两者用途不同。
进度见 `reports/modernbert-l256-s42/status.json`，完成结果见同目录 `result.json`，每轮验证预测保留JSONL。

Rust工具使用原配tokenizer、固定Tokenizers 0.22.2，本机验证工具链为Rust 1.95.0。所有下载和构建缓存位于项目内：

```bash
CARGO_HOME="$PWD/.cache/cargo" CARGO_TARGET_DIR="$PWD/rust-tokenizer/target" cargo build --release --locked -j 2 --manifest-path rust-tokenizer/Cargo.toml
uv run --locked --extra metal python scripts/validate_rust_tokenizer.py
uv run --locked --extra metal python scripts/cache_tokens.py --split train --length 256
uv run --locked --extra metal python scripts/cache_tokens.py --split validation --length 256
uv run --locked --extra metal python scripts/summarize_day3.py
```

包装器只接受train/validation；逐条核对ID、label、token ID、mask及Unicode字符位置后，才发布JSONL与NumPy缓存。
缓存身份包含数据、tokenizer、Rust二进制和源码哈希，重复使用时检查内容哈希。长度512也可生成；官方测试仍不启用。
独立Rust工程不等于独立分词算法，Python接口本身也使用Rust库；本阶段没有端到端加速结论。

## 复现

在此文件所在的 `End-IMDB` 目录执行，推荐Python 3.12、Apple Silicon Mac：

```bash
uv sync --locked --extra metal
uv run --locked --extra metal python scripts/prepare_data.py
uv run --locked --extra metal python scripts/verify_data.py
uv run --locked --extra metal python scripts/analyze_tokenizers.py
uv run --locked --extra metal python scripts/check_environment.py
uv run --locked --extra metal python scripts/summarize_day1.py
uv run --locked --extra metal python -m unittest discover -s tests -v
```

首次需要联网下载约80MB官方数据及原配tokenizer；下载、数据与缓存均保留在项目目录。
`UV_CACHE_DIR` 可按本机写入权限设置，锁文件不依赖缓存位置。Metal验证必须在可访问GPU的本机终端运行。
不支持Metal的机器可执行数据脚本及测试，但不会通过GPU验证；今日未验证Linux/CUDA。

`prepare_data.py` 可用 `--archive /absolute/path/aclImdb_v1.tar.gz` 指定已有官方归档。
重跑会按配置重建划分；实验开始后不要随意修改配置或覆盖既有数据版本。训练前执行 `verify_data.py`。
`analyze_tokenizers.py` 沿用锁定revision，且在数据文件哈希变化时拒绝继续。

## 数据约定

训练22,500，验证2,500，官方测试25,000，标签0=负面、1=正面，各类别均衡。
train/validation按清理文本分组，相同文本不会跨侧。测试集仅作完整性审计，正式评估前不调参、不拟合词表。
官方训练与测试原有相同文本的数量已记录；不静默删除，不声称绝对无泄漏。
JSONL仅将 `text` 送入tokenizer；`id` 文件名中包含评分，严禁作为模型特征。

预处理保留否定词、大小写和标点，只替换HTML换行、解码实体、折叠空白。未使用额外无标签数据。

## 目录

```text
configs/      划分与分析配置、tokenizer锁定版本
docs/         设计、方法来源、实验协议
scripts/      可复现的数据、分析和环境验证命令
tests/        数据隔离与预处理关键测试
rust-tokenizer/ Rust分词CLI、Cargo.toml与Cargo.lock
reports/      审计JSON、划分清单、可读结果
data/         官方归档、派生JSONL（不纳入源码包）
artifacts/    环境验证权重（不属于正式分类模型）
.cache/       tokenizer缓存
checkpoints/  最佳模型及带优化器/RNG的续训状态
```

数据来源：[Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)。
模型、论文及第三方库的来源见阅读清单；后续适配时记录具体许可证和引用。
