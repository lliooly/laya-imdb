# CUDA 训练前的本地准备

日期：2026-10-10。用户尚未租用服务器，本阶段完成本地分析、输入核验和冒烟配置；CUDA 计算与跨设备核对仍待真实设备验证。

本文件保留租机前的准备记录；后续实际执行结果见 [CUDA 验证与测速](../../reports/cuda-preparation-v1/REPORT.md)。复验需使用新的运行编号，原始 transfer manifest 是初次交付快照，不代表修改后的源码哈希。

## 已完成

- [Laya-CE 错误分析](../../reports/laya-ce-error-analysis-v1/REPORT.md)：完整统计 106 条错误，另记录 16 条定性抽查。
- `scripts/check_cuda_environment.py`：核对原始 Laya 锁定文件、train/validation 哈希、依赖版本、冒烟配置和 CUDA 可用性，不加载权重、不训练、不读取测试集。
- `scripts/train_laya.py --check-only`：验证训练配置，并在请求的 CUDA 不可用或设备索引越界时拒绝继续，不再错误报告预检成功。
- 两份独立配置：`configs/laya-cuda-smoke-ce-fp32.json` 和 `configs/laya-cuda-smoke-ce-amp-fp16.json`，使用独立 run_id，均从锁定的原始 Laya 权重开始。
- 15 项 CPU 单元测试通过；真实 Mac 预检明确拒绝 CUDA 请求，并且未创建训练输出目录。

本机 PyTorch 没有 CUDA 构建，`cuda_available=false` 是预期状态，不是服务器兼容性结论。原始权重与冻结 train/validation 的 7 项文件哈希检查通过，记录见 `reports/local-preparation-v1/readiness.json`。

## 上游训练器的实际限制

当前锁定 `laya==0.4.1`。已检查本地安装源码：`amp=true` 的训练前向显式使用 `torch.float16` autocast；CUDA 使用 GradScaler。checkpoint 中的 `amp_dtype: bf16` 不决定该训练函数的实际精度，不能将当前 AMP 方案描述为 BF16 训练。

训练器使用 cosine 调度，尚无 warmup 配置字段。当前按轮保存模型，不保存足以精确恢复的 optimizer/RNG 状态。后续正式训练前还需检查非有限损失、梯度、参数、AMP 跳过更新与调度行为；未完成这些检查前，不把现有入口称为可靠的 CUDA 正式训练方案。

## 服务器需要的材料

同步源码、`pyproject.toml`、`uv.lock`、配置和模型锁文件、`reports/split_manifest.json`，以及以下冻结文件：

```text
data/processed/train.jsonl
data/processed/validation.jsonl
.cache/laya/english/model.safetensors
.cache/laya/english/rl_agent_config.json
.cache/laya/english/encoder/config.json
.cache/laya/english/tokenizer/tokenizer.json
.cache/laya/english/tokenizer/tokenizer_config.json
```

使用 `reports/local-preparation-v1/transfer_manifest.json` 核对传输清单与哈希；不需要传输 `.venv`、整个缓存目录、历史权重或官方测试数据。manifest 中的测试 ID/哈希仅为现有审计元数据，训练脚本只读取 train/validation 文件。

服务器上首次安装依赖需要网络。使用 `uv sync --locked --extra laya` 后确认安装的是 CUDA 可用的 PyTorch。Mac 的环境检查不证明 Linux 的锁定依赖、驱动或 GPU 已兼容；若锁定组合无法安装，先记录具体问题，再决定依赖变更并重新验证。

## 真实设备上的顺序

所有命令从仓库根目录运行。首先只检查输入和设备：

```bash
uv run --offline --locked --extra laya python scripts/check_cuda_environment.py \
  --out reports/cuda-preparation-v1/readiness.json
uv run --offline --locked --extra laya python scripts/train_laya.py \
  --config configs/laya-cuda-smoke-ce-fp32.json --check-only
```

预检通过仅说明文件、配置与设备可访问；还不证明前向、梯度或训练稳定。CUDA 可访问且付费训练预算已记录后，顺序执行 FP32、再执行 AMP FP16 冒烟：

```bash
uv run --offline --locked --extra laya python -u scripts/train_laya.py \
  --config configs/laya-cuda-smoke-ce-fp32.json
uv run --offline --locked --extra laya python -u scripts/train_laya.py \
  --config configs/laya-cuda-smoke-ce-amp-fp16.json
```

每份配置从 8 条训练源样本中留出 2 条校准，6 条做梯度训练，另取 4 条验证；micro-batch 1、累积 2，共 3 次更新。2 条校准不足以可靠拟合温度，相关警告是预期的；该运行仅检查链路，不用于准确率判断，也不能外推正式训练速度。

已有目录会拒绝覆盖，复验需复制配置并修改 run_id。两份配置的差异限定为 AMP 和 run_id；在分别从相同原始权重开始的条件下检查数值有限性和保存重载，而不是要求两条训练轨迹完全一致。

接着核对保存重载、相同验证输入及 Mac MLX FP16 的预测/概率，测量参数更新与完整流程耗时、峰值显存，再制定正式 batch、轮数及费用预算。CUDA 回传权重的跨后端验证仍需执行；历史 `compare_laya_backends.py` 固定了原 Mac CE checkpoint，不能直接当作新 CUDA 权重的验证命令。

## 下一轮研究的约束

先建立同平台 CE 参照，再筛选训练策略。长文本路线用相同总 token 预算比较前缀与首尾保留；当前 24 条截断错误不足以单独补齐达到 98% 所需的 56 条净改正。

正式实验继续使用固定 22,100 条梯度训练、400 条校准、2,500 条开发验证。每次使用新 run_id 并保留失败记录；先验证实现再运行付费实验。CUDA 闭环预算最多 20 元，后续遵守已确认路线的总预算，不因目标未达而无限追加实验。
