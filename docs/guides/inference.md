# Mac 推理与服务器训练

默认部署使用已微调的 Laya-CE + MLX FP16；训练继续采用 PyTorch。模型权重、问题模板和原生校准温度沿用训练结果。

## 安装与推理

在项目目录执行；首次安装依赖需要联网，安装后推理完全离线：

```bash
cd /path/to/laya-imdb
uv sync --locked --extra laya-mlx
uv run --offline --locked --extra laya-mlx python scripts/predict.py \
  'The acting was excellent, and I loved the ending.'
```

无需安装 PyTorch 来运行这个推理入口。默认本地模型目录为 `checkpoints/laya-l512-s42-ce`，MLX cache 上限为 2 GiB，不启用编译和 prompt cache。

输出 `label`、原生温度校准后的 `probabilities`、未经校准的 `probabilities_uncalibrated`、实际输入长度及 `truncation`。概率保留计算精度，不采用上游展示接口的四位小数舍入。模型总输入预算为 512 tokens，包含问题和选项；超长影评只保留开头，是否截断以输出为准。

指定未来从服务器下载的 checkpoint：

```bash
uv run --offline --locked --extra laya-mlx python scripts/predict.py \
  'A disappointing movie with a weak script.' \
  --checkpoint checkpoints/my-cuda-run
```

`--checkpoint` 必须是存在的本地目录；不会回退下载公共预训练模型。目录需要完整包含：

```text
model.safetensors
rl_agent_config.json
encoder/config.json
tokenizer/  # 完整保存的 tokenizer 文件
```

上游 Laya 保存的 checkpoint 可直接由 laya-mlx 加载；不需要再做量化或权重转换。上述入口用于相同 ModernBERT 英文编码器、同一 IMDB 二选一问题模板和 negative/positive 选项顺序。以后若更换模型结构、模板或标签，需要同步调整并重新验证。

排查数值问题时可以增加 `--dtype float32`，但当前默认选择 FP16 来获得本机实测加速。

## 保留的基线入口

```bash
uv run --offline --locked --extra metal python scripts/predict.py \
  'The acting was excellent, and I loved the ending.' --model modernbert
```

也可以继续传入历史基线 checkpoint 路径，脚本会根据目录中的 `run.json` 识别为 ModernBERT。基线概率仍输出为未经校准的概率，保持原实验口径。

## 未来 CUDA 微调

训练继续使用 `scripts/train_laya.py` 与官方 PyTorch 训练器。服务器租好后再根据 GPU 显存选择 micro-batch、梯度累积和混合精度；当前尚未验证实际 CUDA 机器或测量其训练速度。

迁移流程：

1. 服务器安装 uv，使用训练 extra `laya`，确认 PyTorch CUDA 可用。Mac 推理 extra `laya-mlx` 不用于 CUDA 训练。
2. 同步代码、锁文件、固定数据划分与 manifest，以及原始 Laya 本地模型缓存；保持现有问题模板、tokenizer 和标签顺序。
3. 复制训练配置，设置新的 `run_id` 与 `device: "cuda"`，按服务器显存调整 batch。先做短流程验证，再由用户开始正式训练与录屏。现有 Mac 训练配置继续保留。
4. 训练完成后取回完整 checkpoint 目录，以及报告、配置和日志。CUDA 优化器状态无需参与 Mac 推理。
5. 在 Mac 重新核对 CUDA/PyTorch 与 MLX FP16 的相同验证样本预测和概率，之后将新目录用于 `--checkpoint`。

已训练权重可迁移，不代表不同 GPU、精度和 batch 的训练轨迹完全一致。新训练结果需要独立记录，最终测试集仍只用于冻结方案后的正式评估。

当前后端实验见 [推理对照报告](../../reports/laya-backend-comparison/REPORT.md)。
