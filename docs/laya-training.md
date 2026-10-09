# Laya 领域微调与离线运行

采用已讨论并获同意的路线：基于开源 Laya 英文模型做 IMDB 领域微调，以原有 ModernBERT 基线作参考，再通过受控实验判断方法是否有效。Jev 仅作产品思路参考，不调用其闭源服务。

## 模型与实现边界

模型是 [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)，固定 revision 为 `7b928d828b7b0e022f929d9bd2e44165aa270148`；编码器为 ModernBERT-large，另含已有的 typed decision head。上游代码为 [Laya](https://github.com/NandhaKishorM/laya)，uv 锁定 PyPI `laya==0.4.1`。并非从零发明新的基础编码器。

当前 `scripts/train_laya.py` 实现 IMDB→typed choice 数据适配、冻结划分检查、离线保护、运行隔离与日志记录；损失、反向传播、优化器、温度校准和模型保存调用上游训练器。为了控制 MPS 内存，通过适配器将其评估 batch 从默认 16 改为 1。自主方法改进的效果还需后续对照证明，不能仅凭换用新模型声称创新有效。

## 当前选择：全量 CE 训练

用户已选择 CE 作为后续训练方法。下一轮使用 `configs/laya-full-ce.json`，从锁定的原始 Laya 权重开始，进行一轮全量训练；不从 pilot 权重继续，以免 pilot 样本被重复学习。默认命令和省略 `--config` 的行为均指向此配置：

```bash
cd /path/to/laya-imdb
caffeinate -i uv run --offline --locked --extra laya python -u scripts/train_laya.py \
  --config configs/laya-full-ce.json
```

固定训练划分共 22,500 条：400 条只作校准，22,100 条参与梯度训练；完整验证 2,500 条，测试集不读取。训练一轮，共 2,763 次参数更新。其余学习率、batch、长度与 CE pilot 相同，余弦学习率按新的更新总数重新规划。

输出目录为 `checkpoints/laya-l512-s42-ce/` 与 `reports/laya-l512-s42-ce/`。正式训练已由用户执行完成，训练结束报告的完整验证准确率为 95.76%，校准后 NLL=0.1233、ECE=0.0168、Brier=0.0675，温度为 1.4808，无校准边界警告。原始 Laya 在同一验证集上的准确率为 92.32%。当前上游训练器只能在每轮结束保存模型，不支持精确中断恢复。重新复现应复制配置并修改 run_id。

此前 ModernBERT-base 基线在同一 2,500 条验证样本上的准确率为 94.64%。两套配置差 1.12 个百分点；两者模型规模、输入预算、训练轮数与校准设置不同，属于方案比较，不能单独证明某个结构设计更优。基线 NLL 未校准，Laya 的上述 NLL 已校准，不作为未经控制的结构优劣证据。模型重载验证与逐样本对比保存到 `reports/laya-l512-s42-ce/saved_checkpoint_validation.json`。

## 已完成的先导实验与复现

```bash
cd /path/to/laya-imdb
caffeinate -i uv run --offline --locked --extra laya python -u scripts/train_laya.py \
  --config configs/laya-pilot-rlcd.json
```

首轮 1,024 条按标签平衡抽样，128 条从中留出做温度校准、896 条用于参数学习；另取 256 条平衡验证样本。两类监督输出是 negative/positive，包含选项的语义说明。训练一轮，共 112 次 optimizer 更新，有效 batch 为 8（micro-batch 1 × 梯度累积 8）。全编码器与决策头均参与微调，float32、MPS、梯度检查点，encoder lr=2.5e-5、head lr=1e-4。总长度 512 包含问题与选项，因此实际影评预算小于 512，不可直接视为基线的 512 个纯影评 token。

编码器与决策头使用不同学习率；上游使用 cosine 衰减，无 warmup。它与原有 ModernBERT 基线的模型规模、长度和优化器设置不同，先导成绩不能归因于 RLCD 单一因素。

先加载本地权重并评估原模型，然后开始训练，每 16 个 micro-step 打印 loss。RLCD 的奖励项允许负 loss，训练过程中应用同一目标观察趋势。训练完成后评估微调模型，打印 Accuracy、NLL、ECE 与 Brier 的前后变化；目前上游结果不包含 Macro-F1，不能将 Accuracy 字段冒充 Macro-F1。

输出包括：

- `reports/laya-l512-s42-pilot-rlcd/console.log`：完整控制台日志。
- 同目录 `manifest.json`：抽样 ID、梯度训练/校准 ID、源数据与模型哈希、版本、配置。
- 同目录 `status.json` 与 `result.json`：完成状态、耗时和指标。
- `checkpoints/laya-l512-s42-pilot-rlcd/`：最终模型、tokenizer、问题定义及上游训练报告。

测试集不读取、不训练、不用于调参。原训练/验证划分的文件哈希必须匹配 Day1 manifest。校准样本不参与梯度学习，验证样本不用于温度拟合。首轮验证子集属于开发验证集，后续仍需完整验证和最终测试。

## 对照与后续全量配置

soft-CE 对照保持同一模型、数据、随机种子、长度、训练预算与选项顺序增强，仅改变目标函数及输出目录：

```bash
caffeinate -i uv run --offline --locked --extra laya python -u scripts/train_laya.py \
  --config configs/laya-pilot-ce.json
```

RLCD 全量配置 `configs/laya-full-rlcd.json` 保留作后续对照，当前主线使用上方 CE 配置。两种全量配置均训练一轮，固定训练划分为 22,500 条（400 条校准、22,100 条梯度训练），独立完整验证 2,500 条。仅在需要另做 RLCD 全量对照时使用：

```bash
caffeinate -i uv run --offline --locked --extra laya python -u scripts/train_laya.py \
  --config configs/laya-full-rlcd.json
```

两轮 pilot 已由用户完成：RLCD 334.4 秒，CE 347.5 秒；两者验证准确率都是 93.75%。训练结束报告中的 CE NLL=0.1830、Brier=0.1038，RLCD NLL=0.2191、Brier=0.1088；CE 温度为 1.361，RLCD 为 5.0 且触及上限。保存重载核对后准确率均一致，CE 的 Brier 因 fp16 舍入为 0.1039。结果支持当前优先选择 CE，但单种子、256 条验证不足以断言普遍优势。

后续进行完整验证、检查选项顺序敏感性及校准；多种子、蒸馏和自适应长度属于后续阶段。历史 pilot 输出目录已经存在，直接重跑会被覆盖保护拒绝；复现应复制配置并选择新的 run_id。

## 断网、录屏与中断

依赖、Python、数据、权重、encoder config 与 tokenizer 均已在这台电脑准备。上述 `uv run --offline --locked` 命令禁止 uv 联网解析/下载；脚本设置 Hugging Face 离线标志，使用绝对本地模型目录，并拦截 Python IPv4/IPv6 connect/sendto。无需 API key、Jev 服务或远程 GPU。在另一台电脑复现时，首次安装与缓存下载仍需联网：`uv sync --locked --extra laya` 后运行 `uv run --locked --extra laya python scripts/prepare_laya.py`；数据需另外准备。

可先只检查缓存、依赖、数据与 MPS，不启动训练：

```bash
uv run --offline --locked --extra laya python scripts/train_laya.py --check-only
```

当前 Laya 上游训练器在每轮结束保存 `checkpoint_latest`，不保存完整 optimizer/RNG 状态，**不支持基线训练脚本那种精确断点续训**。这次 pilot 只有一轮，中途停止可能丢失本轮进度；录屏期间保持插电、开盖。重新试验请复制配置并改变 `run_id`，程序拒绝覆盖已有运行。不要给它加基线脚本的 `--resume` 参数。

上游保存权重为 fp16，训练为 float32，因此保存重载存在量化舍入；训练结束的评估是内存中 float32 模型，交付前还需核对已保存模型。录屏建议保留启动、loss 进度、验证比较表和完成路径，最终按课程要求剪辑/加速至 5 分钟内。

## 已完成的准备验证

RLCD 与 soft-CE 各进行 8 条训练源样本、4 条验证样本的离线 MPS 冒烟测试：内部 2 条校准、6 条梯度训练，完整模型各执行 3 次参数更新，保存与验证成功。2 条校准不足以拟合温度，上游按预期警告；这不是训练失败，也不代表正式模型效果。实际记录在 `reports/laya-smoke-rlcd/` 和 `reports/laya-smoke-soft-ce/`。

另已离线重载两种保存权重，检查全部参数为有限数，并在对应 4 条样本上核对保存前后分类一致；记录见 `reports/laya_preflight.json`。该检查不能替代正式训练后的完整保存重载验证。模型实际参数量为 421,293,827。原有 6 项测试通过，抽样可复现性、标签平衡、校准互斥及 CE/RLCD 配置仅目标不同的检查也通过。

离线保护是本项目 Python 路径与库配置的约束，不是操作系统防火墙。验证过程中未要求关闭整台电脑的 Wi-Fi。
