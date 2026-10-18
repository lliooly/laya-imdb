# 本地限定算力下的精度优化：可行性与实验计划

日期：2026-10-18（北京时间）。作者：experiment-feasibility planner。

**范围声明（先读）**：本文只做只读盘点与计划，**没有启动任何训练、没有执行新的开发/测试推理、没有读取官方测试正文**，只读取了仓库内已归档的报告、缓存 JSON 元数据、配置与日志，以及本机 `sysctl`/`df` 硬件信息。除本文件外未修改任何文件。所有时间/成本要么给出报告或日志中的实测值，要么明确标注为"外推估计"。数字来源统一列在第 8 节。

本计划与同目录的 `headroom-analysis.md`（任务 1，量化余量）和 `literature.md`（任务 2，文献依据）互补；本文件不重复它们的逐样本分析，只在需要时引用其结论位。

---

## 0. 结论摘要

1. **付费算力不可用（已核实）**：用户 2026-10-17 21:30 确认余额 7.03 美元（[余额记录](../../delivery/deberta-balance-confirmed-7p03-20261017.json)）；随后 DeBERTa 两轮 budget7 正式训练完成，应用生命周期 7,974 秒，**按保守上界计费约 5.60 美元**（[ACCEPTANCE.md](../deberta-heterogeneous-bf16-budget7-s42-20261017-v1/ACCEPTANCE.md)），账本状态 `completed_verified_local`、`compute_closed=true`（[budget json](../../delivery/deberta-heterogeneous-bf16-budget7-s42-20261017-v1-budget.json)）。因此**最坏情形下的余额下界为 7.03 − 5.60 = 1.43 美元**（5.60 是花费的保守上界，实际余额只可能更高；无实际账单可查，见 1.1）。而一次 Laya 两轮 L40S 训练外推约 2.0–2.6 美元、一次 DeBERTa 两轮约 5.6–9.0 美元。→ **按最坏情形余额下界 1.43 美元判断，付不起一次正式两轮训练；实际余额可能更高但没有账单证据，在用户确认余额/追加预算前，任何新付费 GPU 都不可行**。
2. **本地资产比预期多**：`checkpoints/` + `artifacts/remote-*-backup/` 合计有 **20 个以上完整 Laya 检查点**（含 s42/s43 多配方、6 个稳健损失模型、512/2048 上下文对、翻标数据模型）、**DeBERTa-v3-large 2 个完整 epoch + 4 个周期快照**、ModernBERT-base 3 个 epoch；大部分模型已有开发集（2500）与校准集（400）概率缓存。**零训练 ensemble/TTA/stacking 可以在本机几小时内离线完成**（第 2、6 节）。
3. **本地训练实测结论**：Laya 全参 2 轮微调在本机 M5 Pro MPS **实测 4.04 小时**（14,547 秒，[status.json](../laya-mps-ce-epochs2-l512-s42-fp32-v1/status.json)）；DeBERTa-v3-large 2 轮本地外推 **10–13 小时**（按 L40S 实测 63.0 分钟/轮 × 本机/L40S 慢 4.93 倍外推）；RoBERTa-large、1–3B 解码器 LoRA 在本机要么权重不在本地（需联网）、要么 10–40 小时，性价比与可行性都不足（第 3.2 节）。
4. **98% 目标判断（全覆盖）**：开发集 98% = 2450/2500，当前探索最高 2435/2500（97.40%），**还差 15 条净修正**。任务 1 的独立复核（[headroom-analysis.md](headroom-analysis.md) §3–§6，[verification.md](verification.md)）给出三点关键事实：(i) 57 个去重视角的标签知情 oracle 为 **2481/2500 = 99.24%**（去掉事后组合后的 46 个单模型/单分支视角仍是 2481），68 条融合错误中 **49 条可被至少一个现有视角救回、19 条全错**——因此从 2435 到 2450 只需捕获 **15/46 = 32.6%** 的可修复错误（按 2432 的 68 错口径为 **18/49 = 36.7%**），并非此前假设的高比例；(ii) 但**所有只用 400 校准拟合的可部署规则在开发集上都到顶于 2435**（等权/ridge/多数/门控全部 ≤2435，最好 +3 条、精确 McNemar p=0.25、条件 bootstrap 95% 下限 0）；(iii) "Laya 主融合 + DeBERTa"两视角 oracle 只有 **2447/2500 = 97.88%**（[ACCEPTANCE.md](../deberta-heterogeneous-bf16-budget7-s42-20261017-v1/ACCEPTANCE.md)），**在现有两个模型之间做完美选择也到不了开发 98%**。结论：瓶颈不是后处理或权重搜索，而是缺少**新的独立模型/训练信号**；本地限定预算下现实最好情形仍是开发 97.5–97.8%、官方测试 97.1–97.5%，98% 全覆盖应判定为不可达（第 4.4 节）。
5. **推荐**：零训练缓存集成审核已由任务 1 实质执行（57 视角、全部 400 校准规则 ≤2435、判断为饱和），因此 P0-1 现在只是"用新 run_id 复现并封存门槛"，按已知结果预计不会通过 97.50% 门槛。**默认建议按第 5.2 节零新增算力收尾**；只有用户明确要求留正式 run 留痕、或愿意接受 1–1.5 小时 DeBERTa 快照/TTA 探索时，才按第 5.1 节执行，且门槛不通过即停。任何情况下都不应再按开发集反复搜索，也不应把 98% 写成承诺。

---

## 1. 预算与约束核查

### 1.1 Modal 余额（核实过程与假设）

| 项 | 数值 | 来源 |
|---|---|---|
| 用户确认余额（2026-10-17 21:30） | 7.03 USD | [deberta-balance-confirmed-7p03-20261017.json](../../delivery/deberta-balance-confirmed-7p03-20261017.json) |
| budget7 两轮 DeBERTa 应用生命周期 | 7,974 秒 ≈ 2 小时 13 分 | [ACCEPTANCE.md](../deberta-heterogeneous-bf16-budget7-s42-20261017-v1/ACCEPTANCE.md) |
| budget7 本次保守计算上界 | 5.60 USD（非平台账单） | 同上 |
| budget7 账本状态 | `completed_verified_local`、`compute_closed=true` | [budget json](../../delivery/deberta-heterogeneous-bf16-budget7-s42-20261017-v1-budget.json) |
| 账本"预留 7 美元后剩余" | 0.03 USD | 同上（注意这是**预算封顶口径**，不是余额口径） |
| 本计划采用的余额假设 | **最坏情形余额下界 ≥ 1.43 USD**（7.03 − 5.60）；实际余额可能更高但无账单 | 推导，见下 |

推导：`delivery/` 与 `reports/` 均未包含 Modal 实际发票（[modal-budget.json](../../delivery/modal-budget.json) 明确 `budget_not_invoice`，budget7 同样注明"不是平台实际账单"）。5.60 是**花费的保守上界**，故实际花费 ≤5.60、**余额 ≥1.43**（最坏情形下界；若实际扣费更低则余额更高，但没有账单可作依据）。按任务硬约束"未经用户明确批准视为不可用"，本计划一律把新付费 GPU 视为不可用。

对照新训练成本（外推，见 3.2）：Laya 两轮 L40S ≈ 2,951 秒 ×（0.00066544 + 0.0000182325）USD/秒 ≈ 2.02 USD，加启动/清理余量约 **2.0–2.6 USD**；DeBERTa 两轮按实测上界 **5.6 USD**（保守公式 9.03 USD，[余额记录](../../delivery/deberta-balance-confirmed-7p03-20261017.json)）。两者都**高于最坏情形余额下界 1.43 USD**（Laya 2.0–2.6 亦高于该下界）。→ **最坏情形下 1.43 美元不够任何一次正式两轮训练；实际余额可能更高，但在用户确认/账单出现前仍按"无付费算力"处理。**

### 1.2 本地硬件与环境（本会话实测）

| 项 | 观测值 | 命令 |
|---|---|---|
| CPU | Apple M5 Pro | `sysctl -n machdep.cpu.brand_string` |
| 内存 | 68,719,476,736 B = 64 GiB | `sysctl -n hw.memsize` |
| 磁盘可用 | 82 GiB（926 GiB 卷已用 815 GiB） | `df -h .` |
| 运行时 | PyTorch（MPS/CPU）+ MLX 已装在 `.venv`；`environments/bow-cpu` 为 CPU 锁定环境 | 仓库文档与历史报告 |
| 离线 | 依赖、模型、数据已本地就绪；历史命令均用 `uv --locked --offline` | [training.md](../../docs/guides/training.md) |
| 本地模型缓存 | Laya（807 MB）、ModernBERT-base、DeBERTa-v3-large（836 MB）、bert-base-uncased；**没有 RoBERTa / 解码器 LLM** | `.cache/` 盘点 |

注意：本会话直接运行 `uv` 曾因沙箱无法写 `~/.cache/uv` 失败。执行实验时应沿用历史报告的写法：`UV_CACHE_DIR=/private/tmp/<name>-uv-cache uv run --locked --offline --no-sync ...`（[bow-nb REPORT](../bow-nb-complementarity-cpu-20261018-v1/REPORT.md) 与 [token-query REPORT](../laya-token-query-mps-fp32-20261018-v1/REPORT.md) 均这样运行）。

### 1.3 冻结数据与评估协议（不可变）

- 划分：原训练 22,500 = 梯度训练 22,100 + 校准 400；开发 2,500；官方测试 25,000（冻结、历史上已用过一次正式评估，**不是全项目未见数据**）。
- 校准 400 只用于选超参数/拟合温度与融合权重；开发 2,500 只做最终对照；**不得用开发集拟合、不得用测试集调参**。
- 新实验必须新 `run_id`、新目录、不覆盖历史报告（[AGENTS.md](../../AGENTS.md)、[training.md](../../docs/guides/training.md)）。
- 官方测试只允许"先冻结候选、后一次性评估"，且必须披露其非独立性质（[收尾记录](../../docs/experiments/2026-10-18-accuracy-search-closeout.md)）。

### 1.4 本任务的执行边界

- 未启动训练、未新增推理、未调用 Modal/GPU/新服务。
- 未修改本文件以外的任何文件；同目录 `headroom-analysis.md`、`literature.md` 由队友撰写，本计划不触碰。
- 未读取官方测试正文；仅引用已归档的聚合指标与缓存文件元数据。

---

## 2. 本地可用资产清单

### 2.1 可加载的完整权重（含已知开发集 Accuracy）

开发 Acc 一律为 2,500 条全覆盖、历史报告口径；不同训练/推理后端不可直接当作单因素收益。

| # | 模型/检查点路径 | 配方/输入 | 开发 Acc | 权重是否本机 | 证据 |
|---|---|---|---:|---|---|
| W1 | `checkpoints/laya-mps-ce-epochs2-l512-s42-fp32-v1` | Laya CE，Mac MPS FP32，2 轮，512 首尾 | **96.96%** | 是（0.84 GB） | [all-metrics M005](../all-experiments-metrics-20261016-v1/REPORT.md) |
| W2 | `checkpoints/laya-mps-ce-epochs2-lr06-l512-s42-fp32-v1` | 同上，LR×0.6 | 96.12% | 是 | [all-metrics M006](../all-experiments-metrics-20261016-v1/REPORT.md) |
| W3 | `checkpoints/laya-l512-s42-ce` | Laya CE，MPS FP32，1 轮 | 95.76% | 是 | [all-metrics M002](../all-experiments-metrics-20261016-v1/REPORT.md) |
| W4 | `checkpoints/laya-cuda-reference-ce-l512-s42-fp32-v1` | Laya CE，CUDA FP32，1 轮 | 96.28% | 是 | [all-metrics M007](../all-experiments-metrics-20261016-v1/REPORT.md) |
| W5 | `checkpoints/laya-rtx4090-ce-epochs2-l512-s42-fp32-v1` | Laya CE，4090，2 轮 | 95.88% | 是 | [all-metrics M008](../all-experiments-metrics-20261016-v1/REPORT.md) |
| W6 | `checkpoints/laya-context-pair-s42-modal-20261016-v1/context-512-s42` | Laya CE，L40S，2 轮，512 | 96.32% | 是（9.4 GB 含优化器） | [context-pair evaluation](../laya-context-pair-s42-modal-20261016-v1/evaluation/REPORT.md) |
| W7 | 同目录 `context-2048-s42` | 同上，2048 上下文 | 96.32% | 是（9.4 GB） | [length-router](../laya-length-router-20261017-v1/REPORT.md) |
| W8 | `checkpoints/laya-reviewed-data-s42-modal-20261017-v1/context-512-s42` | 用户翻标 v2 数据，2 轮 | 96.40% | 是（9.4 GB） | [reviewed-data evaluation](../laya-reviewed-data-s42-modal-20261017-v1/evaluation/REPORT.md) |
| W9-W11 | `checkpoints/laya-robust-loss-modal-deferred-ce-s{42,43,44}-strict-fp32-20261015-v2/epoch-002` | CE 同期控制，3 seed | 96.24 / 96.16 / 96.56% | 是（每个 6.4 GB 含优化器） | [robust-loss 分析](../laya-robust-loss-analysis-20261016-v1/REPORT.md) |
| W12-W14 | `checkpoints/laya-robust-loss-modal-deferred-mixed-s{42,43,44}-.../epoch-002` | 0.5CE+0.5GCE，3 seed | 96.40 / 96.16 / 96.64% | 是（每个 6.4 GB） | 同上 |
| W15 | `artifacts/remote-backup-20261011/checkpoints/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1` | 基线 s43，2 轮 | 96.48% | 是（0.84 GB） | [all-metrics M013](../all-experiments-metrics-20261016-v1/REPORT.md) |
| W16 | `artifacts/remote-backup-20261011/checkpoints/laya-rtx4090-ce-combined-s43-epochs2-fp32-v1` | warmup+LS s43，2 轮 | 96.48% | 是 | [all-metrics M014](../all-experiments-metrics-20261016-v1/REPORT.md) |
| W17-W21 | `artifacts/remote-backup-20261011/checkpoints/laya-rtx4090-ce-epochs2-{l512,ls005,warmup005,warmup005-ls005,warmup005-ls005-rdrop1}-...` | s42 五种正则配方 | 95.88 / 96.16 / 96.32 / 96.44 / 96.08% | 是（各 0.84 GB） | [all-metrics M008-M012](../all-experiments-metrics-20261016-v1/REPORT.md) |
| W22-W24 | `artifacts/remote-cleanup-backup-20261014-v1/checkpoints/semantic30-abc-s42-strict-20261013-v1-{A,B,C}` | 语义增强 A/B/C，2 轮 | 96.52 / 96.32 / 96.44% | 是（各 0.84 GB） | [all-metrics M019-M021](../all-experiments-metrics-20261016-v1/REPORT.md) |
| W25 | `checkpoints/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/epoch-002` | DeBERTa-v3-large，2 轮，512 首尾，BF16 前向 | **97.04%** | 是（1.74 GB） | [budget7 evaluation](../deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/REPORT.md) |
| W26 | 同目录 `epoch-001` | 同上第 1 轮权重 | **未评估**（新信息） | 是（1.74 GB） | 本计划建议 |
| W27-W30 | 同目录 `periodic/periodic-epoch-00{1,2}-step-0{256,512}` | 训练中期快照 | **未评估**（新信息） | 是（各 1.74 GB） | 同上 |
| W31 | `checkpoints/modernbert-l256-s42/best-epoch3.safetensors` | ModernBERT-base，256 token，3 轮 | 94.64%（e1/e2 为 94.16/94.48） | 是 | [all-metrics M001](../all-experiments-metrics-20261016-v1/REPORT.md) |
| W32 | `checkpoints/laya-l512-s42-pilot-ce` / `-pilot-rlcd` | 小预算 pilot | 93.52 / 93.72% | 是 | [all-metrics M003/M004](../all-experiments-metrics-20261016-v1/REPORT.md) |

不可用于集成/评估的"伪检查点"（**必须排除**）：
- `checkpoints/laya-head-control-original-s42-strict-fp32-20261015-v2`：空目录（0 B）。
- `checkpoints/laya-decision-window-*`（96.64/96.44%）、`laya-checkpoint-average-prefix-s42-*`（96.64%）：**只有 config/tokenizer/预测，没有 `model.safetensors`**。
- `checkpoints/laya-long-context-pilot-20261016-v1/pilot-2048`、两个 DeBERTa `pilot-boundary`、`laya-smoke-*`、`laya-*-admission-r1`：3 步工程准入快照，不是训练完成的模型，不可当成员或成绩。
- 任务 1 已确认 L2 头部对照 12 份权重未回传（[head-control 交付缺口](../laya-head-control-delivery-status-20261015-v1/REPORT.md)），不要再尝试恢复。

### 2.2 开发/校准预测缓存（零训练候选的原料）

| 缓存 | 覆盖 | 内容 | 路径 |
|---|---|---|---|
| Laya 主融合 + DeBERTa 概率 | 400 校准 + 2,500 开发 | `neural`（Laya 内部融合后概率）、DeBERTa 概率、逐样本 ID/标签 | `reports/laya-deberta-calibration-weight-cpu-20261018-v1/{calibration,development}-predictions.json` |
| DeBERTa 概率/温度 | 400 + 2,500 | 原始与校准概率、温度 1.137497 | `reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/{calibration,development}.json` |
| Laya 512/2048 分支 | 400 + 2,500 | 各分支温度校准概率 | `reports/laya-context-pair-s42-modal-20261016-v1/evaluation/{512,2048}-{calibration,development}.json` |
| 翻标数据模型 | 400 + 2,500 | 512 概率 | `reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation/512-{calibration,development}.json` |
| 长文 B/C 融合输入 | 2,500 | Laya/2048/旧融合概率与长文掩码 | `reports/laya-long-fusion-bc-20261017-v1/predictions.json` |
| 稳健损失 6 模型 | 2,500（校准需按 2.4 补齐） | 各模型逐样本概率/指标 | `reports/laya-robust-loss-analysis-20261016-v1/{summary.json,comparison.png}`、各 `checkpoints/.../epoch-002/` |
| 4090 各配方模型 | 2,500 | `validation_predictions`、原生温度 | `reports/laya-rtx4090-*/cuda-saved-validation*.json`（每模型约 0.2–0.4 MB） |
| 固定权重多视角 | 400 | `head`、`head_tail`、`reverse` 概率 | `reports/laya-posthoc-{baseline,combined}-s{42,43}-fp32-v1/calibration-*.json` |
| 语义 A/B/C | 2,500 | 逐样本结果 | `reports/semantic30-abc-completed-20261013-v1/` |
| 词袋/NB 分支 | 400 + 2,500 | TF-IDF/NB 概率与融合 | `reports/bow-nb-complementarity-cpu-20261018-v1/{calibration,development}-predictions.json` |
| 冻结表示头 | 400 + 2,500 | 1024 维特征 + 两个 MLP 头 | `artifacts/laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1/{calibration,validation}.npz`、`artifacts/final-cpu-alternatives-20261018-v2/` |
| token 查询头 | 2,500 | 均匀/稠密/Top32 头输出 | `reports/laya-token-query-mps-fp32-20261018-v1/development-{heads,probabilities}.json` |
| **官方测试概率（仅冻结候选读数用）** | 25,000 | DeBERTa CPU FP32 logits；Laya 主融合/两模型概率 | `reports/deberta-laya-final-cpu-evaluation-20261018-v1/test-predictions.json`、`reports/laya-frozen-official-test-20261014-v1/test-actual.json` |

说明：上表"官方测试缓存"只允许**检验已冻结候选**（见 4.1 P0-4 的治理条件），不得用于选择成员/权重/阈值。

### 2.3 Token / 表示缓存（决定"还能做什么零训练工作"）

| 缓存 | 规模 | 内容 | 路径 |
|---|---|---|---|
| Laya baseline-s43 全 token 表示 | 27.25 GB + 3.13 GB + 0.50 GB | 22,100 / 2,500 / 400 条的逐 token 1024 维 FP32 | `artifacts/laya-token-query-mps-fp32-20261018-v1/{gradient_train,validation,calibration}-tokens.npy` |
| 冻结表示（候选标记差向量） | 202 MB | 22,100 / 400 / 2,500 条 1024 维特征 | `artifacts/laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1/*.npz` |
| ModernBERT 分词缓存 | 693 MB | L256/L512 训练/验证 token id | `.cache/tokenized/`（[rust_cache_*.json](../rust_cache_train_l512.json)） |
| DeBERTa 训练 token id | 28 MB | 22,100 条 512 首尾 token id | `artifacts/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/train-tokens.jsonl` |
| Laya 全参训练数据 | 每次训练复制 | `reports/<run>/data/{train,validation}.jsonl` | 各 run 目录 |

注意：磁盘只剩 82 GiB，这三份 token 缓存合计约 31 GB（du 计 29 GiB）已占相当比例；本计划**不建议**再生成第二个全 token 缓存（除非明确需要新的 token 级方法）。

### 2.4 缺失项（必须写明，不能猜）

- **稳健损失 6 模型的校准集（400）概率**：报告只给出开发集与校准温度；若要把它们放进集成，需要本机对 400 条做一次前向（Laya 在 MPS 约 20 条/秒，400 条约 20–30 秒/模型），成本可忽略。
- **DeBERTa epoch-001 / 4 个周期快照的开发集概率**：未评估，需要本机 CPU 前向（6.1 条/秒，见 3.1）。
- **语义 A/B/C 与 4090 各配方的校准集概率**：部分只有开发集；同样可用 400 条本机前向补齐（若权重可加载，均在 2.1）。
- **RoBERTa-large / 任意解码器 LLM 权重**：`.cache` 中没有；离线规则下不可得（除非用户批准联网下载，见 4.3）。
- **官方测试逐样本预测**只对 DeBERTa 与 Laya 主融合/两模型存在；其余成员若要在测试集评估，需要新推理（见 4.1 P0-1 的成本行）。

### 2.5 可复用的入口与配置

| 用途 | 现有入口 | 备注 |
|---|---|---|
| Laya 本地训练（MPS 已实测） | `scripts/train_laya.py` + `configs/laya-mac-ce-epochs2-fp32.json`（及 `-lr06`） | 本机两次完整运行均 `completed`；命令见 [training.md](../../docs/guides/training.md) |
| 冻结权重多视角导出（head/head_tail/reverse） | `scripts/laya_decision_views.py`、`scripts/laya_posthoc.py`、`scripts/export_laya_posthoc.py` | 已有 400/2,500 缓存 |
| 二模型/多系数概率融合 | `scripts/laya_probability_fusion.py`、`scripts/run_laya_probability_fusion.py` | 400 校准拟合单系数 |
| 校准概率融合（Laya+DeBERTa） | `scripts/run_laya_deberta_calibration_weight.py` | 产生 2.2 节主缓存 |
| 冻结候选的 CPU 复核与 sklearn 重算 | `scripts/run_final_cpu_alternatives_v2.py`（fit/evaluate 两阶段） | 4 候选、Holm 校正、bootstrap；**新实验应复制其结构写新脚本** |
| 词袋/NB 分支 | `scripts/run_bow_nb_study.py` + `environments/bow-cpu` | CPU 秒级 |
| DeBERTa 本地切片推理 | `scripts/deberta_final_evaluation.py:infer/prepare`（冻结脚本，只服务官方测试） | 复用于开发集需新脚本（见 6.2） |
| DeBERTa 训练入口 | `scripts/deberta_budget_training.py`（仅 CUDA 路径） | 本地 MPS 需新脚本 |
| Laya MLM 预训练 | **不存在** | 需新脚本 |
| 官方测试前的冻结契约 | `configs/deberta-laya-final-cpu-evaluation-20261018-v1.json` 模式 | 新候选必须新建契约 |

---

## 3. 真实吞吐与成本证据

### 3.1 已实测数字（可直接引用）

| # | 测量项 | 实测结果 | 来源 |
|---|---|---|---|
| E1 | Laya 全参 2 轮微调，本机 MPS FP32，micro-batch 1×accum 8 | **14,547.30 秒 = 4.04 小时**（含训练+开发评估+保存），5,526 更新；峰值 allocated 8.13 GiB | [status.json](../laya-mps-ce-epochs2-l512-s42-fp32-v1/status.json)（`elapsed_seconds`）、[result.json](../laya-mps-ce-epochs2-l512-s42-fp32-v1/result.json) |
| E2 | Laya 全参 1 轮，本机 MPS FP32 | **8,738.10 秒 = 2.43 小时**；峰值 allocated 9.12 GiB | [status.json](../laya-l512-s42-ce/status.json) |
| E3 | Laya 全参 2 轮，L40S CUDA FP32 | 进程区间 **2,951.37–2,980.98 秒（49.2–49.7 分钟）**/模型，5,526 更新；峰值 allocated 7.83 GiB | [robust-loss §5.2](../laya-robust-loss-analysis-20261016-v1/REPORT.md) |
| E4 | Laya 全参 2 轮，RTX 4090 CUDA FP32 | 训练阶段 **2,611–2,621 秒（43.5–43.7 分钟）** | [4090 seed43 status](../rtx4090-seed43-completed-v1/status.json)（stages 时间戳） |
| E5 | **本机 / L40S 慢速比**（同 Laya 2 轮任务） | 14,547 / 2,951 = **4.93×** | E1 ÷ E3（推导） |
| E6 | DeBERTa-v3-large 2 轮 L40S（BF16 前向，FP32 主权重），有效 batch 32 | 每轮更新 3,780.99 / 3,779.38 秒 = **63.0 分钟/轮**，共 7,560.4 秒；峰值 allocated 8.97 GiB；准入单次更新 6.6414 秒/32 条 | [training.json](../../checkpoints/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/epoch-002/training.json)、[ACCEPTANCE.md](../deberta-heterogeneous-bf16-budget7-s42-20261017-v1/ACCEPTANCE.md)、[准入 REPORT](../deberta-heterogeneous-bf16-s42-modal-20261017-v1/REPORT.md) |
| E7 | DeBERTa L40S 推理 | 400 条 4.94 秒；2,500 条 27.02 秒（≈92.5 条/秒） | [ACCEPTANCE.md](../deberta-heterogeneous-bf16-budget7-s42-20261017-v1/ACCEPTANCE.md) |
| E8 | DeBERTa 本机 **CPU FP32** 官方测试推理 | 25,000 条 **4,095.14 秒（68.25 分钟）= 6.10 条/秒** | [verification.json](../deberta-laya-final-cpu-evaluation-20261018-v1/verification.json)、[ACCEPTANCE.md](../deberta-laya-final-cpu-evaluation-20261018-v1/ACCEPTANCE.md) |
| E9 | Laya 本机 **MLX FP16** 推理 | 2,500 条 **44.29 秒 = 56.4 条/秒** | [mlx-fp16-eval.json](../laya-mps-ce-epochs2-l512-s42-fp32-v1/mlx-validation/mlx-fp16-eval.json) |
| E10 | Laya 本机 **MPS FP32** 推理 | 2,500 条 **131.55 秒 = 19.0 条/秒** | [export.json](../laya-mps-ce-epochs2-l512-s42-fp32-v1/saved-validation/export.json) |
| E11 | Laya 冻结编码器全 token 前向（MPS FP32） | 22,100 条 / 6,653,327 token / **1,578.16 秒**（4,215 token/秒，14.0 条/秒）；400 条 24.55 秒；2,500 条 181.36 秒 | [gradient_train-extraction.json](../laya-token-query-mps-fp32-20261018-v1/gradient_train-extraction.json)、[extract.log](../laya-token-query-mps-fp32-20261018-v1/extract.log) |
| E12 | 缓存表示上的轻量头训练 | final-cpu-alt：65,793 参数 MLP，CPU 训练/保存/重载 0.64–0.84 秒；单头开发 96.84%，并入融合 97.24%（**无净增益**）。token-query：9 个头 × 5 轮，MPS 每轮 691 更新约 10–11 秒；分支 96.80–96.84%，融合 97.24–97.28%（**无净增益**） | [final-cpu-alternatives-v2](../final-cpu-alternatives-20261018-v2/REPORT.md)、[token-query train.log](../laya-token-query-mps-fp32-20261018-v1/train.log) |
| E13 | 词袋/NB CPU 全流程 fit | 22.16 秒（含特征抽取、训练、校准、保存、重载） | [bow-nb](../bow-nb-complementarity-cpu-20261018-v1/REPORT.md) |
| E14 | Modal CPU（4/8/12 线程）64 条 DeBERTa 推理外推 | 14.44/13.48/13.16 秒 → 25,000 条约 5,142–5,642 秒（云 CPU；本机 CPU 实测更快，E8） | [cpu-throughput-options.json](../deberta-heterogeneous-bf16-budget7-s42-20261017-v1/cpu-throughput-options.json) |
| E15 | DeBERTa L40S 2 轮预算外推 | 平均速度 2.73 小时 / 6.82 USD；保守 3.63 小时 / 9.03 USD | [余额记录](../../delivery/deberta-balance-confirmed-7p03-20261017.json) |

由 E1/E3 还可得到 Laya 训练吞吐：本机端到端约 **3.0 条/秒**（44,200 曝光/14,547 秒），纯训练边际轮约 **3.8 条/秒**（5,810 秒/22,100 条 ≈1,145 token/秒）；L40S 约 **15.0 条/秒**。

### 3.2 M5 Pro 本地训练/推理外推（**估计值**，非实测）

| 任务 | 外推依据 | 估计本机时长 | 内存可行性 | 不确定性/备注 |
|---|---|---|---|---|
| (a) Laya 2 轮微调 | **实测 E1** | **4.04 小时** | 实测峰值 8.13 GiB，可行 | 无需外推；配置/权重/数据本地齐备 |
| (b) DeBERTa-v3-large 2 轮微调 | E6 每轮 63.0 分钟 × E5 的 4.93× | **10.4 小时**（区间 **10–13 小时**） | FP32 主权重+梯度+Adam ≈ 6.96 GB，+ 激活/临时约 1–2 GB，峰值估计 **10–14 GB**，64 GiB 统一内存可行；L40S 实测峰值仅 8.97 GiB | 4.93× 来自 Laya（不同算子的 MPS/CUDA 比值可能不同）；若 MPS batch 8/512 显存吃紧，需降到 micro-batch 4 并加倍累积，时间再 +10–30%；MPS 无 AMP 准入先例，必须 FP32 |
| (c) RoBERTa-large 2 轮 | 参数量 355M vs Laya 421M（≈0.84×）× E1 | **3.4 小时**（区间 3.4–4.5 小时） | 约 8–10 GB，可行 | **权重不在本机**，需联网下载（违反离线规则，需用户批准）；单模型历史文献 IMDB 约 96% 量级，作集成分支期望低 |
| (d) Laya MLM 继续预训练（22.5k 评论） | E1 边际训练吞吐 ≈1,145 token/秒；22.1k 条 ≈6.65M token/轮 | **1.6 小时/轮**；3 轮 ≈4.9 小时，再加 4.04 小时微调 ≈ **9 小时** | 与全参微调同量级，可行 | 数据/分词必须新脚本；MPS 上 MLM 损失数值稳定性未验证；无本地先例 |
| (e) 1–3B 解码器 LoRA | 按 token 曝光 22.6M×2 轮、每 token 约 9 GFLOP（前向 2N＋反向 4N，N≈1.5B；LoRA 不降激活计算）× M5 Pro 有效 3–6 TFLOPS | 1.5B：**10–19 小时**；3B：**20–40 小时** | 1.5B FP32 LoRA 约 12–20 GB；3B 约 24–40 GB，64 GiB 理论可行但 MPS 上 QLoRA/4bit 不可用 | 权重不在本机（需联网）；HF 解码器分类需改写成 scoring/prompt，工程量大；本地无证据表明能超过 97.5%（预期 96–97% 量级）→ **不建议** |
| (f) 缓存表示上的头重训 | E12 实测 | **分钟级（已做过）** | 可行 | **已穷尽**：MLP 单独 96.84%、融合 97.24%；token 查询头 96.80–96.84%、融合 97.24–97.28%，净增益 ≤0 |
| (g) DeBERTa 本机开发集推理（零训练用） | E8 | 2,900 条（400+2,500）≈ **7.9 分钟/检查点**；6 个检查点约 48 分钟 | CPU 内存 ~5–8 GB | 若走 MPS 可能 2–4× 更快，但**未实测**，计划按 CPU 保守值 |
| (h) Laya 本机开发集推理 | E9/E10 | 2,900 条：MLX FP16 ≈51 秒；MPS FP32 ≈153 秒 | 2.5–9 GB | 已有 MLX/MPS 换算入口，需新 run_id |

### 3.3 磁盘与显存预算

- 磁盘：现有可用 82 GiB；实测目录口径：一个完整 Laya 两轮 run（含逐轮优化器）约 6.4 GB（如 robust-loss 目录），一个 DeBERTa 完整 run 约 10.4 GB（2×(1.74 模型+3.48 优化器)）。本计划最坏情形（DeBERTa 新 seed + 3 个 Laya 新模型）约新增 **30 GB**，仍在 82 GiB 内但需预留；若只做 P0（零训练）则新增 <1 GB。**不建议删除/移动任何历史权重**（AGENTS.md 要求保留）。
- 显存/统一内存：Laya 实测峰值 8.13–9.12 GiB；DeBERTa 估计 10–14 GB。64 GiB 足够，但 MPS 高水位需按 PyTorch 建议设置，且禁止用 BF16/AMP 绕过（CUDA AMP FP16 梯度溢出未通过准入，MPS 无对等证据）。

---

## 4. 候选实验矩阵

### 4.0 目标分解与可观测的边际收益（为什么边际收益都很小）

| 参照 | 开发正确/2500 | Accuracy | 备注 |
|---|---:|---:|---|
| 原 Laya 主融合（冻结） | 2424 | 96.96% | 2 模型内部融合 |
| DeBERTa 单模型 | 2426 | 97.04% | [budget7 eval](../deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/REPORT.md) |
| 固定 50:50 异构融合 | 2432 | 97.28% | 正式测试 97.028% 的候选 |
| **探索最高（regularized_logit）** | **2435** | **97.40%** | 校准集 3 参数，[final-cpu-alt-v2](../final-cpu-alternatives-20261018-v2/REPORT.md) |
| TF-IDF 辅助融合 | 2433 | 97.32% | 稀疏分支仅 +1 |
| 校准权重融合（拟合 α） | 2429 | 97.16% | **拟合反而 -3**，证明 400 条上拟合易过拟合 |
| 开发 98% 门槛 | 2450 | 98.00% | 需 +15 条净修正 |

已观测的边际收益（配对口径）：
- 异构模型（DeBERTa 加入）：融合相对主融合修复 13/退化 5，**净 +8 条（+0.32 pp）**，McNemar p=0.096（[ACCEPTANCE.md](../deberta-heterogeneous-bf16-budget7-s42-20261017-v1/ACCEPTANCE.md)）。
- 单个正则/输入策略：warmup+LS s42 +0.56 pp 但 s43 未复现；首尾窗口 +0.16～+0.36 pp；稳健损失 +0.08 pp；重采样 +0.08 pp；语义增强 -0.08～-0.20 pp；R-Drop -0.36 pp（[全实验指标表](../all-experiments-metrics-20261016-v1/REPORT.md) §4）。
- 集成类：两模型拟合融合相对固定 50:50 **-0.12 pp**；八视角 OLS -0.16 pp；等权八视角 +0.04 pp（同上）。
- oracle 上限：两模型 2447（97.88%）；六稳健损失模型 48 条全错 → 2452（98.08%）；18 视角 40 条全错 → 2460（98.40%）；TF-IDF/NB 分支 oracle 2454/2459（98.16/98.36%）但单分支只有 91/92%。

**可实现的增益 ≈ 决策规则捕获率 × 可修复错误数**。任务 1 已把两侧都量化（[headroom-analysis.md](headroom-analysis.md) §3–§4、§6）：57 个去重视角的标签知情 oracle 是 **2481/2500（99.24%，19 条全错）**，去掉事后组合的 46 个单模型/单分支视角仍是 2481；68 条融合错误中 **49 条**至少被一个其它视角答对、**19 条**全错。因此从 2435（65 错）到 2450 需要捕获 15/46 = **32.6%** 的可修复错误（按 2432 的 68 错口径：18/49 = **36.7%**）——比例本身并不高。真正的问题在**可部署规则一侧**：所有只用 400 校准拟合的规则（等权、ridge logit 2/3/4/5/9 输入、多数投票、低置信门控）在开发集上全部 ≤2435，最好的也只有 +3 条、McNemar p=0.25、区间下限 0；且两视角 oracle 2447 < 2450。**可修复信息存在 ≠ 固定规则能取出**。→ **开发 98% 需要新的独立模型/训练信号，而不是继续在现有冻结输出上做后处理。**

### 4.1 P0：零训练候选（优先，成本最低）

| 编号 | 假设 | 协议（数据/输入/校准/评估） | run_id + 入口 | 预期增益（依据） | 本地成本 | 风险 | 决策门槛 | 能否到 98% |
|---|---|---|---|---|---|---|---|---|
| **P0-1** | 多配方/多种子 Laya + DeBERTa 的分层等权集成 ≥ 现有 97.40% | 成员**预先按"独立训练家族"规则**声明（每个家族只取 1 个代表，多 seed 家族先家族内平均）：Laya 主融合、Laya 等权辅助、DeBERTa e2、context-512、reviewed、robust-CE-s42、semantic-A、s43 基线、Mac-2ep、ModernBERT-e3、TF-IDF、NB；层次等权（家族内平均→家族间等权）；仅用 400 校准拟合至多 1 个稀疏分支权重（[0,0.25]）或 3 参数正则 logit；先冻结再评估 2,500 开发一次 | 新 `local-ensemble-audit-20261018-v1`；新脚本 `scripts/run_local_ensemble_audit.py`（复制 `run_final_cpu_alternatives_v2.py` 结构）+ 新配置 `configs/local-ensemble-audit-20261018-v1.json` | **+0.00～+0.40 pp**，中位 +0.10；依据：异构融合 +0.32、拟合权重 -0.12、历史集成 ±0.16 | **离线 10–60 分钟**（多数缓存已存在；缺缓存的家族用本机前向补齐 400/2,500 条：Laya ≈21 秒/2.2 分钟，DeBERTa ≈66 秒/6.8 分钟） | 缓存格式/输入视角不一致；400 条拟合过拟合；开发集反复使用导致乐观 | 开发 ≥ **97.50%** 且相对 2435 净修正 ≥+3 且校准内嵌套 CV 波动 <0.30 pp；98% 目标需 +15 条（=46 条可修复错误的 32.6%），当前可部署规则未显示此能力 | 仅增量 |
| **P0-2** | DeBERTa 的 epoch-001/周期快照提供额外多样性；快照概率平均或 2 轮权重 soup 有小增益 | 本机 CPU FP32 对 6 个检查点各推理 400+2,500；预声明三种聚合：①e1+e2 概率平均；②6 快照概率平均；③e1/e2 权重平均（soup）；均只用 400 校准拟合 1 个权重；评估开发一次 | 新 `deberta-local-snapshots-20261018-v1`；新脚本 `scripts/eval_deberta_checkpoints.py`（复用 `deberta_inputs.encode` 与 `deberta_final_evaluation.py` 的模型加载/分块推理模式） | **-0.10～+0.24 pp**；依据：Laya N8 epoch 概率平均 0.00、权重平均 0.00；e1 训练损失 0.191（未充分收敛） | **约 1 小时**（6×7.9 分钟 + 聚合） | 中期快照欠拟合；CPU/后端数值与 BF16 缓存不完全一致；权重 soup 可能跨盆地失效 | 合并进 P0-1 后开发 ≥ **97.55%** | 仅增量 |
| **P0-3** | DeBERTa 的窗口/选项顺序 TTA 与 Laya 类似，能带来小幅净修正 | 对 DeBERTa 做 2 种 TTA：①选项反序；②前缀 vs 首尾窗口（训练用首尾，推理加前缀视图做 logit 平均）；每视图各 2,900 条前向；只用 400 校准拟合 1 个权重；评估开发一次 | 新 `deberta-tta-20261018-v1`；扩展 `scripts/eval_deberta_checkpoints.py` 或复用 `scripts/laya_decision_views.py` 的视图逻辑 | **-0.10～+0.16 pp**；依据：Laya 选项反序 96.56/96.32（-0.04～-0.2）、首尾窗口 +0.16～+0.36 | **约 0.5–1 小时**（2–4 个视图 × 8 分钟） | TTA 视图可能与训练不一致（DeBERTa 训练固定首尾+选项顺序）；多视图=多次开发读取 | 与 P0-2 合并后 ≥97.55%；否则停止 | 仅增量 |
| **P0-4** | 已冻结的 97.40% 候选在官方测试上有多少（**测量，不是提升**） | 不改任何参数：把冻结的 `regularized_logit` 系数作用到已缓存的 Laya 主融合测试概率与 DeBERTa 测试概率；一次性读数 | 新 `frozen-candidate-test-readout-20261018-v1`；**需用户明确批准**；新脚本 `scripts/evaluate_frozen_candidate_test.py`（只读缓存，不推理） | 预期 **97.0–97.5%**（97.40 dev − 历史 0.25～0.6 pp 落差） | **分钟级** | 官方测试已被使用，不是独立留出；任何"再试一个融合公式"都会变成测试集调参 | 仅作最终报告数字；不据此改选模型/权重 | 不能提升 |
| **P0-5** | 开发错题可分类，用于**分析**而非调参 | 复用 `common-47.md`、`CASE-STUDY.md`、68/65 条错误分类；不重训、不改标签、不把开发错题加入训练 | 由任务 1（headroom-analyst）交付；本计划只引用 | 不产生精度 | 分钟级 | 把分析结论误用为训练信号 | 仅写报告/局限 | 不能 |

P0-1 的成员选择规则（预先声明，避免"按开发分数挑选"）：
1. 每个**独立训练配方家族**选一个代表（Laya-CE 主、Laya 等权辅助、Laya robust-CE、Laya robust-mixed、Laya context-512、Laya context-2048、Laya reviewed-data、Laya semantic、Laya s43 基线、DeBERTa、ModernBERT-base）；
2. 同家族多 seed 先家族内概率平均；
3. 家族间等权（或只拟合一个收缩系数）；
4. 稀疏分支（TF-IDF/NB）只能以校准集拟合的 ≤0.25 权重加入；
5. 全部参数在开发评估前封存；开发只评估一次。

### 4.2 P1：本地训练候选（仅在 P0 门槛通过或用户明确要求时）

| 编号 | 假设 | 协议 | run_id + 入口 | 预期增益 | 成本 | 风险 | 决策门槛 | 能否到 98% |
|---|---|---|---|---|---|---|---|---|
| **P1-1** | 第二个 DeBERTa seed 提供当前最强的异构多样性 | 同 budget7 配方（512 首尾、2 轮、有效 batch 32）改 seed 43；先本机 FP32 训练（新脚本），冻结后统一用 400 校准、评估 2,500 开发；与现有 DeBERTa s42 概率平均或用 1 个校准权重 | 新 `deberta-local-s43-mps-fp32-20261018-v1`；新脚本 `scripts/deberta_local_training.py`（从 `deberta_budget_training.py` 改设备/mps，保留非有限值保护与逐轮保存）；新配置 | 单模型预期 96.8–97.2%（不同 seed 方差）；集成相对 P0 最好候选 **+0.05～+0.25 pp** | **10–13 小时**（估计，见 3.2b） | MPS 数值/内存问题；无法从 CUDA 严格路径迁移；中途无精确续训需自建；时间成本高 | 单模型开发 ≥96.9 且集成开发 ≥ **97.55%**；否则不跑测试 | **仍不能**（最好情形开发 97.7–97.9） |
| **P1-2** | 新增 Laya seed 45 在 7 个 Laya 模型之后仍有微小集成收益 | 复制 Mac 2 轮 CE 配置；seed 45；其余相同；最终 epoch2 | 新 `laya-mps-ce-epochs2-l512-s45-fp32-20261018-v1`；`scripts/train_laya.py`；新配置 | 单模型 96.2–96.6%；集成 **+0～+0.12 pp** | **4.0 小时**（实测 E1） | 与已有 6 个 Laya seed 高度相关，边际递减 | 集成开发 ≥97.5% 且相对 P0 增益 ≥+0.05 pp；否则停 | 不能 |
| **P1-3** | 领域内 MLM 继续预训练（DAPT）能小幅提升下游 | 22.1k 梯度文本（不含校准/开发）做 15% MLM，1–3 轮；再标准 2 轮 CE；全部只用校准选超参 | 新 `laya-dapt-mlm-20261018-v1` + `laya-dapt-ce-s42-20261018-v1`；新脚本 `scripts/laya_mlm_pretrain.py` | 0～+0.3 pp（无本地证据，文献量级）；自身需 ≥97.4% 才有意义 | **8–10 小时**（3 轮 MLM 4.9h + 微调 4h） | MPS MLM 数值稳定、无先例；收益不确定；不能与 CUDA 结果混谈 | 单模型开发 ≥97.3% 且集成 ≥97.55%；否则停 | 不能 |
| **P1-4** | 更好的微调配方（层间 LR 衰减/EMA/SWA/对抗）还有空间 | 单因素、2 个 seed 预声明；每因素 4 小时 | 新配置 + `scripts/train_laya.py`（部分已有 warmup/LS/R-Drop 支持） | 0～+0.2 pp；warmup+LS 仅 s42 有效、R-Drop 负 | 每因素 4 小时（2 seed 8 小时） | 重复已失败的搜索；dev 选择偏差 | 两 seed 平均 ≥+0.20 pp 才继续；否则停 | 不能 |
| **P1-5** | 5 折交叉训练 + bagging 能最大化本地精度 | 5 折（每折 17.7k 梯度）各 2 轮，开发集上平均概率；每折校准独立 | 新 `laya-cv5-s42-20261018-v1` 等 5 个 run_id | **+0.10～+0.40 pp**（bagging 文献量级）；开发最好情形 97.6–97.9% | **5×≈3.5 小时 ≈17–20 小时** + 测试推理 2–3 小时 | 长时间占用本机；仍是同分布数据 | 只有前两折平均即 ≥97.5% 才继续；否则停 | 理论上限仍 <98% |
| **P1-6** | 更多人工标签清理（翻标/删除）能带来真实增益 | 在已核验的 61 翻标 + 3 删除基础上，需**用户人工**继续审核训练分歧样本；再按 reviewed-data 协议重训（22097→更少）并对照 | 新 `laya-reviewed-data-v3-...` + 新 `laya-94-user-label-revision-v3` 配置；入口沿用 `scripts/apply_laya_label_revision_v2.py` 模式 | 已做一次：96.32→96.40%（**+0.08 pp，区间跨零**）；扩展量级预计 +0.0～+0.2 pp | 标注人力（不可自动）+ 4 小时训练/轮；本地只能做 1 轮 | 依赖人工判断质量；样本量小；可能引入噪声 | 新一轮翻标 ≥50 条且重训开发 ≥97.5% 才继续 | 不能 |

**已做且无增益、不应重复**（详见 4.5）：缓存表示上的 MLP/查询头（E12、token-query）、Epoch 参数平均（N8 = 0.00 pp）、选项反序作为独立方法、线性头/重初始化头（L2 +0.03 pp、区间跨零）、稳健损失 mixed（+0.08 pp、门槛未过）、45 条重采样与 30 源语义增强、随机窗口训练（-0.20 pp）、长文 2048 分支融合（C 方案 +0.00）、长度路由（96.56%，低于主融合）、标签翻标重训（+0.08 pp，不显著）。此外，**零训练缓存集成审核（§4.1 P0-1）已由任务 1 完成等价分析**：57 视角、400 校准上所有等权/ridge/多数/门控规则 ≤2435，判断为"融合权重空间已饱和"（[headroom-analysis.md](headroom-analysis.md) §6）；因此 P0-1 若执行，定位应是"新 run_id 下复现并封存"，其门槛按已知结果预计不通过。

### 4.3 P2：超出本地/当前预算的候选（明确列为不可执行或需用户批准）

| 编号 | 候选 | 为什么不可执行 | 若获得资源后的成本/预期 |
|---|---|---|---|
| P2-1 | Modal 续跑 DeBERTa seed43 / 正式 2048 Laya 对照 | 最坏情形余额下界 1.43 USD < 单次 Laya 2 轮 ≈2.0–2.6 USD、DeBERTa 2 轮 ≈5.6–9.0 USD（实际余额可能更高但无账单，仍按不可用处理） | 每 DeBERTa seed 5.6–9.0 USD；预期集成 +0.05～0.25 pp |
| P2-2 | RoBERTa-large 本地微调 | 权重不在本地，需联网下载（违反离线规则） | 约 3.4–4.5 小时；单模型约 95.8–96.5%，集成边际很小 |
| P2-3 | 1–3B 解码器 LoRA/QLoRA | 权重不在本地；MPS 无 QLoRA 生态；10–40 小时 | 预期 96–97% 量级，不确定能超过现有融合；不建议 |
| P2-4 | 购买外部标注/更多无标签数据 | 需要预算与人工；[收尾记录](../../docs/experiments/2026-10-18-accuracy-search-closeout.md) 已列为未完成方向 | 不可估；这是唯一可能真正突破标签上限的路径，但不在本次范围 |

### 4.4 98% 可达性判断（把"目标"和"可实现"分开）

- **全覆盖开发 98%**（2450/2500）：任务 1 已量化可修复性——57 视角/46 单模型池的标签知情 oracle 为 2481（99.24%，19 全错），从 2435 到 2450 只需捕获 15/46 = 32.6% 的可修复错误（68 错口径 18/49 = 36.7%）；经典 18 视角 oracle 2460 也只比 98% 高 10 条。但可部署侧已饱和：(a) 400 校准上所有规则 ≤2435（最好 +3、p=0.25、区间下限 0）；(b) 两视角 oracle 2447 < 2450，**完美选择器也达不到开发 98%**；(c) 因此缺口只能由**新的独立模型/训练信号**补上，而不是后处理/权重搜索（[headroom-analysis.md](headroom-analysis.md) §4、§6、§7；[verification.md](verification.md) 独立复核一致）。现实最优开发 ≈**97.5–97.8%**。→ **概率低（<10%）；本地计划不应把它作为目标承诺。**
- **全覆盖官方测试 98%**（24500/25000）：需要从当前 24257 再净修正 243 条；历史开发→测试落差 0.25–0.6 pp，即需要开发 ≈98.3% 才可能有 98% 测试。→ **本地不可达（概率≈0）。**
- **"高精度 + 部分覆盖率"**：校准目标 90% 覆盖率对应的**实际测试覆盖率为 88.932%**（保留 22,233 条，其中正确 22,062 条 → **99.2309%**；[正式测试 ACCEPTANCE](../deberta-laya-final-cpu-evaluation-20261018-v1/ACCEPTANCE.md)）。如果课程目标允许"拒答/选择性分类"，这条线已超过 98%；但**它不是全覆盖成绩**（且 90% 是校准目标、不是实际覆盖率），不得混写。
- 结论：**本计划只追求"在 97.0–97.4% 基础上做可复核的小幅提升"，同时如实说明 98% 未达到。** 任何"98%"表述必须是研究目标/上限讨论，不能是实验预期。

### 4.5 明确"不要做"的清单

1. **不要重复已失败方向**：语义增强/45 条重采样、R-Drop、随机窗口训练、N8 epoch 参数平均、线性头/重初始化头对照、稳健损失比例搜索、长文 2048 融合、长度路由、缓存表示 MLP/查询头、线性/共享头重训、TF-IDF/NB 大权重融合。均有冻结证据且未过各自门槛。
2. **不要在开发集上反向选择**：不按开发分数挑 seed/epoch/融合权重/TTA 组合；不把开发错题加入训练或改训练标签；不把 2,500 开发当验证集反复调参。所有拟合只用 400 校准，并做校准内嵌套 CV。
3. **不要按官方测试调参**：不因测试结果改权重、阈值、TTA、模型选择或部署；官方测试最多对**已冻结候选**做一次性读数（P0-4），且需用户批准。
4. **不要在没有用户批准时花付费算力**；不要因为"只差一点"再开 Modal。
5. **不要把损失/后端差异当单因素结论**：本机 MPS 训练与历史 CUDA 训练不可逐位比较；Mac 推理与 CUDA/BF16 缓存的概率差已多次超过阈值（[MLX 精度诊断](../laya-mlx-precision-diagnosis-20261014-v1/REPORT.md)）。
6. **不要尝试恢复未回传权重**（L2 头部对照、部分历史模型），也不要使用空目录/3 步准入快照当模型。
7. **不要删除历史权重/缓存来"腾空间"**；AGENTS.md 要求保留逐轮检查点。
8. **不要宣称 98% 或 SOTA**；不要把探索性开发结果写成官方测试结果。

---

## 5. 推荐行动

### 5.1 如果用户要 (i)"在本地预算内尽量提高全覆盖精度"

按顺序只做 3 件事，总预算 **≤6–8 小时本地墙钟**：

1. **动作 1（≤1 小时，零训练；仅在用户要求正式留痕时执行）**：P0-1 缓存概率集成分层审核的**复现**（任务 1 已给出等价结果：57 视角、全部 400 校准规则 ≤2435、饱和）。产出：新 run_id 下的复现记录、可修复错误与捕获率。
   - **门槛 G1**：开发 ≥97.50%（≥2438）**且**相对 2435 净修正 ≥+3 且配对区间支持、校准内嵌套 CV 波动 <0.30 pp。按任务 1 已知结果预计不通过 → 直接进入 5.2 收尾（若以 98% 为目标，需净修 15 条 = 46 条可修复错误的 32.6%，当前可部署规则未显示此能力）。
2. **动作 2（1–1.5 小时，零训练）**：P0-2 DeBERTa 快照评估 + P0-3 DeBERTa TTA；把新成员并入动作 1 的集成，仍只用 400 校准拟合 1 个权重，开发评估一次。
   - **门槛 G2**：开发 ≥97.55%（≥2439）。不通过 → 停止，写收尾。
3. **动作 3（2–4 小时，条件执行）**：冻结候选 + 契约，按 P0-4 模式对官方测试做**一次性**本地评估：Laya 成员用 MLX/MPS 缓存/重算（约 7–22 分钟/模型，仅补算新成员），DeBERTa 用 CPU 68 分钟/次；报告 dev/test 与落差。
   - **前置条件**：用户明确同意"再对官方测试做一次冻结候选读数"（因为测试已非独立）；候选在开发 ≥97.75% 才有意义（否则测试 98% 毫无可能）。
   - **预期**：测试 97.1–97.5%；不承诺 98%。

若用户坚持"必须冲 98%"，则 1–3 动作应改为：**P1-1（DeBERTa seed43，10–13 小时）→ 集成门槛 → 最多一次测试**；并明确告知预期测试上限约 97.6%，98% 概率 <2%。**不建议**；若时间允许，P1-5 5 折 bagging（17–20 小时）是本地理论上限更高的路线，但仍是"提高 0.1–0.4 pp"，不是 98%。

### 5.2 如果用户要 (ii)"诚实收尾、把报告写完"

- **不再新增算力**：直接采用已核验数字：正式测试 **97.028%（24257/25000，固定 50:50 异构融合）**、DeBERTa 单模型 96.932%、探索开发最高 **97.40%**（新候选，无官方测试）。证据链见 [正式测试报告](../deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md)、[收尾记录](../../docs/experiments/2026-10-18-accuracy-search-closeout.md)。
- 可选、分钟级：若用户要求"97.40% 候选的官方数字"，执行 P0-4 一次性冻结读数（需批准），并明确它不是独立留出。
- 把本文 4.4 的 98% 边界、4.5 的未做/失败清单、任务 1 的 oracle 分析和任务 2 的文献对照写进课程报告。
- 交付清单仍按 [收尾记录](../../docs/experiments/2026-10-18-accuracy-search-closeout.md) §收尾交付清单执行（实验总表、Word 报告、视频、源码包、权重要核验）。

### 5.3 停止规则（预声明）

- 动作 1 若执行，按任务 1 已知结果预计开发 ≤2435、达不到 G1（≥2438）：**停止全部增益搜索**，不再训练、不再跑官方测试。
- 若动作 2 后开发 <97.55%：停止；若 ≥97.55% 但 <97.75%：不跑官方测试（预期测试仍 <97.4%），除非用户明确要求"要一个数字"。
- 任一训练实验（P1）若单模型开发低于其基线（Laya <96.5%、DeBERTa <96.9%）或集成增益 <+0.05 pp：立即停止，不追加 seed。
- 本机总训练预算硬上限建议 **14 小时**（一次 DeBERTa seed 或两次 Laya seed + 集成）；超过则回到收尾。

### 5.4 预期结果范围（全部为估计）

| 路线 | 开发 Acc 预期 | 官方测试预期 | 说明 |
|---|---|---|---|
| 现状（已冻结） | 97.40%（探索） | 97.028%（已测） | 无需新算力 |
| P0 全部零训练 | **97.40–97.80%** | **97.1–97.5%** | 2–4 小时本地；最好情形也不到 98 |
| P0 + P1-1（DeBERTa seed2） | 97.45–97.90% | 97.15–97.55% | 10–13 小时；增益边际 |
| P0 + P1-5（5 折 bagging） | 97.50–97.90% | 97.2–97.55% | 17–20 小时；本地理论上限最高，仍不到 98 |
| 只有"完美选择器"（oracle，不可部署） | 97.88–98.40% | 不可部署 | 仅用于说明余量，不是成绩 |

---

## 6. 最便宜的下一步实验（精确步骤）

### 6.1 定义（全部为预声明，避免开发集选择）

- **实验名/run_id**：`local-ensemble-audit-20261018-v1`。
- **定位说明**：任务 1 已用同一批缓存完成等价分析（57 视角、全部 400 校准规则 ≤2435、饱和；[headroom-analysis.md](headroom-analysis.md) §6）。本节正式 run 的唯一增值是**用新 run_id 留存可审计记录**，不是发现新机会；按已知结果预计停止（见 6.3）。
- **数据**：只读现有 400 校准 + 2,500 开发缓存；不新增推理（若某家族缺校准或开发概率，允许本机前向补齐 400 或 2,500 条，成本 ≤7 分钟/模型）。
- **成员（家族代表，家族内先平均）**：
  1. Laya 主融合（`laya-deberta-calibration-weight-cpu-20261018-v1` 里的 `neural`）
  2. Laya 等权辅助 50:50（`laya-probability-fusion-s43-s42-20261014-v1`）
  3. DeBERTa s42 e2（`deberta-heterogeneous-bf16-budget7-.../evaluation`）
  4. Laya context-512 与 context-2048（家族内平均）
  5. Laya reviewed-data-512
  6. Laya robust-CE（s42/s43/s44 家族内平均）
  7. Laya robust-mixed（家族内平均）
  8. Laya semantic A/B/C（家族内平均）
  9. Laya s43 基线 + combined-s43（家族内平均）
  10. Laya Mac 2 轮 s42（W1）
  11. ModernBERT-base e3
  12. 稀疏分支：TF-IDF 与 NB（只允许 1 个校准拟合权重 w∈[0,0.25]，可取 0）
- **融合规则（三选一预声明，按顺序报全部，不按开发挑）**：
  - M1：家族间等权 logit 平均（无拟合参数）；
  - M2：等权 logit 平均 + 1 个收缩系数（向等权 L2）；
  - M3：正则 logit（≤3 参数，L2 向等权），仅用 400 校准拟合。
- **校准内验证**：400 条上做重复 5 折（5×5）嵌套 CV，记录拟合参数波动与 OOF NLL/Accuracy；开发集只在参数冻结后评估一次。
- **统计**：相对 2435 的配对修复/退化、精确 McNemar、20,000 次类别分层配对 bootstrap；同时报告"相对约束 oracle 的捕获率"。

### 6.2 需要的新代码/配置（本轮不创建，只声明）

| 文件 | 作用 | 复用 |
|---|---|---|
| `configs/local-ensemble-audit-20261018-v1.json` | 成员列表、各缓存路径与 SHA、融合规则、拟合参数约束、预声明门槛 | 仿照 `configs/final-cpu-alternatives-20261018-v2.json` |
| `scripts/run_local_ensemble_audit.py` | `fit`（只用 400）与 `evaluate`（只用 2,500）两阶段；读访问审计；拒绝覆盖 | 复制 `scripts/run_final_cpu_alternatives_v2.py` 的两阶段/封存/审计结构 |
| 输出目录 | `reports/local-ensemble-audit-20261018-v1/`、`artifacts/local-ensemble-audit-20261018-v1/` | 新目录 |

执行命令（与历史 CPU 实验一致）：

```bash
cd /Users/shishishi/Dev-Project/NLP-Homework/End-IMDB
UV_CACHE_DIR=/private/tmp/end-imdb-ensemble-uv-cache \
uv run --project environments/bow-cpu --locked --offline --no-sync \
  python scripts/run_local_ensemble_audit.py fit
UV_CACHE_DIR=/private/tmp/end-imdb-ensemble-uv-cache \
uv run --project environments/bow-cpu --locked --offline --no-sync \
  python scripts/run_local_ensemble_audit.py evaluate
```

（**本轮不执行**；需用户批准后由执行者创建脚本/配置。命令结构参照 [bow-nb REPORT](../bow-nb-complementarity-cpu-20261018-v1/REPORT.md) 与 [final-cpu-alternatives REPORT](../final-cpu-alternatives-20261018-v2/REPORT.md)。）

### 6.3 什么结果值得继续

- **值得继续（进入动作 2/3）**：M1/M2 的冻结开发 Acc **≥97.50%（≥2438）**，相对 2435 净修正 ≥+3，20,000 次配对 bootstrap 的条件 95% 下界 ≥0，校准嵌套 CV 的 Acc 波动 <0.30 pp。**按任务 1 已知结果（可部署规则全部 ≤2435）预计不满足**，即停止是默认结局；若以 98% 为目标，需要净修 15 条（=46 条可修复错误的 32.6%），当前规则未显示此能力。
- **可以"记录但停止"**：开发 97.30–97.49%（与现有 97.28–97.40 同水平）；说明集成本地已近饱和，写入报告即可。
- **明确停止**：开发 <97.30%（连现有 50:50 都不如）或捕获率 <30%；这说明现有视角的凸组合已到顶，任何本地训练（4–13 小时）的期望收益 <0.1 pp，应直接收尾。

### 6.4 如果动作 1 通过，动作 2 的具体追加（仍低成本）

- 用 3.2(g) 的本机 CPU DeBERTa 推理评估 6 个检查点（约 48 分钟）与 2 个 TTA 视图（约 16 分钟），全部只写新 run 目录。
- 将最佳预声明聚合并入 6.1 的成员，**重新只拟合 1 个权重**，再评估开发一次（不是反复试）。
- 若开发 ≥97.55%：冻结候选（新契约 + SHA），按 P0-4 模式对官方测试一次性读数；否则停止。

---

## 7. 风险、治理与审计要求

| 风险 | 影响 | 缓解 |
|---|---|---|
| 400 条校准拟合过拟合 | 集成权重在开发/测试上反而变差（已有证据：拟合 α 后 2429 < 2432） | 只用 ≤1–3 参数、强正则向等权、嵌套 CV、预声明 |
| 开发集反复使用 | 开发分数乐观，测试落差放大 | 每个候选只评估一次；报告所有预声明候选；用配对区间而非单点 |
| 官方测试非独立 | 新测试数字不是"未见数据"成绩 | 报告显式披露历史使用；不据此调参/选模；不重新包装为 SOTA |
| 缓存/输入视角不一致 | 概率不可比，集成结论错 | 校验样本 ID/标签/温度/SHA；只混合同输入协议或显式多层融合；写访问审计 |
| MPS 训练数值/稳定性 | 新模型不可比、可能 NaN | 保持 FP32、非有限值保护、固定种子、逐轮保存；不声称与 CUDA 单因素可比 |
| 磁盘/内存 | 新 run 失败 | 预留 ≥15 GiB；先跑 `--check-only` 与短程 smoke |
| 时间沉没 | 4–13 小时换来 <0.1 pp | 预声明停止规则；先做零训练门槛，再决定是否训练 |
| 误用未回传/丢失权重 | 结果不可复现 | 不使用 2.1 列出的"伪检查点"；所有新结果绑定权重 SHA |

审计要求（沿用仓库惯例）：
- 每个新 run 新目录 + `run_id`；`manifest` 记录数据/权重/源码 SHA 与后端；拒绝覆盖已有目录。
- 概率缓存与最终预测追加 SHA；`test_accessed` 字段逐阶段记录。
- 训练/评估分离进程：`fit` 禁读开发，`evaluate` 才读开发（参照 final-cpu-alternatives 与 calibration-weight 报告的做法）。
- 不删除、不改写任何历史报告/权重/缓存。

---

## 8. 证据索引（数字 → 来源）

| 编号 | 数值/结论 | 来源 |
|---|---|---|
| S1 | 本机 M5 Pro、64 GiB、82 GiB 可用 | 本会话 `sysctl -n machdep.cpu.brand_string`、`sysctl -n hw.memsize`、`df -h .` |
| S2 | Laya 本机 2 轮 14,547.30 s；1 轮 8,738.10 s | `reports/laya-mps-ce-epochs2-l512-s42-fp32-v1/status.json`、`reports/laya-l512-s42-ce/status.json` |
| S3 | Laya L40S 2 轮 2,951.37–2,980.98 s；4090 2 轮 2,611–2,621 s | `reports/laya-robust-loss-analysis-20261016-v1/REPORT.md` §5.2；`reports/rtx4090-seed43-completed-v1/status.json` |
| S4 | DeBERTa L40S 每轮 3,780.99/3,779.38 s、峰值 8.97 GiB、准入 6.6414 s/32 条 | `.../budget7.../epoch-002/training.json`；`.../budget7.../ACCEPTANCE.md`；`.../deberta-heterogeneous-bf16-s42-modal-20261017-v1/REPORT.md` |
| S5 | 余额 7.03 USD、花费保守上界 5.60 USD（→ 最坏情形余额下界 ≥1.43）、projection 2.73/3.63 h | `delivery/deberta-balance-confirmed-7p03-20261017.json`、`.../budget7.../ACCEPTANCE.md`、`.../budget7.../budget.json` |
| S6 | 开发 97.40% = 2435/2500，四候选结果与参数 | `reports/final-cpu-alternatives-20261018-v2/REPORT.md` |
| S7 | 开发 97.28% = 2432/2500，正式测试 97.028% = 24257/25000 | `reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/ACCEPTANCE.md`、`reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md` |
| S8 | 两模型 oracle 2447/2500（97.88%）；57 视角 oracle 2481/2500（99.24%，19 全错；46 单模型池同值）；68 错中 49 可救/19 全错；经典 18 视角 oracle 2460（40 全错）；六模型 48 全错（oracle 2452） | `headroom-analysis.md` §3–§4、`verification.md`、`.../budget7.../ACCEPTANCE.md`、`reports/laya-robust-loss-analysis-20261016-v1/REPORT.md` §6、`reports/all-experiments-metrics-20261016-v1/REPORT.md` §9 |
| S9 | DeBERTa 本机 CPU 25,000 条 4,095.14 s | `reports/deberta-laya-final-cpu-evaluation-20261018-v1/verification.json` |
| S10 | Laya MLX FP16 2,500 条 44.29 s；MPS FP32 131.55 s | `.../laya-mps-ce-epochs2-l512-s42-fp32-v1/mlx-validation/mlx-fp16-eval.json`、`.../saved-validation/export.json` |
| S11 | Laya 全 token 前向 22,100 条 1,578.16 s（6.65M token） | `reports/laya-token-query-mps-fp32-20261018-v1/gradient_train-extraction.json`、`extract.log` |
| S12 | 校准权重拟合 97.16%（-3 条）、固定 50:50 97.28% | `reports/laya-deberta-calibration-weight-cpu-20261018-v1/REPORT.md` |
| S13 | 缓存头/查询头无增益（97.24%、96.80–96.84%） | `reports/final-cpu-alternatives-20261018-v2/REPORT.md`、`reports/laya-token-query-mps-fp32-20261018-v1/REPORT.md` |
| S14 | 稳健损失 +0.08 pp（门槛未过）；45 重采样 +0.08；语义 -0.08～-0.20 | `reports/laya-robust-loss-analysis-20261016-v1/REPORT.md`、`reports/all-experiments-metrics-20261016-v1/REPORT.md` §4 |
| S15 | 校准目标 90% 覆盖率 → 实际测试覆盖 88.932%、保留 22,233 条中正确 22,062 条 = 99.2309% | `reports/deberta-laya-final-cpu-evaluation-20261018-v1/ACCEPTANCE.md`、`verification.md` |
| S16 | 历史 dev→test 落差 97.28→97.028（-0.25 pp）；历史通用约 0.6 pp | `.../budget7.../ACCEPTANCE.md`、`reports/all-experiments-metrics-20261016-v1/REPORT.md` §2 |
| S17 | 资产可用性（权重/SHA/空目录/伪检查点） | 本会话 `checkpoints/` 与 `artifacts/` 全量扫描；`checkpoints/*/model.safetensors` 大小清单 |
| S18 | 缓存路径与内容（概率/特征/token） | 本会话读取各 run 的 `evaluation/*.json`、`calibration-*.json`、`*.npz`、`*tokens.npy` 元数据 |
| S19 | 本地模型缓存只有 Laya/ModernBERT/DeBERTa/bert-base | 本会话 `.cache/`、`.cache/huggingface/hub` 盘点 |
| S20 | 训练入口与环境（`train_laya.py`、`scripts/deberta_*`、`environments/bow-cpu`、`uv --locked --offline`） | `scripts/` 列表、`environments/` 列表、[training.md](../../docs/guides/training.md) |

---

## 9. 一句话建议

**零训练缓存集成审核已由任务 1 等价完成（57 视角、全部 400 校准规则 ≤2435、饱和），按已知结果预计无法通过 97.50% 门槛；可修复错误只需 32.6% 捕获率，但可部署规则取不出——默认应立即按第 5.2 节收尾。只有用户明确要求正式复现留痕、或愿做 1–1.5 小时 DeBERTa 快照/TTA 探索时，才按第 6 节执行且不通过即停；冻结候选后最多做一次官方测试读数。本地限定预算下合理预期是开发 97.5–97.8%、测试 97.1–97.5%；98% 全覆盖不可达，不应作为承诺或实验目标写入结果。**
