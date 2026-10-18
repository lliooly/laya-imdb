# Accuracy 提升空间定量分析：冻结开发产物的误差结构、oracle 与 98% 距离

- **run_id**：`accuracy-improvement-research-20261018-v1/headroom`
- **日期**：2026-10-18（北京时间）
- **范围**：只使用本地冻结的开发（2500）/校准（400）预测与已发表报告中的官方测试**聚合计数**；**未训练、未推理、未打开官方测试逐样本文件**。
- **一句话结论**：当前最好的冻结融合 2432/2500（97.28%，68 条错误）距离开发 98% 的 2450/2500 还差 **+18 条**。在 57 个去重后的冻结视角里，**49/68** 条错误至少被一个其它视角答对，但 400 条校准集上拟合的所有简单组合最多只多对 **3 条（2435，97.40%）**，精确 McNemar p=0.25、条件 bootstrap 95% 区间 [0.00, +0.28] pp——**没有可分辨的统计收益**。剩余的可修复性主要是"换一个真正不同的模型/训练信号"，而不是继续调融合权重或输入窗口。

## 0. 结论摘要（先看数字）

| 对象 | 数值 | 证据 |
|---|---:|---|
| 当前最好开发融合（Laya+DeBERTa 50:50） | 2432/2500 = **97.28%**，68 错 | `reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/REPORT.md`；本轮逐样本复算 |
| 已发表 97.40% 候选（regularized_logit） | 2435/2500 = **97.40%**，65 错 | `reports/final-cpu-alternatives-20261018-v2/REPORT.md`；本轮从冻结概率**完全复现** |
| 22 文件 → 18 去重视角的标签知情 oracle | **2460/2500 = 98.40%**，40 条全错 | 本轮复算，与 `docs/superpowers/specs/2026-10-14-laya-stage2-diagnosis-and-next-directions.md` 完全一致 |
| 本轮扩大到 57 个去重视角的 oracle | **2481/2500 = 99.24%**，19 条全错 | 见 §4；含事后组合视角，仅作上限参考 |
| 68 条错题中至少被 1 个其它视角答对 | **49/68** | 见 §3；68 条中 47 条属经典"三流水线共错"，19 条（全部在这 47 条内）在所有 57 视角中全错 |
| 68 条错题中在所有 57 个视角里全错 | **19/2500 = 0.76%** | 见 §3 |
| 400 校准集拟合的最佳组合 | 2435（= 已发表候选），净 +3，p=0.25 | 见 §6；等权/多数投票/门控均 ≤2433 |
| 达到开发 98% 所需净修正 | **+18 条 = 68 条错误的 26.5%** | 见 §7 |
| 官方测试 98% 所需净修正（仅用已发表聚合） | **+243 条 = 743 条错误的 32.7%** | 见 §7；24257→24500 |

## 1. 方法与证据边界

**读取了什么**

- 开发真值与正文：`data/processed/validation.jsonl`（2500 条，SHA-256 `dbc3205…d2094`）。
- 校准划分：`reports/laya-deberta-calibration-weight-cpu-20261018-v1/frozen.json` 的 400 个 `calibration_ids`（SHA-256 `c692c1a…d2b3`）；标签从 `data/processed/train.jsonl` 取回（该 400 条原文位于训练文件中，SHA-256 `f39b1e5…acd7`）。三份 manifest 的 `calibration_ids` 交叉一致。
- `reports/**`、`artifacts/**` 下所有可解析的 JSON/JSONL：登记 75 个来源、自动补扫 2,035 个非登记文件、再补入 10 条自动识别视图；逐文件判断是否含 2500 行、`sample_id` 可对齐的预测向量。

**没有读取什么（硬约束）**

- 未打开任何官方测试逐样本文件。扫描时路径含 `test` 的 **587 个文件被跳过并计数**（`skipped_test_paths`）。
- 未训练、未推理、未重新拟合任何模型；唯一"拟合"是 §6 的 400 条校准集集成参数，且只在开发集评分一次。
- 官方测试只引用 `reports/deberta-laya-final-cpu-evaluation-20261018-v1/REPORT.md` 的聚合计数：laya main 24086、DeBERTa 24233、50:50 融合 24257（均 /25000）。

**对齐与去重规则**

1. 每个来源先按 `sample_id` 对齐到 `validation.jsonl` 顺序；行数必须 2500 且样本集合完全一致。
2. 来源自带标签时逐条比对：**50/50 个带标签来源全部一致**（`verification.json` 检查 `dev_label_agreement_all_sources=true`）；不一致的自动来源被丢弃。
3. 预测类别优先用文件中的 `pred_label`，否则用 `probabilities[1]` 或 `logits` softmax；概率列方向用仓库惯例（index 1 = 正类），并对每个来源做"若整体翻转后准确率>90%、原方向<50% 则标记"的保护（本轮无来源触发）。
4. 去重按**2500 字节预测类别向量**的 SHA-256 精确去重：**85 条有效视图记录 → 57 个互异预测向量**。同一向量可能来自不同导出（如 posthoc 与 decision-study 的两份 `validation-head.json`）。
5. 所有来源的准确率与已发表报告交叉核对：**31/31 个可核对视图完全一致**（`reports/accuracy-improvement-research-20261018-v1/verification.json`）。

## 2. 视图清单：现有冻结预测池有多大、有多强

完整 85 条记录（含路径）在 `artifacts/accuracy-improvement-research-20261018-v1/headroom-analysis-data.json` 的 `inventory.views`；下表按家族汇总。

| 家族 | 记录数 | 最好开发准确率 | 代表来源 |
|---|---:|---:|---|
| Laya-512 固定权重 × 窗口/选项序（posthoc 8 + decision-study 6 + 窗口训练 6 + checkpoint 平均 1） | 21 | 96.84%（2421） | `reports/laya-posthoc-*`、`reports/laya-decision-study-s43-s42-strict-20261013-v1/`、`reports/laya-decision-execution-20261014-v1-final-study/` |
| Laya 原始/线性头对照（6 个 strict 模型） | 6 | 96.72%（2418） | `reports/laya-head-control-*-strict-fp32-20261015-v2/validation.json` |
| 冻结表示探针（full/encoder/post_head） | 3 | 96.84%（2421） | `reports/laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1/validation-predictions.json` |
| Laya 单模型历史（Mac 1 轮 / 2 轮 / lr0.6） | 3 | 96.96%（2424） | `reports/laya-l512-s42-ce/`、`reports/laya-mps-ce-epochs2-*/saved-validation/raw.jsonl` |
| 其他历史模型（ModernBERT、Laya-CE/RLCD pilot） | 5 | 95.76%（2394） | `reports/model-comparison/*/raw.jsonl`、`reports/modernbert-l256-s42/` |
| Laya 双模型融合（70:30 主融合、等权辅助、重复导出） | 4 | 97.00%（2425） | `reports/laya-probability-fusion-s43-s42-20261014-v1/validation-fused.json`、`reports/laya-fusion-inference-cuda-fp32-20261014-v1/validation-actual.json` |
| CUDA 真实推理复核（与 posthoc 同权重的另一次导出，向量与 posthoc 不同） | 4 | 96.84%（2421） | `reports/laya-fusion-inference-cuda-fp32-20261014-v1/validation-actual.json` |
| DeBERTa-v3-large（预算 7，CUDA BF16） | 1 | 97.04%（2426） | `reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/predictions.json` |
| **Laya+DeBERTa 50:50 异构融合** | 1 | **97.28%（2432）** | 同上 `positive_probabilities.fusion_50_50` |
| 新 Laya 分支：512 / 2048 / 用户修订标签 512 / 长度路由 | 4 | 96.56%（2414） | `reports/laya-context-pair-s42-modal-20261016-v1/evaluation/`、`reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation/`、`reports/laya-length-router-20261017-v1/` |
| 长文融合 B/C（事后离线组合） | 6 | 97.00%（2425） | `reports/laya-long-fusion-bc-20261017-v1/predictions.json` |
| CPU 新候选（regularized_logit、long_context、2 个 MLP 头） | 4 | **97.40%（2435）** | `reports/final-cpu-alternatives-20261018-v2/development-predictions.json` |
| 词袋/TF-IDF/NB-ngram（含固定权重融合） | 5 | 97.32%（2433） | `reports/bow-nb-complementarity-cpu-20261018-v1/development-predictions.json` |
| 全序列查询头（mean/dense/top32 × 分支/融合） | 6 | 97.28%（2432） | `reports/laya-token-query-mps-fp32-20261018-v1/development-probabilities.json` |
| 单权重重校准融合（ldw；50:50 参考与校准 NLL 权重） | 2 | 97.28%（2432） | `reports/laya-deberta-calibration-weight-cpu-20261018-v1/development-predictions.json` |
| 自动补扫 10 条记录（MLX FP16 复跑、token-query 单种子头、DeBERTa 另一次导出等） | 10 | 97.04%（2426） | 见 `inventory.views[*].origin == "autoscan"` |

> 说明：记录数≠独立训练数。57 个互异向量中仍包含同权重不同输入、同模型不同 epoch、事后组合与重复导出；"去重"只能去掉**完全相同**的预测向量。因此下文 oracle 一律标注为上限参考。

## 3. 当前最好融合的 68 条错题：边际可修复性

50:50 融合 = `(p_Laya_main + p_DeBERTa)/2`，阈值 0.5。**68 条错误**（复算 = 已发表 2432/2500），与历史 `reports/deberta-laya-final-analysis-20261018-v1/` 的 47 条"三流水线共错"关系如下：

- 68 条中 **47 条**属于 `common-47.md` 的共错集合（Laya 主融合、2048 视角、DeBERTa 同时错）；**21 条**是融合特有的（至少一方本来判对）。
- 68 条中 **49 条**至少被一个现有互异视角答对；**19 条在 57 个视角中全错**。
- 21 条非共错：**全部 21 条**都能被至少一个其它视角修复；47 条共错中 28 条可被修复，19 条不可。
- 19 条全错集合 **⊂ common-47**。也就是说，新增视角（DeBERTa、新 Laya 512/2048、修订标签、词袋）救回了旧 40 条全错中的 21 条，但剩下的 19 条没有任何现有视角能救。
- 错误置信度并不低：68 条中 **29 条**融合置信度 ≥0.95、36 条 ≥0.90（中位数 0.92）。这不是"只差低置信度样本"的问题。
- 截断：Laya 侧 10 条、DeBERTa 侧 8 条、双截断 8 条（68 条中），截断不是主体。

**修复这些错题最多的视角**（对 68 条错误逐条统计）：

| 视角 | 答对 68 错题数 | 独立开发准确率 | 证据 |
|---|---:|---:|---|
| NB-ngram LR（单分支） | **27** | 92.24%（2306） | `reports/bow-nb-complementarity-cpu-20261018-v1/` |
| TF-IDF LR（单分支） | **22** | 91.12%（2278） | 同上 |
| Mac 两轮 Laya 单模型 | 14 | 96.96%（2424） | `reports/laya-mps-ce-epochs2-l512-s42-fp32-v1/` |
| Laya-RLCD pilot | 14 | 93.72%（2343） | `reports/model-comparison/laya_rlcd_pilot/raw.jsonl` |
| combined-s43/head、head_tail | 各 12 | 96.48/96.64% | `reports/laya-posthoc-combined-s43-fp32-v1/` |
| head-control-original-s44 | 12 | 96.36%（2409） | `reports/laya-head-control-original-s44-strict-fp32-20261015-v2/` |
| ModernBERT-best | 12 | 94.64%（2366） | `reports/modernbert-l256-s42/validation-best.jsonl` |
| Laya 修订标签 512 | 12 | 96.40%（2410） | `reports/laya-reviewed-data-s42-modal-20261017-v1/` |

**关键矛盾**：修复能力最强的 NB/TF-IDF 与强融合的逐样本翻转率为 7.2%/7.9%（180/198 条），独立准确率只有 91–92%；它们与 Laya 的翻转远多于错误修复，固定权重融合会回收掉大部分收益（见 §6）。逐条"哪些视角答对了这条错误"的完整 68×≤57 矩阵见 `headroom-analysis-data.json` 的 `fusion_errors_68.per_error_correct_views`。

## 4. Oracle：标签知情上限（不是可部署成绩）

**4.1 经典 22 文件池（复现历史结论）**

按 `docs/superpowers/specs/2026-10-14-laya-stage2-diagnosis-and-next-directions.md` 的口径遍历 `reports/**/validation*.json` 的 rows 型 2500 行文件（posthoc 8 + decision-study 6 + 窗口训练 6 + checkpoint 1 + 主融合 1 = 22 文件），按预测向量去重：

| 在多少视角中判对 | 样本数 | 占比 |
|---|---:|---:|
| 0（全错） | **40** | 1.60% |
| 1–3 | 15 | 0.60% |
| 4–9 | 24 | 0.96% |
| 10–16 | 84 | 3.36% |
| 17 | 12 | 0.48% |
| 18（全部判对） | 2325 | 93.00% |

→ **2460/2500 = 98.40%**，40 条全错。与历史文档 **完全一致**（22 文件、18 向量、40/15/24/84/12/2325）。

**4.2 本轮扩大池（57 个互异向量，含 DeBERTa/新 Laya/词袋/新候选/事后组合）**

- oracle **2481/2500 = 99.24%**，全错 **19 条**，2050 条（82.00%）在 57 个视角中全部判对。
- 仅去掉融合/事后组合视角（46 个"单模型/单分支"视角）仍是 **2481/2500、19 条全错**：新覆盖来自 DeBERTa、新 Laya 512/2048、修订标签、词袋/查询头等**独立来源**，不是组合技巧。

**4.3 逐视角加入的累积 oracle（每步是标签知情上限）**

| 加入视角 | oracle 正确数 | 占比 |
|---|---:|---:|
| Laya 主融合 | 2424 | 96.96% |
| + DeBERTa | 2447 | 97.88% |
| + 新 Laya 2048 | 2453 | 98.12% |
| + 修订标签 Laya | 2457 | 98.28% |
| + NB-ngram | 2470 | 98.80% |
| + TF-IDF | 2473 | 98.92% |
| + token-query mean | 2474 | 98.96% |
| + MLP 头 | 2474 | 98.96% |

这张表说明：**98% 所需的信息在 4 个独立来源后就已在池子里存在**（97.88%→98.12% 之间跨过 98%），但不代表任何固定决策规则能把它取出来——下节与 §6 显示正好相反。

> 所有 oracle 都是"知道真值后逐条挑一个正确视角"，**不可部署**；池中含事后组合、重复导出与已被多轮开发集选择过的视角，因此 99.24% 不是"新方法可达上限"，也不是任务上限。它只能说明"现有冻结输出里还留着多少标签可区分的正确信息"。

## 5. 68 条错题的质量审计（AI 定性分类，非人工独立裁定）

**方法**：逐条阅读 2500 条开发集全文（`data/processed/validation.jsonl`；68 条全文与元数据另存 `artifacts/accuracy-improvement-research-20261018-v1/fusion-68-error-evidence.json`），结合 `common-47.md`、`CASE-STUDY.md`、Laya/DeBERTa 截断标记与校准概率/margin 分类。分类一次确定，没有针对任何准确率数字调整。**这是 AI 定性分析，不是第二标注者的独立人工裁定，也不是已确认的错标。**

| 类别 | 数量/68 | 说明 |
|---|---:|---|
| (a) 标签—正文冲突候选 | **11** | 正文有作者自评分或明确总评与标签相反，如 H006、H034 |
| (b) 评价歧义/反讽/评价轴错位 | **46** | 混合评价、价格/反讽、"烂但好看"、工艺 vs 观看体验 |
| (c) 截断缺失关键证据 | **1** | 双方 tokenizer 均截断且可见文本缺第一人称总评（H026） |
| (d) 文本支持标签的模型误判 | **7** | 如 H001（明确负评）、H018（明确贬损）等可见文本已足够判对 |
| (e) 不确定 | **3** | 纯叙事或作者自述无法评价 |

- (a) 11 条 ID：`train/neg/3155_1`、`train/neg/7397_4`、`train/neg/7647_1`、`train/neg/7927_1`、`train/neg/9409_4`、`train/pos/1230_10`、`train/pos/1454_7`、`train/pos/1710_7`、`train/pos/3160_7`、`train/pos/7770_7`、`train/pos/8647_7`（.txt）。
- (c)：`train/pos/12208_8.txt`。(d)：`train/neg/10566_3`、`train/neg/4632_4`、`train/neg/6081_4`、`train/neg/810_1`、`train/pos/10091_7`、`train/pos/5506_10`、`train/pos/9_7`。(e)：`train/neg/12240_1`、`train/pos/12122_7`、`train/pos/8527_8`。完整 (a)–(e) 列表与逐条理由见 `headroom-analysis-data.json` 的 `error_classification_ai`。
- **19 条"57 视角全错"的核心**以 (a)/(b) 为主：a=6、b=11、c=1、d=1。其中 7 条（a+c）有"标签/输入"层面的解释，11 条是强歧义。这与 `CASE-STUDY.md` 的结论一致：**模型一致错误 ≠ 标签错误，但确实存在少量强冲突候选**。
- 47 条共错分类：a=10、b=32、c=1、d=2、e=2；21 条非共错：a=1、b=14、d=5、e=1（非共错里"文本支持标签的明显模型误判"比例更高）。

**与 47 条历史材料的关系**：`common-47.md` 记录 43/47 在 Laya 与 DeBERTa 两侧都未截断；本轮 68 条中 Laya 截断 10 条、DeBERTa 截断 8 条，最高频的解释仍不是"文章没读全"。类别 (c) 只有 1 条是"主要归因于截断"，与历史判断方向一致。

## 6. 400 条校准集上的约束集成（诚实但探索性）

**6.1 同一个候选的独立复现（先做这一步）**

冻结参数公式：`p = sigmoid(0.5319458737546259·clip(logit(p_Laya),-12,12) + 0.5727065867827512·clip(logit(p_DeBERTa),-12,12) − 0.1161031700961201)`。用 50:50 融合的上游两路概率（`deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/predictions.json` 的 `frozen_main` 与 `deberta`）重算：

- **2435/2500 = 97.40%**，与 `reports/final-cpu-alternatives-20261018-v2/development-predictions.json` 的 `regularized_logit` **逐样本完全一致**（概率最大差 1.1e-16）。
- 相对 2432：**修复 3、退化 0**，净 +3（+0.12 pp）；20,000 次配对 bootstrap 95% 区间 **[0.00, +0.28] pp**，精确 McNemar **p=0.25**。
- 修复的 3 条：`train/neg/2572_1`、`train/neg/5026_3`、`train/pos/5506_10`。剩余 65 错。
- 另外，用本轮独立实现的 ridge logistic（目标函数与 `scripts/run_final_cpu_alternatives_v2.py` 相同：mean NLL + 0.01·Σ(w−prior)²，L-BFGS 换成 Newton/IRLS）在 400 校准行上重新拟合，得到 **相同的冻结系数 [0.5319, 0.5727, −0.1161] 与相同的 2435**；说明该候选不是实现巧合，但**它比 2432 只多对 3 条**。

**6.2 有哪些模型的 400 条校准集预测可用**

| 校准视图 | 来源文件 | SHA-256（前 12） |
|---|---|---|
| Laya 主融合、DeBERTa | `reports/final-cpu-alternatives-20261018-v2/calibration-predictions.json` | `67bffcee4e66` |
| 新 Laya 512 / 2048 | `reports/laya-context-pair-s42-modal-20261016-v1/evaluation/{512,2048}-calibration.json`（按报告冻结温度还原概率） | `83424732…` / `ca0ff7a3…` |
| 修订标签 Laya | `reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation/512-calibration.json` | `4c34006d…` |
| TF-IDF / NB-ngram | `reports/bow-nb-complementarity-cpu-20261018-v1/{tfidf_lr,nb_ngram_lr}-calibration.json` | `19905e42…` / `77916b8f…` |
| 两个 MLP 头 | `reports/final-cpu-alternatives-20261018-v2/calibration-predictions.json` | `67bffcee4e66` |

**缺失的校准概率（报告缺口，不臆造）**：token-query 三种子头只有 `heads-frozen.json` 的校准分支概率与 index（`reports/laya-token-query-mps-fp32-20261018-v1/`），且其已发表校准融合权重为 mean/dense 0.0000%、top32 0.368%（净 −1），再拟合无意义；frozen-probe 只有校准特征缓存（需重训读出，不在本轮范围）；length-router / long-fusion B/C / 事后组合只有开发预测，无对应 400 行概率缓存。故 §6.3 只用上表 9 个视图。

**6.3 结果（全部只在 400 校准集拟合/选择，开发集只评分一次）**

| 规则 | 开发正确/2500 | 净变化 | 修复/退化 | 条件 95% 区间（pp） | 精确 McNemar p |
|---|---:|---:|---:|---|---:|
| 50:50 参照（等权 Laya+DeBERTa） | 2432 | — | — | — | — |
| 等权 Laya+DeBERTa（复算参照） | 2432 | 0 | 0/0 | [0.00, 0.00] | 1 |
| **ridge logit [Laya, DeBERTa]，L2=0.01（=已发表候选）** | **2435** | **+3** | 3/0 | [0.00, +0.28] | 0.25 |
| ridge logit [Laya, DeBERTa]，L2=0 | 2434 | +2 | 3/1 | [−0.08, +0.24] | 0.625 |
| ridge logit [Laya, DeBERTa, 2048] | 2435 | +3 | 3/0 | [0.00, +0.28] | 0.25 |
| ridge logit [+修订标签] | 2431 | −1 | 1/2 | [−0.20, +0.08] | 1 |
| ridge logit [+TF-IDF] | 2433 | +1 | 3/2 | [−0.12, +0.24] | 1 |
| ridge logit [+NB +2×MLP]（9 参数） | 2434 | +2 | 3/1 | [−0.08, +0.24] | 0.625 |
| 等权 [Laya, DeBERTa, 2048] | 2426 | −6 | 4/10 | [−0.52, +0.04] | 0.18 |
| 等权 [+修订标签] | 2424 | −8 | 6/14 | [−0.68, +0.04] | 0.12 |
| 等权 [+TF-IDF] | 2422 | −10 | 5/15 | [−0.76, −0.04] | 0.04 |
| 多数投票 [Laya, DeBERTa, 2048] | 2427 | −5 | 5/10 | [−0.52, +0.08] | 0.30 |
| 多数投票 [Laya, DeBERTa, 修订] | 2422 | −10 | 4/14 | [−0.76, −0.08] | 0.03 |
| 低置信门控 → NB（阈值 0.02，校准拟合） | 2433 | +1 | 3/2 | [−0.12, +0.20] | 1 |
| 低置信门控 → TF-IDF（阈值 0.11） | 2431 | −1 | 7/8 | [−0.36, +0.28] | 1 |
| 低置信门控 → 新 2048（阈值 0.02） | 2432 | 0 | 1/1 | [−0.12, +0.12] | 1 |

**读法**

- 加入 2048 后系数为 0.026，几乎被自动忽略（2435 与两输入候选逐样本相同）；加入修订标签/TF-IDF/NB/MLP 后**没有**超过 2435。9 参数模型在 400 行上已在过拟合边缘（见下）。
- 校准集重代入准确率 98.8–99.0%，400 行留一（LOO）估计 98.5–99.0%；同一批模型在开发集只有 97.28–97.40%。**校准集本身对模型偏乐观（约 +1.4 pp）**，用它拟合阈值/权重会放大乐观性。这不是新发现，但与"校准集拟合 = 可泛化收益"的直觉相反。
- 结论：**在现有冻结输出上，融合权重空间已经饱和**。2435 是"多对 3 条"的探索性上限，不能写成"稳定提升"；p=0.25、区间下界 0 都支持这个谨慎结论。

## 7. 98% 还差多少：算术、修复率与 dev–test 差距

**7.1 开发集（2500）**

- 98% = 2450/2500。当前 2432 → **净 +18**；当前 68 错 → 需要修正其中 **18/68 = 26.5%**（净）。
- 已发表候选 2435（65 错）→ 还差 **+15**（15/65 = 23.1%）。
- 修复/退化形式：设新方法修复 F 条、新错 B 条，需要 **F − B ≥ 18**。
- 在现有 57 个冻结视角里：错题最多能救回 49/68；**但实测最好的可部署组合只多对 3 条**（已发表的 CPU 候选与 bow/nb 候选，加上本轮的等权/ridge/多数/门控规则，全部 ≤2435）。若要用现有视角补 18 条，等于要求决策规则在"明显翻转"的弱分支上精准取用——本轮没有找到这样的规则。

**7.2 官方测试（25000，仅引用已发表聚合）**

| 方案 | 正确/25000 | 准确率 | 98% 所需净修正 | 需要修正的现存错误比例 |
|---|---:|---:|---:|---:|
| Laya 主融合（历史） | 24086 | 96.344% | +414 | 914 中 45.3% |
| DeBERTa CPU FP32 | 24233 | 96.932% | +267 | 767 中 34.8% |
| **Laya+DeBERTa 50:50** | **24257** | **97.028%** | **+243** | **743 中 32.7%** |

> 开发 98% 需净修 26.5% 的错，官方测试 98% 需净修 32.7% 的错（分母不同，不能混用）。

**7.3 dev–test 落差（只引用已发表成绩对）**

| 开发方案 | 开发 | 官方测试 | 差（dev − test） |
|---|---:|---:|---:|
| Laya 主融合 | 2424（96.96%） | 24086（96.344%） | **+0.616 pp** |
| DeBERTa | 2426（97.04%，CUDA BF16） | 24233（96.932%，CPU FP32） | +0.108 pp（混后端，不宜作单因素） |
| Laya+DeBERTa 50:50 | 2432（97.28%） | 24257（97.028%） | **+0.252 pp** |

历史总表记录"开发—测试约 0.6 pp"是跨方案的粗略规律（`reports/all-experiments-metrics-20261016-v1/REPORT.md` §2），并明确不能当作所有新模型的统一扣减。若用 0.25–0.6 pp 作敏感性：要让官方测试达到 98%，开发集可能需要在 **98.25%–98.6%（约 2457–2465）**——即比当前再净多 25–33 条。**这是假设，不是预测。**

## 8. 标签天花板敏感性（估算，不是已证明）

**分类锚点**（§5，AI 定性）：严格"强冲突" (a)=11；(a)+(c)=12；把全部"歧义"也视为不可恢复则 (a)+(b)+(c)=58。

如果 68 条错误中有 x 条是"标签/歧义不可恢复"，则全开发覆盖的最高准确率 = (2500−x)/2500：

| x（不可恢复的错题数） | 开发最高正确数 | 开发最高准确率 | 开发 98% 仍可达？ | 若剩余错题全修复，所需修复率 18/(68−x) | 同误差结构下官方测试的上限* |
|---:|---:|---:|:---:|---:|---:|
| 20 | 2480 | 99.20% | 是 | 37.5% | 99.13% |
| 30 | 2470 | 98.80% | 是 | 47.4% | 98.69% |
| 40 | 2460 | 98.40% | 是 | 64.3% | 98.25% |
| 47 | 2453 | 98.12% | 是 | 85.7% | 97.95% |
| 50 | 2450 | 98.00% | 仅当其余全对 | 100% | 97.81% |
| 55 | 2445 | 97.80% | **否** | >100%（不可达） | 97.60% |

\* 按"官方测试 743 条错误与开发 68 条错误结构相同"的同比例假设：不可恢复测试错误 ≈ 743·x/68。

**怎么读这张表**

- 用最严格的 (a)=11 作锚点：开发上限 99.56%，98% 不受标签噪声阻断；官方测试上限约 99.52%（同比例假设）。
- 用最宽的 (a+b+c)=58 作锚点：开发上限 97.68%，**98% 不可能**；官方测试上限约 97.47%。但把 46 条"歧义/反讽/评价轴"全部当作不可恢复是**过度悲观**——文献与常识上反讽、混合情感是可学的，本轮也没有证据证明它们一律不可学。
- 因此更合理的表述是：**已确认的强标签冲突只有 11 条（0.44%），不足以把 98% 排除；但"高歧义"池（(a)+(b)=57 条，2.3%）意味着 98% 需要模型在真实困难样本上有实质性能力提升，而不是靠清理标签或调阈值就能达到。** 历史 94 条训练候选复核（61 翻转、23 保持、3 删除、7 待定，`reports/laya-94-user-label-revision-20261017-v2/REPORT.md`）支持"训练池确有标签噪声"，但那不能直接折算成开发集错误率；两者不能混用。

## 9. 对下一阶段的可操作含义

1. **停止在现有冻结输出上搜索融合权重/结构。** 已发表的最后一轮 CPU 候选（4 个）与 bow/nb 候选（2 个）+ 本轮 6 组 ridge/等权拟合与多数/门控规则全部在 2431–2435 之间；最好的 +3 条也不显著（p=0.25，区间下界 0）。真正要做的是**新的独立证据来源**。
2. **互补性存在但不可融合。** NB-ngram/TF-IDF 单分支能救 27/22 条融合错题（远多于任何神经视角），但它们与强融合的翻转率是 7–8%，固定权重融合只回收 1 条左右。可探索的是**结构化路由/拒答**（例如只在低置信区使用弱分支），本轮门控最好也只有 +1 条；如果做，应作为预先声明的新实验，而不是继续拟合。
3. **长度/截断不是瓶颈。** 68 条错误中双截断只有 8 条，(c) 类只 1 条；已发表的风险—覆盖与长短分组也显示截断组并不更难（融合截断组 97.54% vs 未截断组 97.23%，`reports/bow-nb-complementarity-cpu-20261018-v1/REPORT.md`）。继续加长输入的期望收益应按具体样本数（个位数）而不是百分点来估计。
4. **98% 的两种可能路径**：
   - 全覆盖：需要新的模型/训练信号把错题净修 18 条（oracle 显示信息最多支持 49 条可救，但需要泛化决策规则）。按已发表方案成本，这至少是一次新的训练实验，而不是 CPU 后处理。
   - 选择性：已发表的拒答口径在 90% 覆盖率下保留准确率 **98.84%（2224/2250）**、80% 覆盖率 **99.55%（1991/2000）**（`reports/all-experiments-metrics-20261016-v1/REPORT.md` §6，主融合 2424 口径）。如果任务允许覆盖率 <100%，98% 已由既有结果达到；这与"全覆盖 98%"是不同目标，不能混写。
5. **标签治理的定位**：11 条强冲突候选应作为数据质量研究/独立盲审对象，而不是改开发标签后重报准确率。改标签会让历史结果与官方测试口径不可比；本篇已发布候选都不依赖标签修改。
6. **官方测试已消耗。** 任何新方案都没有独立确认集；新的开发集成绩需按"探索性"报告，不能宣称 98% 已达成。

## 10. 本报告证明了什么 / 没证明什么

**证明（可复现）**

- 复现了 97.28%（2432/2500，68 错）、97.40%（2435/2500，3 修复 0 退化）与 98.40%（18 视角 oracle，40 全错）三个关键数字；全部输入对齐与标签一致。
- 量化了 68 条错题的边际可修复性：49 条可被现有视角救、19 条全错；最强修复来源是词袋/稀疏分支而非神经视角。
- 在 400 校准集上系统检验了等权、ridge logit（2/3/4/5/9 输入）、多数投票、低置信门控；**没有任何规则在开发上超过 2435**，所有增益 ≤+3 条且不显著。
- 给出 98% 的开发/测试算术、所需修复率、dev–test 落差区间与标签天花板敏感性表。

**没证明**

- 没有证明"现有池的 99.24% oracle 可通过任何可部署方法接近"；也没有证明 98% 不可能达到（分类里的 (a) 只有 11 条）。
- 68 条分类不是人工独立裁定，更没有访问外部原影评；"冲突候选"≠确认错标。
- 没有 97.40% 候选的官方测试成绩；用开发 0.12 pp 增益（小于 0.25–0.6 pp 的 dev–test 落差）推断测试会达到 97.4% 是不成立的。
- 没有做新训练、新推理、新结构；所有"下一步"都是基于现有证据的方向判断，不是已验证收益。
- 官方测试只有聚合计数，无法核对 68 条开发错误与测试错误的一致性；所有测试侧数字都来自已发表报告，未经本轮逐样本复核。

## 11. 复现与证据路径

```bash
# 生成全部分析数字（无训练/推理；只读冻结产物）
UV_CACHE_DIR=/private/tmp/end-imdb-headroom-uv-cache \
uv run --locked --offline --no-sync python scripts/analyze_accuracy_headroom.py
```

| 产物 | 路径 | 说明 |
|---|---|---|
| 本报告 | `reports/accuracy-improvement-research-20261018-v1/headroom-analysis.md` | 本文件 |
| 机器可读摘要（<40 KB） | `reports/accuracy-improvement-research-20261018-v1/headroom-summary.json` | 关键数字 + 输入 SHA + 命令 |
| 独立核验 | `reports/accuracy-improvement-research-20261018-v1/verification.json` | 13 项自检 + 31 个"已发表 vs 复算"准确率核对（31/31 一致） |
| 全量分析数据 | `artifacts/accuracy-improvement-research-20261018-v1/headroom-analysis-data.json` | 85 条视图记录、57 向量、oracle、68 错题逐条修复矩阵、分类、集成与算术 |
| 68 错题证据 | `artifacts/accuracy-improvement-research-20261018-v1/fusion-68-error-evidence.json` | 全文 + 概率 + margin + 截断标记 + common-47 编号 |
| 分析脚本 | `scripts/analyze_accuracy_headroom.py` | 本报告与上述 JSON 的唯一生成入口 |

输入 SHA-256：`data/processed/validation.jsonl` = `dbc3205038851d78e3e31ec5e5985005bfc0b53854371bed7eb9c03f764d2094`；`data/processed/train.jsonl` = `f39b1e522d3d6375a5f7fc6c254ec3cb2795f956774f09086c6fd755f747acd7`；校准 manifest = `c692c1a45dc78a80e233ee3933dc1014cbd1e0340db736b6b02e8dfdbc81d2b3`。

## 附录 A：57 个互异预测向量（去重后的完整开发准确率表）

"×N"表示有 N 条记录使用完全相同的预测类别向量（去重后只计一次）。路径见 `headroom-analysis-data.json` 的 `inventory.distinct_records`。

| # | 代表视角 | 类别 | 正确/2500 | 准确率 |
|---:|---|---|---:|---:|
| 1 | baseline-s42/head | laya512-posthoc-window | 2397 | 95.88% |
| 2 | baseline-s42/head_tail | laya512-posthoc-window | 2402 | 96.08% |
| 3 | baseline-s43/head ×3 | laya512-posthoc-window | 2412 | 96.48% |
| 4 | baseline-s43/head_tail ×4 | laya512-posthoc-window | 2421 | 96.84% |
| 5 | combined-s42/head ×3 | laya512-posthoc-window | 2411 | 96.44% |
| 6 | combined-s42/head_tail ×3 | laya512-posthoc-window | 2417 | 96.68% |
| 7 | combined-s43/head | laya512-posthoc-window | 2412 | 96.48% |
| 8 | combined-s43/head_tail | laya512-posthoc-window | 2416 | 96.64% |
| 9 | baseline-s43/reverse | laya512-decision-study | 2414 | 96.56% |
| 10 | combined-s42/reverse | laya512-decision-study | 2408 | 96.32% |
| 11 | prefix-trained/head | laya512-window-training | 2409 | 96.36% |
| 12 | prefix-trained/head_tail | laya512-window-training | 2416 | 96.64% |
| 13 | prefix-trained/reverse | laya512-window-training | 2408 | 96.32% |
| 14 | random-window-trained/head | laya512-window-training | 2406 | 96.24% |
| 15 | random-window-trained/head_tail | laya512-window-training | 2411 | 96.44% |
| 16 | random-window-trained/reverse | laya512-window-training | 2407 | 96.28% |
| 17 | checkpoint-average-s42/head_tail | laya512-checkpoint-average | 2416 | 96.64% |
| 18 | frozen-probe-s43/encoder | laya512-frozen-probe | 2417 | 96.68% |
| 19 | frozen-probe-s43/post_head | laya512-frozen-probe | 2420 | 96.80% |
| 20 | laya-main-fusion-70_30 ×3 | fusion | 2424 | 96.96% |
| 21 | equal-50_50(laya) | fusion | 2425 | 97.00% |
| 22 | head-control-original-s42 | laya512-head-control | 2413 | 96.52% |
| 23 | head-control-original-s43 | laya512-head-control | 2417 | 96.68% |
| 24 | head-control-original-s44 | laya512-head-control | 2409 | 96.36% |
| 25 | head-control-linear-s42 | laya512-head-control | 2418 | 96.72% |
| 26 | head-control-linear-s43 | laya512-head-control | 2412 | 96.48% |
| 27 | head-control-linear-s44 | laya512-head-control | 2407 | 96.28% |
| 28 | mac-one-epoch-s42 ×3 | laya512-single-model | 2394 | 95.76% |
| 29 | mac-two-epoch-s42 ×2 | laya512-single-model | 2424 | 96.96% |
| 30 | mac-two-epoch-s42-lr06 ×2 | laya512-single-model | 2403 | 96.12% |
| 31 | modernbert-best ×2 | other-models | 2366 | 94.64% |
| 32 | laya-ce-pilot | other-models | 2338 | 93.52% |
| 33 | laya-rlcd-pilot | other-models | 2343 | 93.72% |
| 34 | deberta-s42 ×2 | deberta | 2426 | 97.04% |
| 35 | fusion-50_50(laya+deberta) ×6 | fusion | 2432 | 97.28% |
| 36 | laya-new-512(orig-data) ×2 | laya-new | 2408 | 96.32% |
| 37 | laya-new-2048(orig-data) ×2 | laya-new | 2408 | 96.32% |
| 38 | laya-revised-labels-512 | laya-new | 2410 | 96.40% |
| 39 | length-router(512/2048 routed) ×2 | laya-new | 2414 | 96.56% |
| 40 | long-fusion-B | fusion-posthoc | 2413 | 96.52% |
| 41 | long-fusion-C | fusion-posthoc | 2424 | 96.96% |
| 42 | long-fusion-C_aux | fusion-posthoc | 2425 | 97.00% |
| 43 | cpu-alt/regularized_logit | cpu-alternatives | 2435 | 97.40% |
| 44 | cpu-alt/mlp_dropout1 ×3 | cpu-alternatives | 2431 | 97.24% |
| 45 | bow-nb/tfidf_lr/calibrated | bow-nb | 2278 | 91.12% |
| 46 | bow-nb/tfidf_lr/fusion | bow-nb | 2433 | 97.32% |
| 47 | bow-nb/nb_ngram_lr/calibrated | bow-nb | 2306 | 92.24% |
| 48 | bow-nb/nb_ngram_lr/fusion | bow-nb | 2432 | 97.28% |
| 49 | token-query/mean/branch ×2 | token-query | 2421 | 96.84% |
| 50 | token-query/top32/branch | token-query | 2420 | 96.80% |
| 51 | ldw/p_calibration_nll_weight | fusion | 2429 | 97.16% |
| 52 | token-query mean-s42（单种子头） | autoscan | 2419 | 96.76% |
| 53 | token-query top32-s42（单种子头） | autoscan | 2418 | 96.72% |
| 54 | token-query mean-s43（单种子头） | autoscan | 2421 | 96.84% |
| 55 | token-query top32-s43（单种子头） | autoscan | 2422 | 96.88% |
| 56 | token-query dense-s44（单种子头） | autoscan | 2422 | 96.88% |
| 57 | token-query top32-s44（单种子头） | autoscan | 2423 | 96.92% |

**方法备注**：McNemar 为精确双侧二项检验；bootstrap 为固定 2500 行开发集上的配对样本 bootstrap（20,000 次，seed=20261018），不含训练/校准选择不确定性；所有区间、p 值与候选增量均为探索性，不构成显著性声明。
