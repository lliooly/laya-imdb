# 独立核验报告：accuracy-improvement-research-20261018-v1

- **角色**：独立核验者（非三名作者）；**日期** 2026-10-18（北京时间）。
- **范围**：只读冻结产物重算；不训练、不推理、**未打开官方测试逐样本**（测试侧只用已发表聚合计数）。
- **方法**：不复用队友脚本。自写 NumPy 代码（`/private/tmp/end-imdb-verifier/verify{,2,3}.py`）从原始 JSON/JSONL 重新提取 2,500 行预测向量并重算；仅“85 条视图记录”的**来源清单**取自 `headroom-analysis-data.json` 的 provenance（逐条重载源文件重算数值，不采信其数字）。未完成项标 PARTIAL。
- **只新增本文件**；deliverable SHA：headroom-analysis.md `6b511742…d6aa`、headroom-summary.json `564ddebb…f933`、experiment-plan.md `64ed62ba…1365`、literature.md `be01326b…0580`。

## 1. 逐条核验（expected=deliverable 值；recomputed=本轮自算）

| # | 声称 | expected | recomputed（自算） | 判定 | 证据（SHA-256 前 12 位/方法） |
|---:|---|---|---|---|---|
| 1 | 50:50 融合 正确/错误；公式一致性 | 2432/68 | **2432/68**；`(p_laya+p_deb)/2` 与存储 `fusion_50_50` 逐样本一致（max|Δ|=0） | PASS | `…budget7…/evaluation/predictions.json` `117f708768c0` |
| 2 | 两视角 oracle | 2447 | **2447** | PASS | 同上重算 |
| 3 | 68 错置信 ≥.95/≥.90/中位 | 29/36/0.92 | **29/36/0.9198** | PASS | `max(p,1−p)` |
| 4 | 68 错∩common-47 | 47 | **47**（21 条非共错） | PASS | `common-47.json` `aa342599bdab` |
| 5 | 可修复性：≥1 视角救回/全错/19⊂47/47 中 28 可救/21 非共错全可救 | 49/19/是/28/是 | **49/19/是/28/是** | PASS | 57 向量×68 错矩阵 |
| 6 | 68 错截断 Laya/DeBERTa/双；common-47 双侧未截断 | 10/8/8；43/47 | **10/8/8；43/47** | PASS | `bow-…/development-predictions.json`、`…/evaluation/development.json` `408656c1d810` |
| 7 | 最强修复视角 | NB27/TFIDF22/mac2ep14/RLCD14/head 类 12 | **27/22/14/14/12**（combined-s43 head 与 head_tail、head-control-orig-s44、modernbert-best、revised 各 12；零修复视角 2 个） | PASS | 同上矩阵 |
| 8 | 错误 AI 分类计数/拆分 | 11/46/1/7/3=68；核心 19=6/11/1/1/0；共错 47=10/32/1/2/2；非共 21=1/14/0/5/1 | **全部算术一致，68 个 ID 唯一且均在 68 错内，核心 19** | PASS（算术）；**分类=AI 定性非人工**→§3 | headroom-summary.json |
| 9 | 视图去重 | 85 记录→57 向量 | **85/85 记录重载+向量重算成功，57 互异向量，逐记录 correct 与 inventory 值 85/85 一致** | PASS（条件于来源清单）；**完整性 PARTIAL**（未重扫 2,035 文件） | artifact `c55e8aba534d` + 自写提取器 |
| 10 | 经典 22 文件池 | 22→18 向量；oracle 2460/98.40%；40 全错；分组 40/15/24/84/12/2325 | **完全一致** | PASS | 自写 glob：`reports/**/validation-{head,head_tail,reverse}.json`+`validation-fused.json` |
| 11 | 57 视角/单模型池 oracle | 2481（99.24%）/19；46 视角仍 2481/19 | **2481/19；46 视角 2481/19** | PASS | 自提取 57 向量 |
| 12 | 累积 oracle | 2424→2447→2453→2457→2470→2473→2474→2474 | **8 步逐值一致** | PASS | 固定顺序：Laya→DeBERTa→2048→修订→NB→TFIDF→token-query→MLP |
| 13 | 带标签源一致/自动补扫/31-31 核对 | 50/50；2,035/587；31/31 | 自扫 **79 个全标签源 100% 一致**；10 条 autoscan 正确；85/85 记录 correct 与 inventory 一致 | **PARTIAL**：三项计数口径未逐一回查 | verification.json |
| 14 | 97.40% 候选公式复现 | 2435；maxΔp 1.1e-16 | **2435；max|Δp|=1.11e-16**（v2 frozen 系数+clip(logit)±12） | PASS | `final-cpu-alt-v2/frozen.json` `10e3e9a7fcda`、dev `3b4059b5cc9e` |
| 15 | 相对 2432：修复/退化、McNemar、bootstrap CI、修复 ID | 3/0；p=.25；[0,+0.28] pp；3 条 | **3/0；精确 p=0.25；[0.00,+0.28] pp**（20,000 次配对 bootstrap，seed=20261018）；3 个 ID 逐一吻合 | PASS | 同上 |
| 16 | 400 校准集独立重拟合 | 系数[0.5319,0.5727,−0.1161]、dev 2435 | 自写 Newton/IRLS（mean NLL+0.01Σ(w−prior)²，clip12，prior[.5,.5,0]）：**[0.53194587,0.57270659,−0.11610317]，dev=2435**（与冻结系数差<1e-9） | PASS | `final-cpu-alt-v2/calibration-predictions.json` `67bffcee4e66` |
| 17 | 仅 400 校准行、无开发集拟合 | 400 行、无 dev | cal 文件 **400 行，ID 序列==`frozen.json` calibration_ids，与 dev ID 交集 0，标签与 train.jsonl 400/400 一致；仅 400 行重拟合即复现冻结候选**；v2 frozen.json 含 `development_opened=False` 审计字段 | PASS（数据口径）；进程级“未读 dev”离线不可证 | 同上+`laya-deberta-calibration-weight…/frozen.json` `c692c1a45dc7` |
| 18 | §6.3 其余规则 | L2=0:2434(3/1)；equal:2432/cal.99；其余均≤2435 | **L2=0: 2434(3/1)，系数[0.51631,0.61041,−0.34143]；equal 2432、校准 0.99(396/400)；logit2 重代入 99.0%、LOO 98.75%（自跑 400 次留一）**；其余 12 条未逐条重算 | PASS（4 条自算，23/24 行）**+PARTIAL** | headroom-summary.json |
| 19 | 时间/成本事实 | 14,547.30s/8,738.10s；2,951–2,981s；63.0min/7,560.4s；7.03/5.60/closed/1.43 | **14,547.3019s(4.041h)/8,738.1001s；2,951.37–2,980.98s；63.003min/7,560.37s；7.03/5.60/`completed_verified_local`+`compute_closed=true`/1.43（5.60=7,974s×费率+0.13=5.5816）** | PASS（算术）；**“余额上界”方向错**→§2-D3 | `…/status.json`×2、robust-loss REPORT §5.2 `77d678c896f9`、`epoch-002/training.json`、budget `9b6d7298c49b`、balance `817e4a576fdd`、ACCEPTANCE.md |
| 20 | 两视角/六稳健损失 oracle | 2447/2452(48 全错) | **2447/2452（独立读 6 个 `evaluated/validation.json`，全错=48）** | PASS | 6 个 `checkpoints/laya-robust-loss-…-v2/evaluated/validation.json` |
| 21 | 官方测试聚合；选择性（target 90%） | 24086/24233/24257（/25000）；88.93% 覆盖→99.23% | **96.344%/96.932%/97.028%（仅聚合）；实际覆盖 88.932%（retained 22,233）、correct 22,062→99.2309%** | PASS（literature 准确；plan 写“90% 覆盖率”不精确→§2-D2） | `…final-cpu-evaluation…/summary.json` `2f2fb8f181b0` |
| 22 | TF-IDF/NB 分支 oracle | 2454/2459（98.16/98.36%） | **2454/2459**（自算：以 2432 融合为基修复 22/27；以 2424 为基则为 2452/2456，引用须写基线） | PASS | bow-nb `development-predictions.json`+budget7 predictions |
| 23 | 异构融合 13/5、净+8、p | 13/5/+8/+0.32pp/0.096 | **13/5/+8/+0.32pp/精确 p=0.09625** | PASS | 自算配对 |
| 24 | 吞吐/外推 E5/E8/E11/E14 | 4.93×；6.10/s；14.0/s+4,215 tok/s；5,142–5,642s | **4.9290；6.1048；14.004+4,215.9；5,141.2–5,641.8** | PASS | status/报告/`cpu-throughput-options.json` |
| 25 | 98% 算术（dev/test）与 dev–test 落差 | +18=26.5%/68；+243=32.7%/743；+15; 0.252/0.616/0.108pp | **18/68=26.47%；243/743=32.71%；15/65=23.08%；0.2520/0.6160/0.1080** | PASS | 纯算术 |
| 26 | 标签天花板敏感性表 | x=20…55 的 dev/test 上限 | 公式 `(2500−x)/2500`、`1−743x/68/25000` **逐值一致**（x=11→test 99.52%，x=55→97.60%） | PASS（算术）；x=AI 锚点→§3 | headroom md §8 |
| 27 | literature 本地：94 候选 61/23/3/7；94/22100=0.4253%；损失占比 80.40%；dev 90%/80% 覆盖 98.84%/99.55%；八视角 96.88%；两模型 97.00%；+24 条=+0.096pp；743−725=18 | 同左 | **报告原文逐项吻合；61/94=64.9%、61/22100=0.28%；2224/2250=98.844%、1991/2000=99.55%；24/25000=0.096pp；1−725/25000=97.10%** | PASS（本地部分） | `laya-94-user-label-revision-v2/REPORT.md`、`laya-training-loss-diagnosis/REPORT.md`、`all-experiments-metrics/REPORT.md` |
| 28 | literature 外部文献数字（XLNet 96.80、RoBERTa 96.54/96.68、2.90% 标签噪声、PWC 前 10、CFA 97.072、QLoRA 91.2、2026 新论文等） | 原样引用 | **离线无法重抓，未复现**；内部算术自洽，方向合理；QLoRA/SentimentBERT 2026 等引用无法核实 | **PARTIAL/不可核验** | §3 |

## 2. 差异清单（三个 deliverable 之间）

- **D1（中，实质性）**：`experiment-plan.md` §4.0 用**过时 oracle 假设**“暂按 98.0–98.4%（2455）→ 从 2435 到 2450 需捕获 15/20=75%”；但 `headroom-analysis.md` 实测全视角 oracle=**2481（99.24%）**（46 单模型池仍 2481）⇒ 需捕获 **15/46=32.6%**；按“68 错中 49 可救”口径为 **18/49=36.7%**。plan 用 75% 支撑“98% 概率很低”，与 task-1 不一致；谨慎结论仍可由“实测最好组合仅 +3、两视角 oracle 2447<2450”支撑，但该段论证必须改写。
- **D2（低）**：plan 称“**测试 90% 覆盖率**时已达 99.231%”；实际是**校准目标**90%、**实际覆盖 88.932%**。literature 写“实际覆盖 88.93%”准确，应统一。
- **D3（低，方向性）**：plan 称“剩余余额**上界**≈7.03−5.60=1.43”；5.60 是**花费**保守上界 ⇒ 余额应为 **≥1.43（下界/最坏情形）**。算术正确、决策不变。
- **D4（低）**：literature 速览 2 称“**唯一更高**的 97.072%”，但同文 §2.2 列有 97.42%/97.67%/98.12%（协议无效被排除）；“唯一”需限定为“唯一声称官方 25k/25k 协议的更高声明”。
- **D5（低/口径）**：headroom §5 硬核叙述漏 d=1（6+11+1+1=19，总数一致）；literature“比基准 25k 少 2.9k”只算梯度集 22,100（若含 400 校准，训练池 22,500→少 2.5k）；`verification.json` 的“50/50 带标签源”“31/31”未精确复现（我扫得 79 个全标签源全一致、85/85 记录值一致），方向一致非矛盾。
- **无致命差异**：无关键数字被推翻；以上为论证/措辞/口径层面。

## 3. 裁决：可引用 vs 应软化/丢弃

**可直接引用（本轮独立复算）**：融合 **2432/68**、两视角 oracle **2447**；97.40% 候选 **2435、3 修复/0 退化、McNemar p=0.25、CI [0.00,+0.28] pp**，且**仅用 400 校准行**重拟合系数一致、同样得 2435；L2=0 变体 2434(3/1)；equal 2432；**49/19** 可修复性、common-47 的 47/21/28、置信 29/36/0.9198、截断 10/8/8、NB27/TFIDF22；经典池 **22→18、oracle 2460、40 全错、分组 40/15/24/84/12/2325**；全池 **57 向量、oracle 2481/19**；累积曲线 8 值；官方测试 **24086/24233/24257**；选择性 **88.932% 覆盖、22062/22233=99.2309%**；时间/成本 **14,547.30s、8,738.10s、2,951.37–2,980.98s、63.003min/轮、余额 7.03、花费上界 5.60、closed、1.43**；六稳健损失 **oracle 2452（48 全错）**；TFIDF/NB oracle **2454/2459（基线 2432；基线 2424 时 2452/2456，引用须写基线）**。

**必须软化**：① **99.24%/98.40% 只能写“标签知情 oracle 上限”**——含事后组合/重复导出/被多轮开发集选择过的视角，**不可部署**；49/68 可救≠可被固定规则取出（实测最好融合仅 +3）。② §5 的 11/46/1/7/3 与 19 条核心为 **AI 定性分类**，非人工裁定、非确认错标；标签天花板表是敏感性假设。③ plan 的“75% 捕获率”论证按 D1 改为 32.6%（或 36.7%），理由改为“可部署规则全部 ≤2435 且增量不显著（p=0.25、CI 下界 0）”。④ §6.3 其余 12 条、§6.2 校准可用性、2,035/587 扫描数、50/50 与 31/31：标注未逐条复核（PARTIAL）。⑤ literature 外部数字（2.90% 噪声、PWC、CFA 97.072、QLoRA/SentimentBERT 2026）标“作者自报、本轮离线不可复现”，不得写成经本项目核验。

**遗留局限（最终报告必须披露）**：①官方测试非独立（历史已用），测试数字均为聚合复制、未逐样本核验；②开发集被反复使用（选择偏差），开发增量为探索性；③oracle 为标签知情上限；④68 错分类/噪声锚点为 AI 定性；⑤外部文献离线不可重抓；⑥视图池完整性（85/2,035/587）依赖作者 provenance；⑦bootstrap/温度还原/§6.3 其余规则未全量审计。

## 4. 复现命令与输入 SHA-256

```bash
cd /Users/shishishi/Dev-Project/NLP-Homework/End-IMDB
UV_CACHE_DIR=/private/tmp/end-imdb-verifier-uv-cache uv run --locked --offline --no-sync \
  python /private/tmp/end-imdb-verifier/verify.py   # 主核验 -> results.json
# verify2.py/verify3.py：截断/common47/分支 oracle/SHA 前缀/落差/天花板算术/分类计数
```
输入 SHA-256（前 12 位；全值见 `results.json`）：`data/processed/validation.jsonl`=`dbc320503885`、`train.jsonl`=`f39b1e522d3d`；`laya-deberta-calibration-weight…/frozen.json`=`c692c1a45dc7`；budget7 `evaluation/{predictions,development}.json`=`117f708768c0`/`408656c1d810`；`final-cpu-alt-v2/{frozen,development-predictions,calibration-predictions}.json`=`10e3e9a7fcda`/`3b4059b5cc9e`/`67bffcee4e66`；`…final-cpu-evaluation/summary.json`=`2f2fb8f181b0`；`common-47.json`=`aa342599bdab`；`headroom-analysis-data.json`=`c55e8aba534d`；`delivery/{balance,budget}.json`=`817e4a576fdd`/`9b6d7298c49b`；`robust-loss REPORT.md`=`77d678c896f9`。
判定口径：`pred=1[p≥0.5]`（index 1=正类；logits 用 z1>z0）；去重=2,500 字节向量 SHA-256；oracle=逐样本任一视角判对；McNemar=精确双侧二项；bootstrap=固定开发集配对样本 20,000 次 seed=20261018；校准重拟合=mean NLL+0.01Σ(w−prior)²，prior=[.5,.5,0]，logit clip ±12。

## 5. 机器可读结果（JSON）

```json
{"auditor":"verifier","date":"2026-10-18","checks_total":63,"checks_pass":62,"checks_fail":0,
"key_numbers":{"fusion_correct":2432,"fusion_errors":68,"two_view_oracle":2447,"err_conf_ge095":29,"err_conf_ge090":36,"err_conf_median":0.9198,"errors_in_common47":47,"fixed_by_any":49,"wrong_in_all":19,"distinct_vectors":57,"classic":{"files":22,"vectors":18,"oracle":2460,"all_wrong":40},"all_oracle":2481,"all_oracle_all_wrong":19,"single_pool_views":46,"truncation":{"laya":10,"deberta":8,"both":8},"common47_both_untruncated":43,"v2":{"correct":2435,"fixed":3,"broke":0,"max_prob_diff":1.11e-16},"v2_paired":{"delta_pp":0.12,"ci95_pp":[0.0,0.28],"mcnemar_p":0.25},"calibration":{"rows":400,"ids_equal_manifest":true,"no_dev_overlap":true,"labels_ok":400,"own_coef":[0.53194587,0.57270659,-0.11610317],"own_dev_correct":2435,"resub_acc":0.99,"loo_acc":0.9875},"logit_l2_0":{"dev_correct":2434,"fixed":3,"broke":1},"equal_dev_correct":2432,"robust_loss":{"all_wrong":48,"oracle":2452},"oracle_fusion_plus":{"tfidf":2454,"nb":2459},"official_test":{"original_laya_main":24086,"deberta_cpu":24233,"fusion_50_50":24257,"n":25000},"selective":{"target":0.9,"actual_coverage":0.88932,"retained":22233,"correct":22062,"accuracy":0.9923087},"timings":{"laya_mps_2ep_s":14547.3019,"laya_mps_1ep_s":8738.1001,"laya_l40s_2ep_s":[2951.37,2980.98],"deberta_min_per_epoch":63.003},"budget":{"balance":7.03,"spent_upper":5.60,"residual_arith":1.43,"compute_closed":true}},
 "discrepancies":[
  {"id":"D1","severity":"medium","where":"experiment-plan §4.0","issue":"capture argument vs stale oracle 2455; headroom 2481 => required 15/46=32.6% (or 18/49=36.7%), not 75%"},
  {"id":"D2","severity":"low","where":"experiment-plan §4.4","issue":"'90% coverage' is calibration target; actual 88.932%"},
  {"id":"D3","severity":"low","where":"experiment-plan §0/§1.1","issue":"'residual upper bound 1.43' direction reversed; remaining >=1.43"},
  {"id":"D4","severity":"low","where":"literature 速览2","issue":"'only higher 97.072%' vs §2.2's 97.42/97.67/98.12 invalid claims"},
  {"id":"D5","severity":"low","where":"headroom §5","issue":"hard-core narrative omits d=1 (sums still 19)"},
  {"id":"D6","severity":"low","where":"literature §1 note","issue":"'2.9k fewer' counts gradient 22100 only; pool 22500 => 2.5k"},
  {"id":"D7","severity":"info","where":"verification.json","issue":"50/50 and 31/31 counts not exactly reproduced; 79 labelled sources all agree"}],
 "verdict":"All headline numbers reproduce; oracle/classification/external-literature caveats per sections 1-3."}
```
