# 训练共同错题台账合并与补审报告

日期：2026-10-15（Asia/Shanghai）

run_id：`laya-train-common-error-review-20261015-v1`

状态：**94 条候选台账合并完成；复用旧记录 32 条，新增逐条文本审阅 62 条。未改标签、未训练、未做模型推理。**

## 1. 主要结论

这批高 margin 共同误判不宜整体当作“模型缺乏语言能力”的训练池：**58 条属于标签—文本态度冲突候选，23 条存在态度歧义或缺少本人评价，6 条缺少可见的决定性证据，7 条为可见输入上的模型错误候选。**

这些是本次 AI 审阅与旧记录归并后的工作分类，不是已证实的错标数量。尤其 7 条模型错误候选中，4 条沿用旧审阅，新增 3 条只有 R033 被首遍文本审阅标为高置信；另两条为中等置信。它们不足以直接支撑新结构或重采样训练的收益承诺。

还纠正了一处来源错误：两份训练导出来自 **baseline-s43 与 combined-s43，训练 seed 都是 43**。此前诊断把 combined 写成 s42、并称跨 seed 一致，均不成立。94 条筛选结果不受影响，但不能作为跨种子复现证据。

## 2. 数据来源与核验

| 项目 | 核验结果 |
|---|---|
| 梯度训练分区 | 原 manifest 的 22100 个 ID；与 400 条校准和开发集 ID 无交集 |
| 基线来源 | `laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1`，seed 43 |
| 组合来源 | `laya-rtx4090-ce-combined-s43-epochs2-fp32-v1`，seed 43；warmup 0.05 + label smoothing 0.05 |
| 筛选规则 | 两模型均预测给定标签的相反类别，错误类别相对给定类别的原始 logit margin 均 > 1 |
| 候选组成 | 94 条，原负类 49 / 正类 45；其中 16 条超过 466-token 评论预算 |
| 原始训练文件 | 94/94 个训练成员的 raw SHA、清洗后正文和路径标签与处理数据一致 |
| 模型实际输入 | 重建输入，无模型前向；94/94 个输入 SHA 与原预测导出一致 |
| 旧记录证据 | 32/32 条旧证据引文存在于当前冻结正文中 |
| 完全重复正文 | 这 94 条在原 22100 梯度训练池中均无相同清洗正文副本；未做近重复检测 |
| 数据边界 | 仅读取训练源及所选原始训练成员正文；未读开发集或官方测试正文 |

原始文件核对排除了这 94 条在本项目清洗/合并阶段发生正文与标签错配的解释。**它不能证明原评分与文字态度完全一致。** IMDB 原类别来自评分区间，不能把所有文字冲突直接叫作标签录入错误。两模型的 logit 尺度不同，margin > 1 是筛选规则，不代表同一概率置信水平。

## 3. 合并方式与审阅协议

先按 ID 对齐两份预测，核对标签与输入 SHA，再与累计 223 条旧台账求交，得到旧记录 32 条、新记录 62 条。

新增文本用固定随机顺序分配 R001–R062，首遍展示全文和 466-token 前缀边界，隐藏源 ID、原标签、模型分数和旧处置。逐条记录全文态度、前缀态度、置信度、精确引文及理由；62 条标注文件 SHA 在揭示元数据之前已输出，后续归并验证其未变。

这属于部分匿名化的单 AI 审阅：审阅者知道所有文本来自共同误判筛选，不能消除会话既有信息；全文与前缀由同一会话一起阅读，也不是独立的两次盲评。旧 32 条只做处置映射和证据核验，没有重新匿名审判。不能称为独立人工复核。

归并优先级为：全文不确定 → 态度歧义；全文与原标签相反 → 标签—文本态度冲突候选；全文与标签一致但前缀不足 → 截断缺证据；其余 → 可见模型错误候选。此顺序使四类互斥，不表示其他因素不存在。

旧台账中 `ambiguous_irony_or_enjoyment` 的最终处置优先于首遍正负判断；`frozen_candidate` 仅映射为沿用的模型错误候选，不直接加入任何新的训练池。完整旧记录保存在 `selection.json`，没有覆盖历史台账。

## 4. 分类结果

| 工作分类 | 复用旧 32 条 | 新审 62 条 | 合计 | 占本次候选 |
|---|---:|---:|---:|---:|
| 标签—文本态度冲突候选 | 19 | 39 | **58** | 61.70% |
| 态度歧义或缺少本人评价 | 6 | 17 | **23** | 24.47% |
| 决定性证据在截断外 | 3 | 3 | **6** | 6.38% |
| 可见输入上的模型错误候选 | 4 | 3 | **7** | 7.45% |
| 合计 | 32 | 62 | **94** | 100.00% |

新增 39 条标签冲突候选的首遍置信度为 **35 条 high、4 条 medium**；high 表示本次文本判断较明确，仍不等于“已确认错标”。新增 17 条歧义项均为 low。新增 3 条可见模型错误为 1 条 high、2 条 medium；新增 3 条截断缺证据为 2 条 high、1 条 medium。

58 + 23 = 81 条进入冲突或不确定队列，占本次筛选池 86.17%。这个比例受模型共同错误与 margin 阈值富集影响，不能外推为原训练集噪声率，也不能拿来推算模型准确率天花板。旧记录与新记录的审阅方式、置信标注不同，合计仅用于工作队列管理。

## 5. 有区分力的具体案例


### R033：可见的全局反转：适合作为理解能力探针

- ID：`train/neg/3654_1.txt`；原标签：负向；全文判断：负向；前缀判断：负向。
- 长度：329 token；处置：可见输入上的模型错误候选；置信度：high。
- 原文证据：“Please note: All of the above is opposite for the film in question.”

末句仍在可见输入内。它比泛泛的反讽归类更容易设计可证伪对照：保留/删除反转声明后，预期极性应改变。这里只提出对照，尚未生成或推理反事实样本。


### R048：反讽声明被截断：不能只归因语义理解

- ID：`train/neg/8278_1.txt`；原标签：负向；全文判断：负向；前缀判断：不确定。
- 长度：531 token；处置：决定性证据在截断外；置信度：high。
- 原文证据：“I'm being sarcastic. 1/10”

负向锚点在截断外。可在后续固定权重、固定提示下比较不同覆盖策略，但当前不能认定任意首尾策略一定修好。


### R062：评价对象在长文后部才明确

- ID：`train/pos/9175_10.txt`；原标签：正向；全文判断：正向；前缀判断：不确定。
- 长度：1201 token；处置：决定性证据在截断外；置信度：high。
- 原文证据：“All the faults of the '73 version stand corrected in the TV mini series of '83” / “this is the definitive and only true "Jane Eyre".”

前缀主要批评 2006 与 1973 两版；全文后部集中赞扬 1983 版。对象切换与截断叠加，既不是纯长度问题，也不是已证明的局部语义问题。


### R031：其他改编差评与当前动画的温和肯定

- ID：`train/pos/1005_10.txt`；原标签：正向；全文判断：正向；前缀判断：正向。
- 长度：130 token；处置：可见输入上的模型错误候选；置信度：medium。
- 原文证据：“Verhoeven's movie was utter and complete garbage.” / “This anime isn't perfect, but it's at least mostly accurate, as best I can tell.”

正文中最强烈的辱骂指向另一部真人电影。当前动画的正向判断较弱，因此此条仅列中等置信模型错误候选。


### R006：制作评分与娱乐态度冲突

- ID：`train/pos/11393_7.txt`；原标签：正向；全文判断：负向；前缀判断：负向。
- 长度：312 token；处置：标签—文本态度冲突候选；置信度：high。
- 原文证据：“It does get a 7 for very high production values, but as for entertainment values, it is rather poor.” / “it just doesn't satisfy in the long run.”

正文自己给制作价值 7 分，原正类并非已证明录入错误；文本对娱乐性却明显负向。应先决定研究目标是复现评分标签还是判断文字整体态度。


### R058：血腥情绪与教育价值不能混为一谈

- ID：`train/neg/9050_1.txt`；原标签：负向；全文判断：正向；前缀判断：正向。
- 长度：253 token；处置：标签—文本态度冲突候选；置信度：high。
- 原文证据：“this one should be required viewing for everyone preparing to learn to drive.” / “I only wish I could have found a copy when my children were attending”

作者希望所有学车者观看，负面词描述事故内容与生理反应。原负类与文字推荐存在强张力，但仍需独立复核，不能自动改标签。


### R049：只有剧情梗概，没有本人评价

- ID：`train/neg/8443_4.txt`；原标签：负向；全文判断：不确定；前缀判断：不确定。
- 长度：179 token；处置：态度歧义或缺少本人评价；置信度：low。
- 原文证据：“This is my translation of the DVD cover text synopsis.”

作者明确说明在翻译 DVD 封底简介。此类文本不适合用语义二分类强行解释原评分，也不能依据剧情悲惨推定负评。


### R060：他人的好评引文不是作者自己的好评

- ID：`train/neg/1892_3.txt`；原标签：负向；全文判断：不确定；前缀判断：不确定。
- 长度：181 token；处置：态度歧义或缺少本人评价；置信度：low。
- 原文证据：“He wrote : " A Fantastic documentary of 1924.” / “Bakhtiari region of Zardkuh is in Iran not in Irak”

作者主要纠正地理归属；Fantastic 属于被引用的另一评论者。应保留为缺少明确本人评价。


## 6. 对下一阶段的建议

1. **先核验监督目标的一致性。** 对 58 条冲突候选进行独立复核，优先看新审 35 条文本态度较明确的样本；区分原评分与文字态度张力、评价维度差异和可能的源标注问题。评分维度差异如 R006 不能自动翻标签。旧 19 条与新中等置信 4 条也需复核，不能用旧处置当独立背书。
2. **把 7 条可见模型错误作为探针候选，不作为训练收益证据。** 优先以 R033 的明确全局反转设计最小对照；R031 的对象区分和 R002 的讽刺解释仍需复核。旧 4 条的置信度未重新评级，不能按新 high 等同处理。
3. **单独处理 6 条截断缺证据样本。** 新增为 R026、R048、R062；旧记录为 `train/neg/4006_4.txt`、`train/neg/4407_4.txt`、`train/pos/5945_10.txt`。后续可检验文本覆盖与目标对象，但本次未做新推理，不声称已修复。
4. **23 条歧义项保持隔离。** 细分反讽不确定、质量与娱乐维度冲突、评分与文字张力及无本人评价；保留原数据与标签，不因两个模型同向就重采样或自标注。
5. **继续既定归因路线。** 本轮已完成共同错题候选集合的整理，并未测得表示层瓶颈。风险—覆盖正式报告、冻结表示线性探针和同期多 seed 头部对照仍是后续独立工作。

## 7. 局限与未执行项

- 只有同一训练 seed 的两个配方，不是跨 seed 或折外评估；两模型都见过这些训练样本。
- 新旧审阅均为 AI 证据整理，没有独立人工标注，也没有盲审一致率。
- 高 margin 共同错误是有偏抽样；未检查未入选样本，未估计总体噪声率或漏检率。
- 引文存在、原始文件匹配与输入哈希匹配可程序验证；语义判断正确与否不能由这些校验自动证明。
- 未改任何标签、训练样本、采样权重或实验配置；未进行训练、新模型推理、云端操作或测试评估。
- 本次补审结果不证明删除、改标、增加上下文或更换头部后开发集会改善；干预效果须另行设计同期对照。

## 8. 产物与复现

| 文件 | 内容 |
|---|---|
| [summary.json](summary.json) | 分类、来源、置信度与数据边界的结构化汇总 |
| [merged-review-ledger.csv](merged-review-ledger.csv) | 可筛选的 94 条逐条台账 |
| [merged-review-ledger.json](merged-review-ledger.json) | 精确引文、原标签、来源、全文/前缀判断、margin 与处置 |
| [selection.json](selection.json) | 94 条筛选元数据与 32 条完整旧记录 |
| [new-text-annotations.jsonl](new-text-annotations.jsonl) | 元数据揭示前记录的 62 条文本判断 |
| [new-text-pass-seal.json](new-text-pass-seal.json) | 首遍标注的 SHA 固定值及范围说明 |
| [review-queues.json](review-queues.json) | 四类互斥后续复核队列，不是重采样清单 |
| [boundary-audit.json](boundary-audit.json) | 原始训练文件、实际输入与旧引文核验 |
| [protocol.json](protocol.json) / [provenance.json](provenance.json) | 筛选协议、真实模型来源、输入与产物哈希 |

完整原始文本的匿名阅读包仅保存在本地临时目录，不复制进报告。仓库忽略规则默认仅纳入新报告的 `REPORT.md` 与 `summary.json`，其他逐条诊断和脚本留在本地。

在仓库根目录运行：

```bash
UV_CACHE_DIR=/private/tmp/end-imdb-review-uv-cache uv run --locked --offline --no-sync python reports/laya-train-common-error-review-20261015-v1/prepare_review.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_CACHE_DIR=/private/tmp/end-imdb-review-uv-cache uv run --locked --offline --no-sync python reports/laya-train-common-error-review-20261015-v1/audit_inputs.py
UV_CACHE_DIR=/private/tmp/end-imdb-review-uv-cache uv run --locked --offline --no-sync python reports/laya-train-common-error-review-20261015-v1/finalize_review.py
```

复现只校验和重建归并结果，不重新进行语义审阅。已有准备与最终产物若不同会拒绝覆盖；改变标注、源文件或分类规则需新建版本。

## 9. 94 条逐条处置

下表引文为摘要，完整引文与字段见 JSON/CSV。R 编号为本轮匿名首遍；L 编号为本轮给旧记录分配的引用号。原标签保持不变。

### 标签—文本态度冲突候选

| 编号 / ID | 原标签 | 来源 / 置信度 | 证据摘要 | 判断依据 |
|---|---|---|---|---|
| L002 / `train/neg/2355_3.txt` | 负向 | 旧记录 / legacy_unrated | the film, on the whole, is truly a rare treat to watch. | 批评Black而赞Page3，明确8/10和推荐，原负标签与总评冲突。 |
| L003 / `train/neg/2985_4.txt` | 负向 | 旧记录 / legacy_unrated | you can't make a film this bad,so good,on purpose. | 制作缺点与camp享受并存但总体强烈赞美，负标签张力；关键喜爱已在前缀。 |
| L005 / `train/neg/3932_4.txt` | 负向 | 旧记录 / legacy_unrated | Horror fans should check this one out. | routine story是轻微缺点，整体solid/best及推荐明确正面，负标签张力。 |
| L007 / `train/neg/4239_4.txt` | 负向 | 旧记录 / legacy_unrated | take 2 hours and enjoy it. | cheesy/bad特效描述预期及别片，不是本片实际缺陷；正文总体好评，负标签张力。 |
| L009 / `train/neg/6038_3.txt` | 负向 | 旧记录 / legacy_unrated | That is exactly why I enjoyed the movie so much. | camp缺点正是享受原因，通篇肯定并期待续作，负标签张力。 |
| L011 / `train/neg/6473_3.txt` | 负向 | 旧记录 / legacy_unrated | I think this is a good family film though overall! | 原作更好但本片总体好评，负标签与明确总评冲突。 |
| L013 / `train/neg/801_4.txt` | 负向 | 旧记录 / legacy_unrated | 'Fatal Error' is an entertaining flick | 低预算、套路与娱乐评价混合，总评alright且一直可看，负标签存在张力。 |
| L017 / `train/pos/10876_7.txt` | 正向 | 旧记录 / legacy_unrated | Don't see this film | good仅为观看前预期且明确推翻，正文全负面，与正类7分冲突。 |
| L019 / `train/pos/12276_7.txt` | 正向 | 旧记录 / legacy_unrated | a superficial, shallow film | 技术演技尚可但改编核心失败，正类7分与总体失望冲突。 |
| L020 / `train/pos/1419_7.txt` | 正向 | 旧记录 / legacy_unrated | the movie simply doesn't deliver. | 少量摄影认可不抵消通篇内容与呈现差评，正标签有显著张力，不能直接升权。 |
| L021 / `train/pos/1778_10.txt` | 正向 | 旧记录 / legacy_unrated | This movie is a ridiculous farce of film-making. | Bravo夸另一个评论者而非影片，导演另一部作品好但本片差，正类10分张力。 |
| L022 / `train/pos/3372_9.txt` | 正向 | 旧记录 / legacy_unrated | This is a dud of a film and I wouldn't even recommend it | 其他Heston作品好，本片明确不推荐；正类9分与正文冲突。 |
| L023 / `train/pos/3653_10.txt` | 正向 | 旧记录 / legacy_unrated | the direction is extremely boring | 演员演得好不等于电影好，正文批评强烈却正类10分，标签张力。 |
| L024 / `train/pos/5086_7.txt` | 正向 | 旧记录 / legacy_unrated | very slow | 只认可结尾，整体批冗长并建议其他版本，正类7分有明显张力。 |
| L025 / `train/pos/5149_8.txt` | 正向 | 旧记录 / legacy_unrated | this disappointing production | Carlyle表演好但总体docudrama差，正类8分与正文冲突。 |
| L027 / `train/pos/6038_10.txt` | 正向 | 旧记录 / legacy_unrated | This film is terrible | 反驳其他人好评并讽刺制作，整体差评明确，正类10分冲突。 |
| L028 / `train/pos/7538_7.txt` | 正向 | 旧记录 / legacy_unrated | tired serio-comedy that wastes the talents | 明星才华被浪费，电影通篇差评，与正类7分张力。 |
| L029 / `train/pos/9216_10.txt` | 正向 | 旧记录 / legacy_unrated | I give it NO stars out of *****. | 演员有才但本片浪费才华，明确零星，与正类10分冲突。 |
| L031 / `train/pos/9824_10.txt` | 正向 | 旧记录 / legacy_unrated | Bad acting, bad direction | 仅剧情尚可，其余制作全差，sterling example讥讽类型现状，正类10分冲突。 |
| R001 / `train/pos/7782_7.txt` | 正向 | 新审 / high | The move felt chaotic and disjointed. | 摄影优点没有抵消叙事混乱、人物疏离和整体不满足；负面总评明确。 |
| R004 / `train/neg/11947_4.txt` | 负向 | 新审 / high | All up the movie is enjoyable. | 前半节奏慢是局部保留；整体明确评价为 enjoyable，并肯定表演与结尾。 |
| R005 / `train/neg/1435_4.txt` | 负向 | 新审 / high | I think this is a good big screen movie to add to my collection. | wooden character 描述历史人物形象，紧接着肯定表演；导演、表演、收藏意愿和结尾均有正评。 |
| R006 / `train/pos/11393_7.txt` | 正向 | 新审 / high | it just doesn't satisfy in the long run. | 7 分被限定为制作价值，不能当作整体喜爱；娱乐性、故事和总体满足感均明确否定。 |
| R008 / `train/neg/3677_3.txt` | 负向 | 新审 / high | Ordinarily, I don't choose to watch horror films, but this is an exception. Good story, excellent acting. | 平常不看恐怖片与本片构成让步，末句直接肯定故事和表演。 |
| R011 / `train/pos/245_9.txt` | 正向 | 新审 / high | Other than that, just don't see it. It's neither good enough nor bad enough to be entertaining. | 演员吸引力与制作合格只是保留，明确建议其他观众不要看，整体无娱乐价值。 |
| R013 / `train/neg/5558_4.txt` | 负向 | 新审 / high | Darkman 3 is awesome. | 明确赞扬当前第三部并收藏系列；后半 DVD 区码说明不改变整体正评。 |
| R014 / `train/neg/3270_1.txt` | 负向 | 新审 / high | It is very slick and vibrant | 演员实际年龄不符的质疑被表演可信化解；整体节奏、表演与视觉均肯定。 |
| R016 / `train/neg/8314_1.txt` | 负向 | 新审 / high | makes its points beautifully | 全文与前缀都持续赞扬城市质感、摄影和对话；尾部故事赞语加强已有正评，不是唯一决定性证据。 |
| R017 / `train/neg/8656_4.txt` | 负向 | 新审 / high | Overall 3/6 stars and worth a watch. | 虽批评动作不足、承认 B 级片，仍明确总体好看、期待续集和值得观看；数值中间分不推翻文字推荐。 |
| R018 / `train/neg/6601_4.txt` | 负向 | 新审 / high | I enjoyed it. | 不满被归于原作纯粹主义者，作者自己明确欣赏并推荐给儿童。 |
| R019 / `train/pos/9646_7.txt` | 正向 | 新审 / high | But as a love story or character study it is lacking and unsatisfying. | 承认角色间情感和身体吸引力，不抵消叙事缺失及爱情/人物研究失败的总评。 |
| R021 / `train/neg/970_1.txt` | 负向 | 新审 / high | i would recommend everyone to see this movie | 恐惧和惊吓被明确视为恐怖片有效性的优点，作者直接推荐。 |
| R022 / `train/neg/6461_4.txt` | 负向 | 新审 / high | I would like to see it again | 明确好评与重看意愿；不真实是轻微保留，最后反问其他评论者不改变自己的好评。 |
| R024 / `train/pos/8682_7.txt` | 正向 | 新审 / high | Maybe I just didn't care for this one at all. | 喜爱演员和系列不等于喜欢本集；本集乏味、故事较弱、换台意愿明确。 |
| R025 / `train/neg/11940_4.txt` | 负向 | 新审 / high | Rent it and check it out. It's worth seeing at least once | 承认剧情、表演和连续性问题，仍明确说有趣、适合家人并推荐观看；推荐句在前缀内，尾部只补充客串赞语。 |
| R027 / `train/neg/3843_3.txt` | 负向 | 新审 / high | For persons interested in this genre, it is a must see. | 开头就称同类最佳并赞扬哥特气氛；尾部推荐加强已有好评，不构成缺证据。死亡、谋杀和恐惧词来自剧情。 |
| R028 / `train/neg/12061_4.txt` | 负向 | 新审 / high | it is well made and indeed entertaining. | sleazy 和 disturbing 主要描述题材；结尾明确制作好、有娱乐性。 |
| R029 / `train/neg/1201_2.txt` | 负向 | 新审 / medium | FANTASTIC! | 夸张恐怖修辞与肯定节奏、结尾赞叹一致；文本没有明确反转证据，不能仅凭夸张把好评倒读为讽刺差评。 |
| R030 / `train/neg/4661_3.txt` | 负向 | 新审 / medium | I enjoyed the movie from a curiosity stand point | 推荐主要基于历史与社会议题的观看价值，赞扬程度有限，但 worth watching 与 enjoyed 均明确。 |
| R032 / `train/neg/11755_3.txt` | 负向 | 新审 / high | i loved it it was pretty funny too. | 当前电影的好评、喜爱和幽默感评价反复明确，没有反转或对象歧义。 |
| R034 / `train/pos/10921_10.txt` | 正向 | 新审 / high | Don't be fooled just becuase this movie has a plot, it does not mean it's a good one. | 开头期望与局部风格赞语被剧情可预测、打斗不足的实际体验推翻，末尾明确否定。 |
| R036 / `train/pos/10112_7.txt` | 正向 | 新审 / high | It is interesting (somewhat) but was this really needed? | 对纪录片整体失望和必要性质疑明确；对 Wood 本人及旧片的差评是另一对象，末尾仅承认粉丝观看价值。 |
| R037 / `train/pos/7300_8.txt` | 正向 | 新审 / high | his introspective and dull treatment of his subjects is a constant disappointment. | 虽先谈此前作品，随后明确批评当前纪录片技术浪费、景别单调与持续失望；海洋生物本身的优美不等于影片好评。 |
| R038 / `train/neg/11325_4.txt` | 负向 | 新审 / medium | has all the right moves in this high-speed adventure that delivers plenty of action. | 主体是暴力剧情梗概，末段为动作表现的宣传式肯定。可读为温和正向，但缺少第一人称总体体验，置信度为中等。 |
| R039 / `train/neg/9370_4.txt` | 负向 | 新审 / high | See it if you have the opportunity. | 不是伟大电影的让步之后，明确推荐、感人、难忘并喜欢角色。 |
| R040 / `train/neg/2356_4.txt` | 负向 | 新审 / high | The film is fun to watch due to the fabulous dancing of Eleanor Powell. | 明确娱乐性好评，并持续肯定演员；间谍与 Trouble 描述情节，不是作者的质量否定。 |
| R041 / `train/pos/5206_7.txt` | 正向 | 新审 / high | The acting was rated a D | 对制作、演技、角色反应均明确否定；末尾仅限定极端血腥爱好者，不推翻整片差评。 |
| R042 / `train/pos/1359_7.txt` | 正向 | 新审 / high | Worth seeing for Posey and Townsend but that's about it. | 对两位演员的好感与电影坏明确区分；仅为演员值得看，不是总体推荐。 |
| R043 / `train/pos/4235_7.txt` | 正向 | 新审 / high | If you want a good ape movie watch the original. | 过去喜欢导演和原作不改变当前改编的明确负评；2/4 星与失望语气一致。 |
| R044 / `train/pos/12232_8.txt` | 正向 | 新审 / high | If this movie was meant to scare its audience, I think it missed its calling. | 海报收藏价值与整片差评明确分开；部分笑点不足以抵消 trite、garbage 与恐怖效果失败。 |
| R051 / `train/pos/960_7.txt` | 正向 | 新审 / high | nice visuals, the kid screams a lot, the river princess is terrible cute and the rest is crap. | 承认文化理解限制和视觉优点，但作者本人总评明确负面；不据此推断模型具有同样的文化理解缺陷。 |
| R052 / `train/pos/11693_7.txt` | 正向 | 新审 / medium | It all seems real fake and the editing is just bad. | 好评主要给旧版本和当前片末三分之一，主体否定当前片制作、表演和剪辑；建议等录像带是有限观看建议，保留中等置信。 |
| R053 / `train/neg/11037_3.txt` | 负向 | 新审 / high | Great writing and great casting along with great acting makes this a must see. | 明确赞扬该单集编剧、演员和表演；后半求剧照的附带问题不改变推荐。 |
| R054 / `train/neg/11315_4.txt` | 负向 | 新审 / high | Kathy Long's fights are extremely impressive. | 结局失望是明确让步，主体直言喜欢，并赞扬动作和摄影。 |
| R057 / `train/neg/10087_4.txt` | 负向 | 新审 / high | A creepy film; worth seeing! | 前文主要为情节介绍，末句直接推荐；creepy 为悬疑类型效果，不是负向质量判断。 |
| R058 / `train/neg/9050_1.txt` | 负向 | 新审 / high | I only wish I could have found a copy when my children were attending | 血腥、恶心反应被用来说明交通安全教育有效；作者明确希望让学生和孩子观看，不能把情绪不适等同于电影差评。 |
| R059 / `train/neg/8576_1.txt` | 负向 | 新审 / high | generates a fast pace that carries proceedings along in zestful tempo | 节奏与喜剧娱乐性直接肯定；sinister、villainy、horror 描述情节类型与战争背景，不是整体负评。 |
| R061 / `train/neg/2287_4.txt` | 负向 | 新审 / high | wrote and directed an outstanding biopic | 演技与整部传记片均明确好评；wrong、evils 描述人物道德争议，不是作者对影片的否定。 |

### 态度歧义或缺少本人评价

| 编号 / ID | 原标签 | 来源 / 置信度 | 证据摘要 | 判断依据 |
|---|---|---|---|---|
| L001 / `train/neg/1576_1.txt` | 负向 | 旧记录 / legacy_unrated | the epic feel of say, 'the godfather' | 跨名片和明星的过度夸奖疑似反讽，缺少明确片名或反向具体证据，先隔离不强定。 |
| L004 / `train/neg/2990_2.txt` | 负向 | 旧记录 / legacy_unrated | No attention to detail was left out! | 可能是反讽差评，也可能真心享受烂片；没有明确本人负面总评锚点，不凭原负标签倒推。 |
| L010 / `train/neg/6315_1.txt` | 负向 | 旧记录 / legacy_unrated | it surely defines the word 'masterpiece'. | 可能对性教育影片反讽，但文本缺少足够明确的反向线索，不能靠原负标签倒推讽刺。 |
| L014 / `train/neg/8620_2.txt` | 负向 | 旧记录 / legacy_unrated | My personal rating is 10/10. | 通篇夸张赞誉与低分冲突，可能反讽，但仅凭正文不足确定反向整体态度，隔离标签张力。 |
| L016 / `train/neg/9245_2.txt` | 负向 | 旧记录 / legacy_unrated | major on-screen chemistry when they talk about what the Start button does. | 软件教学被电影化夸赞可能是玩笑，但不等于明确不喜欢；正文还写10/10，与负标签张力未消除。 |
| L018 / `train/pos/11585_10.txt` | 正向 | 旧记录 / legacy_unrated | This is one of the most bewilderedly trance-state inducing bad movies | 全篇差评，尾部Enjoy和Favourite可能camp享受也可能讥讽，正类10分不宜仅据此强定；前缀没有享受总评。 |
| R003 / `train/neg/2734_4.txt` | 负向 | 新审 / low | it was much more of an end of season episode | 对剧集整体和下一季的喜爱明确，但对当前收尾单集有保留。不能把系列好评直接等同于目标单集的明确正评。 |
| R007 / `train/neg/1428_1.txt` | 负向 | 新审 / low | a better nostalgia rip-off made-for- T.V. movie | masterpiece、outstanding、建议录制与 rip-off、Direct-to-video 并置，既可是真实的类型欣赏，也可能反讽；缺少消除歧义的总评锚点。 |
| R009 / `train/pos/6107_7.txt` | 正向 | 新审 / low | the film was kind of childish... 6/10 | 文字偏负，动作场面有肯定，结尾 6/10 提供温和正向数值但与总评语气有张力。保留为混合/边界，不单凭评分或批评自动定类。 |
| R010 / `train/pos/6128_7.txt` | 正向 | 新审 / low | The scene where Jack Frost attacks Shannon Elizabeth is worth watching a few times. | 承认低成本和俗套，同时欣赏一场戏；没有明确整片推荐或否定。分级提醒也不等于质量差评。 |
| R012 / `train/pos/90_7.txt` | 正向 | 新审 / low | movies that don't all revolve around the poor little rich boys and girls | 明确评价是 OK/不突出，后半愤怒主要指向富人题材与叙事选择；缺少清晰整片正负倾向，不能把角色敌意直接等同于影片差评。 |
| R015 / `train/pos/9894_8.txt` | 正向 | 新审 / low | All in all it's true that "Slaughter High" is a piece of garbage,but I enjoyed it. | 电影质量被强烈否定而个人娱乐体验明确正向，两种评价维度并存；仅凭 enjoyed 不能判定原质量评分标签错误。 |
| R020 / `train/pos/4745_7.txt` | 正向 | 新审 / low | Black Mask is a decent film but do yourself a favor and pick up Fist of Legend | 对当前片说 decent，批评剪辑与火爆场面，并更推荐另一部。可理解为温和正评或失望保留，不以替代片推荐自动判为本片负评。 |
| R023 / `train/pos/7898_10.txt` | 正向 | 新审 / low | worthwhile for something different. | 主体强烈批评剧本与尴尬感，结尾又承认幽默并称观看 worthwhile；不推荐与值得看同现，保留总评歧义。 |
| R035 / `train/neg/1577_2.txt` | 负向 | 新审 / low | My one regret? That I didn't read the novel first. | 持续夸张赞语中夹有忘记女演员名字、漏油曲轴箱比喻等反讽线索；没有像 R033 那样的明确反转，文本本身不足以高置信确定最终极性。 |
| R045 / `train/pos/4041_7.txt` | 正向 | 新审 / low | Now if the payoff was half as good as the build up was I would have rated this so much higher. My Grade: C+ | 铺垫喜欢、结局失望、C+ 评分并存，属于温和/边界评价；没有预先约定字母评分到二分类标签的映射，不强制归正或归负。 |
| R046 / `train/pos/1358_10.txt` | 正向 | 新审 / low | Watch with caution and maintain a conscience. | 承认恶意围观的娱乐感，同时批评节目伦理和参赛者；没有清晰的本人整体推荐或否定，不能把讨厌人物直接映射为节目负评。 |
| R047 / `train/pos/10886_10.txt` | 正向 | 新审 / low | what makes it more enjoyable than any of the other neo-slasher flicks. | 常规质量标准明确否定，类型趣味标准又把它列为同类最值得看；两个评价维度冲突，不依据某一句推荐自动认定标签错误。 |
| R049 / `train/neg/8443_4.txt` | 负向 | 新审 / low | This is my translation of the DVD cover text synopsis. | 作者明确只翻译 DVD 封底剧情梗概，没有本人质量或喜爱评价；剧情中的死亡、疏离不能充当电影负评。 |
| R050 / `train/pos/10027_7.txt` | 正向 | 新审 / low | Out of a possible 100% I give this film a mere 71%. | 重看失望、爱情线批评与表演/导演肯定并存，最终 71% 又属偏高数值。不能在没有标签评分映射的前提下只取负面文字或数值一端。 |
| R055 / `train/pos/8512_7.txt` | 正向 | 新审 / low | aside from that I can't really remember anything. | 虽以坏电影开头，作者承认尚未看过成片，正文主要担忧自己的醉酒出镜与拍摄经历；不足以形成有依据的成片极性判断。 |
| R056 / `train/pos/773_7.txt` | 正向 | 新审 / low | Thus,I guess its worth seeing if you don't get annoyed | 多数文字批评俗套、演技与平庸，结尾又给条件性值得看的判断；总评边界不清，保留混合评价。 |
| R060 / `train/neg/1892_3.txt` | 负向 | 新审 / low | Bakhtiari region of Zardkuh is in Iran not in Irak | Fantastic 是引用另一评论者；本人主要纠正地理归属并致谢，未给出明确整片质量或喜爱判断。 |

### 决定性证据在截断外

| 编号 / ID | 原标签 | 来源 / 置信度 | 证据摘要 | 判断依据 |
|---|---|---|---|---|
| L006 / `train/neg/4006_4.txt` | 负向 | 旧记录 / legacy_unrated | So, all in all, I was a little disappointed. | 前缀几乎全是对原系列和角色回归的赞誉，实质批评剧情政治化及失望在截断后，不能仅当可见语义难例。 |
| L008 / `train/neg/4407_4.txt` | 负向 | 旧记录 / legacy_unrated | Make no mistake, this is a pretty bad movie! | 前缀赞想象力和趣味，明确差评及演技批评在截断后，存在输入缺失解释。 |
| L026 / `train/pos/5945_10.txt` | 正向 | 旧记录 / legacy_unrated | this is absolutely awesome stuff! | 前缀主要泛批好莱坞选角，当前短片明确好评在尾部，属于对象切换伴随截断。 |
| R026 / `train/pos/7287_7.txt` | 正向 | 新审 / medium | you need to snap this one up and now! Four stars for its cheesy-bizarreness alone! | 全文末尾明确为猎奇趣味打四星并推荐同好购买；前缀主要是荒唐剧情与制作错误，未见明确总体欣赏。决定性推荐在截断外。 |
| R048 / `train/neg/8278_1.txt` | 负向 | 新审 / high | I'm being sarcastic. 1/10 | 全文末尾明确声明反讽并给 1/10，负向无歧义；该声明在截断外，前缀主要为夸张赞语，仅凭前缀难以稳妥反向定类。 |
| R062 / `train/pos/9175_10.txt` | 正向 | 新审 / high | this is the definitive and only true "Jane Eyre". | 全文重点推崇 1983 版，前缀却主要批评 2006 与 1973 两版；针对 1983 版的决定性赞扬均在截断外。前缀缺少当前目标的正评，不能当作可见输入上的明确模型理解失败。 |

### 可见输入上的模型错误候选

| 编号 / ID | 原标签 | 来源 / 置信度 | 证据摘要 | 判断依据 |
|---|---|---|---|---|
| L012 / `train/neg/7461_2.txt` | 负向 | 旧记录 / legacy_unrated | Reminiscent of John Reynolds' performance in `Manos, the Hands of Fate,' | 夸张天才赞誉配合荒唐剧情及放屁体现社会不公，前缀足以识别持续反讽。 |
| L015 / `train/neg/9137_1.txt` | 负向 | 旧记录 / legacy_unrated | I never even knew they had palm trees in Vermont | 佛蒙特棕榈树等具体穿帮与不可能的全能天才吹捧构成反讽。 |
| L030 / `train/pos/9240_10.txt` | 正向 | 旧记录 / legacy_unrated | the movie is so bad, it's real fun to watch. | 差制作但真实有趣，末尾明确建议观看。 |
| L032 / `train/pos/9966_10.txt` | 正向 | 旧记录 / legacy_unrated | this movie must get a ten rating because it has to be seen to be believed. | 十分彩用离奇吸引力标准且每年重看，不能将worst字面等同不喜欢。 |
| R002 / `train/neg/455_4.txt` | 负向 | 新审 / medium | should be offering it. | 把观影与脑损伤、电击治疗并置，形成持续的挖苦语境；开头的最高成就和剧情赞语不能孤立读取。讽刺文本仍保留中等置信，不作客观错标确认。 |
| R031 / `train/pos/1005_10.txt` | 正向 | 新审 / medium | This anime isn't perfect, but it's at least mostly accurate, as best I can tell. | 绝大多数差评针对另一部真人电影；当前动画仅在末句获得忠实原著的温和肯定。目标识别是关键，不能把前文辱骂移到当前动画。 |
| R033 / `train/neg/3654_1.txt` | 负向 | 新审 / high | Please note: All of the above is opposite for the film in question. | 末句明确要求将此前全部赞语反读为相反评价，是可直接验证的全局反转；该句仍在 466-token 前缀内。 |
