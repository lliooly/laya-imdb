# 用户人工裁定：训练数据修订 v2

原始训练集、上一版修订均保留；本版按用户两轮裁定累计翻转 61 条、确认保持 23 条、删除 3 条、待定 7 条。最终决定归属于用户人工裁定，先前 AI 注释仅为背景资料。

**C012 已澄清：保持 neg。** 用户选择“按反讽判断，保持 neg”，覆盖先前“翻转”字样。C010 按用户明确的反讽负向理由保持 neg。

## 本轮新增操作

| 操作 | 数量 | 编号 |
|---|---:|---|
| 翻转 | 3 | C018, C057, C081 |
| 保持 | 14 | C010, C012, C013, C033, C043, C047, C050, C054, C070, C077, C078, C089, C093, C094 |
| 删除 | 3 | C014, C041, C084 |
| 待定，暂留原标签 | 7 | C020, C026, C061, C068, C076, C083, C086 |

## 训练版本与备份

- [新训练数据](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/data/label-revisions/laya-94-user-label-revision-20261017-v2/train.jsonl)；[新 manifest](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/data/label-revisions/laya-94-user-label-revision-20261017-v2/manifest.json)。
- [原始训练数据](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/data/processed/train.jsonl)：22500 条；[上一版](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/data/label-revisions/laya-94-user-label-revision-20261017-v1/train.jsonl)：22500 条。本版保留 22497 条。
- 原梯度 22100 条中移除 3 条，现为 22097 条；原 400 条校准完整保留，开发 / 测试不改、不重新划分。
- [删除的 3 条正文备份](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/data/label-revisions/laya-94-user-label-revision-20261017-v2/removed-training-rows.jsonl)；[全部 94 条原标签备份](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/data/label-revisions/laya-94-user-label-revision-20261017-v2/original-labels-94.json)。
- [累计修改补丁](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/data/label-revisions/laya-94-user-label-revision-20261017-v2/cumulative-label-patch.json)；[剩余 7 条单独文档](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/reports/laya-94-user-label-revision-20261017-v2/unresolved-7.md)。

## 本轮逐条裁定

| 编号 | 样本 ID | 操作 | 原标签 → 本版 | 用户理由 |
|---|---|---|---|---|
| C010 | `train/neg/1428_1.txt` | 保持 | neg → neg | 反讽而非真心赞美；nostalgia rip-off、made-for-TV、direct-to-video 构成贬义及拆台。 |
| C012 | `train/neg/1576_1.txt` | 保持 | neg → neg | 补充澄清明确选择保持 neg；借经典作品和演员形象的错位进行反讽，不是真心赞美。此补充决定覆盖前文“翻转”。 |
| C013 | `train/neg/1577_2.txt` | 保持 | neg → neg | what's-her-name、leaky crankcase 及经典情爱作品与失控汽车的错位构成反讽，未见烂片真欣赏锚点。 |
| C014 | `train/neg/1892_3.txt` | 删除 | neg → 删除 | 作者仅纠正地理与致谢；Fantastic documentary 是引用他人，不含本人对影片的评价。 |
| C018 | `train/neg/2734_4.txt` | 翻转 | neg → pos | 改正；本集问题被称可原谅的 hiccups，全文肯定剧集并期待第七季，缺少负向总评。 |
| C020 | `train/neg/2990_2.txt` | 待定，暂留原标签 | neg → neg | 恶搞差评与烂片粉丝真欣赏两读均成立，缺少明确总评锚点。 |
| C026 | `train/neg/4006_4.txt` | 待定，暂留原标签 | neg → neg | 角色回归盛赞与政治化批评并存；a little disappointed 与 recommended viewing / still good TV 都是作者总评。 |
| C033 | `train/neg/6315_1.txt` | 保持 | neg → neg | 性教育片题材与 sleek, sexy、edge of your seat、masterpiece 等赞词错位，是恶搞。 |
| C041 | `train/neg/8443_4.txt` | 删除 | neg → 删除 | DVD 封底梗概翻译，无主观判断；剧情中的死亡不能作为作者差评。 |
| C043 | `train/neg/8620_2.txt` | 保持 | neg → neg | 情色 B 级片被当学院奖遗珠并与经典作品并列；10/10 属于假仰慕腔的包袱。 |
| C047 | `train/neg/9245_2.txt` | 保持 | neg → neg | Windows 95 教程讲开始菜单被吹成杰作和金球奖级表演；Overall: 10/10 是包袱。 |
| C050 | `train/pos/10027_7.txt` | 保持 | pos → pos | 二刷厌烦及演员批评不等于全片否定；肯定女主与导演，作者总分 71% 与正类同侧。 |
| C054 | `train/pos/10886_10.txt` | 保持 | pos → pos | but 将常规质量缺点收为类型趣味，且明确 only one worth seeing；观看体验正向。 |
| C057 | `train/pos/11585_10.txt` | 翻转 | pos → neg | 改负；配乐、拟音、剧本、剪辑、醉酒导演与 incoherent mess 构成主体差评；单独 Enjoy 不足以判正。 |
| C061 | `train/pos/1358_10.txt` | 待定，暂留原标签 | pos → pos | 围观快感与伦理反感、奖励缩水和 Watch with caution 并存；缺少整季明确推荐或否定。 |
| C068 | `train/pos/4041_7.txt` | 待定，暂留原标签 | pos → pos | 喜欢铺垫、结局失望，自评分 C+；未约定字母评分与二分类的对应。 |
| C070 | `train/pos/4745_7.txt` | 保持 | pos → pos | 总评 isn't terrible / a decent film；更推荐另一部作品不取消本片的温和肯定。 |
| C076 | `train/pos/6107_7.txt` | 待定，暂留原标签 | pos → pos | 演技人物负向、打斗及不用动脑可看正向；6/10 不足以形成干净的正类或负类。 |
| C077 | `train/pos/6128_7.txt` | 保持 | pos → pos | 承认 B 级及俗套但肯定不装神作与重看价值，是有保留的正向评价。 |
| C078 | `train/pos/7287_7.txt` | 保持 | pos → pos | 结尾面向 Ed Wood / Russ Meyer 影迷推荐购买，并给 Four stars for its cheesy-bizarreness alone。 |
| C081 | `train/pos/773_7.txt` | 翻转 | pos → neg | 改负；failed attempt、俗套、演技无力与 sunk into mediocrity 为作者总判断；条件推荐未取消作者自己的厌烦。 |
| C083 | `train/pos/7898_10.txt` | 待定，暂留原标签 | pos → pos | I wouldn't recommend this film 与 quite funny / worthwhile 都明确，无法单边裁定。 |
| C084 | `train/pos/8512_7.txt` | 删除 | pos → 删除 | 作者写明 I was in the movie and I haven't seen it yet；正文是拍摄经历，不是已看成片的评价。 |
| C086 | `train/pos/90_7.txt` | 待定，暂留原标签 | pos → pos | 对影片只有 OK...nothing really outstanding；之后转向批评富家子弟，不支持干净正类或翻负，也未明确要求删除。 |
| C089 | `train/pos/9240_10.txt` | 保持 | pos → pos | so bad, it's real fun to watch. Go check it out! 明确肯定观看体验。 |
| C093 | `train/pos/9894_8.txt` | 保持 | pos → pos | 总评 a piece of garbage, but I enjoyed it，面向烂砍杀片影迷，体验正向。 |
| C094 | `train/pos/9966_10.txt` | 保持 | pos → pos | 离奇到应给十分、has to be seen，并每年邀人重看，负面工艺词不取消真实欣赏。 |

## 核验与后续使用

- 原始版与 v1 的文件哈希均未变化。相对原始版，只翻转指定 61 条标签、删除指定 3 条；相对 v1，仅新增 3 条翻转和 3 条删除。其余行字节一致、保留样本顺序与正文不变。
- 本版类别数：neg 11243、pos 11254；梯度部分 neg 11058、pos 11039。
- 校准 400 条不受修改；开发与官方测试正文未读取。待定 7 条原样保留。样本 ID 和文件名评分仍表示原始来源，训练标签以本版 label 字段为准。
- 已具备后续原版 / 改版对照所需的数据版本与出处；下次须用新 run_id 绑定本版 manifest。本轮没有启动训练，也没有因本次修改重算历史模型成绩。
- 复核：`uv run --locked --offline --no-sync python scripts/apply_laya_label_revision_v2.py --verify`。
