# 94 条训练高冲突候选：完整信息

只含原梯度训练分区的 94 条候选，没有开发样本。保留原标签与原文；未重新训练或推理。

筛选：baseline-s43 与 combined-s43 均预测为给定标签的相反类别，且反向原始 logit margin 均 >1。两者训练 seed 都是 43，属于不同配方，不能称跨种子一致。导出发生在已参与训练的样本上，不是 out-of-fold 预测。

模型输入为旧 512 总长度下的原始前缀，评论预算 466 token；16 条在这个规则下截断。其余正文也完整保留供复核。margin 是原始 logits 差，不是校准概率，也不表示不同模型同等置信。

原标签：负向 49 条、正向 45 条。文件名原评分与原标签对齐。旧 AI 审阅分类：冲突候选 58、歧义 23、截断缺证据 6、可见模型错误候选 7。旧记录复用 32 条，新 AI 审阅 62 条；尚无独立人工复核，确认错标数量未知。

## 清单索引

| 编号 | 原审阅号 | 样本 ID | 原评分 | 原标签→两模型预测 | tokens | 前缀截断 | baseline / combined 反向 margin | 旧 AI 工作分类 |
|---|---|---|---:|---|---:|---|---|---|
| [C001](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c001) | R057 | `train/neg/10087_4.txt` | 4 | 负向→正向 | 119 | 否 | 4.7590 / 3.4389 | 标签—文本态度冲突候选 |
| [C002](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c002) | R053 | `train/neg/11037_3.txt` | 3 | 负向→正向 | 142 | 否 | 8.8703 / 3.7486 | 标签—文本态度冲突候选 |
| [C003](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c003) | R054 | `train/neg/11315_4.txt` | 4 | 负向→正向 | 67 | 否 | 7.4061 / 3.7162 | 标签—文本态度冲突候选 |
| [C004](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c004) | R038 | `train/neg/11325_4.txt` | 4 | 负向→正向 | 216 | 否 | 2.5229 / 3.5361 | 标签—文本态度冲突候选 |
| [C005](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c005) | R032 | `train/neg/11755_3.txt` | 3 | 负向→正向 | 63 | 否 | 8.7807 / 3.6556 | 标签—文本态度冲突候选 |
| [C006](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c006) | R025 | `train/neg/11940_4.txt` | 4 | 负向→正向 | 476 | 是 | 4.6563 / 3.4219 | 标签—文本态度冲突候选 |
| [C007](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c007) | R004 | `train/neg/11947_4.txt` | 4 | 负向→正向 | 140 | 否 | 7.6636 / 3.6866 | 标签—文本态度冲突候选 |
| [C008](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c008) | R029 | `train/neg/1201_2.txt` | 2 | 负向→正向 | 72 | 否 | 8.9230 / 3.7052 | 标签—文本态度冲突候选 |
| [C009](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c009) | R028 | `train/neg/12061_4.txt` | 4 | 负向→正向 | 90 | 否 | 8.0815 / 3.7055 | 标签—文本态度冲突候选 |
| [C010](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c010) | R007 | `train/neg/1428_1.txt` | 1 | 负向→正向 | 123 | 否 | 7.0895 / 3.7682 | 态度歧义或缺少本人评价 |
| [C011](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c011) | R005 | `train/neg/1435_4.txt` | 4 | 负向→正向 | 344 | 否 | 5.3302 / 3.7260 | 标签—文本态度冲突候选 |
| [C012](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c012) | L001 | `train/neg/1576_1.txt` | 1 | 负向→正向 | 204 | 否 | 9.0711 / 3.7569 | 态度歧义或缺少本人评价 |
| [C013](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c013) | R035 | `train/neg/1577_2.txt` | 2 | 负向→正向 | 530 | 是 | 8.7856 / 3.7914 | 态度歧义或缺少本人评价 |
| [C014](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c014) | R060 | `train/neg/1892_3.txt` | 3 | 负向→正向 | 181 | 否 | 4.2169 / 3.5230 | 态度歧义或缺少本人评价 |
| [C015](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c015) | R061 | `train/neg/2287_4.txt` | 4 | 负向→正向 | 151 | 否 | 2.9514 / 2.8385 | 标签—文本态度冲突候选 |
| [C016](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c016) | L002 | `train/neg/2355_3.txt` | 3 | 负向→正向 | 459 | 否 | 8.8284 / 3.7805 | 标签—文本态度冲突候选 |
| [C017](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c017) | R040 | `train/neg/2356_4.txt` | 4 | 负向→正向 | 163 | 否 | 4.0297 / 3.5061 | 标签—文本态度冲突候选 |
| [C018](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c018) | R003 | `train/neg/2734_4.txt` | 4 | 负向→正向 | 153 | 否 | 4.8174 / 3.6277 | 态度歧义或缺少本人评价 |
| [C019](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c019) | L003 | `train/neg/2985_4.txt` | 4 | 负向→正向 | 529 | 是 | 8.8586 / 3.7072 | 标签—文本态度冲突候选 |
| [C020](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c020) | L004 | `train/neg/2990_2.txt` | 2 | 负向→正向 | 408 | 否 | 8.2871 / 3.0945 | 态度歧义或缺少本人评价 |
| [C021](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c021) | R014 | `train/neg/3270_1.txt` | 1 | 负向→正向 | 174 | 否 | 8.7474 / 3.7192 | 标签—文本态度冲突候选 |
| [C022](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c022) | R033 | `train/neg/3654_1.txt` | 1 | 负向→正向 | 329 | 否 | 8.8174 / 3.7655 | 可见输入上的模型错误候选 |
| [C023](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c023) | R008 | `train/neg/3677_3.txt` | 3 | 负向→正向 | 72 | 否 | 8.6878 / 3.6556 | 标签—文本态度冲突候选 |
| [C024](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c024) | R027 | `train/neg/3843_3.txt` | 3 | 负向→正向 | 683 | 是 | 8.6062 / 3.7917 | 标签—文本态度冲突候选 |
| [C025](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c025) | L005 | `train/neg/3932_4.txt` | 4 | 负向→正向 | 138 | 否 | 7.1911 / 3.7660 | 标签—文本态度冲突候选 |
| [C026](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c026) | L006 | `train/neg/4006_4.txt` | 4 | 负向→正向 | 1197 | 是 | 8.7518 / 3.7759 | 决定性证据在截断外 |
| [C027](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c027) | L007 | `train/neg/4239_4.txt` | 4 | 负向→正向 | 154 | 否 | 4.2342 / 3.5967 | 标签—文本态度冲突候选 |
| [C028](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c028) | L008 | `train/neg/4407_4.txt` | 4 | 负向→正向 | 1018 | 是 | 3.8291 / 3.5072 | 决定性证据在截断外 |
| [C029](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c029) | R002 | `train/neg/455_4.txt` | 4 | 负向→正向 | 583 | 是 | 8.5508 / 3.6961 | 可见输入上的模型错误候选 |
| [C030](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c030) | R030 | `train/neg/4661_3.txt` | 3 | 负向→正向 | 193 | 否 | 5.0974 / 3.2796 | 标签—文本态度冲突候选 |
| [C031](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c031) | R013 | `train/neg/5558_4.txt` | 4 | 负向→正向 | 173 | 否 | 8.6433 / 3.7005 | 标签—文本态度冲突候选 |
| [C032](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c032) | L009 | `train/neg/6038_3.txt` | 3 | 负向→正向 | 124 | 否 | 4.5578 / 3.6617 | 标签—文本态度冲突候选 |
| [C033](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c033) | L010 | `train/neg/6315_1.txt` | 1 | 负向→正向 | 154 | 否 | 9.1047 / 3.7564 | 态度歧义或缺少本人评价 |
| [C034](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c034) | R022 | `train/neg/6461_4.txt` | 4 | 负向→正向 | 51 | 否 | 8.1278 / 3.7017 | 标签—文本态度冲突候选 |
| [C035](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c035) | L011 | `train/neg/6473_3.txt` | 3 | 负向→正向 | 224 | 否 | 1.6537 / 3.3092 | 标签—文本态度冲突候选 |
| [C036](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c036) | R018 | `train/neg/6601_4.txt` | 4 | 负向→正向 | 76 | 否 | 6.0179 / 3.4593 | 标签—文本态度冲突候选 |
| [C037](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c037) | L012 | `train/neg/7461_2.txt` | 2 | 负向→正向 | 657 | 是 | 9.0020 / 3.7760 | 可见输入上的模型错误候选 |
| [C038](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c038) | L013 | `train/neg/801_4.txt` | 4 | 负向→正向 | 197 | 否 | 7.5375 / 3.6331 | 标签—文本态度冲突候选 |
| [C039](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c039) | R048 | `train/neg/8278_1.txt` | 1 | 负向→正向 | 531 | 是 | 9.0234 / 3.7664 | 决定性证据在截断外 |
| [C040](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c040) | R016 | `train/neg/8314_1.txt` | 1 | 负向→正向 | 606 | 是 | 8.9540 / 3.7770 | 标签—文本态度冲突候选 |
| [C041](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c041) | R049 | `train/neg/8443_4.txt` | 4 | 负向→正向 | 179 | 否 | 3.9650 / 2.9291 | 态度歧义或缺少本人评价 |
| [C042](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c042) | R059 | `train/neg/8576_1.txt` | 1 | 负向→正向 | 142 | 否 | 5.0934 / 3.0014 | 标签—文本态度冲突候选 |
| [C043](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c043) | L014 | `train/neg/8620_2.txt` | 2 | 负向→正向 | 317 | 否 | 9.2422 / 3.8021 | 态度歧义或缺少本人评价 |
| [C044](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c044) | R017 | `train/neg/8656_4.txt` | 4 | 负向→正向 | 197 | 否 | 2.6733 / 3.1430 | 标签—文本态度冲突候选 |
| [C045](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c045) | R058 | `train/neg/9050_1.txt` | 1 | 负向→正向 | 253 | 否 | 5.7688 / 3.7107 | 标签—文本态度冲突候选 |
| [C046](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c046) | L015 | `train/neg/9137_1.txt` | 1 | 负向→正向 | 356 | 否 | 9.0479 / 3.7702 | 可见输入上的模型错误候选 |
| [C047](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c047) | L016 | `train/neg/9245_2.txt` | 2 | 负向→正向 | 86 | 否 | 9.2317 / 3.7466 | 态度歧义或缺少本人评价 |
| [C048](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c048) | R039 | `train/neg/9370_4.txt` | 4 | 负向→正向 | 136 | 否 | 4.8258 / 3.5520 | 标签—文本态度冲突候选 |
| [C049](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c049) | R021 | `train/neg/970_1.txt` | 1 | 负向→正向 | 112 | 否 | 8.9481 / 3.7831 | 标签—文本态度冲突候选 |
| [C050](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c050) | R050 | `train/pos/10027_7.txt` | 7 | 正向→负向 | 300 | 否 | 4.2644 / 2.0223 | 态度歧义或缺少本人评价 |
| [C051](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c051) | R031 | `train/pos/1005_10.txt` | 10 | 正向→负向 | 130 | 否 | 6.2355 / 3.6875 | 可见输入上的模型错误候选 |
| [C052](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c052) | R036 | `train/pos/10112_7.txt` | 7 | 正向→负向 | 277 | 否 | 4.9854 / 3.6573 | 标签—文本态度冲突候选 |
| [C053](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c053) | L017 | `train/pos/10876_7.txt` | 7 | 正向→负向 | 204 | 否 | 8.6635 / 3.7215 | 标签—文本态度冲突候选 |
| [C054](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c054) | R047 | `train/pos/10886_10.txt` | 10 | 正向→负向 | 662 | 是 | 6.3499 / 3.5980 | 态度歧义或缺少本人评价 |
| [C055](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c055) | R034 | `train/pos/10921_10.txt` | 10 | 正向→负向 | 283 | 否 | 5.1081 / 3.4117 | 标签—文本态度冲突候选 |
| [C056](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c056) | R006 | `train/pos/11393_7.txt` | 7 | 正向→负向 | 312 | 否 | 6.4394 / 1.5581 | 标签—文本态度冲突候选 |
| [C057](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c057) | L018 | `train/pos/11585_10.txt` | 10 | 正向→负向 | 596 | 是 | 8.3331 / 3.7023 | 态度歧义或缺少本人评价 |
| [C058](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c058) | R052 | `train/pos/11693_7.txt` | 7 | 正向→负向 | 180 | 否 | 4.4914 / 3.7206 | 标签—文本态度冲突候选 |
| [C059](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c059) | R044 | `train/pos/12232_8.txt` | 8 | 正向→负向 | 213 | 否 | 4.5431 / 3.7038 | 标签—文本态度冲突候选 |
| [C060](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c060) | L019 | `train/pos/12276_7.txt` | 7 | 正向→负向 | 112 | 否 | 7.1556 / 3.7015 | 标签—文本态度冲突候选 |
| [C061](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c061) | R046 | `train/pos/1358_10.txt` | 10 | 正向→负向 | 201 | 否 | 7.0574 / 3.5901 | 态度歧义或缺少本人评价 |
| [C062](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c062) | R042 | `train/pos/1359_7.txt` | 7 | 正向→负向 | 186 | 否 | 7.7372 / 3.6984 | 标签—文本态度冲突候选 |
| [C063](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c063) | L020 | `train/pos/1419_7.txt` | 7 | 正向→负向 | 992 | 是 | 8.5312 / 3.6727 | 标签—文本态度冲突候选 |
| [C064](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c064) | L021 | `train/pos/1778_10.txt` | 10 | 正向→负向 | 223 | 否 | 6.7651 / 3.6761 | 标签—文本态度冲突候选 |
| [C065](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c065) | R011 | `train/pos/245_9.txt` | 9 | 正向→负向 | 238 | 否 | 6.3535 / 3.7051 | 标签—文本态度冲突候选 |
| [C066](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c066) | L022 | `train/pos/3372_9.txt` | 9 | 正向→负向 | 415 | 否 | 8.7934 / 3.7332 | 标签—文本态度冲突候选 |
| [C067](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c067) | L023 | `train/pos/3653_10.txt` | 10 | 正向→负向 | 103 | 否 | 8.8541 / 3.7570 | 标签—文本态度冲突候选 |
| [C068](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c068) | R045 | `train/pos/4041_7.txt` | 7 | 正向→负向 | 297 | 否 | 5.5963 / 3.6281 | 态度歧义或缺少本人评价 |
| [C069](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c069) | R043 | `train/pos/4235_7.txt` | 7 | 正向→负向 | 105 | 否 | 5.6699 / 3.6597 | 标签—文本态度冲突候选 |
| [C070](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c070) | R020 | `train/pos/4745_7.txt` | 7 | 正向→负向 | 200 | 否 | 2.2057 / 3.2050 | 态度歧义或缺少本人评价 |
| [C071](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c071) | L024 | `train/pos/5086_7.txt` | 7 | 正向→负向 | 179 | 否 | 7.3661 / 3.6249 | 标签—文本态度冲突候选 |
| [C072](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c072) | L025 | `train/pos/5149_8.txt` | 8 | 正向→负向 | 246 | 否 | 6.6824 / 3.6717 | 标签—文本态度冲突候选 |
| [C073](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c073) | R041 | `train/pos/5206_7.txt` | 7 | 正向→负向 | 178 | 否 | 5.5240 / 3.7069 | 标签—文本态度冲突候选 |
| [C074](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c074) | L026 | `train/pos/5945_10.txt` | 10 | 正向→负向 | 864 | 是 | 8.1246 / 3.6841 | 决定性证据在截断外 |
| [C075](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c075) | L027 | `train/pos/6038_10.txt` | 10 | 正向→负向 | 194 | 否 | 9.0201 / 3.7421 | 标签—文本态度冲突候选 |
| [C076](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c076) | R009 | `train/pos/6107_7.txt` | 7 | 正向→负向 | 98 | 否 | 3.8046 / 3.6047 | 态度歧义或缺少本人评价 |
| [C077](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c077) | R010 | `train/pos/6128_7.txt` | 7 | 正向→负向 | 256 | 否 | 4.2979 / 3.5246 | 态度歧义或缺少本人评价 |
| [C078](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c078) | R026 | `train/pos/7287_7.txt` | 7 | 正向→负向 | 679 | 是 | 7.0902 / 3.7242 | 决定性证据在截断外 |
| [C079](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c079) | R037 | `train/pos/7300_8.txt` | 8 | 正向→负向 | 256 | 否 | 5.5806 / 3.3955 | 标签—文本态度冲突候选 |
| [C080](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c080) | L028 | `train/pos/7538_7.txt` | 7 | 正向→负向 | 68 | 否 | 7.4481 / 3.6884 | 标签—文本态度冲突候选 |
| [C081](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c081) | R056 | `train/pos/773_7.txt` | 7 | 正向→负向 | 151 | 否 | 5.6753 / 3.6954 | 态度歧义或缺少本人评价 |
| [C082](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c082) | R001 | `train/pos/7782_7.txt` | 7 | 正向→负向 | 135 | 否 | 5.7451 / 3.5915 | 标签—文本态度冲突候选 |
| [C083](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c083) | R023 | `train/pos/7898_10.txt` | 10 | 正向→负向 | 264 | 否 | 6.2449 / 3.6376 | 态度歧义或缺少本人评价 |
| [C084](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c084) | R055 | `train/pos/8512_7.txt` | 7 | 正向→负向 | 187 | 否 | 6.4066 / 3.7002 | 态度歧义或缺少本人评价 |
| [C085](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c085) | R024 | `train/pos/8682_7.txt` | 7 | 正向→负向 | 199 | 否 | 4.9030 / 3.3757 | 标签—文本态度冲突候选 |
| [C086](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c086) | R012 | `train/pos/90_7.txt` | 7 | 正向→负向 | 202 | 否 | 5.1218 / 3.4289 | 态度歧义或缺少本人评价 |
| [C087](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c087) | R062 | `train/pos/9175_10.txt` | 10 | 正向→负向 | 1201 | 是 | 8.0934 / 3.7948 | 决定性证据在截断外 |
| [C088](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c088) | L029 | `train/pos/9216_10.txt` | 10 | 正向→负向 | 82 | 否 | 8.6826 / 3.7414 | 标签—文本态度冲突候选 |
| [C089](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c089) | L030 | `train/pos/9240_10.txt` | 10 | 正向→负向 | 162 | 否 | 8.8396 / 3.7337 | 可见输入上的模型错误候选 |
| [C090](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c090) | R051 | `train/pos/960_7.txt` | 7 | 正向→负向 | 180 | 否 | 5.7976 / 3.6453 | 标签—文本态度冲突候选 |
| [C091](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c091) | R019 | `train/pos/9646_7.txt` | 7 | 正向→负向 | 164 | 否 | 3.5174 / 3.3664 | 标签—文本态度冲突候选 |
| [C092](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c092) | L031 | `train/pos/9824_10.txt` | 10 | 正向→负向 | 133 | 否 | 7.9252 / 3.6992 | 标签—文本态度冲突候选 |
| [C093](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c093) | R015 | `train/pos/9894_8.txt` | 8 | 正向→负向 | 255 | 否 | 1.8237 / 2.1186 | 态度歧义或缺少本人评价 |
| [C094](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md#c094) | L032 | `train/pos/9966_10.txt` | 10 | 正向→负向 | 244 | 否 | 8.6916 / 3.6804 | 可见输入上的模型错误候选 |

## 完整材料

- [94 条全文、模型分数与已有 AI 审阅](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates-full.md)
- [完整 JSON 数据](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/artifacts/laya-94-candidates-information-20261017-v1/94-candidates.json)

信息来源为原训练导出及 2026-10-15 已完成的 AI 审阅台账。本次只做整理和一致性核验，未把任何条目标成已确认错标，也未删样本、改标签或分配训练权重。此 94 条是模型冲突富集样本，不能估计整套 IMDB 的标签噪声率。
