# 94 条训练候选：按用户逐条裁定修订标签

**完成：58 条翻转（neg→pos 32、pos→neg 26）、9 条确认保持、27 条单列待定。未启动训练。**

逐条列举与标题数量不符：标题写“翻转共54条／负转正29条／正转负22条”，实际明确列出32＋26＝58条。执行以唯一的样本编号及 ID 清单为准，没有增补或漏改。待定27条与附件中实际列出的27个编号完全一致。

## 修改后的数据

- [修订版 train.jsonl](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/data/label-revisions/laya-94-user-label-revision-20261017-v1/train.jsonl)：22500 条、原顺序及 ID 保持，仅指定 58 行的 label 字段变化。
- [新数据 manifest](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/data/label-revisions/laya-94-user-label-revision-20261017-v1/manifest.json)：原 22100 梯度／400 校准分区保持；全部翻转位于梯度训练分区。
- [完整修改补丁](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/data/label-revisions/laya-94-user-label-revision-20261017-v1/label-patch.json)；[94 条原标签备份](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/data/label-revisions/laya-94-user-label-revision-20261017-v1/original-labels-94.json)。
- 原冻结数据 `data/processed/train.jsonl` 及其 manifest、历史报告均保留。修订版不直接覆盖旧实验源哈希；下次训练需用新 run_id 和本版本 manifest。
- 样本 ID 中 neg/pos 目录及文件名评分保留为原始出处，不按新语义标签重命名，也不据路径重新推导有效标签。

## 单独待定文档

- [27 条待定：全文、附件意见及先前 AI 审阅](/Users/shishishi/Dev-Project/NLP-Homework/End-IMDB/reports/laya-94-user-label-revision-20261017-v1/unresolved-27.md)。
- 待定全部暂留原标签。附件中“改为／或保持／或删除”等拟定意见不覆盖用户最终明确清单。

## 已执行的 58 条翻转

| 编号 | 样本 ID | 原标签 | 修订标签 |
|---|---|---|---|
| C001 | `train/neg/10087_4.txt` | neg（负向） | pos（正向） |
| C002 | `train/neg/11037_3.txt` | neg（负向） | pos（正向） |
| C003 | `train/neg/11315_4.txt` | neg（负向） | pos（正向） |
| C004 | `train/neg/11325_4.txt` | neg（负向） | pos（正向） |
| C005 | `train/neg/11755_3.txt` | neg（负向） | pos（正向） |
| C006 | `train/neg/11940_4.txt` | neg（负向） | pos（正向） |
| C007 | `train/neg/11947_4.txt` | neg（负向） | pos（正向） |
| C008 | `train/neg/1201_2.txt` | neg（负向） | pos（正向） |
| C009 | `train/neg/12061_4.txt` | neg（负向） | pos（正向） |
| C011 | `train/neg/1435_4.txt` | neg（负向） | pos（正向） |
| C015 | `train/neg/2287_4.txt` | neg（负向） | pos（正向） |
| C016 | `train/neg/2355_3.txt` | neg（负向） | pos（正向） |
| C017 | `train/neg/2356_4.txt` | neg（负向） | pos（正向） |
| C019 | `train/neg/2985_4.txt` | neg（负向） | pos（正向） |
| C021 | `train/neg/3270_1.txt` | neg（负向） | pos（正向） |
| C023 | `train/neg/3677_3.txt` | neg（负向） | pos（正向） |
| C024 | `train/neg/3843_3.txt` | neg（负向） | pos（正向） |
| C025 | `train/neg/3932_4.txt` | neg（负向） | pos（正向） |
| C027 | `train/neg/4239_4.txt` | neg（负向） | pos（正向） |
| C030 | `train/neg/4661_3.txt` | neg（负向） | pos（正向） |
| C031 | `train/neg/5558_4.txt` | neg（负向） | pos（正向） |
| C032 | `train/neg/6038_3.txt` | neg（负向） | pos（正向） |
| C034 | `train/neg/6461_4.txt` | neg（负向） | pos（正向） |
| C035 | `train/neg/6473_3.txt` | neg（负向） | pos（正向） |
| C036 | `train/neg/6601_4.txt` | neg（负向） | pos（正向） |
| C038 | `train/neg/801_4.txt` | neg（负向） | pos（正向） |
| C040 | `train/neg/8314_1.txt` | neg（负向） | pos（正向） |
| C042 | `train/neg/8576_1.txt` | neg（负向） | pos（正向） |
| C044 | `train/neg/8656_4.txt` | neg（负向） | pos（正向） |
| C045 | `train/neg/9050_1.txt` | neg（负向） | pos（正向） |
| C048 | `train/neg/9370_4.txt` | neg（负向） | pos（正向） |
| C049 | `train/neg/970_1.txt` | neg（负向） | pos（正向） |
| C052 | `train/pos/10112_7.txt` | pos（正向） | neg（负向） |
| C053 | `train/pos/10876_7.txt` | pos（正向） | neg（负向） |
| C055 | `train/pos/10921_10.txt` | pos（正向） | neg（负向） |
| C056 | `train/pos/11393_7.txt` | pos（正向） | neg（负向） |
| C058 | `train/pos/11693_7.txt` | pos（正向） | neg（负向） |
| C059 | `train/pos/12232_8.txt` | pos（正向） | neg（负向） |
| C060 | `train/pos/12276_7.txt` | pos（正向） | neg（负向） |
| C062 | `train/pos/1359_7.txt` | pos（正向） | neg（负向） |
| C063 | `train/pos/1419_7.txt` | pos（正向） | neg（负向） |
| C064 | `train/pos/1778_10.txt` | pos（正向） | neg（负向） |
| C065 | `train/pos/245_9.txt` | pos（正向） | neg（负向） |
| C066 | `train/pos/3372_9.txt` | pos（正向） | neg（负向） |
| C067 | `train/pos/3653_10.txt` | pos（正向） | neg（负向） |
| C069 | `train/pos/4235_7.txt` | pos（正向） | neg（负向） |
| C071 | `train/pos/5086_7.txt` | pos（正向） | neg（负向） |
| C072 | `train/pos/5149_8.txt` | pos（正向） | neg（负向） |
| C073 | `train/pos/5206_7.txt` | pos（正向） | neg（负向） |
| C075 | `train/pos/6038_10.txt` | pos（正向） | neg（负向） |
| C079 | `train/pos/7300_8.txt` | pos（正向） | neg（负向） |
| C080 | `train/pos/7538_7.txt` | pos（正向） | neg（负向） |
| C082 | `train/pos/7782_7.txt` | pos（正向） | neg（负向） |
| C085 | `train/pos/8682_7.txt` | pos（正向） | neg（负向） |
| C088 | `train/pos/9216_10.txt` | pos（正向） | neg（负向） |
| C090 | `train/pos/960_7.txt` | pos（正向） | neg（负向） |
| C091 | `train/pos/9646_7.txt` | pos（正向） | neg（负向） |
| C092 | `train/pos/9824_10.txt` | pos（正向） | neg（负向） |

## 确认保持的 9 条

| 编号 | 样本 ID | 保持标签 |
|---|---|---|
| C022 | `train/neg/3654_1.txt` | neg（负向） |
| C028 | `train/neg/4407_4.txt` | neg（负向） |
| C029 | `train/neg/455_4.txt` | neg（负向） |
| C037 | `train/neg/7461_2.txt` | neg（负向） |
| C039 | `train/neg/8278_1.txt` | neg（负向） |
| C046 | `train/neg/9137_1.txt` | neg（负向） |
| C051 | `train/pos/1005_10.txt` | pos（正向） |
| C074 | `train/pos/5945_10.txt` | pos（正向） |
| C087 | `train/pos/9175_10.txt` | pos（正向） |

## 核验

- 新训练集类别数：负向 11244、正向 11256。原数据各11250；正向净增6。
- 对22500条逐行比较：仅58个指定label改变，全部正文／ID／文本哈希不变；其余22442行连序列化字节也保持。400条校准不变，开发与官方测试正文未读取。
- 用户裁定作为此次标签修订的依据；不将旧AI分类更新成独立人工一致性证明。原始评分和旧标签已保留用于版本比较。
- 复核命令：`uv run --locked --offline --no-sync python scripts/apply_laya_94_user_label_revision.py --verify`。
