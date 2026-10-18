# 本地 CPU 词袋 / NB-ngram 互补性实验

固定 22,100 梯度训练、400 校准、2,500 开发；原标签和分区保持。两个候选的模型、校准参数、融合权重和主方案均在独立 fit 进程封存，evaluate 进程仅推理与统计。官方测试正文、预测和分布未读取。

校准集预选主方案：`tfidf_lr`。原 Laya + DeBERTa 固定 50:50 概率缓存准确复现 2432/2500 = 97.28%。DeBERTa 为历史 CUDA BF16 开发缓存，没有新神经网络推理。

| 方案 | 正确 / 2500 | Accuracy | NLL | ECE |
|---|---:|---:|---:|---:|
| 原神经融合 | 2432 | 97.28% | 0.088768 | 0.916% |
| tfidf_lr 单独 | 2278 | 91.12% | 0.223719 | 1.852% |
| tfidf_lr 固定权重融合 | 2433 | 97.32% | 0.088023 | 0.721% |
| nb_ngram_lr 单独 | 2306 | 92.24% | 0.199979 | 1.566% |
| nb_ngram_lr 固定权重融合 | 2432 | 97.28% | 0.087274 | 0.677% |

## 校准和权重

不搜索训练超参数。TF-IDF 为 word 1–2 gram、C=4；NB 分支为 binary word 1–3 gram、C=1；两者 min_df=3、最多 300000 特征，全文输入，保留否定词与单字符数字，tokenizer 保留 8/10 评分形式。词表、IDF、NB 类内统计只用梯度训练文本。

400 条校准拟合有界 sigmoid，NLL 加固定 0.001 正则；再以校准 NLL 最小化拟合 p=(1-w)p_neural+w*p_sparse，w∈[0,0.25]，可取零。原神经融合内部比例保持不变。主候选按校准融合 NLL 选择，未按开发成绩反选。

| 候选 | 特征数 | sigmoid a / b | 稀疏权重 | 校准融合 NLL |
|---|---:|---:|---:|---:|
| tfidf_lr | 242335 | 2.237079 / -0.285802 | 2.3179% | 0.054734 |
| nb_ngram_lr | 300000 | 0.987273 / -0.126629 | 2.4165% | 0.054976 |

## 实际融合的修复与退化

| 候选 | 修复 / 退化 | 净变化 | Accuracy 差值及条件 95% 区间（pp） | 精确 McNemar p |
|---|---:|---:|---|---:|
| tfidf_lr | 3 / 2 | +1 | +0.040 [-0.120, +0.200] | 1 |
| nb_ngram_lr | 2 / 2 | +0 | +0.000 [-0.160, +0.160] | 1 |

## 独立分支的互补性与 oracle

| 分支 | 答对原 68 条错误 | 判错原正确样本 | 共同错误 | 标签知情 oracle |
|---|---:|---:|---:|---:|
| tfidf_lr | 22 | 176 | 46 | 98.16% |
| nb_ngram_lr | 27 | 153 | 41 | 98.36% |

oracle 按已知正确标签逐条选模型，仅描述上限，不能部署。单独分支答对某条错题，不代表小权重融合能跨过 0.5 边界。样本概率见 complementarity-cases.json，未人工查看正文后修改特征。

## 长短文分组

| 候选 / 分组 | 样本数 | 原神经准确率 | 融合准确率 | 修复 / 退化 |
|---|---:|---:|---:|---:|
| tfidf_lr / laya_truncated | 407 | 97.543% | 97.297% | 0 / 1 |
| tfidf_lr / laya_complete | 2093 | 97.229% | 97.324% | 3 / 1 |
| nb_ngram_lr / laya_truncated | 407 | 97.543% | 97.297% | 0 / 1 |
| nb_ngram_lr / laya_complete | 2093 | 97.229% | 97.277% | 2 / 1 |

## 结论与边界

校准预选主方案在开发集相对 97.28% 净变化 +1 条，准确率 97.32%。这是本轮预定流程的结果，不把开发上较高的次级候选反选为主方案。

20,000 次类别分层、共享 ID 配对 bootstrap（seed=20261018）固定已训练模型与参数，区间不包含校准拟合、模型选择、训练随机性和多轮开发复用的不确定性。两个候选及分组完整报告，次级 p 值未作多重比较校正。NLL 是预定权重目标，不能保证 Accuracy 增加；无新测试或默认部署切换。

准备阶段偏离：父会话寻找来源时误读旧测试契约，其中有官方测试样本 ID（文件名含评分）。没有打开测试正文 / 预测，未计算或使用测试分布。因此不声称整个会话未接触测试元数据；fit/evaluate 的测试文件访问禁止与读路径分别记录在 access-audit 文件中。

CPU fit 共 22.16 秒，包含特征提取、训练、校准、模型保存及重载验证；不包括首次依赖下载。两模型重载后 400 条校准 margins 逐位一致。完整模型与词表在 artifacts/bow-nb-complementarity-cpu-20261018-v1/，环境锁在 environments/bow-cpu/uv.lock。

复现需新 run_id，拒绝覆盖已有目录。命令：

```bash
UV_CACHE_DIR=/private/tmp/end-imdb-bow-uv-cache uv run --project environments/bow-cpu --locked --offline --no-sync python scripts/run_bow_nb_study.py fit
UV_CACHE_DIR=/private/tmp/end-imdb-bow-uv-cache uv run --project environments/bow-cpu --locked --offline --no-sync python scripts/run_bow_nb_study.py evaluate
```
