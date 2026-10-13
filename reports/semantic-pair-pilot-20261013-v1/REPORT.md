# 语义改写首轮试制与固定权重诊断

2026-10-13，分支 `dev/semantic-pair-pilot`。完成 30 组三元组、90 个版本的固定权重诊断，以及 A/B/C 全量曝光计划检查。没有新增微调，没有修改现有模型权重或接触官方测试。

## 结果

使用本地 Mac 两轮 CE checkpoint，经 MLX FP16、原前缀输入对三个版本逐条预测。权重 SHA 为 `e3fde5c96111b478ab19067b6a11291c0a4577d5c1754f898eccb3f445adf92a`。

下表是相对于原标签的预测一致数。改写标签尚未经独立人工确认，且来源均为模型见过的训练样本；不是独立准确率，更不是增强训练后的收益。

| 草稿主类别 | 数量 | 原文 | 普通改写 B | 语义改写 C |
| --- | ---: | ---: | ---: | ---: |
| 反讽 | 6 | 5 | 5 | 5 |
| 否定、转折等复杂关系 | 13 | 10 | 7 | 9 |
| 评价对象或观点切换 | 11 | 9 | 9 | 9 |
| 合计 | 30 | 24 | 21 | 23 |

普通改写新增 3 条错误，语义改写新增 1 条错误，两者均未修正原文已有错误。普通改写的变化涉及恐怖效果与品质评价、劣质效果带来的戏仿乐趣、制作缺陷与观看乐趣。语义改写回退的是最后一类。全部变化 ID 和 logits 保留在 [固定权重诊断](../semantic-pair-probe-20261013-v1/summary.json) 和其 predictions.jsonl，不依结果调整或重选这批样本。

这些结果提示对措辞的敏感性值得进一步核查；不能证明我们的语义解释一定正确，也不能证明增强训练无效。当前证据不支持“把句子说直白就能直接纠错”，更不支持已经提高反讽识别能力。

## 已完成的检查

- 全部来源属于冻结的 22,100 条梯度训练分区；400 条校准与 2,500 条开发验证来源排除，未读取开发验证正文或官方测试。
- 每个版本只改动一段原文，其余文字保留；原文证据及修改片段位于可见评论前缀中。这不代表所有决定性证据都已被模型看到。
- 90 个输入用 PyTorch 上游编码器独立重建，与实际 MLX 推理输入哈希逐一一致。核对的是输入，不是两个后端的 logits。
- B/C token 数量差的均值为 +2.7，范围 -11 至 +21；长度接近但没有完全匹配，显式情感词和解释强度仍可能构成混杂。
- A/B/C 每组预览 44,200 次曝光，2,210 个增强位置（总量 5%，集中在第二轮），其余 41,990 个位置完全一致；每轮标签数量一致，首轮覆盖全部 22,100 个原始来源。这里验证的是数据计划，尚未验证训练循环实际消费这些位置。
- 60 个改写与全部原训练正文没有完全重复。近似重复审计尚未完成，不能据此宣称排除了语义重叠。
- 8 项单元测试覆盖分区隔离、错误来源/标签/哈希、编辑歧义与重叠、截断外证据、人工审核门槛、A/B/C 预算，以及纠正和回退计数。

独立检查见 [integration-verification.json](integration-verification.json)；可复现的复核入口为 `scripts/verify_semantic_pairs.py`，其输出另存，不覆盖本轮历史核验。

## 样本质量与训练准入

类别与标签分布是 6/13/11 和负 13、正 17，未为了满足最初每类 10 条而制造确定性。复杂关系类别不等于经过句法标注的“长难句”类别，且反讽样本全部为负类。本轮不能证明对正向反讽或广泛复杂句的覆盖。

样本由同一 AI 起草，已见原标签和历史候选描述，尚未获得独立人工审核。高歧义反讽、可能改变态度强度的显式解释，以及“劣质但享受”样本优先人工核查。训练导出要求逐条人工接受，并将审核绑定到三个版本的 SHA；仅填写审核者类型不是客观语义保证。

本轮停止于已确认方案的样本审核阶段。下一步先审核这 30 组、修正或淘汰存疑改写、补齐可靠样本与近似重复检查；再实现和验证训练循环适配，做实际设备测速与预算核对。审核通过不自动证明样本足够，也不直接启动三组付费全量训练。

## 本地材料与复现

- [完整原文与改写审核材料](data/REVIEW.md)
- [待填写的人工审核表](data/review-decisions.json)
- [三元组数据](data/triplets.json)
- [阶段汇总与哈希](summary.json)

`data/` 与起草 recipes 含原始影评，按仓库规则只保留本地，不提交 Git。源码、汇总与无正文的逐样本预测提交到分支。复现需要本地冻结训练数据、tokenizer、checkpoint 和上述材料；克隆源码不会自动获得这些文件。

```bash
UV_CACHE_DIR=/private/tmp/end-imdb-uv-cache uv run --offline --locked --extra laya python -m unittest discover -s tests -p test_semantic_pair_pilot.py -v

UV_CACHE_DIR=/private/tmp/end-imdb-uv-cache uv run --offline --locked --extra laya python scripts/semantic_pair_pilot.py --recipes reports/semantic-pair-recipes-20261013-v1/data/recipes.json --out reports/semantic-pair-pilot-reproduction

UV_CACHE_DIR=/private/tmp/end-imdb-uv-cache uv run --offline --locked --extra laya-mlx python scripts/probe_semantic_pairs.py --pack reports/semantic-pair-pilot-20261013-v1/data/triplets.json --out reports/semantic-pair-probe-reproduction

UV_CACHE_DIR=/private/tmp/end-imdb-uv-cache uv run --offline --locked --extra laya python scripts/verify_semantic_pairs.py --pack reports/semantic-pair-pilot-20261013-v1/data/triplets.json --predictions reports/semantic-pair-probe-20261013-v1/predictions.jsonl --out reports/semantic-pair-input-verification-reproduction.json
```

Metal 首次在受限沙箱内不可见；在批准使用本机 GPU 后，离线诊断完成。没有连接云实例或产生云训练费用。
