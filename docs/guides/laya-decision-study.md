# Laya 决策因素实验与训练交接

依据：[原提案](../superpowers/specs/2026-10-13-laya-decision-factor-attribution-design.md)；审查、范围与结论修正见[实现计划](../superpowers/plans/2026-10-13-laya-decision-factor-implementation.md)。2026-10-14 用户已授权远端执行，真实结果以新 run_id 下的报告为准。

固定权重与完整训练结果见[执行报告](../../reports/laya-decision-execution-20261014-v1/README.md)：两组均完成 5526 次更新，最终同用首尾时控制 96.64%、随机窗口 96.44%，N4 未达到成功条件；未自动追加 seed43。

## 固定权重诊断

`configs/laya-decision-study-20261013-v1.json` 预声明 baseline-s43、combined-s42 两个最终 checkpoint 及权重/manifest SHA256，使用 CUDA strict FP32、batch=8，窗口 128/256/466，步长为窗口一半。配置检查不读取数据、模型或初始化 GPU：

```bash
uv run --offline --frozen --extra laya --extra reference python scripts/run_laya_decision_study.py --config configs/laya-decision-study-20261013-v1.json --validate-config
```

核验 checkpoint 目录存在和哈希，再执行同一命令去掉 `--validate-config`。本次远端历史目录缺少大权重文件，先从本地归档恢复与配置预声明 SHA256 完全一致的原权重，再运行诊断。

结果写入全新 `reports/laya-decision-study-s43-s42-strict-20261013-v1/`。每模型导出校准/开发的前缀、反序前缀、首尾和三种滑窗结果；保存单视图及融合 Accuracy、Macro-F1、NLL、ECE、截断分层、翻转 ID、精确 McNemar、实际前向行数、窗口数、耗时。

N2/N6 只在超过 466 token 时等权平均前缀/首尾概率；其他样本直接复用前缀，输入哈希、概率都严格相同。N1 从 scorer 输入捕获表示，mean/max 集合余弦特征的唯一系数只在原校准 400 上拟合，写入 `frozen-fit.json` 后才读取开发文本。N1 的 `head_training_eligible` 是校准信号门槛，不能视为泛化证明；没有信号不启动新决策头训练。

EXP-1 与 EXP-2 按提案成功/停止条件报告，三个窗口全部保留，不根据开发集挑选窗口。EXP-2 的 ECE 使用决定窗口的原生概率；不是整篇重新校准值。提案未给出延迟阈值，因此代码报告时间，`latency_acceptable=null`，不自动宣布延迟成功。未知信号与统计不确定性均留给实验判断。

## 同期窗口训练对照

只有前置诊断支持继续时才运行 EXP-3。两个正式配置仅 run_id 和 `window_training.mode` 不同：

| 项目 | 控制 | 实验 |
| --- | --- | --- |
| 配置 | `laya-decision-window-prefix-s42-strict-20261013-v1.json` | `laya-decision-window-random_head_tail-s42-strict-20261013-v1.json` |
| 训练窗口 | 前缀 | 每条原始 ID、每轮确定性 50% 前缀/首尾 |
| 主评估输入 | 首尾 | 首尾 |
| 辅助评估输入 | 前缀 | 前缀 |

共同设置：seed42、strict、FP32、AMP 关闭、22100 梯度/400 校准/2500 开发、两轮、max_len=512、head_max_len=192、encoder_lr=2.5e-5、head_lr=1e-4、warmup=0.05、label_smoothing=0.05、原选项乱序。未添加一致性损失、重采样或语义改写。

先对两份配置执行 `scripts/train_laya.py --config <配置路径> --validate-config`，服务器再执行 `--check-only` 做设备、模型、数据准入。正式启动使用同一入口去掉检查标记；用各自新 run_id，不能覆盖已有结果。

训练窗口 hook 只作用于梯度子集；校准仍用原前缀与硬目标。`window_consumption.jsonl` 记录实际 sample ID、epoch、选择的窗口与编码哈希，`window_audit.json` 验证每轮准确消费 22100 条。控制组保留与原前缀相同的输入和随机数流。

开发集前后评估在这个训练协议中关闭。两轮固定预算，不用开发集早停，保留逐轮 checkpoint 与训练损失。训练后只绑定最终 checkpoint；绑定器会拒绝未完成、分区改变或增加其他训练因素的对照：

```bash
uv run --offline --frozen --extra laya --extra reference python scripts/prepare_laya_window_evaluation.py --prefix-run laya-decision-window-prefix-s42-strict-20261013-v1 --random-run laya-decision-window-random_head_tail-s42-strict-20261013-v1 --run-id laya-decision-window-evaluation-s42-strict-20261013-v1 --out configs/laya-decision-window-evaluation-s42-strict-20261013-v1.json
uv run --offline --frozen --extra laya --extra reference python scripts/run_laya_decision_study.py --config configs/laya-decision-window-evaluation-s42-strict-20261013-v1.json
```

报告 `exp3_training_factor` 的主比较是两组同用首尾；成功要求整体至少 +0.2pp 且未截断组不退化。前缀对前缀为辅助比较。不会自动挑选表现最好的视图或 checkpoint，不自动追加 seed43；需要先按冻结规则解释结果。

## N8 与后续门槛

`scripts/average_laya_checkpoints.py` 接收同一次训练的 epoch-001/002 目录、两份预声明权重 SHA256 和新输出目录。验证配置、tokenizer、encoder 配置、参数键/形状/dtype/有限值后，做固定 50:50 参数平均；温度清为 1，保留平均来源与哈希。不会训练或推理。

2026-10-14 用户批准执行 N8，方案见[固定平均计划](../superpowers/plans/2026-10-14-laya-checkpoint-average.md)。入口 `scripts/run_laya_checkpoint_average.py` 对前缀控制的 epoch-001/002 做固定 50:50 平均，只拟合温度（不拟合偏置），原校准 400 冻结结果后再读取开发 2500；重载已校准权重后评估。主比较同用首尾，控制 logits 复用上次冻结导出，所有输入哈希写入配置，不根据开发集选比例或窗口。

```bash
uv run --offline --frozen --extra laya --extra reference python scripts/run_laya_checkpoint_average.py --config configs/laya-checkpoint-average-prefix-s42-strict-20261014-v1.json
```

报告在 `reports/laya-checkpoint-average-prefix-s42-strict-20261014-v1/`，原始平均权重和已校准权重分别保存在同名 `-uncalibrated` 与同名 checkpoint 目录。此阶段不运行梯度训练，不改变默认部署。

N8 已于北京时间 2026-10-14 13:04:26 完成，用时 66.47 秒；本地与远端相关测试各 37 项通过。平均和控制同为 96.64%，修正/退化各 12 条，McNemar p=1；同协议重校准后 NLL/ECE 均未改善。本次固定 epoch-001/002 平均没有正向证据，不追加比例搜索或自动部署。详见[N8 结果](../../reports/laya-checkpoint-average-prefix-s42-strict-20261014-v1/REPORT.md)。

N1 新结构训练、N7 自蒸馏保持提案的前置门槛，本轮不提前启动；新增代码不改变当前默认部署权重。官方测试保持封存；原提案和历史 reports 均保留。

## N2/N6 多模型融合

2026-10-14 用户批准继续固定少量模型融合，见[执行方案](../superpowers/plans/2026-10-14-laya-probability-fusion.md)。离线入口复用历史CUDA FP32导出，只在原400条校准集拟合baseline-s43／combined-s42间的一个系数；每个模型内部使用固定截断门控及前缀／首尾等权。配置绑定28个输入文件，不加载GPU或访问官方测试。

```bash
UV_CACHE_DIR=/tmp/end-imdb-decision-uv uv run --offline --frozen python scripts/run_laya_probability_fusion.py --config configs/laya-probability-fusion-s43-s42-20261014-v1.json --validate-config
```

实际执行去掉`--validate-config`，仅在输出目录不存在时运行；已有run拒绝覆盖，不必重复本轮开发回放。`scripts/laya_probability_fusion.py`中的`apply_gated_mixture`可使用`frozen-fit.json`保存的系数对已核验、相同顺序的新预测应用同一规则。它只混合概率，不负责神经网络推理或数据读取。

本轮系数为baseline70.2033%／combined29.7967%，主方案96.96%，相对固定baseline-s43首尾96.84%净+3条、p=.5078125；未通过事前继续门槛。预声明等权辅助对照97.00%，p=.541256，未反选为主方案或部署。8视角等权按本轮输入定义为96.88%，无截距OLS为96.68%，均未复现原提案数字。原LS缺少完整方法记录，不能据此宣称任何LS定义均无效。详见[融合结果](../../reports/laya-probability-fusion-s43-s42-20261014-v1/REPORT.md)。

新代码相关21项检查通过；校准拟合先持久化、重载，再读取开发预测。开发集已反复探索，本轮p值仅作描述，不构成独立泛化证明。不自动追加组合搜索或训练，不改变默认部署。

## 远端顺序执行与停止门槛

`scripts/run_laya_decision_pipeline.py` 顺序执行两组准入、固定权重诊断、N4 证据判断、两组训练和保存权重的最终研究。每阶段独立日志、PID、超时和状态均写入全新 `artifacts/<run_id>/`，断开 SSH 后仍可完成；失败停止，不自动重跑开发研究。

N4 的继续门槛在正式诊断前固定：baseline-s43 和 combined-s42 均须满足首尾相对前缀的整体与截断准确率严格改善。N2 等权融合是否成功独立报告，不替代 N4 的覆盖假设。门槛不成立则停止完整训练；成立也不改变学习率、轮数或视图定义。两轮后必须验证每轮 22100 条消费和合计 5526 次参数更新，主比较固定最终轮、两组同用首尾。

```bash
uv run --offline --frozen --extra laya --extra reference python scripts/run_laya_decision_pipeline.py --study-config configs/laya-decision-study-20261013-v1.json --prefix-config configs/laya-decision-window-prefix-s42-strict-20261013-v1.json --random-config configs/laya-decision-window-random_head_tail-s42-strict-20261013-v1.json --run-id laya-decision-execution-20261014-v1
```

## 本地验证

使用锁定依赖、离线模式；受沙箱缓存目录权限限制时可设置 `UV_CACHE_DIR=/tmp/end-imdb-decision-uv`。算法和集成测试仅用合成文本、CPU 小模型及临时 checkpoint，覆盖真实上游训练循环/`predict_long`，不产生真实训练结果。

本轮核验：相关测试 **58 项通过**，远端最初 55 项及新增顺序执行 3 项分别通过；两组正式配置通过 RTX 4090 的 `--check-only`，两组 CUDA 冒烟各完成 3 次参数更新。GPU 冒烟不作为真实开发成绩。

```bash
PYTHONPATH=scripts:tests uv run --offline --frozen --all-extras python -m unittest test_laya_decision_views test_laya_decision_pipeline test_laya_optimization test_laya_posthoc test_laya_posthoc_inputs test_laya_runtime test_laya_preflight test_training -q
```
