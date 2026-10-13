# Laya-CE 标签平滑与 warmup 独立对照

日期：2026-10-11。状态：两组正式训练和完整保存权重验证已完成；标签平滑 96.16%，warmup 96.32%，Mac 核对尚未进行。分析见 [完成报告](../../../reports/rtx4090-regularization-completed-v1/REPORT.md)。组合配置已通过 CUDA 冒烟并启动。

## 依据与实验范围

已有 RTX 4090 两轮参照保存重载 Accuracy 为 95.88%，103 条错误中 84 条未截断、19 条截断；78 条错误的未校准置信度至少为 0.95。Mac 两轮最佳候选为 96.96%，76 条错误中 63 条未截断、13 条截断。两者有 63 条共同错误，其中 53 条未截断。

这些结果支持先检验训练策略和泛化，再推进长文本策略；高置信错误不单独证明过拟合。平台、batch 分组、dropout 随机轨迹的贡献尚未隔离。不存在已证实的 batch 实现故障。

两组都从锁定的原始权重开始，保持 22,100 条梯度训练、400 条校准、2,500 条开发验证、seed 42、两轮、512 总 token、FP32、micro-batch 8、累积 1、原学习率和问题模板。官方测试继续封存，不修改任何标签。

| 组 | 修改 | 配置 | 状态 |
|---|---|---|---|
| A：参照 | 原 cosine，标签平滑 0 | `configs/laya-rtx4090-ce-epochs2-fp32.json` | 已完成，95.88% |
| B：标签平滑 | `label_smoothing=0.05`，原 cosine | `configs/laya-rtx4090-ce-epochs2-ls005-fp32.json` | 完成，96.16% |
| C：warmup | `warmup_ratio=0.05`，标签平滑 0 | `configs/laya-rtx4090-ce-epochs2-warmup005-fp32.json` | 完成，96.32% |

先分别比较 B 对 A、C 对 A，不提前组合两个因素。标签平滑与 warmup 是已有方法，不直接表述为自主创新；是否有 Accuracy 收益由本项目实验决定。

## 适配与边界

`training.label_smoothing` 仍由上游 `TrainConfig` 校验。上游原本先平滑再划出校准集；新实验通过 `scripts/laya_optimization.py` 先构造原始目标，在上游划分完成后只对 `train_model` 收到的梯度训练子集平滑。二分类目标为 0.975/0.025，校准和开发验证仍使用原始目标；数据 ID、输入 token 和样本内容不变。

新增顶层 `optimization` 对象仅接受 `warmup_ratio` 和 `diagnostics_every`，不向上游 `TrainConfig` 传入未知字段。没有该对象的旧配置完全沿用原路径；`warmup_ratio=0` 委托原 `CosineAnnealingLR`，不采用自定义近似。适配器不重写训练循环、loss、AdamW、梯度裁剪、选项增强、保存或评估。

调度器以实际 optimizer 更新计数，包含每轮末尾的部分累积窗口。总更新 U=5,526，warmup 更新 W=ceil(0.05×U)=277。第 j 次更新使用：

- j≤W：每组峰值学习率乘以 j/W，第一次更新为峰值/277，第 277 次达到原峰值。
- W<j≤U：`min_lr + (peak_lr-min_lr) × (1+cos(pi×(j-W)/(U-W)))/2`，最后一次更新使用 `min_lr=1e-6`。

W≥U 的过短运行被拒绝；不会按 micro-batch 推进调度。零 warmup 使用上游原时序，保持历史轨迹兼容；两种时序都在报告中明确记录。

临时 hooks 仅在上游 `train_model` 内启用，退出或异常时恢复。CUDA 非有限 loss、梯度和参数保护继续启用。记录：

- `manifest.json`：实验配置、适配器哈希和仅对梯度训练平滑的范围。
- `optimizer_updates.jsonl`：前 16 次、warmup 结束前后各一次、每 16 次和最后一次成功更新所使用的学习率、裁剪前梯度范数和是否裁剪；冒烟配置逐次记录。
- `optimization_diagnostics.json`：训练循环状态、预期/实际更新次数、warmup 长度、梯度均值/最大值和裁剪比例。该文件的 completed 只代表梯度训练循环完成，完整流程以 `status.json` 为准。

诊断不抽取随机数、不改变梯度；GPU 同步和文件写入可能增加耗时，因此训练成本据实际记录比较。

## 本地验证与远端执行

所有命令在仓库根目录执行。`--validate-config` 只检查 JSON、锁定的 Laya 版本及配置，不读取数据或模型，不检查 CUDA，不启动训练：

```bash
uv run --offline --locked --extra laya python scripts/train_laya.py \
  --config configs/laya-rtx4090-ce-epochs2-ls005-fp32.json --validate-config
uv run --offline --locked --extra laya python scripts/train_laya.py \
  --config configs/laya-rtx4090-ce-epochs2-warmup005-fp32.json --validate-config
uv run --offline --locked --extra laya python -m unittest discover -s tests -v
```

开机前确认 SSH 入口、当前单价和剩余预算。按已有 4090 完整入口约 42 分钟粗估，两组正式训练约 84 分钟，另计传输、冒烟和重载评估；这不是保证耗时。同步当前源码、新配置、锁定依赖及已有冻结数据和原始权重，保留历史 run 目录。

两份独立冒烟配置均为 128 条训练源样本、其中 16 条仅作校准、32 条验证、micro-batch 2、一轮。C 产生 56 次更新，其中 3 次 warmup，可覆盖升温和衰减边界；这两组的小规模结果只验证可行性，不作为正式准确率证据。

```bash
uv run --offline --locked --extra laya python scripts/train_laya.py \
  --config configs/laya-rtx4090-smoke-ce-ls005-fp32.json --check-only
uv run --offline --locked --extra laya python scripts/audit_cuda_training.py \
  --config configs/laya-rtx4090-smoke-ce-ls005-fp32.json
uv run --offline --locked --extra laya python scripts/train_laya.py \
  --config configs/laya-rtx4090-smoke-ce-warmup005-fp32.json --check-only
uv run --offline --locked --extra laya python scripts/audit_cuda_training.py \
  --config configs/laya-rtx4090-smoke-ce-warmup005-fp32.json
```

检查两组 `numerical_audit.json` 的 passed、保存重载标签一致、optimizer/scheduler 更新数，以及 C 的实际学习率。任一数值保护或重载检查失败时，保留失败证据，先修复并使用新 run_id 重验，不启动对应正式组。

通过后先运行 B，重载完整验证，再运行 C：

```bash
uv run --offline --locked --extra laya python scripts/train_laya.py \
  --config configs/laya-rtx4090-ce-epochs2-ls005-fp32.json
uv run --offline --locked --extra laya python scripts/export_cuda_laya_predictions.py \
  --checkpoint checkpoints/laya-rtx4090-ce-epochs2-ls005-l512-s42-fp32-v1 \
  --out reports/laya-rtx4090-ce-epochs2-ls005-l512-s42-fp32-v1/cuda-saved-validation.json
uv run --offline --locked --extra laya python scripts/train_laya.py \
  --config configs/laya-rtx4090-ce-epochs2-warmup005-fp32.json
uv run --offline --locked --extra laya python scripts/export_cuda_laya_predictions.py \
  --checkpoint checkpoints/laya-rtx4090-ce-epochs2-warmup005-l512-s42-fp32-v1 \
  --out reports/laya-rtx4090-ce-epochs2-warmup005-l512-s42-fp32-v1/cuda-saved-validation.json
```

`--smoke-test` 是更小的 8/4 快速检查，生成 `<正式run_id>-smoke`，不能替代上面的完整数值审计配置。已有目录继续拒绝覆盖；失败后修改 run_id，不删除原报告后重跑。

## 选择与退出条件

以实际保存权重在完整开发验证上的 Accuracy 为首要指标，比较 A/B/C 的逐样本改正、退步、长度分层和错误置信度。Accuracy 相同才比较校准 NLL，再比较部署成本；只改善校准或训练 loss 不算准确率收益。

Mac 的 96.96% 继续作为全项目最佳候选，不因新组超过 CUDA 参照就替换它。有效候选回传 Mac，核对所有验证输入、标签和概率偏差，再按预算用其他 seed 复验。初次单 seed 小幅提升只作为候选信号，不宣称稳健收益。

两组均无 Accuracy 收益就结束本轮搜索，评估首尾保留或一致性正则化；不连续遍历大量平滑比例和 warmup 比例。组合实验只有在单因素结果支持且预算允许时新增。

## 实现与验证清单

1. 保留旧入口默认路径；新增独立配置、训练子集标签平滑、warmup 与诊断适配。
2. CPU 测试实际上游训练循环：零 warmup 诊断后的参数逐值一致、部分累积窗口计数、升温/衰减边界、标签范围、异常时恢复 hooks。
3. 验证所有新配置并运行已有 CPU 测试；不读取官方测试集、不训练完整模型。
4. 4090 上依次完成新冒烟审计和两组正式训练、保存重载、Mac 核对；未执行项目保留为未验证。

参考：[标签平滑](https://arxiv.org/abs/1906.02629)、[微调稳定性](https://arxiv.org/abs/2006.04884)。论文为方法依据，不代表当前模型已取得收益。

## 2026-10-11 完成后的组合验证

独立因素相对 CUDA 参照分别净增 7 与 11 条正确，配对差异尚不显著。保持历史整体最佳 96.96% 不变，在同一预训练起点、划分、种子和两轮预算下验证 `warmup005-ls005` 组合。新增两份配置均使用新 run_id；组合冒烟通过后才能开始正式训练。平台 runner 带阶段超时与余额时间门槛，正式结束后自动执行完整保存权重验证。本轮不自动追加更多比例搜索。
