# Laya 首步梯度差异定位与确定性验证

**首个可见分歧已定位到 `encoder.embeddings.tok_embeddings.weight`；严格协议使原路径、控制路径及两者之间的完整 16 步逐位一致。** 默认、关闭梯度检查点重计算和 math 注意力三个条件均重复出现首步词嵌入梯度差异；单独关闭重计算或改注意力后端没有解决问题。

GPU 诊断通过用户已连接的 Ghostty 会话完成；2026-10-13 用户明确授权重新连接 SSH 后，通过新连接收齐并校验全部诊断文件。没有新建窗口或 worktree。共 10 次独立进程短程训练，每次 16 更新，总计 160 更新；每次使用完整 22,100 梯度样本顺序和两轮 5,526 更新学习率预算。保留启动前开发评估，在下一次训练前向前停止。没有启动新的全量训练，没有保存/替换权重，没有访问官方测试。

## 成对结果

“首步不同参数”按裁剪前梯度哈希统计；参数遍历顺序不是反向算子执行顺序。

| 比较 | 完整 16 步 | 首步不同参数数 | 首个不同更新 |
| --- | --- | ---: | ---: |
| 默认原路径 | 不一致 | 1 | 1 |
| 关闭重计算 | 不一致 | 1 | 1 |
| 仅 math 注意力 | 不一致 | 1 | 1 |
| 严格原路径 | 一致 | 0 | 无 |
| 严格控制路径 | 一致 | 0 | 无 |
| 严格原路径 vs 控制路径 | 一致 | 0 | 无 |

前三种条件的首步仅词嵌入参数不同，其余参数的梯度记录相同（包括无梯度项）。首步前向与损失均相同。三个条件都使用相同模型、编码批、学习率预算；这里只比较各条件内部重复，不要求不同后端之间的输出或 RNG 逐位相同。

严格协议下，三个比较的输入、初始全状态、Torch CPU/CUDA RNG、前向 logits、损失、裁剪前后梯度、梯度范数、学习率和更新后全模型状态均一致。控制协议仍为冻结的 45 条候选池之 control 模式，不涉及重采样组。

## 生效的运行时协议

进程启动后、导入 Torch/CUDA 初始化前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；启用 `torch.use_deterministic_algorithms(True, warn_only=False)`，设置 cuDNN deterministic=true、benchmark=false。这是一个明确的多设置协议；本矩阵不能单独区分其中每项设置的贡献。FP32、AMP 关闭、原有限值保护、默认注意力后端及梯度检查点保持原状。

CPU profiler 的首步算子记录确认：默认和严格协议都实际执行 efficient SDPA；math 条件实际执行 math SDPA；关闭重计算时注意力前向逻辑调用 30 次，对比启用重计算时 60 次。首次冷编译记录中含编译/自动调优试运行，不能把所有底层事件次数直接当成逻辑训练调用次数。

默认词嵌入前向/反向存在 Triton 编译融合的 embedding/LayerNorm 内核。真实训练记录定位到了参数级边界，尚未捕获该层真实上游梯度用于逐算子重放；因此具体因果定位还需结合下方合成复验的适用范围。

## 独立嵌入层合成复验

复用当前安装 Transformers 的 `ModernBertEmbeddings` 实现和锁定 encoder 配置，随机初始化 50368×1024 嵌入矩阵，固定 8×512 合成 token ID 和 8×512×1024 上游梯度。每种条件在独立进程中重复四次反向，不读训练数据或原模型权重。记录全部参数梯度与首轮的数值差异；下表最大差值和元素数取各后续重复及参数中的最大值。

| 嵌入路径 | 状态 | 四次输出一致 | 四次梯度一致 | 最大不同元素数 | 最大绝对差值 |
| --- | --- | --- | --- | ---: | ---: |
| compiled-strict | completed | True | True | 0 | 0 |
| eager | completed | True | True | 0 | 0 |
| compiled | completed | True | False | 155,824 | 1.90734863e-06 |

合成输入只验证对应算子/编译路径在该环境中的行为，不能冒充真实首批训练张量的重放，也不能据此分摊历史准确率变化。

## 代码与验证

扩展诊断 `short_trace` 的可选逐参数记录和首步 CPU profiler；新增运行时协议 helper、顺序矩阵驱动和独立嵌入层复验。正式训练入口和正式配置未修改。原入口和控制入口继续按历史 SHA 校验，新增配置使用独立 run_id，源码快照保存在准备报告 reproduction 中。

本地和远端的 3 项准入测试通过：实际上游 trainer/数值保护/scheduler/control 组合，插桩前后真实 dropout+AdamW 参数与 RNG 一致，停止边界与异常 hook 恢复。新增详细观测也包含在参数/RNG 保持测试内。GPU 上的配置效果由上述实际运行及算子记录验证。

本地分析另外核对实际入口与快照哈希、首步逐参数哈希与全梯度哈希一致、冻结分区计数/顺序、模型锁、超参数、实际更新和完整 scheduler 预算；重算比较与远端结果一致。165 个导出文件及整包 SHA 校验通过。

## 限制与下一步

这证明本机本协议下带观测的前 16 步一致，尚未证明完整两轮逐位一致；哈希和 profiler 增加同步、改变耗时。训练入口记录 failed 是主动 DiagnosticStop，独立 trace/result 标记 diagnostic_completed，不能视作完成全量训练或数值溢出。

历史 96.48% 与 95.96% 的差值仍不能全部归因于本次定位的数值漂移；45 条重采样净改善 5 条也尚未证明稳定。下一次完整配对复验应显式锁定本次严格协议、使用新 run_id，保留逐轮检查点和运行时记录；先同 seed 比较控制/重采样，再决定是否增加种子。不要为解决复现问题关闭重计算或强制 math 后端。

参考：[PyTorch 确定性算法](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html)将 CUDA Embedding 反向列入确定性协议覆盖；[SDPA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)说明后端切换及数值差异边界。

原始诊断：`reports/laya-determinism-cuda-20261012-v1/`。完整本地重算：[summary.json](summary.json)，输入文件校验：[source-provenance.json](source-provenance.json)，备份清单：`artifacts/ghostty-determinism-results-20261012-v1/verified-files.json`。

```bash
UV_CACHE_DIR=/private/tmp/end-imdb-uv-cache uv run --locked --offline python reports/laya-determinism-completed-20261012-v1/analyze.py
```
