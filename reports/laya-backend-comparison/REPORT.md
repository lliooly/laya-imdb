# Laya-CE 同权重推理对照

同一份微调权重、相同 2,500 条开发验证样本；未读取测试集、未重新训练。输入 token、mask、选项顺序和位置逐条核对。

FP32 关闭 MLX TF32；全部 eager、无 prompt cache。温度使用原 checkpoint 值，没有重新拟合。复现脚本默认两个阶段均使用 2 GiB cache。
数值核对 MLX cache limit: runtime default；分组测速 cache limit 见各 backend 的 bench JSON。

| Backend | Accuracy | Macro-F1 | 预测差异（对 MPS） | 最大校准概率差 |
|---|---:|---:|---:|---:|
| mps-fp32 | 95.7600% | 0.957600 | 0 | 0 |
| mlx-fp32 | 95.7600% | 0.957600 | 0 | 1.10238e-05 |
| mlx-fp16 | 95.7600% | 0.957600 | 0 | 0.00434625 |

## 延迟和吞吐

每个后端/阶段独立进程，顺序运行。每组 8 条真实影评，5 次预热、50 次同步计时；长度为包含问题与选项后的实际序列长度。P50 为每批延迟，吞吐为 batch_size / 平均耗时。

prepared_forward 包括 CPU 输入转设备、模型、同步、输出转 CPU 和公共结果格式化；end_to_end 另加各后端的分词、模板及 padding。两者均不含模型加载和首次预热。

| Length | Batch | Scope | MPS FP32 ms | MLX FP32 ms | MLX FP16 ms | FP32 speedup | FP16 speedup* |
|---|---:|---|---:|---:|---:|---:|---:|
| short | 1 | prepared_forward | 43.25 | 18.62 | 11.90 | 2.32× | 3.63× |
| short | 1 | end_to_end | 42.03 | 39.99 | 16.67 | 1.05× | 2.52× |
| short | 8 | prepared_forward | 242.65 | 273.67 | 79.11 | 0.89× | 3.07× |
| short | 8 | end_to_end | 280.15 | 262.80 | 78.22 | 1.07× | 3.58× |
| medium | 1 | prepared_forward | 80.27 | 81.56 | 24.45 | 0.98× | 3.28× |
| medium | 1 | end_to_end | 82.00 | 86.97 | 26.22 | 0.94× | 3.13× |
| medium | 8 | prepared_forward | 701.72 | 615.82 | 195.35 | 1.14× | 3.59× |
| medium | 8 | end_to_end | 659.88 | 617.35 | 191.56 | 1.07× | 3.44× |
| long | 1 | prepared_forward | 158.99 | 182.85 | 46.24 | 0.87× | 3.44× |
| long | 1 | end_to_end | 151.07 | 184.04 | 43.23 | 0.82× | 3.49× |
| long | 8 | prepared_forward | 1186.55 | 1081.91 | 321.34 | 1.10× | 3.69× |
| long | 8 | end_to_end | 1200.03 | 1032.11 | 326.02 | 1.16× | 3.68× |

*FP16 加速同时包含框架与精度变化，不能全部归因为 MLX。原始 JSON 保留 P95、吞吐及全部计时。

## 缓存诊断

初始 MLX 默认设置全量评估结束时缓存超过 51 GB；原始结果保存在 default-cache-diagnostic。它与 FP32 评估变慢同时出现，但未做受控全量重复实验，不能单独证明缓存是原因。分组测速另设 2 GiB 缓存上限；不把默认设置评估耗时与受限缓存分组测速混为同一实验。

## 限制

- 单设备单次运行，受温度、系统负载和运行顺序影响，不能当作跨设备结论。
- MLX allocator peak 与 MPS current driver memory 口径不同，仅记录，不比较内存优劣。
- 这是推理对照，不能推断训练速度。该 MLX 包当前不提供微调入口。
- 验证集此前参与开发与 checkpoint 选择，准确率不是最终测试结果。

复现：`uv run --offline --locked --extra laya --extra metal --extra laya-mlx python -u scripts/compare_laya_backends.py`
