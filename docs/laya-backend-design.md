# Laya 同权重后端对照方案

用户已选择先验证推理，后续是否迁移训练由实测决定。

- 固定现有 `checkpoints/laya-l512-s42-ce` 微调权重，不重训、不修改权重与原生温度。
- 使用原有 2,500 条开发验证样本；保留最终测试集。
- 对照 MPS FP32、MLX FP32、MLX FP16。MLX FP32 关闭 TF32；eager 执行、不启用 prompt cache。
- 逐样本检查 token ID、attention mask、选项 marker 和 qtype；核对全部概率、logits、预测标签与分类指标。
- 每后端/阶段独立进程，GPU 顺序执行。预先选定相同真实评论，按输入长度分组；batch 1 和 8；每组 5 次预热和 50 次同步计时。
- prepared_forward 包含设备传输、同步和输出格式化；end_to_end 另加各后端分词、模板与 padding。加载时间另外记录。
- MPS current memory 与 MLX allocator peak 口径不一致，仅分别保存，不据此宣称内存优势。
- FP16 对 MPS FP32 的差异包含精度与框架因素；FP32 对照用于判断框架收益。

产物：`scripts/compare_laya_backends.py` 和 `reports/laya-backend-comparison/` 下的原始预测、原始计时、指标、数值对齐报告、文件哈希及中文结论。

依赖通过 uv extra `laya-mlx` 固定为 0.3.0；不安装其 reference extra，避免改变现有 Transformers 版本。推理直接读取本地 checkpoint，启用 Hugging Face 离线模式和 Python 网络拦截。

当前 laya-mlx 是第三方推理实现，不提供微调入口。迁移训练需要另行验证训练态 dropout、损失、梯度、更新与恢复流程；不能根据推理结果推断训练加速。

来源：[项目](https://github.com/mizorewww/laya-mlx)、[作者基准](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md)。

实际执行记录：全量数值核对先采用 MLX 默认 cache；发现 cache 累积超过 51 GB 后，分组测速改为 2 GiB 上限。缓存策略属于运行条件，报告分别披露；没有重新进行受限缓存的完整评估计时。复现脚本默认所有阶段均使用 2 GiB cache。

## 部署决定

用户已批准 Mac 推理迁移到 MLX FP16，保留 PyTorch 训练；未来在租用的 NVIDIA CUDA 服务器上训练。默认 `scripts/predict.py` 加载现有全量 CE checkpoint，输出未舍入的概率、原生温度和截断信息；MLX cache 限制 2 GiB。显式 `--model modernbert` 或历史 baseline checkpoint 路径仍可调用基线。新 CUDA checkpoint 用 `--checkpoint` 指定本地目录，不要求转换或重新微调。

验收：默认入口正负影评推理、原验证样本与已保存 FP16 预测概率核对、历史 baseline 命令兼容、空文本拒绝；不训练、不访问测试集。
