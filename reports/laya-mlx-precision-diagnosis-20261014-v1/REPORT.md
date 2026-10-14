# MLX 数值误差诊断

本轮完成4个超限样本的精度／批量对照，并对其中MLX FP32偏差最大的样本追踪逐层输出和首层QKV计算。全部属于事后数值诊断，不能作为独立准确率或泛化证据。原FP16完整验证失败记录保留，未修改默认部署、融合系数、权重或概率门槛，未训练、未访问官方测试集。

## 已确认

1. 原FP16 batch8上下文可以逐值重现上一轮logits，排除了无法复现的偶发结果。
2. 改为batch1后仍有概率差超过0.01；单纯减小批量未消除误差。
3. 本轮样本上MLX FP32部分误差反而更大，不能把切换FP32当作已验证修复。
4. 对baseline-s43首尾视图、`train/neg/12416_3.txt`：PyTorch CPU FP32与CUDA概率差仅0.00003826，MLX GPU FP32与CUDA差0.07742859。输入哈希完全一致。
5. 该样本的输入嵌入跨后端最大差约9.54e-7；首层QKV投影即出现更明显差异，随后传播到后续编码层和决策头。
6. 进一步将**完全相同的PyTorch FP32嵌入输出和逐值相同的QKV权重**交给MLX CPU/GPU矩阵乘法：CPU与PyTorch输出逐值一致；GPU最大绝对差0.00351572、RMS差0.00044102。因此在这一固定输入上，差异可以独立复现于MLX GPU矩阵乘法路径，不需要融合或温度计算参与。

最后一项没有确定具体Metal内核、累加精度或硬件机制，也不证明所有超限样本都由同一个机制产生。这里的“定位到执行路径”不能写成已修复上游缺陷。

## 固定矩阵对照

4个样本 × 2模型 × 2视图，每种设置16个视图记录（未截断首尾复用前缀，因此部分记录重复）。下表是单视图误差，不能与上一轮主融合最大概率差0.01971072混为一谈。

| 设置 | 最大概率绝对差 vs CUDA | 概率差>0.01记录数 | 类别翻转 |
| --- | --- | --- | --- |
| float16/original_batch8 | 0.02807905 | 6/16 | 0 |
| float16/batch1 | 0.01256347 | 3/16 | 0 |
| float32/original_batch8 | 0.07742859 | 5/16 | 0 |
| float32/batch1 | 0.06225318 | 3/16 | 0 |

本轮4条异常样本中各设置均无类别翻转；这不替代完整2900条MLX FP16验证，也不代表FP32已经通过完整核对。

## 证据与复现

- 本目录 `summary.json`：64条诊断记录、原始logits／概率、批量成员、填充长度及误差。
- `../laya-mlx-layer-trace-20261014-v1/summary.json`：CPU／CUDA／MLX输出及逐层误差。
- `../laya-mlx-operator-trace-20261014-v1/summary.json`：首层QKV、attention、MLP各阶段。
- `../laya-mlx-qkv-control-20261014-v1/summary.json`：完全相同输入／权重的CPU与GPU矩阵乘法对照。
- 各目录 `protocol.json`、`reproduction/`、`evidence-manifest.json` 保存协议、源文件和结果哈希。脚本拒绝覆盖已有run目录。
- 脚本分别为 `scripts/diagnose_laya_mlx_precision.py`、`scripts/trace_laya_mlx_precision.py`、`scripts/trace_laya_mlx_operators.py`、`scripts/trace_laya_mlx_qkv.py`，使用锁定的laya/reference/laya-mlx/metal依赖与本机GPU执行，四个进程均正常退出。

## 决策

暂不以“改成FP32”或“batch1”作为修复，也不需要为这个问题重新训练。现有CUDA FP32仍是严格复现参照；Mac FP16保留“完整样本类别一致、严格概率门槛未通过”的结论。

若下一步需要消除概率超限，应针对已隔离的GPU矩阵乘法准备最小复现，研究运行库精确计算路径或受控替代实现；任何候选修复都必须使用新run_id重新完整核对2900条及测量性能，不能仅凭这4条异常样本替换部署。
