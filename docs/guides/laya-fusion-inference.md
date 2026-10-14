# 冻结双模型融合推理

推理入口加载baseline-s43与combined-s42两份实际权重，使用保存的原生温度和已冻结的模型间系数70.2033%／29.7967%。每模型对未截断评论直接复用前缀，对超过466 token的评论平均前缀／首尾概率。输入是影评文本，不需要标签，也不会重新拟合系数。

## 服务器推理

在仓库根目录，使用锁定依赖和已有完整checkpoint：

```bash
UV_CACHE_DIR=/root/sj-tmp/uv-cache uv run --offline --frozen --extra laya --extra reference \
  python scripts/predict_laya_fusion.py \
  'The acting was excellent, and I loved the ending.' \
  --config configs/laya-fusion-inference-cuda-fp32-20261014-v1.json
```

输出主融合的类别、正负概率、截断信息和实际样本前向行数。权重与配置文件必须通过SHA256检查；不下载模型或回退CPU。两模型常驻GPU，使用PyTorch FP32、eval和inference_mode，关闭AMP与TF32。

默认的`scripts/predict.py`入口不受影响。此入口的CUDA FP32结果不等于Mac MLX FP16部署核验。备用MPS配置采用本地已校验备份路径，必须另行运行并记录结果。

## 完整一致性与性能验证

```bash
UV_CACHE_DIR=/root/sj-tmp/uv-cache uv run --offline --frozen --extra laya --extra reference \
  python scripts/validate_laya_fusion_inference.py \
  --config configs/laya-fusion-inference-cuda-fp32-20261014-v1.json
```

已有run目录拒绝覆盖；检查配置时加`--validate-config`，不会加载模型。完整验证包括原校准400条及开发2500条，逐条核对输入哈希、长度、logits、概率和类别。绝对容差预先固定为logits≤1e-3、概率≤1e-4，类别必须完全一致。官方测试集保持封存。

性能使用校准集前128条，batch=8，两轮预热、五轮测量。分别测单模型首尾和双模型门控融合；计时包含分词、编码、数据传输、前向和聚合，排除权重加载。每条耗时是批量摊销值，不等于单请求延迟。CUDA显存采用分配器峰值计数，MPS仅报告采样峰值。

完整协议见[执行计划](../superpowers/plans/2026-10-14-laya-fusion-inference.md)。报告输出至`reports/laya-fusion-inference-cuda-fp32-20261014-v1/`，以完成状态、原始测量和逐条对比为准。

## 已完成的验证

2026-10-14，RTX4090 CUDA FP32已通过完整2900条样本核对，单视图logits、单视图及融合概率最大绝对差均为0，类别全部一致。实际开发结果复现主方案96.96%、等权辅助对照97.00%、首尾单模型96.84%。

128条、batch=8的中位总耗时：单模型1.916秒，融合4.133秒（2.16倍）；峰值allocated显存分别1.91／3.48 GiB。对应吞吐为66.82／30.97条每秒。本地与服务器相关检查各14项通过，真实命令行推理也已通过。详见[完整验证报告](../../reports/laya-fusion-inference-cuda-fp32-20261014-v1/REPORT.md)。

## Mac MLX FP16 入口

使用同一对本地备份权重和冻结系数，模型前向在 Metal GPU 上执行。CPU 输入准备复用已核验的 token 规则，因此这个融合入口仍需要 `laya`／`reference` 依赖；PyTorch 不加载模型。默认单模型入口不变。

```bash
UV_CACHE_DIR=/tmp/end-imdb-decision-uv uv run --offline --frozen \
  --extra laya --extra reference --extra laya-mlx --extra metal \
  python scripts/predict_laya_fusion.py \
  'The acting was excellent, and I loved the ending.' \
  --config configs/laya-fusion-inference-mlx-fp16-20261014-v1.json
```

完整验证使用相同依赖，将脚本换为 `scripts/validate_laya_mlx_fusion.py`，删除影评位置参数，保留配置参数。已有 run 目录拒绝覆盖；重新执行必须另取 run_id。MLX 的预声明门槛为概率绝对差≤0.01、logits绝对差≤0.1，并且所有视图与融合类别完全一致。详见[MLX 执行计划](../superpowers/plans/2026-10-14-laya-mlx-fusion-inference.md)。

2026-10-14实测：Mac 2900条所有类别与CUDA一致，主融合96.96%、辅助97.00%；但主融合1条概率绝对差0.01971超过预设0.01，严格核对未通过。M5 Pro吞吐单模型57.36、融合23.70条/秒，MLX峰值活跃分配1.69／2.47 GiB。详见[Mac验证报告](../../reports/laya-fusion-inference-mlx-fp16-20261014-v1/REPORT.md)。

后续数值诊断已在固定相同输入和权重的条件下，将一个异常样本的早期差异定位到MLX GPU首层QKV矩阵乘法执行路径；具体内核原因尚未确定。切换FP32或batch1均未消除诊断样本的超限，未作为修复部署。见[诊断报告](../../reports/laya-mlx-precision-diagnosis-20261014-v1/REPORT.md)。

## 显式 MLX 精确 FP32 候选

MLX 官方说明，部分硬件上的FP32矩阵运算默认可使用降低精度路径。必须在启动新进程时设置 `MLX_ENABLE_TF32=0`，再使用全FP32模型；单独修改dtype不充分。来源：[MLX数值精度说明](https://ml-explore.github.io/mlx/build/html/usage/precision.html)。本入口拒绝缺失开关，不自动修改已有进程的精度状态。

```bash
MLX_ENABLE_TF32=0 UV_CACHE_DIR=/tmp/end-imdb-decision-uv uv run --offline --frozen \
  --extra laya --extra reference --extra laya-mlx --extra metal \
  python scripts/predict_laya_fusion.py \
  'The acting was excellent, and I loved the ending.' \
  --config configs/laya-fusion-inference-mlx-fp32-precise-20261014-v1.json
```

完整核对脚本为 `scripts/validate_laya_mlx_precise.py`，使用相同环境变量与配置，拒绝覆盖已有run目录。门槛沿用原Mac协议（logits≤0.1、概率≤0.01、所有类别一致），并报告实测误差。新候选不等于原FP16方案已被修复，也不会自动替换默认模型。

精确FP32完整2900条核对已通过原Mac数值门槛；全部对照最大概率差6.2526942e-05，主融合准确率96.96%。单模型／融合吞吐为17.03／7.53条每秒。详见[精确FP32报告](../../reports/laya-fusion-inference-mlx-fp32-precise-20261014-v1/REPORT.md)。
