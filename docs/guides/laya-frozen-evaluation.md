# 冻结后的官方测试评估

2026-10-14，经用户明确授权，固定候选后开启原25000条官方测试集做一次性推理评估。后续不能再把本次测试集称为“从未评估的封存集”，也不能依据其分数调参或换模型。

主方案为baseline-s43／combined-s42的原生温度门控融合，冻结系数0.7020326697613102；参照为baseline-s43首尾单模型。开发集成绩分别96.96%与96.84%。等权融合保留历史辅助身份，本次不评分、不反选。

冻结协议：[配置](../../configs/laya-frozen-official-test-20261014-v1.json)，SHA256为 `25abdd853c3b236d70861ca26e2612dc3d8c97be3620eb3183df21f09dcb92e8`。协议绑定源码、权重配置、融合系数、数据哈希、主指标与98%研究目标。完整测试需24500条正确才达到该目标。

运行使用RTX4090 CUDA FP32、AMP／TF32关闭、batch8。原400校准集先做工程一致性预检，通过后才读取测试正文／标签。所有视图保持原全局batch8分组，主方案与参照共享前向结果，不新增训练或拟合。

```bash
UV_CACHE_DIR=/root/sj-tmp/uv-cache uv run --offline --frozen --extra laya --extra reference \
  python scripts/evaluate_laya_frozen_test.py \
  --freeze configs/laya-frozen-official-test-20261014-v1.json --backend cuda
```

这是复现命令，不是建议重复评测。已有run目录拒绝覆盖；首次完成结果存入 `reports/laya-frozen-official-test-20261014-v1/`。状态及进度在status.json／progress.json，汇总在summary.json。后台监督进程独立于SSH连接，90分钟超时保护。

正式报告包括主／参照指标、配对检验、95% Wilson区间、截断分层及目标判定。已有数据审计显示官方原训练与测试有123个唯一文本重叠，保留完整测试口径并披露该限制。测试分数不用于训练配置选择；ModernBERT实验方法仍须由训练／开发证据确定。

本次已完成：主方案96.344%（24086/25000），参照96.244%，未达到98%目标。完整结果见[正式报告](../../reports/laya-frozen-official-test-20261014-v1/REPORT.md)。
