# 实验配置与模型锁

本目录保存训练、推理、评估配置与模型版本。历史配置按原运行身份保留，包含先导、冒烟、失败及已完成实验；最终状态由对应报告确认。

| 配置 | 用途 |
| --- | --- |
| `laya_model_lock.json`、`deberta_model_lock.json` | 模型来源、revision 和文件哈希 |
| `model_lock.json`、`tokenizer_lock.json` | 早期 ModernBERT 与分词基线 |
| `laya-full-ce.json` | 历史一轮 Laya-CE，开发 95.76% |
| `laya-rtx4090-ce-baseline-s43-epochs2-fp32.json` | 双 Laya 的 baseline-s43 配方 |
| `laya-rtx4090-ce-epochs2-warmup005-ls005-fp32.json` | 双 Laya 的 combined-s42 配方 |
| `laya-fusion-inference-*.json` | 双 Laya 多视图推理；设备、精度与资产分别绑定 |
| `deberta-heterogeneous-bf16-budget7-s42-20261017-v1.json` | 原 DeBERTa 两轮训练 |
| `deberta-laya-final-cpu-evaluation-20261018-v1.json` | 原固定 50:50 融合测试评估 |
| `paper-supplementary-preparation-20261019-v1.json` | 九次补训练的准备快照与固定种子；执行完成状态见下方报告 |
| `final-cpu-alternatives-20261018-v2.json` | 97.40% 开发探索，非最终测试方案 |
| `laya-token-query-mps-fp32-20261018-v1.json` | 冻结编码器的查询头对照 |

九次补训练、回传和测速已经完成，见 [补实验报告](../reports/paper-gpu-supplement-20261019-v1/REPORT.md)。准备快照中的“待启动”和原机器字段反映当时状态，不反向改写实验契约。

复跑前核对脚本内运行身份、模型与输入 SHA、设备准入和原报告。单独复制 JSON 或更改输出路径不足以建立新实验。完整历史导航见 [设计与配置索引](../docs/closeout/designs.md)，当前读者入口见 [文档索引](../docs/README.md)。
