# 代码入口

源码按原路径保留，避免破坏 import、日期协议和源码哈希。[完整模块索引](../docs/closeout/code.md)列出职责与静态局部依赖，[研究台账](../docs/closeout/research-ledger.md)关联设计和结果。

- 主线训练：train_laya.py；DeBERTa 最终两轮：deberta_budget_study.py。
- 历史单模型推理：predict.py；旧 Laya 融合：predict_laya_fusion.py。
- 最终异构研究分析：deberta_final_analysis.py、deberta_final_evaluation.py，不作为现成单条推理 CLI。
- 词袋、最终 CPU 候选、token 查询：run_bow_nb_study.py、run_final_cpu_alternatives_v2.py、run_token_query_study.py。
- 收尾资料：build_closeout_catalog.py，提供 build、verify、test、package。

日期入口可能绑定已有 run_id；不能批量执行全部脚本。新实验需新身份和重新准入，最终部署状态见[说明](../docs/closeout/delivery.md)。
