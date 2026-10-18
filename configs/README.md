# 配置与模型锁

保留全部历史 JSON 与 run_id，含先导、冒烟、失败和完整训练；文件存在不等于任务完成。[设计与配置索引](../docs/closeout/designs.md)提供完整导航，[研究台账](../docs/closeout/research-ledger.md)指向最终结果。

model_lock、laya_model_lock、deberta_model_lock 等记录不同模型来源；各日期配置的 SHA 绑定与设备协议按对应报告判断。复跑不能只改输出目录而忽略脚本内硬编码身份或科学绑定。

最新 token 查询配置只用于冻结编码器实验；final-cpu-alternatives-v2 是 97.40% 开发探索候选；deberta-laya-final-cpu-evaluation 是已完成的固定测试评估协议。整理不重新运行这些任务。
