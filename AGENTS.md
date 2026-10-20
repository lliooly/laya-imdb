# Repository instructions

- 不使用视觉伴侣。不创建 worktree，除非用户明确要求；如需 dev 分支，先提醒用户。
- 所有命令在本仓库根目录运行；使用 uv 和锁定依赖，不提交本地环境、权重、原始数据或缓存。
- 当前为交付准备版本：固定方案为双 Laya 多视图分支与 DeBERTa-v3-large 的 50:50 校准概率融合（共三个 checkpoint）。原固定开发 97.28%、测试 97.028%；97.40% 仅为开发探索，不自动切换最终规则。
- 2026-10-20 已完成三组完整重训、后端核验、权重诊断和稳健性分析。三组等权融合平均 Accuracy 与 DeBERTa 同为 97.08%，平均 ECE 0.741% / 1.328%；不宣称稳定 Accuracy 优势。Word/PDF 报告已完成，当前状态见 docs/experiments/current-status.md 与 paper/README.md。
- predict.py 仍为历史 Laya 单模型；随附模型包的三模型 CPU 单条预测已核验，入口不在源码仓库内；用户已确认两个视频放在桌面，未纳入 Git，未独立核验成片。ModernBERT-base / Laya-RLCD 公平迁移、端到端新头等为后续研究，不根据旧路线图自动追加训练、付费计算、测试重评或部署切换。
- CUDA 入口必须保留非有限值保护；Laya FP32、DeBERTa BF16 与 MLX 概率核验分开认定。Laya AMP FP16 的历史梯度溢出未通过准入，不将规划写成已有结果。
- 保留固定数据划分和测试集封存规则。修改实验配置使用新 run_id，不覆盖历史结果，不根据官方测试结果调参或换模型。
- reports 中历史 manifest、哈希、诊断和 reproduction 快照用于审计，不直接改写；源码变化后按相应训练协议重新验证。build_closeout_catalog.py 绑定 10 月 18 日输出，重新检查或封包应使用新的目录与收据，避免覆盖旧证据。
- 本仓库根 LICENSE 为 MIT；保留第三方来源声明及 docs/THIRD_PARTY_APACHE_LICENSE.txt。
