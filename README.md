# Laya IMDB

IMDB 情感分类课程研究：Laya-CE / ModernBERT-large、PyTorch 训练、Mac MLX 推理、DeBERTa-v3 异构融合，以及数据、损失、输入和决策头对照。更新：**2026-10-18**，现进入研究收尾与交付准备。

| 成绩口径 | Accuracy | 状态 |
|---|---:|---|
| 固定 Laya + DeBERTa 50:50，开发 2500 条 | **97.28%** | 开发 DeBERTa 为 CUDA BF16 |
| 固定异构融合，官方测试 25000 条 | **97.028%** | DeBERTa CPU FP32 + 旧 Laya CUDA 概率缓存 |
| 正则化 logit 融合，开发 2500 条 | **97.40%** | 探索候选，净增 3 条、p=.25，未做对应正式测试 |
| 默认 predict.py 的历史单模型 | **95.76%** | 尚未切换为最终研究融合 |

97.40% 不是测试成绩；98% 目标尚未达到，不宣称 SOTA。官方测试已有历史使用，开发与校准也已多轮用于研究。研究最高分不等于默认部署行为。

## 阅读入口

- [收尾总览](docs/closeout/README.md)：结论、资料导航与交付边界。
- [24 类研究台账](docs/closeout/research-ledger.md)：设计、代码、最终结果和验证对应关系。
- [指标索引](docs/closeout/metrics.md)：保留历史 123 条记录并补充后续结果，区分开发、测试和选择性覆盖。
- [源码与测试](docs/closeout/code.md)、[设计与配置](docs/closeout/designs.md)、[报告与核验](docs/closeout/reports.md)、[文件清单](docs/closeout/files.md)。
- [复现与环境](docs/closeout/reproduction.md)、[参考资料](docs/closeout/references.md)、[完整文档入口](docs/README.md)。

## 环境与检查

Python 3.12，使用 uv 与锁定依赖；所有命令在本仓库根目录执行。首次安装需联网，已有环境检查使用 --offline --no-sync。

```bash
uv sync --locked --extra laya --extra laya-mlx
uv sync --project environments/bow-cpu --locked
uv run --locked --offline --no-sync python scripts/build_closeout_catalog.py build
uv run --locked --offline --no-sync python scripts/build_closeout_catalog.py verify
uv run --locked --offline --no-sync python scripts/build_closeout_catalog.py test
```

测试按模块隔离进程，统计词袋相关模块使用独立 CPU 环境；单元测试不代替 GPU 或跨后端核验。已有 run_id 不能直接覆盖，重跑先阅读原协议。

默认历史单模型示例：

```bash
uv run --locked --offline --no-sync python scripts/predict.py \
  'The acting was excellent, and I loved the ending.'
```

它读取 `checkpoints/laya-l512-s42-ce/`，并不复现 97.028% 异构融合。最终单条预测整合与课程交付见[交付清单](docs/closeout/delivery.md)。

## 数据与目录

原训练 22500：梯度 22100、校准 400；开发 2500；官方测试 25000。标签修订使用独立派生版本，原标签文件保留。Laya 总长 512 包含题目，实际评论预算 466 token；DeBERTa 使用自己的分词器与预算。

| 目录 | 内容 |
|---|---|
| scripts/、tests/ | 研究实现、训练/推理入口、验证与测试；保持路径以维护依赖和哈希 |
| configs/、environments/ | 实验配置、模型锁、平台依赖锁 |
| docs/closeout/ | 最新总览、台账与自动生成清单 |
| docs/superpowers/、docs/archive/ | 日期设计、实现计划、原始提案和整理前快照 |
| reports/ | 历史证据；最终结果可能位于 evaluation/，原报告不改写 |
| artifacts/、checkpoints/、delivery/ | 本地权重、缓存和交付材料，不进入 Git |
| archive/legacy-code/、rust-tokenizer/ | 停用入口与历史分词工程 |

## 交付与许可

源码资料包不包含数据目录、权重、缓存和虚拟环境，也不替代 Word 研究报告和训练/测试视频（每个不超过 5 分钟）。原创代码采用[MIT](LICENSE)，第三方材料保留其原条款，见[第三方声明](THIRD_PARTY_NOTICES.md)和[Apache-2.0 文本](docs/THIRD_PARTY_APACHE_LICENSE.txt)。

生成的 `docs/closeout/catalog.json` 与 `local-assets.json` 属于本机清单，不进入 Git；首次检查前先运行 build。
