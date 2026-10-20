# 论文补实验接入准备

日期：2026-10-19。用户已确认补实验及人民币 100–300 元预算；本轮准备完成后结束，等待用户提供实例信息。未租用、启动或连接任何付费实例。

## 本地已执行及可复现入口

- 重复文本敏感性分析：`reports/paper-duplicate-sensitivity-20261019-v1/REPORT.md` 与 `summary.json`。复用已有冻结预测；四种口径分别报告，不修改官方主成绩。
- 完整开发集 CPU FP32 核验：`reports/paper-deberta-cpu-development-parity-20261019-v2/`。已完成并通过独立正确数、NLL、ID 与冻结文件哈希复核。DeBERTa 类别变化 1 条（2426→2427 正确），最终融合类别变化 0 条（2432/2500 不变）；最大概率差 0.01190887，平均差 0.00015210，未声称数值等价。详见该目录的 `REPORT.md`、`summary.json` 和 `verification.json`。
- 实现：`scripts/paper_supplementary_validation.py`；文本分组、代表记录选择和 bootstrap 的四项测试已通过。
- `v1` CPU 准备在分词前因根环境缺少 SentencePiece 失败，没有执行模型前向。`v2` 使用已有锁定的 deberta-prep 分词依赖；与根环境共有的 Transformers、NumPy、tokenizers、HF Hub 版本均相同。没有安装或修改依赖；完整版本及锁文件哈希见 `runtime_context.json`。

本地重复文本分析命令如下。输出路径必须不存在，以免改写已有报告：

```bash
UV_CACHE_DIR=/private/tmp/imdb-uv-cache uv run --locked --offline --no-sync python scripts/paper_supplementary_validation.py analyze --output reports/NEW_RUN_ID
```

恢复已经冻结的 CPU 核验使用：

```bash
PYTHONPATH="$PWD/environments/deberta-prep/.venv/lib/python3.12/site-packages" UV_CACHE_DIR=/private/tmp/imdb-uv-cache uv run --locked --offline --no-sync python scripts/paper_supplementary_validation.py cpu-parity --resume --output reports/paper-deberta-cpu-development-parity-20261019-v2
```

## 建议实例

单张 L40S 48GB、Linux、至少 32GB 系统内存、约 150GB 可用数据盘、SSH、按量计费。先取得实际时价和实例信息，代码和资产准备好后再开机。Ubuntu 版本由平台现有镜像及驱动兼容性决定，Python 和训练依赖以仓库锁定文件为准。

100–300 元是总预算，不是单卡日预算。首次接入核对 GPU、驱动、可用显存和磁盘，做固定小规模准入；将实测速率、准备/评估/保存/传输余量及存储收费一起计入预估。预估无法在 300 元内完成时不启动完整排程，也不自动追加预算。

## 远程科学方案

此前方案见 `docs/superpowers/specs/2026-10-19-paper-supplementary-validation-design.md`。在任何新训练前进一步固定：三个系统重复使用互不重叠的随机种子组，组内保留原组合的相对 seed 关系：

| 重复 | Laya baseline | Laya combined | DeBERTa |
|---|---:|---:|---:|
| 1 | 43 | 42 | 42 |
| 2 | 143 | 142 | 142 |
| 3 | 243 | 242 | 242 |

这替代原草案的逐一加一分组，避免跨重复复用同一 Laya 随机种子；修改发生在付费训练及新候选评估之前，不依据新训练成绩。三个重复都在同一远端环境重新训练，共九次两轮训练；历史原组合仅作参照。三组数量只提供初步训练稳定性证据，不保证检出 0.096 个百分点的小效应。

温度和 Laya 内部系数依照原校准流程估计，外层等权固定；不按开发结果挑种子、轮次或权重。不上传官方测试文本、标签或测试预测到训练目录，不将旧官方测试反复用于新种子筛选。独立确认仍缺新的合格评估数据，租卡不能消除这项缺口。

现有 `deberta_budget_study.py` 是历史 Modal 专用入口，硬编码 study、预算及 seed 42；`deberta_training.load_initial` 也固定初始化 seed 42。不能仅改 JSON 的 seed 就宣称完成多种子重训。接入后使用新入口明确传入初始化、dropout 与数据顺序的 seed，并用小规模运行核验初始状态和输入顺序不同；历史科学源码和绑定哈希保持原样。

同平台成本需对现有冻结的 DeBERTa、双 Laya 和最终三模型实际前向进行统一测量，不能用缓存融合时间替代。采用共同样本、预热、重复次数和计时范围，记录显存及模型加载是否计入。

## 接入信息与资产

等待主机、端口、用户名、卡型及每小时单价。专用 SSH 公钥已生成：`/private/tmp/imdb-paper-ssh-20261019/id_ed25519.pub`。私钥仅保存在本机临时目录，未写入仓库。收到实例信息后先验证连接与主机身份。

迁移采用允许清单，包含训练/开发数据、冻结划分 manifest、源码、锁定环境、预训练模型及所需三份最终模型；不整体上传工作区、Git、凭据或历史杂项。预训练和最终权重应分别核对，不能拿已有微调模型作为新训练的随机起点。

本机可用空间约 25GiB。远端恢复包优先持久化；先回传小型日志、预测和清单，再核对最终权重回传所需空间。任何已有权重和历史报告均不删除。

## 下一次收到实例信息后

1. 只读核对连接、GPU、费率、磁盘与驱动，计算预算。
2. 上传允许清单并核对 SHA256，构建锁定环境；使用新 run_id。
3. 对新增 seed 入口和小规模训练/重载进行准入，确认数值有限与有效预算一致。
4. 预算允许时执行三个完整重复及统一成本测量，先保存再评估，报告全部结果。
5. 生成补充结果和论文修改建议；仅将完成且核验过的证据写入论文，独立数据缺口单列。
