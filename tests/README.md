# 测试与验证

从仓库根目录使用锁定环境运行。测试不需要启动真实训练或重新评分官方测试；模型重载、GPU 准入和跨后端核验有各自的原实验收据。

## 不依赖历史资产的模块检查

```bash
uv run --locked --offline --no-sync python -m unittest discover \
  -s tests -p test_closeout_catalog.py -v

uv run --locked --offline --no-sync python -m unittest discover \
  -s tests -p test_paper_fusion_robustness.py -v
```

论文补充检查包括：

| 模块 | 检查内容 |
| --- | --- |
| `test_paper_supplementary_validation.py` | 相同清洗文本分组、标签冲突和重采样 |
| `test_paper_gpu_supplement.py` | 补训练配置、种子与固定运行规则 |
| `test_paper_fusion_diagnostics.py` | 权重拟合、ID 对齐和纠错方向 |
| `test_paper_fusion_robustness.py` | 分组五折、留出标签隔离、参数与样本扰动 |

将上方 `-p` 替换为对应文件名即可逐模块执行。历史 10 月 18 日的 48 模块 / 234 项测试记录仅代表当时版本，不作为当前模块总数。

## 全部模块

完整测试包含本地集成检查，首次克隆后仅安装依赖还不能全部运行。除根环境与 `environments/bow-cpu` 外，需恢复原始 Laya 分词器（`.cache/laya/english/tokenizer/`）、冻结数据（如 `data/processed/train.jsonl`）和相应历史预测（如 `test_laya_posthoc_inputs.py` 使用的报告输入）。这些资产不进入 Git，准备方式见 [复现指南](../docs/closeout/reproduction.md)，具体路径由各测试的输入契约确认。尚未恢复资产时，先运行上方自包含模块。

部分科学边界测试会安装进程级永久 audit hook，因此每个模块应使用单独进程。资产与环境准备完整后执行：

```bash
for test_module in tests/test_*.py; do
  case "$test_module" in
    tests/test_bow_nb_branch.py|tests/test_laya_deberta_calibration_weight.py)
      uv run --project environments/bow-cpu --locked --offline --no-sync \
        python -m unittest discover -s tests -p "${test_module##*/}" -v || break
      ;;
    *)
      uv run --locked --offline --no-sync \
        python -m unittest discover -s tests -p "${test_module##*/}" -v || break
      ;;
  esac
done
```

该命令将结果写到终端，遇到失败停止，不覆盖历史报告。10 月 18 日的归档工具 `build_closeout_catalog.py test` 会写入旧日期目录，保留用于追溯原流程。

## 验证范围

单元测试覆盖输入契约、统计方法、哈希和覆盖保护等规则，不代表当前设备完成了 GPU 全量训练或全部模型加载。最终报告已有核验见 [document_validation.json](../paper/analysis/document_validation.json)，历史训练与数值证据由 [文档入口](../docs/README.md) 导航。
