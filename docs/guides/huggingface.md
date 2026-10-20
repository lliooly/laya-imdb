# Hugging Face 模型发布与下载

模型仓库：<https://huggingface.co/shishishi3/laya-deberta-imdb-ensemble>。具体发布状态与 revision 以 [发布记录](../../reports/huggingface-release-20261020-v1/REPORT.md) 为准。

本发布沿用原固定三模型系统：两个 Laya 微调模型与 DeBERTa-v3-large，Laya 内部分支系数及外层 50:50 规则保持冻结。模型仓库包含推理权重、分词器、配置、冻结校准参数、20 个绑定哈希的运行模块和 CPU 推理脚本。GitHub 继续保存完整训练、分析和论文工程。

## 下载与运行

使用 Python 3.12、uv 和 Hugging Face CLI。在终端执行：

```bash
hf download shishishi3/laya-deberta-imdb-ensemble --local-dir imdb-ensemble
uv run --python 3.12 --with-requirements imdb-ensemble/requirements.txt python imdb-ensemble/predict_final.py --repo imdb-ensemble/runtime --text "The movie was beautifully acted and thoroughly enjoyable."
```

下载模型和首次安装依赖需要联网；推理读取本地权重，使用 CPU FP32，不重新训练或拟合。完整融合系统使用配套脚本，尚未实现直接通过 `transformers.pipeline()` 加载整个系统。

仅核对文件时执行：

```bash
python imdb-ensemble/predict_final.py --repo imdb-ensemble/runtime --verify-only
```

模型仓库的 `FILE_MANIFEST.json` 列出载荷 SHA-256。为固定复现，可给 `hf download` 增加 `--revision` 并使用发布记录中的提交哈希。

## 结果与许可

原固定官方测试 Accuracy 为 97.028%（24,257/25,000），对应 Laya CUDA FP32 概率与 DeBERTa CPU FP32；随附任意文本入口为全 CPU FP32。三组重训平均 Accuracy 与 DeBERTa 持平，平均 ECE 改善。发布不改变论文口径，不宣称在所有模型组合和数据上取得稳定分类增益。

固定 revision 的 Laya 模型卡标注 Apache-2.0，DeBERTa 标注 MIT。本项目代码使用 MIT；各组件的原许可与署名保留在模型仓库 `LICENSES.md`、`licenses/` 和 `provenance/` 中。模型包不包含原始影评数据、优化器状态、中间训练恢复包或学生学号和班级。
