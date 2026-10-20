# IMDB 情感分类项目报告

本目录提供课程报告的 Word、PDF、XeLaTeX 源码和图表，内容对应 2026-10-20 最终核验版本。正文由 `manuscript.py` 统一维护，可导出 TeX 与 Word。

## 阅读与交付

| 材料 | 文件 | 说明 |
| --- | --- | --- |
| 可编辑报告 | [IMDB项目报告.docx](IMDB项目报告.docx) | 正文、表格与公式；本机渲染 18 页 |
| PDF 报告 | [main.pdf](main.pdf) | XeLaTeX 编译，16 页 |
| TeX 源码 | [main.tex](main.tex) | UTF-8，以 XeLaTeX 编译 |
| 图表 | [figures/](figures/) | 11 幅图，PDF / 300 DPI PNG；架构图另有 SVG |
| 正文源 | [manuscript.py](manuscript.py) | 两种输出共用的段落、表格和参考文献 |
| 身份字段 | [submission.json](submission.json) | 姓名、学号、班级；不显示课程名及指导教师 |
| 核验材料 | [analysis/](analysis/) | 指标、来源 SHA256、图表清单和独立复算 |
| 修订记录 | [REVISION_NOTES.md](REVISION_NOTES.md) | 历次内容与版式变更 |

封面已按用户提供的信息填写姓名、学号与班级，并移除课程名和指导教师；首页附有可点击的 GitHub 项目链接。导出时若必要身份字段缺失或仍为占位符，将停止生成。报告包含 8 条参考文献、7 张表和 8 个公式；当前页数与检查状态以 [document_validation.json](analysis/document_validation.json) 为准。封面更正的最新核验见 [identity_revision_20261020.json](analysis/identity_revision_20261020.json)，早期审查记录及其输出哈希保留为历史版本。

## 报告内容

方法为双 Laya 多视图与 DeBERTa-v3-large 的异构概率融合，最终系统共三个 checkpoint。原固定方案完整官方测试为 **24,257 / 25,000 = 97.028%**；相对双 Laya 净增 171 条正确，相对 DeBERTa 净增 24 条、McNemar p=0.1568。

最终报告已纳入九次训练组成的三组完整重复、完整开发集 CPU 数值核验、RTX 4090 同平台测速、权重诊断、文本组条件区间及后处理分组五折。三组等权融合平均 Accuracy 与 DeBERTa 同为 **97.08%**，平均 ECE 为 **0.741%**（DeBERTa 为 **1.328%**）。补实验没有访问官方测试，事后权重候选未形成独立确认或替换最终规则。

数据使用、重复文本、后端差异、统计限制和计算开销均在正文说明。研究问题、配置与逐项结果可从 [工程文档](../docs/README.md) 继续追溯。

## 编译已有 TeX

安装包含中文支持的 TeX Live / XeLaTeX，在仓库根目录执行：

```bash
TEXINPUTS=paper: xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory=paper paper/main.tex
TEXINPUTS=paper: xelatex -interaction=nonstopmode -halt-on-error \
  -output-directory=paper paper/main.tex
```

使用报告压缩包时，解压后进入包含 `main.tex` 和 `figures/` 的 `IMDB报告/`，执行两次：

```bash
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

两轮编译用于解析交叉引用。中文使用 TeX Live 自带 Fandol 字体；仅编译已有正文和图表不需要模型、数据或原始预测。

## 从完整仓库重建

重算图表需要本机完整冻结预测和实验材料；Git 仓库及报告材料包只包含选定结果，无法单独重算全部图表。根环境需保留模型依赖并安装图表 extra：

```bash
uv sync --locked --extra laya --extra laya-mlx --extra analysis
uv run --locked --offline --no-sync python paper/build_figures.py
```

报告导出额外需要 Python 环境中的 `python-docx`、`Pillow`，以及 PATH 中的 XeLaTeX 和 Poppler。这套文档工具链不包含在根环境依赖锁中；使用已有完整文档环境执行：

```bash
python3 paper/build_report.py
```

该步骤生成 Word 与 TeX，随后按上方命令编译 PDF。`build_architecture.py` 还会核对本机模型配置和依赖源码，单独重建架构图需恢复其绑定资产；结构证据见 [architecture_sources.json](analysis/architecture_sources.json)。

Word 使用宋体／黑体字体名称；本机 LibreOffice 通过 `qa/fonts.conf` 做字体映射。不同设备的字体和分页可能不同，Word/WPS 实机检查仍待完成。

## 报告材料包

确认 Word、PDF 和源码一致后，在仓库根目录执行：

```bash
uv run --locked --offline --no-sync python paper/build_package.py
```

生成本地 `paper/IMDB报告与TeX源码.zip`，包内顶层为 `IMDB报告/`。`analysis/package_manifest.json` 记录每个条目；除清单自身避免自引用外，所有文件均核对 SHA256。该命令会更新现有报告包，需保留旧版本时先另存旧包。

报告包不包含项目全部源码、模型、原始数据或视频。项目总交付状态与历史权重缺口见 [交付说明](../docs/closeout/delivery.md)。

## 核验记录

[final_review_20261020.json](analysis/final_review_20261020.json) 与 [document_validation.json](analysis/document_validation.json) 记录正文、指标、公式、引用、嵌入图像和逐页版式检查。Word 18 页与 PDF 16 页均已检查，二者独立分页。当前 README 的整理不改变报告正文、数据或已有核验结论。

历史审查见 [review/REVIEW.md](review/REVIEW.md)，各轮修订前快照保留在 `review/` 对应子目录。静态审查可在具备其依赖和冻结输入的完整本地环境中运行 `python3 paper/review/audit.py`；它不执行新的模型测试前向。
