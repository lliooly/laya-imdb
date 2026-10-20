# IMDB 课程项目报告

本目录包含 Word 报告、可编译的 XeLaTeX 源文件与 PDF，以及图表和核验材料。正文由 `manuscript.py` 统一维护，修改后可同时导出 TeX 和 Word。

- `IMDB项目报告.docx`：可编辑正文与表格，8 个公式使用同一 TeX 渲染的图片。
- `main.tex`、`main.pdf`：XeLaTeX 源文件及编译结果。
- `figures/`：11 幅图的矢量 PDF 与 300 DPI PNG。前 9 个文件沿用原数据，其中 `06_reliability` 已扩大纵轴以显示原先被裁出的非空箱，`10_training` 补充历史逐轮训练记录，`11_architecture` 为新增的模型总架构与 Laya 局部放大图，并另有可编辑 SVG。报告中架构图排为图 3，后续图号依次顺延。
- `analysis/`：完整精度参数、指标、配色、冻结输入 SHA256、独立复算及本次修订核验记录。
- `submission.json`：姓名、学号、课程名／班级、指导教师。目前按用户要求均保留“待填写”，填写后须重建报告。
- `REVISION_NOTES.md`：历次修订记录。
- `review/REVIEW.md`：2026-10-19 论文审查清单，按风险排序并记录处理状态；`review/audit.py` 可复算核心指标及重复文本，`review/references.json` 记录引用核验。
- `review/original/`：本轮审查前的正文及导出文件快照。

报告包含 8 条参考文献、7 张表、8 个公式。当前页数以 `analysis/document_validation.json` 为准。原固定方案完整测试为 **24,257 / 25,000 = 97.028%**；相对旧 Laya 净增 171 条，相对最强 DeBERTa 单分支净增 24 条、McNemar p=0.1568。原固定结果与公式保持不变。

2026-10-20 修订纳入九次训练形成的三组完整重复、事后权重诊断、重复五折后处理检查、按文本组的条件区间、完整开发集 CPU 数值核验及 RTX 4090 同平台测速。三组等权融合平均准确率与 DeBERTa 均为 97.08%，平均 ECE 为 0.741%（DeBERTa 为 1.328%）。正文以部分配置的小幅增益、误差互补和概率质量为主线，同时说明其条件性。新增实验没有访问官方测试；权重探索没有形成独立确认，也未据开发集选择最好种子或最终策略。本轮改写前快照在 `review/before-supplement-20261020/`。

新增架构图可通过 `uv run --locked --offline --no-sync python paper/build_architecture.py` 单独重建；它核对本机锁定的模型配置及依赖源码，结构证据见 `analysis/architecture_sources.json`，设计范围见 `analysis/architecture_design.md`。

## 仅编译已导出的 TeX

解压 `IMDB报告与TeX源码.zip` 后，先进入 `IMDB报告/`，即包含 `main.tex` 和 `figures/` 的目录，再执行：

```bash
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

也可在 LaTeX 编辑器中打开该目录，以 `main.tex` 为主文件并选择 XeLaTeX。中文使用 TeX Live 自带 Fandol 字体，编译不依赖 macOS 系统中文字体。

## 在完整仓库中重建双格式报告

以下命令仅供完整 `End-IMDB` 仓库使用，均从仓库根目录执行。报告压缩包不包含冻结预测、实验环境或完整 `reports/`，不能单独重新计算图表。

```bash
UV_CACHE_DIR=/private/tmp/imdb-uv-cache uv run --locked --offline --no-sync python paper/build_figures.py

# 使用 Codex 已配置的文档运行时；其他机器须提供 python-docx、Pillow、XeLaTeX、Poppler。
/Users/shishishi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 paper/build_report.py

TEXINPUTS=paper: xelatex -interaction=nonstopmode -halt-on-error -output-directory=paper paper/main.tex
TEXINPUTS=paper: xelatex -interaction=nonstopmode -halt-on-error -output-directory=paper paper/main.tex
```

Word 使用“宋体／黑体”标准字体名称；本机无界面预览通过 `qa/fonts.conf` 映射到系统字体。不同设备的字体字形及分页仍可能不同。当前核验使用本机 XeLaTeX 与捆绑 LibreOffice；实际 Word／WPS 环境未执行验收。预览仅用于内部核验，不打入报告包。

## 打包与交付边界

从完整仓库根目录执行：

```bash
/Users/shishishi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 paper/build_package.py
```

压缩包 `IMDB报告与TeX源码.zip` 仅为报告材料包。清单包含包内每一个文件条目；清单自身为避免自引用不记录自身哈希，其他文件均以 SHA256 核对。包内顶层目录为 `IMDB报告/`。

这不是源码、权重、训练／测试视频的总交付包。ModernBERT-base／Laya-RLCD 公平迁移、两版交付及默认部署整合尚未完成；训练与测试视频尚需独立完成验收。三模型计算链路已在 RTX 4090、64 条固定开发记录上实测，包含分词、传输、前向与融合，不含加载；该批记录均为负类，批处理结果不代表完整在线部署延迟。历史源码包早于本次报告，正式总交付时仍须重建。交付状态保留在工程文档中。

## 最终核验

2026-10-20 最终检查已完成：纠正摘要中校准与融合的顺序，将交叉验证的正向发现具体限定为保守组合保持准确率并降低 NLL，修正表 3 的后端注释。13 项补实验与分析测试通过；Word 18 页和 TeX PDF 16 页均已逐页核验。最终核验记录为 `analysis/final_review_20261020.json`，当前文件页数与结构以 `analysis/document_validation.json` 为准。以下为历史修订说明，其页数对应当时版本。

## 本轮论文审查

已精简交付进度、内部审查口吻与反复的过程说明，补充实际英文模板、训练与推理视图、分支温度、DeBERTa 输入预算及 bootstrap 设置，增加 GCE 原始文献。保留必要的评估限制与未达显著的结果。主指标、原有 8 个公式及 11 幅图的数据均未改变；图 7 的纵轴下限已改为 0，以显示 DeBERTa 的 12/37 分箱点；正文与 Word/PDF 已从统一源重建。

2026-10-20 又进行了一轮表达与语气修订：修复歧义、错词、空格、术语不统一及若干与源记录口径不一致的表述；叙述上突出“部分配置小幅正向、概率质量改善”，同时保留全部数值、p 值、区间与限制事实，未把不显著结果写成显著。主结果、8 个公式、8 条参考文献及 11 幅图的数据均未改动；修订前快照在 `review/before-tone-20261020/`，当前页数以 `analysis/document_validation.json` 为准（TeX 15 页、Word 渲染 18 页）。

```bash
/Users/shishishi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 paper/review/audit.py
```
