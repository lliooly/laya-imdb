# Hugging Face 模型发布记录

日期：2026-10-20。状态：**已公开发布并核验**。

模型仓库：<https://huggingface.co/shishishi3/laya-deberta-imdb-ensemble>。

固定发布 revision：`9a20453f3c48854a456a584e7f69c82f495c5c00`。

## 发布内容

发布沿用原固定系统的三套推理权重、分词器、模型配置和冻结校准参数，另附 CPU FP32 启动脚本、依赖列表、20 个绑定哈希的本地运行模块、模型卡及各组件许可。发布文件共 55 个，载荷约 3.44 GB。运行模块与 GitHub `f39da7f9714a76885ad0dbff1133752549e543ed` 的原文件一致。

没有重新训练、拟合、挑选补实验最好种子或访问官方测试。原论文的后端口径、成绩和限制不变；完整融合使用随附入口，未声称支持默认 `transformers.pipeline()` 或托管推理。

## 验证

1. 固定上游 Laya revision 的模型卡标注 Apache-2.0；固定 DeBERTa revision 标注 MIT。保留来源锁、模型卡、许可正文和署名，统一许可入口为模型仓库 `LICENSES.md`。
2. 本机模型包的三份权重及冻结参数与既有提交材料逐文件一致。独立运行包示例的正类概率为 0.9918030091682009，与此前核验一致。
3. 上传后逐文件核对服务端记录：LFS 文件比对 SHA-256，普通文件核对 Git blob 身份并与本地 SHA-256 清单绑定。
4. 按固定 revision 获取模型快照并复核全部载荷 SHA-256。大文件仅在服务端 LFS 哈希匹配时复用本地字节；小文件从固定 revision 下载。
5. 核验后公开仓库，并通过未认证请求确认该 revision 可读取。详见 [verification.json](verification.json)。

模型包不含原始 IMDB 影评、完整数据集、官方测试预测数组、优化器状态、中间恢复包或学生学号班级。源码与论文保留在 GitHub；下载与运行步骤见 [发布指南](../../docs/guides/huggingface.md)。
