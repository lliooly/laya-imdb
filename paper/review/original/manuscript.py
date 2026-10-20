"""Report content shared by the editable TeX and Word exports."""
TITLE='基于多视图 Laya 与 DeBERTa-v3 异构概率融合的 IMDB 电影评论情感分类'
SUBTITLE='固定数据划分下的分类性能、概率校准与受控消融研究'
BLOCKS=[]
def p(text,src=''):BLOCKS.append(dict(kind='p',text=text,source=src))
def h(text,level=1):BLOCKS.append(dict(kind='h',text=text,level=level))
def table(caption,headers,rows,src):BLOCKS.append(dict(kind='table',caption=caption,headers=headers,rows=rows,source=src))
def fig(name,caption,src):BLOCKS.append(dict(kind='figure',name=name,caption=caption,source=src))
def eq(tex,word):BLOCKS.append(dict(kind='equation',tex=tex,word=word))

h('摘要',0)
p('本文研究 IMDB 电影评论二分类，构建多视图 Laya 与 DeBERTa-v3-large 的异构概率融合系统。官方 25,000 条训练样本固定分为梯度训练、校准和开发三个子集，分别用于参数学习、概率校准与方案比较。Laya 分支融合两个既有微调模型的前缀、首尾输入视图；DeBERTa 分支采用独立分词器和首尾输入。各分支沿用自身校准所得温度，再按开发期预声明、测试前冻结的 50:50 比例平均正类概率。在完整 25,000 条官方测试上，系统正确分类 24,257 条，Accuracy 为 97.028%，Macro-F1 为 0.970280，NLL 为 0.092618，ECE 为 0.650%。相对旧 Laya 主融合，准确率增加 0.684 个百分点，净修正 171 条，错误数减少约 18.71%。主要改进证据来自与旧 Laya 主融合的配对比较；相对最强 DeBERTa 单分支的净增益未达显著。受控实验中，增加上下文、使用复杂决策头或鲁棒损失均未得到稳定提升的证据。本文保留跨后端核验、负结果和错误案例。官方测试已有历史使用，开发集被多轮复用，最终融合也未经多随机种子确认，因此条件统计区间不能视为完整的泛化保证。', 'final;context;heads;robust')
p('关键词：电影评论；情感分类；Laya；DeBERTa-v3；概率融合；温度校准')

h('引言')
p('电影评论情感分类要判断作者对整部电影的态度，局部褒贬词未必与最终立场一致。作者可能先介绍剧情，再在结尾表态；也可能认可演员却否定电影，或用挖苦表达不满。词频和局部片段容易遗漏评价对象及语用关系。本项目因此同时考察分类准确率、长文本信息保留和模型置信度。')
p('IMDB 官方训练集和测试集各有 25,000 条带标签评论，正负类各占一半 [1]。本文保留完整官方测试分区，在训练分区内固定划分开发集与校准集，并用保存的配置、模型权重身份和逐样本预测追踪结果。测试集在此前研究中已被使用。本轮新增评估前虽已冻结候选和融合规则，仍不能据此声称整个项目从未接触测试信息。', 'audit;final')
p('ModernBERT 的编码器设计兼顾计算效率与较长上下文 [2]。Laya 将文本状态与类型化问题联合编码 [3]；本项目锁定的英语模型以 ModernBERT-large 为编码器，软件版本为 laya 0.4.1，模型 revision 为 7b928d828b7b0e022f929d9bd2e44165aa270148。DeBERTa-v3 采用替换词检测和梯度解耦的嵌入共享预训练 [4]，在表示、分词与训练路径上均与 Laya 不同，可作为互补分支。但结构差异是否带来收益，仍须比较同一批样本的预测变化。', 'locks;comparison')
p('本文给出输入多视图、Laya 内部融合和跨模型融合的具体流程与概率计算规则，并用完整测试预测、配对错误变化及概率指标检验效果。三种子决策头和损失对照、512/2048 长度对照以及跨后端核验，则用于区分分类改进、探索性波动与数值实现差异。编码器、类型化决策头和概率平均均有既有来源；本项目的贡献限于 IMDB 任务适配、组合方案和可追溯的验证流程，不主张提出新的基础预训练架构。', 'fusion;heads;robust;context;mlx')

h('数据集与实验协议')
h('数据清洗和固定划分',2)
p('数据来自 Stanford 发布的 Large Movie Review Dataset [1]。预处理将 HTML 换行标记替换为空格、还原 HTML 实体并压缩连续空白；保留大小写与标点，不使用停用词删除或词干化，以保留否定、程度及反讽线索。训练与开发采用清洗文本分组后的分层划分，冲突标签文本组保留在训练侧。审计记录训练—开发的唯一清洗文本交集为 0。', 'audit')
table('固定数据分区及用途',['分区','样本数','用途'],[
 ['梯度训练','22,100','模型参数更新'],['校准','400','温度、既定权重或拒答阈值'],
 ['开发','2,500','方案比较与诊断；已多轮使用'],['官方测试','25,000','冻结方案的完整评分']], 'audit;deberta;final')
p('原官方训练集先形成 22,500 条训练池和 2,500 条开发集，再从训练池留出 400 条校准样本，得到 22,100 条梯度训练样本。官方训练、测试均为 12,500 正类和 12,500 负类；开发为每类 1,250 条。官方原训练与测试之间存在 123 个唯一清洗文本重叠，完整官方测试口径保留这些样本，不将其描述为完全无重复的独立总体。', 'audit;deberta;final')
fig('02_data','数据规模与官方测试评论长度。右图长度来自保存的原 Laya 分词统计，横轴为离散长度区间；长度不代表 DeBERTa 的分词长度。官方测试有 3,713 条超过 Laya 的 466 token 评论预算，占 14.852%。','audit;oldtest')
h('训练、校准与评估的边界',2)
p('梯度训练集仅用于模型更新；400 条校准集用于原生温度以及明确声明的少量后处理参数；开发集用于评估研究方案。重复使用校准、开发数据会引入选择乐观性，本文因此把大部分机制分析标为探索性。最终测试前冻结模型身份、输入策略、温度、融合规则和执行后端；评分时按唯一 sample ID 对齐输入、标签及概率，不按不可靠的数组顺序拼接不同导出。', 'final;fusion;deberta')
p('本文不将不同预算历史模型的差值解释为单个机制的因果效应。例如 ModernBERT-base 历史结果采用 256 token，Laya 采用更大编码器及 512 token 总预算，两者可作方案比较，不能用来单独证明类型化决策头有效。后续头部实验虽然采用共同 Laya 编码器及重新初始化的头，也只回答该预训练编码器与该训练预算下的问题。', 'comparison;heads')

h('模型与方法')
h('Laya 类型化情感决策',2)
p('Laya 将评论作为 state，将“作者对这部电影的整体态度是什么”作为 choice 类型问题，候选为 negative 与 positive，并为两个候选提供自然语言判定标准。模型保留上游两层 Transformer 决策头和候选 MASK 位置评分结构 [3]。设编码器为 E，类型化问题模板为 q，决策头为 H，则类别分数由以下抽象表示给出；该式说明数据流，不替代源码中的 token 布局。', 'comparison;traincode')
eq(r'\mathbf z_{m,v}(x)=H_m\!\left(E_m\left(\operatorname{pack}(v(x),q)\right)\right),\qquad \hat y=\arg\max_{c\in\{0,1\}}z_c.', 'zₘ,ᵥ(x) = Hₘ(Eₘ(pack(v(x), q))),    ŷ = argmax[c∈{0,1}] z꜀')
p('模型主要采用监督交叉熵（cross-entropy，CE）训练。对单条样本，类别目标为 y，预测概率为 p；目标分布为 one-hot 时即为普通 CE。部分组合配方使用 0.05 标签平滑，另有校准决策强化学习（Reinforcement Learning for Calibrated Decisions，RLCD）先导实验 [3]。该先导实验与全量 CE 的训练样本预算不同，两者差值不能作为单独改变损失函数的消融结果。', 'comparison;combinedconfig')
eq(r'\mathcal L_{\rm CE}=-\sum_{c=0}^{1}\widetilde y_c\log p_c,\qquad \widetilde y_c=(1-\varepsilon)\,\mathbb 1[y=c]+\varepsilon/2.', 'LCE = −Σ[c=0,1] ỹ꜀ log p꜀,    ỹ꜀ = (1−ε)1[y=c] + ε/2')
h('长度约束与双视图概率',2)
p('Laya 的最大总长度为 512 token。任务问题及候选模板占用预算后，评论实际可用 466 token。前缀视图保留评论开头 466 token；首尾视图对超长评论保留原分词序列的首 233 与末 233 token，避免先裁字符串再重新分词造成预算漂移。对不超过预算的样本，两种视图使用同一输入并复用前向；超长样本才额外计算首尾视图。', 'fusion;oldtest')
eq(r'\bar p_m(x)=\begin{cases}p_{m,\mathrm{prefix}}(x),& \ell_m(x)\le466,\\[2pt]\frac12\left[p_{m,\mathrm{prefix}}(x)+p_{m,\mathrm{headtail}}(x)\right],&\ell_m(x)>466.\end{cases}', 'p̄ₘ(x) = pₘ,prefix(x)，当长度≤466；否则 p̄ₘ(x) = [pₘ,prefix(x)+pₘ,headtail(x)]/2')
p('式（4）中的 p 为正类概率，各分支沿用自身校准所得温度，未为融合重新拟合。Laya 的两个 checkpoint 使用各自训练阶段保存的校准温度；DeBERTa 的温度由本项目在 400 条校准样本上最小化 NLL 得到。温度缩放将 logits 同除以正温度 T，再计算 softmax [5]。正温度不改变单分支的类别排序，却会改变融合时的概率尺度，因而需要将校准效果与 Accuracy 分开考察。', 'fusion;deberta')
eq(r'p_{m,v}(y=1\mid x)=\operatorname{softmax}\!\left(\mathbf z_{m,v}(x)/T_m\right)_1.', 'pₘ,ᵥ(y=1|x) = softmax(zₘ,ᵥ(x)/Tₘ)₁')
h('Laya 内部融合与 DeBERTa 异构分支',2)
p('Laya 分支使用 baseline-s43 与 combined-s42 两个既有模型：前者为 seed 43 的基线配方，后者为 seed 42 的 warmup 与标签平滑组合配方。内部融合系数仅在 400 条校准样本上拟合，以 NLL 加 0.01(λ−0.5)² 为目标，冻结后的 λ 约为 0.7020。完整精度保存在 paper/analysis/plot_data.json。Laya 分支含两个 checkpoint，加上 DeBERTa 后共需三份 checkpoint，系统成本应按这三份模型计算。', 'fusion;combinedconfig;baseconfig')
eq(r'p_{\rm Laya}(x)=\lambda\bar p_{\rm baseline\text{-}s43}(x)+(1-\lambda)\bar p_{\rm combined\text{-}s42}(x).', 'pLaya(x) = λ p̄baseline-s43(x) + (1−λ) p̄combined-s42(x)')
p('DeBERTa-v3-large 使用独立的 SentencePiece 分词器，在 512 总 token 预算内保留特殊 token，并对超长评论取首尾片段。它不使用 Laya 问题模板，可容纳的评论 token 数也不同。该分支以 seed 42 训练两轮，开发推理采用 CUDA BF16，校准温度约为 1.137（完整精度见 paper/analysis/plot_data.json）。最终测试在通过工程准入后改用本地 CPU FP32，Laya 则复用旧 CUDA FP32 冻结概率。因此开发与测试的差异同时包含数据和执行精度差异，不能全部归因于模型结构。', 'deberta;debertaconfig;final;debertacode')
eq(r'p_{\rm final}(x)=\tfrac12p_{\rm Laya}(x)+\tfrac12p_{\rm DeBERTa}(x),\qquad \hat y=\mathbb 1[p_{\rm final}(x)\ge0.5].', 'pfinal(x) = 0.5 pLaya(x) + 0.5 pDeBERTa(x),    ŷ = 1[pfinal(x) ≥ 0.5]')
p('式（6）的 50:50 比例在开发期预声明、官方测试前冻结，未依据测试结果搜索权重。另一个仅最小化校准 NLL 的单系数实验得到 Laya 21.94%、DeBERTa 78.06% 的权重。开发 NLL 从 0.088768 降至 0.088108，准确率却从 97.28% 降至 97.16%：修复 8 条、退化 11 条，净少正确 3 条，精确 McNemar p=0.6476，差异未达显著。该对照说明概率损失与离散分类正确数对应不同目标，也未提供足够证据替换既定等权规则。', 'weight')
fig('01_pipeline','最终冻结方案的计算流程示意。Laya 分支包含两个模型，DeBERTa 分支包含一个模型；两分支在各自校准后等权融合。该图为方法示意，不表示实测因果机制。','fusion;deberta;final')

p('图 3 进一步展开模型内部结构。每个 Laya checkpoint 使用 28 层 ModernBERT-large 编码器，在各 token 的输出上加入 choice 类型嵌入，再经过两层前置 LayerNorm 的 Transformer 决策块；最后提取两个候选 [MASK] 位置，由共享评分网络分别得到负类与正类 logits。DeBERTa-v3-large 使用 24 层编码器，经首 token 池化与线性分类头输出两类 logits。图中的 Laya 层堆栈表示共同结构，baseline-s43 与 combined-s42 的参数仍彼此独立。右侧放大图保留自注意力与前馈子层的残差连接，未画入不参与本次情感概率融合的上游 action head。', 'architecture;archfacts;locks;debertalock')
fig('11_architecture','最终模型架构与 Laya 决策头局部放大。数据流自下而上：左侧为多视图 Laya、DeBERTa 与冻结概率融合，右侧展开标记 [A] 的决策头。堆栈标明编码器层数，×2 表示两个前置 LayerNorm 决策块；两个 Laya checkpoint 使用独立权重。图据锁定配置和实际源码绘制，未表示新的实验或生成式解码器。','architecture;archfacts;locks;debertalock')


h('实验设置与评价指标')
h('训练配置与运行平台',2)
table('构成最终方案的训练配方摘要',['项目','Laya 两个既有模型','DeBERTa-v3-large'],[
 ['训练轮数 / seed','2 轮；43 / 42','2 轮；42'],['最大总长度','512；评论预算 466','512；扣除自身特殊 token'],
 ['micro-batch / 累积','8 / 1','8 / 4'],['初始学习率','编码器 2.5e−5；头 1e−4','1e−5'],['weight decay / clip','0.01 / 1.0','0.01 / 1.0'],
 ['调度','余弦；组合配方 5% warmup','83 warmup / 1,382 总更新'],['平滑系数','baseline 0；combined 0.05','普通监督分类'],
 ['训练计算精度','CUDA FP32；AMP 关闭','CUDA BF16；FP32 主参数']], 'baseconfig;combinedconfig;debertaconfig')
p('研究在 Apple M5 Pro 的 MPS/MLX、本地 CPU 以及 CUDA GPU 环境进行。历史正则化实验使用 RTX 4090；部分头部、损失和后续训练使用单独锁定的云端环境。各阶段的设备、依赖、精度和计时口径不同，本文仅对同协议的测量作直接比较，不把不同硬件及不同阶段的耗时拼成统一效率排行。', 'comparison;regularization;robust;mlx')
h('分类、概率与不确定性指标',2)
p('主要指标为全覆盖 Accuracy，辅助报告宏平均 F1（Macro-F1）和受试者工作特征曲线下面积（area under the ROC curve，AUROC）。概率质量采用负对数似然（negative log-likelihood，NLL）、正类 Brier score，以及期望校准误差（expected calibration error，ECE）。ECE 使用 15 个等宽置信分箱，按 top-label 口径计算，表中以百分数表示；二分类置信度为 max(p,1−p)，空箱不计入 ECE。AUROC 衡量排序，不能替代 Accuracy 或校准误差。', 'final;analysiscode')
eq(r'\operatorname{NLL}=-\frac1n\sum_{i=1}^{n}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right],\qquad \operatorname{Brier}_{+}=\frac1n\sum_i(p_i-y_i)^2.', 'NLL = −(1/n)Σᵢ[yᵢ log pᵢ + (1−yᵢ)log(1−pᵢ)]； Brier₊ = (1/n)Σᵢ(pᵢ−yᵢ)²')
eq(r'\operatorname{ECE}=\sum_{b=1}^{15}\frac{|B_b|}{n}\left|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\right|.', 'ECE = Σ[b=1,…,15] (|Bᵦ|/n) |acc(Bᵦ) − conf(Bᵦ)|')
p('单方案 Accuracy 报告 Wilson 95% 区间。比较同一批 ID 上的两个固定方案时，同时给出修复数、退化数、精确双侧 McNemar 检验，以及已有实验的类别分层配对 bootstrap 区间。这些区间固定已训练模型及校准参数，只反映样本层不确定性，不覆盖训练种子、温度重新估计、多轮选择和重复文本相关性。多数开发实验未作跨全部实验的多重比较校正；历史模型六组比较及后续融合四候选比较分别做过 Holm 校正，其中四候选调整后 p 值均为 1。各实验内部的校正不能视为整个研究已作全族校正。', 'final;heads;robust')

h('实验结果与分析')
h('开发集和完整官方测试',2)
table('固定主方案的开发与官方测试结果',['数据','方案','正确 / 总数','Accuracy','Macro-F1','NLL','ECE'],[
 ['开发','旧 Laya 主融合','2,424 / 2,500','96.960%','0.969599','0.106593','1.292%'],['开发','DeBERTa BF16','2,426 / 2,500','97.040%','0.970400','0.093473','1.326%'],['开发','固定异构融合','2,432 / 2,500','97.280%','0.972800','0.088768','0.916%'],
 ['测试','旧 Laya 主融合','24,086 / 25,000','96.344%','0.963440','0.114159','1.204%'],['测试','DeBERTa FP32','24,233 / 25,000','96.932%','0.969319','0.095947','1.363%'],['测试','固定异构融合','24,257 / 25,000','97.028%','0.970280','0.092618','0.650%']], 'deberta;final')
p('如表 3 和图 4 所示，最终融合的完整官方测试 Accuracy 为 97.028%，Wilson 95% 区间为 [96.810%, 97.231%]；Macro-F1 为 0.970280，AUROC 为 0.994350。旧 Laya 主融合的 AUROC 为 0.992436，DeBERTa 单分支为 0.994449。融合提高了分类正确率及若干概率指标，但测试 AUROC 略低于 DeBERTa，不能据此宣称各项指标均更优。', 'final')
fig('03_accuracy','固定模型的准确率与 Wilson 95% 区间。横轴采用明确标出的局部尺度；开发与测试分开显示。误差线不包含训练或模型选择不确定性，也不用于单独判断配对差异。','deberta;final')
p('开发集上的正则化 logit 探索曾达到 97.40%，相对固定 97.28% 方案只净多正确 3 条，名义 McNemar p=0.25，四候选 Holm 调整后 p=1，且 NLL/ECE 变差，没有对应的完整正式测试结果。TF-IDF 分支融合达到 97.32%，净增 1 条、p=1；NB-ngram 融合为 97.28%。这些结果可说明探索覆盖范围，却不足以替代预先冻结的最终方案。词袋分支借鉴 NB 加权 n-gram 思路 [6]，不等于原论文所有模型的精确复现。', 'alternatives;bow')
h('配对错误变化与收益来源',2)
p('相对旧 Laya 主融合，最终融合修复 265 条错误，同时将原来正确的 94 条改错，净变化为 +171 条；Accuracy 增加 0.684 个百分点，条件配对 bootstrap 95% 区间为 [+0.536,+0.832] 个百分点，精确 McNemar p=5.538×10⁻²⁰。旧方案错误数 914 条，最终为 743 条，相对错误减少率为 171/914≈18.71%。这一比较支持本项目既有系统的实测改进。', 'final')
p('相对 DeBERTa 单分支，最终融合修复 144 条、退化 120 条，净多正确 24 条，即 +0.096 个百分点；从冻结概率独立复算的精确 McNemar p=0.1568，未达到常用的 0.05 阈值。引入 DeBERTa 后相对旧 Laya 的系统改进，与融合是否稳定优于最强单分支，是两个不同问题。前一比较的极小 p 值不能用于后一比较，本次对后者的证据仍较弱。', 'analysiscode;final')
fig('04_errors','左：最终融合的官方测试混淆矩阵，行是真实标签、列是预测标签，每行分母为 12,500。右：相对旧 Laya 的样本级修复、退化和净增量；二者都保留，避免只统计修复。','final')
p('最终混淆矩阵为负类正确 12,098、负类误报正类 402；正类漏报负类 341、正类正确 12,159。两类召回分别为 96.784% 与 97.272%，差异为 0.488 个百分点。类别平衡使 Accuracy 与 Macro-F1 接近，但仍需关注具体方向的错误；该矩阵本身不能解释模型内部为何误判。', 'final')
h('概率校准和置信度',2)
p('完整测试上，最终融合的 NLL 为 0.092618，低于旧 Laya 的 0.114159 和 DeBERTa 的 0.095947；ECE 为 0.650%，也低于两分支的 1.204% 和 1.363%。正类 Brier score 分别为融合 0.024172、DeBERTa 0.025002、旧 Laya 0.029499。图 6 同时展示这三项指标，以免校准判断只依赖一种分箱统计。', 'final')
fig('05_probability','完整官方测试的三项概率指标。条形轴从零开始；NLL、ECE 与 Brier 越低越好。ECE 是有限样本、依赖分箱的描述量，不是单条预测的正确概率保证。','final')
fig('06_reliability','官方测试可靠性曲线。沿用原始 15 等宽置信分箱，只绘制非空箱；低置信箱样本较少，局部折线波动不能直接解释为稳定规律。每箱分母与数值保存在 analysis/reliability_bins.csv。','final')
p('将置信度不低于 0.95 的误分类样本定义为高置信度错误，其比例以全部错误为分母。最终融合为 244/743=32.8%，旧 Laya 为 375/914=41.0%，DeBERTa 为 308/767=40.2%；最终融合错误样本的平均置信度为 0.812。这些结果表明，本次冻结方案缓和了部分过度自信。置信度排序可用于人工复核与选择性输出，但全覆盖分类、固定覆盖排序曲线和按校准阈值部署后的实际覆盖率具有不同分母，不能混报。', 'final;devanalysis')
h('文本长度与误差分层',2)
table('按原 Laya 评论预算分层的官方测试 Accuracy',['分组','n','旧 Laya','DeBERTa','最终融合'],[
 ['≤466 token','21,287','96.345%','96.970%','97.050%'],['>466 token','3,713','96.337%','96.714%','96.903%']], 'final')
p('最终融合在未截断与截断两组均高于旧 Laya，分别净多正确 150 与 21 条。该分层仅按 Laya tokenizer 的 466 token 预算定义；DeBERTa 使用自己的分词器，因此不能把两组直接叫作两个模型都读全或都截断。分层差值是描述性结果，未单独建立长度交互效应的确认检验。', 'final')
fig('07_strata_ablation','左：官方测试长度分层；右：独立的三随机种子 CE / mixed 开发对照，显示全部种子而非只取最好种子。两面板对应不同实验与样本口径。','final;robust')
h('受控消融与未稳定获益的方向',2)
table('主要受控实验与探索结果概览',['实验','观察结果','证据解释'],[
 ['决策头：3 seeds','原始 96.520%；线性 96.493%','均值差 +0.027 pp；方向不一致'],
 ['鲁棒损失：3 seeds','CE 96.32%；mixed 96.40%','均值差 +0.08 pp；区间跨零'],
 ['512 → 2048','全体 96.32% → 96.32%','长组 +6，短组 −6'],
 ['训练标签修订 v2','96.32% → 96.40%','净 +2；p=0.883'],
 ['校准拟合外层权重','97.28% → 97.16%','NLL 改善，Accuracy 降低'],
 ['正则化 logit 探索','开发 97.40%','净 +3；p=0.25；无正式测试']], 'heads;robust;context;reviewed;weight;alternatives')
p('决策头对照使用相同的预训练编码器、训练输入和预算，比较重新初始化的原始头与线性头。seed 42、43、44 下，原始头减线性头的 Accuracy 差分别为 −0.20、+0.20、+0.08 个百分点，均值 +0.0267，条件 95% 区间为 [−0.2533,+0.3067]。未检出稳定优势，并不等于已证明两者等效。线性头的平均训练进程耗时少 6.81%（包含两轮训练及保存）、峰值 allocated 显存少 4.94%、可训练参数少 6.23%；本轮未测得推理吞吐收益。在这组端到端头部对照（项目实验编号 L2）中，12 份逐轮权重尚未回传，统计证据与完整模型交付状态需分别说明。', 'heads')
p('鲁棒损失使用 0.5 CE + 0.5 GCE(q=0.7)，其中 GCE 为广义交叉熵（generalized cross-entropy）。三个种子的 Accuracy 差为 +0.16、0.00、+0.08 个百分点，平均 +0.08，条件区间为 [−0.174,+0.347]；截断组平均下降 0.328 个百分点。效应较小且区间跨零，尚不足以支持升级默认配方。本文保留全部种子和负向分层结果，不据此推断鲁棒损失普遍无效。', 'robust')
p('长度对照采用固定 seed 42、两轮和共同样本曝光预算，512 与 2048 的全体准确率均为 96.32%；2048 修复 23 条并退化 23 条。截断组净增 6 条、未截断组净减 6 条，说明上下文增加在该配置下产生收益与代价抵消。这里的同预算指训练样本与轮数，不表示 token 运算量、显存或实际训练耗时相同。', 'context')
fig('08_controls','左：共同预算下原始头与线性头的全部种子成绩，纵轴为局部尺度。右：独立长度对照中 2048 相对 512 的净正确数变化；全体净变化为零。','heads;context')
p('训练数据修订单独保存派生版本：94 条复核候选最终包含 61 条翻标、3 条删除，原始标签及划分保留。按原开发标签评估，准确率从 96.32% 到 96.40%，p=0.883。复核可能提高局部语义一致性，但现有实验不足以证明其带来稳定泛化收益。本文没有因开发或测试案例而改写最终评分标签。', 'reviewed')
h('跨后端数值核验与工程成本',2)
p('相同类别预测仍可能对应不同概率，概率融合因此需要核对分支输出尺度。历史 MLX FP16 候选未通过既定概率门槛；后续独立候选改为全 FP32，并在启动时关闭降低精度的矩阵运算路径。在 400 条校准和 2,500 条开发、共 2,900 条输入上，相对 CUDA 缓存的最大 logits 差约为 0.0003574，最大概率差约为 0.00006253，类别差异为 0，均通过原定 logits≤0.1、概率≤0.01 和类别一致门槛。完整差值保存在 paper/analysis/plot_data.json；通过门槛不意味着逐位等价，也不构成新的 Accuracy 增益。', 'mlx')
p('同一历史 MLX 精确 FP32 benchmark 采用 128 条校准样本、batch 8、两轮预热与五轮测量，计入编码、传输、前向及融合，不计模型加载。单 Laya 吞吐为 17.030 条/秒，两个 Laya 融合为 7.531 条/秒；峰值 MLX 活跃分配分别为 3.37 与 4.94 GiB。该测量不包含 DeBERTa，不能据此报告最终异构系统的端到端吞吐；MLX 活跃分配也不等于进程总内存。', 'mlx')
h('选择性输出与错误案例',2)
p('拒答阈值在原 400 条校准集上确定，并于本轮测试前封存。对最终融合，校准目标覆盖 90% 对应阈值 0.9713223，实际测试保留 22,233 条，占 88.932%；其中正确 22,062 条，选择性 Accuracy 为 99.231%。提高选择性准确率的代价是拒绝 2,767 条，因此该数字不能代替全覆盖 97.028%，也不附加有限样本总体风险保证。', 'final')
fig('09_selective_cost','左：以校准集冻结阈值获得的测试实际覆盖率与保留样本准确率；不是在测试上选择阈值。右：独立历史 MLX 精确 FP32 测量，仅比较单 Laya 与两个 Laya 融合，不能解释为最终三 checkpoint 系统的成本。','final;mlx')
table('开发集有目的选取的错误案例：原标签保持不变',['样本 ID（train/ 后）','原标签','融合 p(pos)','定性解释'],[
 ['neg/10566_3.txt','负','0.6833','系列好评干扰当前电影差评'],['neg/11233_2.txt','负','0.9834','廉价锚定中的挖苦与有限认可'],
 ['pos/12230_9.txt','正','0.0580','制作差但观看体验愉快'],['pos/3160_7.txt','正','0.0054','正文自评分与标签冲突候选']], 'cases')
p('以上案例来自原 Laya 主融合、2048 视角和 DeBERTa 三条明确流水线共同错误的 47 条开发样本集合，并非最终融合全部 68 条开发错误的同义集合。47 条中 43 条在 Laya 与 DeBERTa 两种分词器下都未截断，因而不能把所有共错归因于上下文不足。模型一致也不等于标签错误：评价对象、讽刺、观看乐趣与制作质量的分离均可能导致真实语义误判。', 'cases;devanalysis')
p('案例经过有目的选择，用于解释可能的失败方式，没有第二位标注者盲审，也没有从它们估计总体错标率。所谓正文—标签冲突只标记待核实的数据问题，不作为修改开发标签、扣除错误或推导模型能力上限的依据。', 'cases')

h('讨论、局限与可复现性')
h('有效操作与创新边界',2)
p('本项目最直接的有效性证据，是完整测试上相对旧 Laya 主融合的配对改进与概率质量提升：错误数、NLL 和 ECE 同时下降，且全体样本均参与评分。开发与测试都观察到异构融合的准确率增益，但相对最强单分支的测试增益未达显著。原始类型化头、DeBERTa、温度缩放和概率平均均为已有技术，本项目的贡献限于 IMDB 组合方案、实验设计与工程验证。', 'final;deberta;heads')
p('对外部高分结果，应先检查分区及训练信息是否一致。本项目对一份 BERT+BiLSTM 公开代码的静态审计发现，它合并官方 train/test 后重新划分，故不满足本项目固定官方 25k/25k 的直接比较条件；但缺少绑定论文报告分数的精确运行产物，不能由此断言论文数字无效，也未证实单时间步退化或电影级泄漏。本文不制作混合不同协议的“领先排行榜”，不宣称 SOTA。', 'literature')
h('结果的适用范围',2)
p('官方测试曾用于历史研究，本轮冻结规则只能约束本轮直接调参，无法消除历史反馈。开发集和校准集反复使用也带来选择偏差，最终候选并非从全新的独立开发集产生。此外，最终融合缺少多种子确认，头部与损失对照的三种子结果不能代替最终三 checkpoint 系统的训练方差估计。官方数据还存在重复文本和标签争议，当前区间没有建模这些相关性。', 'final;heads;robust;audit')
p('DeBERTa 开发推理采用 CUDA BF16，测试采用 CPU FP32；跨后端准入不保证逐位等价，因此不能将开发与测试的性能差全部解释为分布效应。系统增加了模型数量与推理成本，但尚无包含 DeBERTa 的统一端到端 benchmark。结果也只覆盖二元、强极性的英语电影评论，不能直接推广到中性情绪、多标签评论或其他领域。完整测试距 98% 研究目标仍差 243 条正确预测，该目标尚未达到。', 'final;deberta;mlx')
h('复现材料与报告生成',2)
p('数据准备与核验入口为 scripts/prepare_data.py、scripts/verify_data.py；Laya 训练入口为 scripts/train_laya.py，DeBERTa 固定预算训练由 scripts/deberta_budget_training.py 及对应 study 组织；最终完整测试入口为 scripts/deberta_final_evaluation.py。模型 revision、资产 SHA256、原始数据摘要、配置及环境锁分别位于 configs、reports 和 environments。历史运行产物不覆盖，重跑使用新 run_id。', 'traincode;debertaconfig;final')
p('本报告图表由 paper/build_figures.py 从冻结摘要和逐样本预测生成，独立核对正确数、混淆矩阵及 NLL；数据表与输入 SHA256 存在 paper/analysis。TeX 与 Word 由同一内容源导出，TeX 支持 XeLaTeX 编译。Scientific Writing 与 Scientific Visualization 技能用于证据整理、区间解释及图表审查，相关软件方法来源列于文献 [7]；文档与代码辅助生成不替代作者对事实和最终提交的责任。', 'analysiscode')
p('当前通用 scripts/predict.py 仍指向历史单模型。其 95.76% 是一轮 Laya-CE 在 2,500 条开发样本上的成绩，既非官方测试分数，也非最终融合成绩。课程另要求源码压缩包及各不超过 5 分钟的训练、测试视频，须独立验收；现有材料尚不能作为已完成的视频交付。历史源码包早于本次报告定稿，正式提交前仍需随完整交付重新打包。', 'delivery;comparison')


h('未完成工作与交付边界',2)
p('已确认路线中的 ModernBERT-base / Laya-RLCD 公平迁移尚未执行，也未完成最高准确率版与高效单模型版的两版交付及统一默认预测入口。当前报告对应已有 Laya-CE 与 DeBERTa 的研究结果，不能代替这些计划项。后续训练与部署整合尚未执行，因而也没有两版系统在同一平台上的端到端吞吐、延迟、峰值总内存或可比训练费用；本文仅报告已经保存的局部成本证据。现有记录不足以进一步归因于某项资源限制。', 'delivery;heads;mlx')
h('已保留的逐轮训练记录',2)
p('图 11 展示历史 combined-s43 两轮训练中保存的平均训练损失与原始开发准确率。该运行采用 seed 43，不是最终融合所用的 combined-s42；开发指标取自最终温度校准前的逐轮 checkpoint，不能替代最终方案成绩。图中仅有两个逐轮观测点，连线用于识别顺序，不据此推断逐步收敛过程，也不将训练 CE 与校准后的 NLL 混为一项损失。', 'curve')
fig('10_training','历史 combined-s43 的逐轮训练损失与开发准确率。每轮开发分母为 2,500，指标未经最终温度校准；此运行不是最终融合成员。仅绘制保存的两轮观测，不补造训练步骤。','curve')

h('结论')
p('本文构建并评估了多视图 Laya 与 DeBERTa-v3 的 IMDB 电影评论分类方案。开发期预声明、测试前冻结的异构等权融合，在完整 25,000 条官方测试上获得 97.028% Accuracy、0.970280 Macro-F1、0.092618 NLL 和 0.650% ECE，相对旧 Laya 主融合净多分类正确 171 条。现有证据支持它改善本项目的既有系统，但相对最强 DeBERTa 单分支的净增益未达显著。复杂头、损失和长度对照采用不同实验口径，不能将这些结果合并为“异构融合优于增加模型内部复杂度”的受控因果结论。', 'final')
p('受控实验中，原始复杂头未显示稳定准确率优势，鲁棒损失平均收益较小，长度增至 2048 token 后全体准确率持平。跨后端核验只说明输出通过给定数值门槛。后续仍需新独立评估数据、多种子确认及统一平台的完整系统成本测量。97.40% 开发探索成绩、拒答后的高准确率和未实施的训练方向均不能替代这些证据。', 'heads;robust;context;mlx;alternatives')

REFERENCES=[
 ('maas','Maas A L, Daly R E, Pham P T, Huang D, Ng A Y, Potts C. Learning Word Vectors for Sentiment Analysis. ACL, 2011: 142–150.','https://aclanthology.org/P11-1015/'),
 ('modernbert','Warner B, Chaffin A, Clavié B, et al. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. arXiv:2412.13663, 2024.','https://arxiv.org/abs/2412.13663'),
 ('laya','Convai Innovations. Laya（英语模型）[模型]. Hugging Face. 访问日期：2026-10-19.','https://huggingface.co/convaiinnovations/laya'),
 ('deberta','He P, Gao J, Chen W. DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. ICLR, 2023; arXiv:2111.09543.','https://arxiv.org/abs/2111.09543'),
 ('guo','Guo C, Pleiss G, Sun Y, Weinberger K Q. On Calibration of Modern Neural Networks. ICML, PMLR 70, 2017: 1321–1330.','https://proceedings.mlr.press/v70/guo17a.html'),
 ('wang','Wang S, Manning C. Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. ACL (Volume 2: Short Papers), 2012: 90–94.','https://aclanthology.org/P12-2018/'),
 ('skills','Kassis T, Agarwal V, He Y, Patel D, Brueckner A M. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065v2, 2026.','https://doi.org/10.48550/arXiv.2609.00065'),
]

SOURCES={
 'architecture':'paper/build_architecture.py',
 'archfacts':'paper/analysis/architecture_sources.json',
 'debertalock':'configs/deberta_model_lock.json',
 'curve':'reports/laya-rtx4090-ce-combined-s43-epochs2-fp32-v1/epoch_validation_curve.json',
 'audit':'reports/data_audit.json', 'final':'reports/deberta-laya-final-cpu-evaluation-20261018-v1/summary.json',
 'oldtest':'reports/laya-frozen-official-test-20261014-v1/test-actual.json',
 'deberta':'reports/deberta-heterogeneous-bf16-budget7-s42-20261017-v1/evaluation/summary.json',
 'fusion':'reports/laya-probability-fusion-s43-s42-20261014-v1/REPORT.md',
 'heads':'reports/laya-head-control-analysis-20261015-v1/REPORT.md',
 'robust':'reports/laya-robust-loss-analysis-20261016-v1/REPORT.md',
 'context':'reports/laya-context-pair-s42-modal-20261016-v1/evaluation/REPORT.md',
 'mlx':'reports/laya-fusion-inference-mlx-fp32-precise-20261014-v1/REPORT.md',
 'reviewed':'reports/laya-reviewed-data-s42-modal-20261017-v1/evaluation/REPORT.md',
 'comparison':'reports/model-comparison/REPORT.md','weight':'reports/laya-deberta-calibration-weight-cpu-20261018-v1/REPORT.md',
 'alternatives':'reports/final-cpu-alternatives-20261018-v2/REPORT.md','bow':'reports/bow-nb-complementarity-cpu-20261018-v1/REPORT.md',
 'cases':'reports/deberta-laya-final-analysis-20261018-v1/CASE-STUDY.md','devanalysis':'reports/deberta-laya-final-analysis-20261018-v1/REPORT.md',
 'literature':'reports/bert-bilstm-literature-audit-20261018-v1/REPORT.md',
 'delivery':'docs/closeout/delivery.md', 'locks':'configs/laya_model_lock.json','traincode':'scripts/train_laya.py',
 'analysiscode':'paper/build_figures.py','debertacode':'scripts/deberta_inputs.py',
 'regularization':'reports/rtx4090-regularization-completed-v1/REPORT.md',
 'baseconfig':'configs/laya-rtx4090-ce-baseline-s43-epochs2-fp32.json',
 'combinedconfig':'configs/laya-rtx4090-ce-epochs2-warmup005-ls005-fp32.json',
 'debertaconfig':'configs/deberta-heterogeneous-bf16-budget7-s42-20261017-v1.json',
}
