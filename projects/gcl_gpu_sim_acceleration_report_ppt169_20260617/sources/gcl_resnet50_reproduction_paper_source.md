# gcl-resnet50-reproduction-paper

基于图神经网络与 K-means 的 ResNet-50 GPU 执行轨迹复现方法研究

## 作者：复现实验小组

## 单位：GPU 执行轨迹压缩与模拟复现实验项目组

日期：2026 年 6 月 15 日

摘  要：GPU 程序的完整执行轨迹包含大量 kernel 调用、线程块、warp、指令和访存信息，直接对完整轨迹进行分析和模拟会带来较高的计算成本。为降低后续分析负担，本文复现了一种基于图神经网络和聚类分析的 GPU kernel 表示方法。整体思路是：首先将 ResNet-50 的真实 NVBit 执行轨迹转换为带类型边的图结构；然后使用关系图卷积网络（RGCN）对每个 kernel 图进行编码，得到固定维度的 kernel embedding；最后使用 K-means 对 embedding 进行无监督聚类，并从每个簇中选择靠近聚类中心的代表 kernel。实验使用真实 ResNet-50 full trace，共包含 265 个 kernel invocation 和 124876 条 CTA 记录。RGCN 导出 265 个 256 维 kernel embedding。K-means 通过轮廓系数选择 K=2，得到 263/2 的聚类分布，silhouette 为 0.481866，簇间/簇内距离比为 2.016339。实验说明该流程能够完成从真实 trace 到 kernel 聚类的端到端复现，并在 embedding 空间中观察到初步分离现象；但当前结果仍属于初步实验，尚需通过更充分训练、对照实验和下游模拟误差验证进一步证明聚类语义的正确性。

关键词：GPU 执行轨迹；图神经网络；RGCN；K-means；ResNet-50；kernel 聚类

中图分类号：TP391    文献标识码：A    DOI：待定

A Reproduction Study of ResNet-50 GPU Trace Modeling Based on Graph Neural Networks and K-means

Abstract: Complete GPU execution traces contain large numbers of kernel invocations, thread blocks, warps, instructions and memory references. Directly analyzing or simulating all trace records is expensive. This paper reproduces a graph-based kernel representation and clustering pipeline. The overall idea is to convert a real ResNet-50 NVBit trace into typed execution graphs, encode each kernel graph using a relational graph convolutional network, and then cluster the generated kernel embeddings with K-means. The reproduced full trace contains 265 kernel invocations and 124876 CTA records. The RGCN exports 265 kernel embeddings with 256 dimensions. Silhouette-based K selection chooses K=2, producing a 263/2 cluster distribution. The silhouette score is 0.481866, and the inter/intra distance ratio is 2.016339. The experiment completes the end-to-end reproduction from real trace to kernel clustering and observes an initial separation signal in the embedding space.

Key words: GPU trace; graph neural network; RGCN; K-means; ResNet-50; kernel clustering

## 1 引言

GPU 已经成为深度学习模型训练与推理的重要计算平台。为了研究 GPU 程序的性能瓶颈和模拟器行为，研究者通常需要采集细粒度执行轨迹。完整执行轨迹能够记录 kernel 调用、线程块调度、warp 执行、指令序列以及访存行为，因此具有较高分析价值。然而，完整 trace 规模较大，直接对所有 trace 记录进行处理会增加存储、训练和仿真的开销。

针对这一问题，一个自然思路是从完整轨迹中学习 kernel 的结构化表示，并根据表示结果选择少量具有代表性的 kernel。这样既能保留不同 kernel 之间的执行差异，又能为后续采样式模拟和性能分析减少输入规模。与只使用手工统计特征的方法相比，图神经网络能够直接处理非规则结构数据，把控制流和数据依赖关系纳入表示学习过程。

本文围绕 ResNet-50 GPU 执行轨迹复现任务，构建了“trace 图建模—RGCN 表示学习—K-means 聚类—实验评价”的完整流程。论文首先介绍整体实现思路，然后说明 GNN 模型结构和 K-means 聚类结构，最后给出我们实际完成的实验和结果分析。

## 2 整体实现思路

本文的整体目标是把真实 GPU 执行轨迹转化为可以学习和聚类的 kernel 表示。具体来说，我们不直接对 trace 中的每条记录进行人工判断，而是先把 trace 转换为图，再用 GNN 自动学习每个 kernel 的向量表示，最后用 K-means 在向量空间中发现相似 kernel 组。

整体流程分为六个步骤。第一步，采集 ResNet-50 推理过程中的真实 NVBit trace，并保留 kernel invocation、CTA、SM、warp 和动态执行记录。第二步，对原始 trace 做 adapter 规范化，建立稳定的 kernel invocation 标识和线程块层级信息。第三步，将每个 kernel 的 selected-SM 轨迹转换为 typed canonical graph。第四步，把图结构张量化，形成节点特征矩阵、边索引和边类型。第五步，使用 RGCN 编码每个 kernel 图，得到固定维度 embedding。第六步，用 K-means 对 embedding 进行聚类，并选择每个簇的代表 kernel。

该实现思路的关键在于：trace 的原始形式是线性或层级记录，而 GNN 需要图结构输入；kernel 之间的相似性不直接由 kernel 名称决定，而由执行结构、访存关系和聚合后的 embedding 共同决定；K-means 不参与 GNN 训练，而是在 GNN 输出空间上进行后处理，用于得到可解释的 kernel 分组和代表样本。

表 1 整体实现流程

阶段

输入

处理内容

输出

Trace 采集

ResNet-50 推理过程

记录 kernel、CTA、SM、warp 和动态执行信息

full trace artifacts

Trace 规范化

原始 NVBit trace

建立稳定 kernel invocation 标识和线程块层级信息

adapter bundle

图构建

规范化 trace

构造包含控制流和访存数据关系的 typed canonical graph

canonical graph bundle

图张量化

canonical graph

生成节点特征、边索引、边类型和层级索引

graph tensor bundle

GNN 编码

graph tensor

使用 RGCN 学习 kernel 级 embedding

kernel embedding table

K-means 聚类

kernel embedding

选择 K、聚类并选取中心附近代表 kernel

selector artifacts

## 实验分析

聚类结果

计算 silhouette、DBI、CH 和代表样本距离

## 实验结论

## 3 GNN 模型结构

本文使用的 GNN 是关系图卷积网络（Relational Graph Convolutional Network，RGCN）。选择 RGCN 的原因是 GPU 执行轨迹图中的边并不是单一关系：指令之间存在控制流关系，访存指令与内存引用之间存在数据来源和数据目的关系。如果使用普通 GCN，这些边会被视为同一种连接，难以区分不同关系对节点表示的影响；RGCN 则可以为不同边类型设置独立的关系变换，更适合本文场景。

在图构建中，节点主要表示 kernel 执行中的指令节点和访存相关 pseudo node。边类型包括三类：control_flow 表示指令执行顺序或控制流连接；data_source 表示数据来源关系；data_destination 表示数据写入或使用去向关系。这样构建后，一个 kernel 不再只是一个名称或统计向量，而是一个包含执行结构和访存关系的 typed graph。

RGCN 的消息传递过程可以概括为：每一层从不同关系类型的邻居节点收集信息，经过关系特定的线性变换后进行聚合，再与节点自身表示结合得到新的节点向量。经过多层传播后，节点表示包含了局部多跳结构信息。由于最终任务是比较 kernel，而不是只比较单个节点，模型还需要把节点级表示逐级读出为 kernel 级表示。

## 3.1 RGCN 参数与层级读出

本次复现中的 RGCN 为 3 层结构，节点输入特征维度为 64，隐藏层维度为 128，最终 kernel embedding 维度为 256。边关系数量为 3，对应 control_flow、data_source 和 data_destination 三类关系。

模型输出不是单个节点 embedding，而是整个 kernel 的 embedding。为此，本文使用层级 readout：先把节点表示聚合到 warp，再从 warp 聚合到 CTA，再聚合到 selected SM，最后得到 kernel 级表示。该过程与 GPU 执行层级相对应，避免直接把所有节点简单平均造成结构信息丢失。

训练目标采用图对比学习中的 InfoNCE 损失。训练时对图进行轻量增强，例如节点丢弃、边丢弃和特征噪声，使同一 kernel 的不同增强视图在 embedding 空间中更接近，不同 graph 的表示相互区分。当前训练使用 Adam 优化器，学习率为 0.005，temperature 为 0.2。

## 表 2 RGCN 模型结构参数

## 结构项

取值

说明

GNN 类型

RGCN

区分不同类型边的图卷积网络

层数

3

消息传递层数

输入维度

64

节点初始特征维度

隐藏维度

128

关系卷积隐藏表示维度

输出维度

256

最终 kernel embedding 维度

边关系

control_flow / data_source / data_destination

控制流、数据来源、数据目的三类关系

Readout

node_to_warp_to_cta_to_selected_sm_to_kernel

从节点逐级聚合到 kernel 表示

训练目标

InfoNCE

图对比学习损失

## 4 K-means 聚类结构

K-means 的输入是 RGCN 输出的 kernel embedding。设共有 n 个 kernel，每个 kernel 对应一个 d 维向量 zi，则全部样本可表示为 Z={z1,z2,...,zn}。K-means 希望将样本划分为 K 个簇 C1,C2,...,CK，并使每个样本到所属簇中心的平方距离尽可能小。其目标函数可以写为：J=sum_k sum_{zi in Ck} ||zi - μk||^2，其中 μk 表示第 k 个簇的中心。

算法执行时先初始化 K 个簇中心，然后重复两个步骤：第一，根据欧氏距离把每个样本分配到最近的簇中心；第二，根据每个簇中样本的均值更新簇中心。当前后两次迭代的簇分配不再变化，或簇中心移动小于阈值时，算法收敛。K-means 结构简单、计算效率高，适合在 embedding 空间中做初步 kernel 分组。

由于 K-means 需要预先指定 K，本文使用轮廓系数选择聚类数。对于某个样本，轮廓系数同时考虑它与同簇样本的平均距离和与最近其他簇样本的平均距离。平均轮廓系数越大，说明簇内越紧密、簇间越分离。确定 K 后，本文从每个簇中选择距离簇中心最近的样本作为代表 kernel，用于表示该簇的典型执行结构。

表 3 K-means 候选 K 的轮廓系数

候选 K

轮廓系数

2

0.53412531

3

0.32226622

4

0.4068767

5

0.42345686

6

0.43066267

## 5 实验设计

实验对象为 torchvision ResNet-50 模型的一次推理过程。实验输入要求来自真实 NVBit trace，而不是人工构造 trace、synthetic fixture 或手工选择的局部 kernel 序列。这样可以保证复现流程面对的是实际深度学习 workload 中的 kernel 调用和线程块执行结构。

实验流程包括四部分。首先，读取 ResNet-50 full trace，并通过 adapter 得到规范化执行记录。其次，基于 selected SM 的执行记录构造 canonical graph，并完成图张量化。然后，使用 RGCN 对 kernel 图进行编码，导出 265 个 kernel embedding。最后，在 embedding 上运行 K-means，并用 silhouette、Davies-Bouldin、Calinski-Harabasz、簇间/簇内距离比等指标分析聚类效果。

需要说明的是，当前 RGCN 训练仍是初步复现实验。manifest 显示 train_graph_count 为 4，optimizer_step_count 为 1，因此本文不把实验结果解释为最终收敛模型，而是解释为一次端到端复现和方法可行性验证。

## 表 4 实验设置与主要结果

项目

结果

输入 workload

torchvision ResNet-50 inference full trace

kernel invocation 数量

265

CTA record 数量

124876

导出 embedding 数量

265

embedding 维度

256

selected K

2

selected score

0.53412531

cluster 分布

cluster 0: 263；cluster 1: 2

silhouette

0.48186617

Davies-Bouldin

0.78974237

Calinski-Harabasz

10.30193812

inter/intra distance ratio

2.01633922

## 6 实验结果与分析

从输入规模看，本次实验成功处理 265 个 kernel invocation 和 124876 条 CTA 记录，说明从真实 trace 到图表示的前端处理流程已经跑通。RGCN 最终导出 265 个 kernel embedding，每个 embedding 维度为 256。这表明每个 kernel invocation 都可以被映射为一个固定维度向量，从而满足后续聚类分析的输入要求。

从 K-means 结果看，轮廓系数选择的最佳聚类数为 K=2，对应得分为 0.53412531。聚类结果中 cluster 0 包含 263 个样本，cluster 1 包含 2 个样本，样本分布明显不均衡。这说明当前 embedding 空间确实存在可分结构，但更像是将少量特殊 kernel 与大多数常规 kernel 分开，而不是形成多个均衡的功能语义类别。

从聚类指标看，silhouette 为 0.48186617，Davies-Bouldin 指数为 0.78974237，Calinski-Harabasz 指数为 10.30193812，簇间/簇内平均距离比为 2.01633922。这些指标表明 embedding 空间不是完全随机或完全退化的，簇间距离大于簇内距离，具有初步聚类信号。

从代表样本看，cluster 0 的代表记录为 gcl_embedding:0066，cluster 1 的代表记录为 gcl_embedding:0002。代表样本选择基于到聚类中心的距离，因此能够反映当前 embedding 空间中的中心样本。但由于当前还没有完整的 simulator/measured metric 对照，metric_claim_status 为 unavailable，因此不能进一步证明代表样本能够准确代表同簇 kernel 的运行时间或仿真行为。

表 5 聚类分布

簇编号

样本数

权重占比

结果解释

0

263

0.91765889

表示当前 embedding 空间中的分组结构，不直接等同于最终语义类别。

1

2

0.08234111

表示当前 embedding 空间中的分组结构，不直接等同于最终语义类别。

表 6 代表样本距离统计

簇编号

代表记录

成员数

平均距离

P95 距离

最大距离

0

gcl_embedding:0066

263

0.761327

1.293181

1.783367

1

gcl_embedding:0002

2

0.667733

1.335465

1.335465

## 7 局限性与后续工作

当前实验仍存在三方面不足。第一，GNN 训练规模较小，只能证明端到端路径可运行，不能证明模型已经充分学习到稳定语义表示。后续应扩大训练图数量，增加训练轮数，并记录 loss curve 和 embedding 指标随训练过程的变化。

第二，当前缺少 baseline ablation。后续需要与随机 embedding、opcode 直方图、无边关系 pooling、control-flow-only RGCN 和 data-flow-only RGCN 等方法比较，判断聚类信号究竟来自图结构消息传递，还是来自简单规模特征。

第三，当前缺少下游模拟误差验证。代表 kernel 的最终价值在于能否近似同簇 kernel 的 simulator 指标或 measured metric。后续应补充 runtime、SM cycles、memory throughput 等指标，计算 cluster 级误差和 sampled-vs-full 仿真加速比。

## 8 结论

本文按照论文格式总结了 ResNet-50 GPU 执行轨迹的 GCL 复现过程。整体实现思想是先把真实 trace 转换为 typed graph，再使用 RGCN 学习 kernel embedding，最后使用 K-means 对 kernel embedding 进行聚类并选择代表样本。

实验结果表明，当前流程已经能够处理真实 ResNet-50 full trace，输入规模为 265 个 kernel invocation 和 124876 条 CTA 记录，并导出 265 个 256 维 kernel embedding。K-means 选择 K=2，得到 263/2 的聚类分布，silhouette 为 0.481866，说明 embedding 空间中存在初步分离现象。

总体来看，本文完成了从真实 GPU trace 到 GNN 表示学习和 K-means 聚类分析的端到端复现。当前结果可以作为后续 GCL 采样模拟研究的工程基础，但要进一步证明聚类语义和代表样本有效性，还需要更充分的训练、对照实验和下游性能误差验证。

## 参考文献

[1] Kipf T N, Welling M. Semi-Supervised Classification with Graph Convolutional Networks[C]//International Conference on Learning Representations, 2017.

[2] Schlichtkrull M, Kipf T N, Bloem P, et al. Modeling Relational Data with Graph Convolutional Networks[C]//European Semantic Web Conference, 2018.

[3] Veličković P, Cucurull G, Casanova A, et al. Graph Attention Networks[C]//International Conference on Learning Representations, 2018.

[4] Chen T, Kornblith S, Norouzi M, Hinton G. A Simple Framework for Contrastive Learning of Visual Representations[C]//International Conference on Machine Learning, 2020.

[5] Lloyd S. Least Squares Quantization in PCM[J]. IEEE Transactions on Information Theory, 1982, 28(2):129-137.

[6] Arthur D, Vassilvitskii S. k-means++: The Advantages of Careful Seeding[C]//ACM-SIAM Symposium on Discrete Algorithms, 2007.

[7] Rousseeuw P J. Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis[J]. Journal of Computational and Applied Mathematics, 1987, 20:53-65.

[8] NVIDIA. NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs[EB/OL].

[9] GCL ResNet-50 GNN Acceptance Report. docs/reports/gcl-resnet50-gnn-acceptance-report-2026-06-11.md.

[10] A 线 GCL ResNet-50 GNN 有效性当前状态报告. docs/superpowers/specs/2026-06-13-a-line-gcl-resnet50-gnn-effectiveness-current-status.md.
