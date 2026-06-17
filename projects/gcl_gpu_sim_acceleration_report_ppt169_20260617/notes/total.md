# Slide 01

各位老师、同学大家好，我们这次汇报的题目是“基于图对比学习的 GPU 仿真加速技术”。这项工作的核心问题是：当一个深度学习模型在 GPU 上执行时，会产生大量 kernel 调用，如果全部逐个进行仿真和分析，成本会非常高。我们的思路是先把 kernel 的执行结构表示出来，再用图神经网络和聚类方法找出相似 kernel，最后只选择少量代表 kernel 进行后续分析。这样就可以在尽量保留结构差异的前提下，减少需要处理的模拟对象数量。

# Slide 02

首先看研究动机。GPU kernel 是 GPU 程序执行和性能分析的基本单元。一个模型推理过程里可能会产生很多 kernel invocation，每个 kernel 又包含 CTA、warp、指令和访存行为。完整 trace 虽然信息最全，但直接分析全部 trace 会带来很高的存储和仿真成本。因此，我们希望用代表 kernel 替代全量 kernel。以本次 ResNet-50 trace 为例，原始输入有 265 个 kernel invocation，我们的目标就是把这些 kernel 变成少量结构上有代表性的样本，从而降低后续模拟规模。

# Slide 03

这一页展示整体流程。第一步是采集 ResNet-50 的真实 GPU 执行 trace；第二步是把每个 kernel 的执行轨迹构造成图；第三步是使用 RGCN 对图结构进行编码，得到固定维度的 kernel embedding；第四步通过图对比学习训练编码器，让结构相近的 kernel 在向量空间里更接近；第五步使用 K-means 聚类；最后从每个簇中选出离中心最近的代表 kernel。可以看到，这条链路把原始 trace 转换成了可以聚类和压缩的表示空间。

# Slide 04

接下来介绍方法部分。我们首先把 kernel 执行轨迹转化为异构关系图。图中的节点可以表示指令节点，也可以表示访存相关的 pseudo node。边则表示不同类型的结构关系，包括 control flow、data source 和 data destination。相比只统计指令数量、访存比例这类人工特征，图表示能够保留指令之间的依赖关系、控制关系和访存关系。因此，一个 kernel 不再只是一个名字或一组统计数值，而是一个包含执行结构的图对象。

# Slide 05

有了图之后，我们使用 RGCN，也就是关系图卷积网络，作为图编码器。选择 RGCN 的原因是我们的图中有多种边类型，普通 GCN 会把所有边当成同一种连接，而 RGCN 可以针对不同关系使用不同的变换参数。本次复现中，模型是三层 RGCN，输入维度是 64，隐藏维度是 128，最终输出 256 维 kernel embedding。也就是说，每个 kernel 最后都会被编码成一个 256 维向量，用于后面的聚类分析。

# Slide 06

这里进一步说明 readout 过程。RGCN 最开始得到的是节点级表示，但我们的目标是比较整个 kernel，所以需要把节点表示逐级聚合成 kernel 表示。这个过程按照 GPU 执行层级进行：先从 node 聚合到 warp，再到 CTA，再到 selected SM，最后得到 kernel embedding。这样做比简单地把所有节点平均起来更合理，因为它保留了 GPU 执行结构中的层次信息，也让最终 embedding 更贴近 kernel 的真实执行组织方式。

# Slide 07

这一页是图对比学习。由于我们没有人工标注的 kernel 类别，所以不能直接做监督分类。GCL 的做法是：对同一个 kernel 图构造两个增强视图，这两个视图应该在 embedding 空间中接近；而不同 kernel 的视图则作为负样本，应该相对远离。这里使用的目标是 InfoNCE 损失。直观理解就是，模型在训练过程中学习“哪些图结构像同一个 kernel，哪些图结构差别更大”。这样即使没有标签，也能学习出具有结构区分能力的 embedding。

# Slide 08

得到 kernel embedding 后，我们进入 K-means 聚类和代表样本选择。输入是 265 个 kernel，每个 kernel 对应一个 256 维向量。K-means 会根据向量空间中的距离把这些 kernel 分成若干簇。K 的选择不是手工拍脑袋决定的，而是通过轮廓系数比较不同 K 值后得到。确定 K 之后，每个簇中距离聚类中心最近的样本会被选为代表 kernel。这个代表 kernel 可以看作当前簇中最典型的结构样本。

# Slide 09

这一页汇总实验设置和主要结果。我们的输入 workload 是 torchvision ResNet-50 的一次推理 full trace，共包含 265 个 kernel invocation 和 124876 条 CTA 记录。RGCN 最终导出了 265 个 256 维 embedding。K-means 根据轮廓系数选择 K=2，silhouette 指标为 0.481866，簇间和簇内距离比为 2.016339。这些指标说明 embedding 空间中确实出现了初步分离现象，也说明从真实 trace 到 kernel 聚类的端到端流程已经跑通。

# Slide 10

这里要解释聚类结果。最终两个簇的规模并不均衡：cluster 0 有 263 个样本，cluster 1 有 2 个样本。因此，这个结果更适合解释为“主体 kernel 群组加少量结构特殊 kernel”的分离，而不是直接说得到了两个明确的功能类别。右侧的示意图表达的就是这个含义：大多数 kernel 在当前 embedding 空间中比较接近，少数 kernel 明显偏离主体。这说明 GCL 学到了一定的结构差异，也能帮助我们发现特殊 kernel。

# Slide 11

这一页给出课程报告的结论。通过这套 GCL + RGCN + K-means 方法，我们把 ResNet-50 full trace 中的 265 个 kernel invocation 压缩成 2 个代表 kernel。按模拟对象数量来估算，理论模拟加速约为 265 除以 2，也就是 132.5 倍。这里的含义是：如果原来需要逐个模拟 265 个 kernel，现在只需要围绕 2 个代表 kernel 做分析，模拟对象规模被显著降低。这正是这套方法用于 GPU 仿真加速的价值。

# Slide 12

最后做一个总结。GNN 负责把 kernel 的执行轨迹编码成图结构表示，GCL 负责在无标签条件下学习 kernel 之间的结构相似性，K-means 负责在 embedding 空间中完成聚类并选择代表 kernel。通过这三个部分的结合，我们完成了从真实 trace 到代表 kernel 选择的端到端复现。最终结果是把 265 个 kernel invocation 压缩到 2 个代表 kernel，得到约 132.5 倍的理论模拟加速。以上就是我们的课程报告，谢谢大家。
