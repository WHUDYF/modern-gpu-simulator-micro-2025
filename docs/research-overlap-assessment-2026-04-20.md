# 当前研究重合度评估

日期：2026-04-20

## 1. 文档目的

这份文档用于回答一个现实问题：

**当前是否已经有与我们高度重合的工作在推进，以及这对我们接下来应该投入多少时间精力意味着什么。**

这里的“重合”需要分层看：

1. 是否有人仍在积极推进 GPU sampled simulation / representative kernel compression
2. 是否有人已经在做我们想做的那一段：`representative kernels -> structured simulator objects -> family / regime / validation`

---

## 2. 当前可以明确确认的事实

### 2.1 sampled GPU simulation 这条线仍然是活跃方向

近年的代表工作至少包括：

- `PKA`，MICRO 2021  
- `Sieve`，ISPASS 2023  
- `Photon`，MICRO 2023  
- `STEM+ROOT`，MICRO 2025  
- `GCL-Sampler`，arXiv 2026

这说明：

**“如何从完整 GPU workload 中压缩出更少但更有代表性的模拟对象”这条线没有结束，反而还在继续快速推进。**

### 2.2 新工作仍然主要集中在 sample selection / similarity discovery

从已检索到的近期工作看，它们的主问题仍然主要是：

- 如何减少 sampled simulation 的误差
- 如何提高 sampled simulation 的速度
- 如何更准确地发现 kernel 相似性
- 如何降低 profiling 或 sampling 的开销

也就是说，主战场仍然是：

**representative sampling**

而不是：

**simulator-side structural organization**

### 2.3 我没有检索到与我们“精确重合”的公开工作

基于本轮检索，我没有看到有工作明确提出下面这条主线：

`representative kernels -> phase / family / representative execution regime -> simulator-side structured reasoning / validation`

更具体地说，我没有看到公开论文明确同时主打：

- representative kernel compression 之后
- 再补一层 workload-to-simulator structural interface
- 再把对象组织成 family / regime
- 再服务 simulator reasoning / tuning

这说明：

**和我们完全同一位置的公开工作，目前至少不明显。**

---

## 3. 近期相邻工作的含义

### 3.1 Photon（MICRO 2023）

Photon 的重点是：

- no up-front analysis
- 多层 sampling（kernel / warp / basic-block）
- 加速大规模 GPU simulation

它说明：

**顶会社区仍然接受“GPU simulation acceleration methodology”这类问题。**

但 Photon 仍然主要在做：

**sampled simulation methodology**

### 3.2 STEM+ROOT（MICRO 2025）

STEM+ROOT 的重点是：

- fine-grained error modeling
- hierarchical clustering
- 更可扩展、更可信的大规模 GPU sampled simulation

它说明：

**这条线不但还活着，而且已经发展到更复杂的误差建模和层次聚类。**

但它的主目标仍然是：

**更好的 sampling**

而不是：

**family / regime 级 simulator 结构层**

### 3.3 GCL-Sampler（arXiv 2026）

GCL-Sampler 的重点是：

- 用 graph contrastive learning 自动发现 kernel 相似性
- 继续提高 sampled simulation 的 fidelity 和 speedup

它说明：

**相似性发现与 representative kernel compression 这个方向还在快速演化，甚至已经开始引入新的学习方法。**

这意味着：如果我们只做“比 PKA 更好的代表 kernel 选择”，竞争会很激烈。

### 3.4 GainSight（arXiv 2025）

GainSight 不是 GPU sampled simulation 工作，但它说明另一条趋势：

**workload-guided hardware analysis / profiling-guided design exploration 正在变热。**

这与我们的方法论方向是相容的，因为我们也在做：

- 从 workload 行为出发
- 形成结构化中间层
- 再服务后续硬件侧验证

---

## 4. 当前重合度判断

如果把重合度分成三层，我当前的判断如下。

### 4.1 与 sampled GPU simulation 主线：高重合

在“workload 压缩 / representative kernel selection / sampled simulation”这一层面，我们和 PKA、Sieve、Photon、STEM+ROOT、GCL-Sampler 处在同一大领域。

所以：

**这个大方向并不空白，竞争是存在的。**

### 4.2 与结构化 simulator 输入层：中等重合

我们和这些工作共享：

- workload 压缩
- representative objects
- 进入 simulator 之前的对象挑选

但我们的重点不是继续优化 sample selection 本身，而是：

**在 compression 之后补一层 structured simulator interface。**

这一层与现有 sampled simulation 工作相邻，但不完全重合。

### 4.3 与 family / regime / simulator reasoning 这条线：低重合

当前没有明显公开工作同时把下面四件事合在一起：

1. representative kernel anchors
2. phase-aware family organization
3. representative execution regime
4. simulator-side reasoning / validation lanes

因此，若我们把贡献点锁在这里，精确重合度目前看是：

**低到中等偏低。**

---

## 5. 这对投入决策意味着什么

### 5.1 不应该降低投入

这轮检索得到的结论不是“领域太挤，没必要做”，而是：

**大方向很热，但我们想切入的那一层还没有被明显占住。**

所以不应该因为“有人在做 sampled simulation”就降低投入。

### 5.2 但不能慢悠悠推进

因为近两年这条线明显还在快速进展：

- 2023 有 Photon
- 2025 有 STEM+ROOT
- 2026 已经出现 GCL-Sampler

这说明：

**如果我们的工作迟迟停留在概念层，很容易被相邻方向的后续工作覆盖。**

### 5.3 最好的策略不是退，而是尽快把边界钉死

当前最合理的策略是：

- 不和这些工作竞争“谁的 sampling 更强”
- 迅速把我们的边界钉在：
  - `compression 之后`
  - `simulator 之前`
  - `family / regime / structured interface`

这样我们就不是去和 Photon / STEM+ROOT / GCL-Sampler 正面拼 sampling，而是在它们之后补上：

**simulator-side organization layer**

---

## 6. 当前最建议的对外表述

当前更稳的表述不是：

**我们要做一个更强的 representative kernel selection 方法。**

而应是：

**我们在 representative kernel compression 的基础上，继续构建从 compressed kernels 到 simulator structured objects 的接口层。**

这句话的好处是：

- 承认相邻工作的重要性
- 避免与 sampled simulation 主线正面同质化
- 保住我们自己的独立贡献边界

---

## 7. 当前阶段的简短结论

如果把当前判断压成最短形式，可以写成：

1. sampled GPU simulation / representative kernel compression 仍然是一个活跃且持续推进的方向。
2. 2023-2026 已经出现多篇新工作，说明这个领域有竞争，不能慢。
3. 但我尚未检索到与我们“从 representative kernels 到 family / regime / simulator-side interface”这一精确位置高度重合的公开工作。
4. 因此，这不是一个应该降低投入的信号，而是一个应该尽快把边界钉死并加速推进的信号。

---

## 8. 本轮检索中最关键的参考工作

- `PKA` — MICRO 2021  
  Principal Kernel Analysis: A Tractable Methodology to Simulate Scaled GPU Workloads

- `Sieve` — ISPASS 2023  
  Sieve: Stratified GPU-Compute Workload Sampling

- `Photon` — MICRO 2023  
  Photon: A Fine-grained Sampled Simulation Methodology for GPU Workloads

- `STEM+ROOT` — MICRO 2025  
  Swift and Trustworthy Large-Scale GPU Simulation with Fine-Grained Error Modeling and Hierarchical Clustering

- `GCL-Sampler` — arXiv 2026  
  GCL-Sampler: Discovering Kernel Similarity for Sampled GPU Simulation via Graph Contrastive Learning

- `GainSight` — arXiv 2025  
  GainSight: A Unified Framework for Data Lifetime Profiling and Heterogeneous Memory Composition
