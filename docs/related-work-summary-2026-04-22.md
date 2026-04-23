# 相关工作总结（第一版）

日期：2026-04-22

## 1. 文档目的

这份文档用于把当前已经系统读过和讨论过的相关工作整理成一份统一参考文档。

目标不是列 bibliography，而是回答：

1. 每篇工作到底解决了什么问题
2. 它们各自补上的关键维度是什么
3. 它们停在了哪里
4. 它们与我们工作的关系是什么

---

## 2. 总体判断

当前 sampled GPU simulation / representative compression 这条线非常活跃。

但这些工作大多集中在：

- 如何更好地压缩 full workload
- 如何更准确地得到 representative objects
- 如何控制 sampled simulation 的误差与成本

目前还没有一条明确公开主线，真正把下面这条链补完整：

`representative kernels -> family / representative regime -> importance ratio -> simulator tuning priority`

这正是我们当前最值得继续推进的位置。

---

## 3. PKA

### 3.1 基本信息

- 标题：**Principal Kernel Analysis: A Tractable Methodology to Simulate Scaled GPU Workloads**
- 会议：**MICRO 2021**

### 3.2 它解决什么问题

PKA 的目标是：

**从大量 kernel 中选出少量 representative kernels，以降低大规模 GPU workload 的 simulation cost。**

### 3.3 它的方法核心

- 采集 microarchitecture-independent kernel behavior features
- PCA 降维
- K-means clustering
- 选 representative kernels
- 再做 kernel projection

### 3.4 它新增的关键维度

**行为特征空间**

也就是说，它强调：

- 不能只按 kernel 名字组织对象
- 要在 behavior feature space 中找 representative objects

### 3.5 它停在哪里

它最终停在：

- representative compression
- sampled simulation tractability

它没有继续回答：

- representative kernels 之后如何形成 family / regime
- importance ratio 如何定义

### 3.6 与我们的关系

PKA 是我们当前最稳的：

**frontend anchor**

也就是说，它更适合作为我们前端 compression 的稳定锚点，而不是竞争对象。

---

## 4. Sieve

### 4.1 基本信息

- 标题：**Sieve: Stratified GPU-Compute Workload Sampling**
- 会议：**ISPASS 2023**

### 4.2 它解决什么问题

Sieve 的目标是：

**让 sampled simulation 的 strata 内 execution time variance 更小，从而提高 sampled simulation 的稳定性与效率。**

### 4.3 它的方法核心

- 用 instruction count 作为 work-size proxy
- 对 kernel invocations 进行 stratification
- 通过更稳定的 strata 选择 representative invocation

### 4.4 它新增的关键维度

**工作量尺度 / work-size**

它强调：

- grouping 不能只看行为相似
- 还必须显式控制 work-size 差异

### 4.5 它停在哪里

它最终仍然停在：

- stratified sampling
- representative invocation selection

### 4.6 与我们的关系

Sieve 给我们的最重要启发是：

**工作模式相近还不够，工作量尺度也必须成为 grouping 的必要条件。**

这对我们后续的 regime 构建非常关键。

---

## 5. Photon

### 5.1 基本信息

- 标题：**Photon: A Fine-grained Sampled Simulation Methodology for GPU Workloads**
- 会议：**MICRO 2023**

### 5.2 它解决什么问题

Photon 的目标是：

**在不依赖重 upfront profiling 的情况下，在线决定 sampled simulation 该采用哪一层粒度。**

### 5.3 它的方法核心

- 在线分析 kernel
- 构造 GPU BBV
- 在 kernel / warp / basic-block 三层之间自适应切换采样粒度

### 5.4 它新增的关键维度

**在线执行路径结构**

它强调：

- 执行路径本身可以作为 sampled simulation 的有效特征
- 不必完全依赖离线 hand-crafted features

### 5.5 它停在哪里

它最终停在：

- adaptive sampling control

### 5.6 与我们的关系

Photon 给我们的最重要启发是：

**family / regime 的证据源不一定只能来自离线 counters，也可以吸收在线执行结构信息。**

---

## 6. STEM+ROOT

### 6.1 基本信息

- 标题：**Swift and Trustworthy Large-Scale GPU Simulation with Fine-Grained Error Modeling and Hierarchical Clustering**
- 会议：**MICRO 2025**

### 6.2 它解决什么问题

STEM+ROOT 的目标是：

**解决同名 kernel invocation 也可能高度 runtime heterogeneous 的问题，从而让 sampled simulation 更准、更可控。**

### 6.3 它的方法核心

- 用 execution time distribution 看 cluster 的稳定性
- `ROOT` 递归拆分有 heterogeneity 的 cluster
- `STEM` 根据 CoV 和 error bound 给每个 cluster 分配 sample budget

### 6.4 它新增的关键维度

**runtime distribution heterogeneity**

它强调：

- grouping 不仅有行为空间这一维
- 还有 invocation 级 runtime distribution 这一维

### 6.5 它停在哪里

它最终停在：

- refined sampled simulation clusters
- sample budget optimization

### 6.6 与我们的关系

STEM+ROOT 是当前与我们最接近的前置工作。

它说明：

- 前端 compression 后的对象必须考虑 heterogeneity
- 否则后续结构分析会失真

因此它非常适合作为：

**frontend refinement inspiration**

而不一定适合作为第一版完整前端实现。

---

## 7. GCL-Sampler

### 7.1 基本信息

- 标题：**GCL-Sampler: Discovering Kernel Similarity for Sampled GPU Simulation via Graph Contrastive Learning**
- 状态：**arXiv 2026 预印本**

### 7.2 它解决什么问题

GCL-Sampler 的目标是：

**摆脱 hand-crafted features 的限制，通过 learned similarity 得到更强的 representative compression。**

### 7.3 它的方法核心

- 把 kernel trace 转成 graph
- 用 R-GCN + contrastive learning 学 kernel embeddings
- 在 learned embedding space 中发现 kernel similarity

### 7.4 它新增的关键维度

**learned structural similarity**

它强调：

- 相似性可以由模型学习，而不必完全人工设计

### 7.5 它停在哪里

它仍然停在：

- representative compression
- sampled simulation front-end

### 7.6 与我们的关系

它提醒我们：

**不要把自己的贡献点放在“再做一种更好的 kernel clustering”上。**

我们的边界必须牢牢钉在：

- compression 之后
- simulator-side organization and decision layer

---

## 8. GainSight

### 8.1 基本信息

- 标题：**GainSight: A Unified Framework for Data Lifetime Profiling and Heterogeneous Memory Composition**
- 状态：**arXiv 2025 预印本**

### 8.2 它解决什么问题

GainSight 的目标是：

**从 workload 中提取细粒度 data lifetime profile，用于指导 heterogeneous on-chip memory composition。**

### 8.3 它的方法核心

- profiling data lifetime
- architecture-agnostic analytical frontend
- workload-driven memory composition

### 8.4 它新增的关键维度

**data lifetime**

它强调：

- 数据生命周期本身可以成为硬件设计的一等输入

### 8.5 它停在哪里

它不是 sampled GPU simulation 工作，而是：

- workload-guided hardware composition

### 8.6 与我们的关系

GainSight 不是直接同类工作，但它非常重要，因为它证明了：

**从 workload 中提结构化信号，再让这些信号去驱动硬件决策，这条方法论是成立的。**

这对我们后续把 `squash + batch + importance ratio` 用于 simulator tuning，有很强的支撑作用。

---

## 9. 当前相关工作的统一图景

如果把这些工作放在一起看，可以得到下面这张方法图景：

### PKA

补上：

- 行为特征空间

### Sieve

补上：

- 工作量尺度

### Photon

补上：

- 在线执行路径结构

### STEM+ROOT

补上：

- runtime distribution heterogeneity

### GCL-Sampler

补上：

- learned similarity

### GainSight

证明：

- workload-derived structure 可以进入 hardware decision

### 我们

真正要补的是：

- representative kernels 之后的：
  - family
  - regime
  - importance ratio
  - tuning priority

---

## 10. 当前我们的位置

基于上面这些工作，当前最稳的定位是：

### 前端

- `PKA` 作为前端锚点
- `STEM+ROOT` 作为异质性 refinement 启发

### 中间结构层

- `squash`
- `batch`
- `family`
- `regime`

### 后端决策层

- `importance ratio`
- `tuning priority`
- `simulator validation lane`

也就是说，我们真正工作的价值不在于：

- 重做前端 compression

而在于：

**把 compression output 继续推进成 simulator-side decision layer。**

---

## 11. 当前阶段的简短结论

如果把这份文档压成最短形式，可以写成：

1. 现有相关工作已经从多个维度不断增强 GPU workload compression：行为特征、工作量尺度、在线路径结构、runtime heterogeneity 和 learned similarity。
2. 这些工作主要仍停在 representative compression 或 sampled simulation budgeting 层。
3. GainSight 进一步证明了从 workload 结构化信号到硬件决策的方法论是成立的。
4. 因此，我们当前最合理的位置不是再做 compression 本身，而是在 compression 之后补上 family / regime / importance ratio / tuning priority 这一层。
