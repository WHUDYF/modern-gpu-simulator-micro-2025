# PKA 相关信息与输入接口整理

日期：2026-04-20

## 1. 文档目的

这份文档用于把当前与 `PKA` 相关的关键信息整理清楚，并回答一个当前最重要的问题：

**如果我们希望把工作线改成“完整 workload -> 压缩 -> simulator 结构化分析 -> 验证”，那么 PKA 在这条链路中到底提供什么，以及我们需要从 PKA 得到哪些输入数据。**

这份文档的重点不是完整复述 PKA 论文，而是为后续实现和方法定位服务。

---

## 2. PKA 的基本信息

### 2.1 论文标题

**Principal Kernel Analysis: A Tractable Methodology to Simulate Scaled GPU Workloads**

### 2.2 发表会议信息

- 会议：**MICRO-54**
- 全称：**54th Annual IEEE/ACM International Symposium on Microarchitecture**
- 年份：**2021**

### 2.3 论文中的关键术语

- `PKA`：**Principal Kernel Analysis**
- `PKS`：**Principal Kernel Selection**

需要注意：

- `PKA` 是整套方法
- `PKS` 更接近方法中的代表 kernel 选择步骤

---

## 3. PKA 解决的核心问题

PKA 主要解决的问题不是 simulator 侧的结构化组织，而是：

**当 workload 很大、包含大量 kernel 时，如何只选择少量 representative kernels 进入模拟器，从而降低 simulation cost。**

换句话说，PKA 关心的是：

- 哪些 kernel 最值得模拟
- 如何用更少的 kernel 近似整个 workload
- 如何降低大规模 GPU workload 的模拟成本

因此，PKA 压缩的是：

**simulation samples**

而不是：

- phase-level 时间结构
- shared mechanism family
- representative execution regime

---

## 4. PKA 的核心思路

从当前阶段的理解出发，可以把 PKA 的主流程概括为：

`full workload -> kernel feature extraction -> dimensionality reduction / projection -> clustering -> representative kernel selection -> simulation`

更具体地说，它大致做了四件事：

### 4.1 从完整 workload 中提取 kernel-level 特征

这一阶段的目标是把大量 kernel 表达成可以比较的 feature vectors。

### 4.2 在 feature space 中做降维 / 投影

目的是让 kernel 之间的相似性结构更可处理，也让后续 clustering 和 representative selection 更稳定。

### 4.3 对 kernel 做 clustering / grouping

通过聚类把相似 kernel 放到一起，形成候选 group。

### 4.4 从每个 group 中选择 representative kernel

最终不是模拟全部 kernel，而是只模拟：

- 少数 principal / representative kernels
- 再用这些结果去近似整个 workload

所以 PKA 的最终输出本质上是：

**一组代表 kernel 及其覆盖关系**

---

## 5. PKA 在我们工作中的位置

### 5.1 我们不应该把 PKA 当成竞争目标

当前更合理的定位不是：

**我们的方法替代 PKA。**

而是：

**PKA 负责 workload compression，我们负责把压缩后的 representative kernels 继续组织成 simulator 可用的结构化分析对象。**

也就是说，P KA 更适合作为我们方法链中的前端压缩层。

### 5.2 我们与 PKA 的分工

可以把分工写成：

- `PKA`
  - 解决：`从完整 workload 中选出代表 kernel`
  - 输出：`representative kernels + coverage / weight information`

- `我们的方法`
  - 解决：`这些 representative kernels 在 simulator 侧如何继续被结构化组织`
  - 输出：`phase / family / representative regime / simulator validation lanes`

一句最简化的话：

**PKA 选对象，我们给结构。**

---

## 6. 为什么我们需要从 PKA 入手

如果我们的最终目标是：

**从算法工作负载到 GPU 模拟的端到端分析**

那么中间就不能跳过压缩层。

否则方法线就会变成：

`selected kernels -> family / regime -> simulator`

而不是：

`full workload -> compression -> structured simulator objects -> validation`

因此，引入 PKA 的意义在于：

### 6.1 把完整 workload 压缩成可管理的前端输入

这一步解决的是规模问题。

### 6.2 为后续 family / regime 分析提供代表锚点

这一步解决的是输入对象问题。

### 6.3 让“端到端”叙事真正闭合

也就是：

- 从完整 workload 出发
- 先压缩
- 再组织
- 最后验证

---

## 7. 我们需要从 PKA 得到哪些输入数据

这是当前最关键的部分。

我们后续真正需要的，不是“完整复刻 PKA 论文中的所有细节”，而是复现它对我们有用的**输入输出契约**。

### 7.1 第一类：Representative Kernels

这是最基础的输出。

至少需要知道：

- 被选中的 representative kernel 是谁
- 它的 kernel name / kernel id
- 它对应的是哪类计算对象

这类数据会直接成为后续 family / regime 分析的输入起点。

### 7.2 第二类：Cluster / Membership 信息

这类信息回答的是：

- 每个 representative kernel 代表了哪些原始 kernel
- 原始 workload 中每个 kernel 属于哪个 cluster

如果没有这层 membership 信息，我们只能知道“谁被选出来了”，却不知道：

- 它覆盖谁
- 它为什么重要
- 后续 simulator 结果如何回写到 workload

### 7.3 第三类：Weight / Coverage 信息

这类信息回答的是：

- 每个 representative kernel 覆盖多少样本
- 在 workload 中占多少比例
- 占多少执行时间

对我们后续方法来说，这非常重要，因为它会进一步影响：

- family weight
- regime weight
- simulator 调参优先级

### 7.4 第四类：Kernel Metadata

除了代表对象本身，我们还需要尽可能保留以下 metadata：

- kernel name
- invocation count
- grid dimension
- block dimension
- dynamic instruction count
- execution time / cycle proxy
- opcode statistics
- 访存相关统计

这类 metadata 的作用不是替代后续 family 分析，而是让 representative kernel 后面还能继续接到：

- shared mechanism identification
- shape / size regime
- phase context

### 7.5 第五类：Context Metadata（如果可以获得）

如果条件允许，最好保留 kernel 所在的 workload 上下文信息，例如：

- 出现顺序
- 在 trace 中的位置
- 所属 phase
- 所属 layer / module
- 输入形状信息（M/N/K、sequence length、batch size、head dim）

这类信息不是 PKA 的主目标，但对我们非常关键，因为我们后面要做的不是单纯 sampling，而是：

**representative kernel -> representative execution regime**

而这一步离不开上下文。

---

## 8. 我们理想中的 PKA 输出接口

从当前方法线出发，我们最希望得到一张代表 kernel 表。

建议的最小字段如下：

| 字段 | 含义 |
|---|---|
| `rep_kernel_id` | representative kernel 标识 |
| `kernel_name` | kernel 名称 |
| `cluster_id` | 所属 cluster |
| `covered_kernels` | 被它代表的原始 kernel 列表 |
| `coverage_count` | 覆盖样本数 |
| `coverage_weight` | 覆盖比例 |
| `time_weight` | 时间占比 |
| `grid_dim` | grid 配置 |
| `block_dim` | block 配置 |
| `inst_count` | 动态指令量 |
| `opcode_summary` | opcode 分布摘要 |
| `memory_summary` | 访存摘要 |
| `trace_position` | 在 trace 中的位置或顺序信息 |
| `phase_hint` | 可选，所属 phase 提示 |
| `shape_hint` | 可选，M/N/K / seq length / batch 等形状提示 |

如果我们能得到这张表，那么后续就能比较自然地接到我们的方法层：

`representative kernels + metadata -> family analysis -> regime extraction -> simulator`

---

## 9. 我们不需要复现 PKA 的哪些部分

为了避免范围失控，当前阶段应明确：

### 9.1 不需要先追求完整论文级复现

我们当前不是要重新发明一个新的 PKA 复现工程，而是要把它作为前端压缩层接入自己的方法。

### 9.2 不需要一开始证明 PKA 本身最优

当前最重要的是：

- 让 PKA 风格的 compression 跑通
- 拿到 stable representative kernels
- 获得后续可用的 membership / weight / metadata

### 9.3 不需要先把所有压缩误差分析做满

那是 PKA 自己的核心评价问题，不是我们当前第一阶段的主任务。

---

## 10. PKA 接入后，我们的方法主线应该怎么改

当前更合理的方法线可以改写为：

`full workload -> PKA-style representative kernel compression -> representative kernels + weights -> family / representative regime organization -> simulator validation`

这条主线比原来的好处在于：

### 10.1 端到端故事更闭合

因为我们不再从“已经挑好的几个 kernel”开始，而是从完整 workload 出发。

### 10.2 我们的创新点更清楚

因为可以明确说：

- PKA 负责 sample compression
- 我们负责 simulator-side structural organization

### 10.3 后续定量验证更自然

因为可以直接比较：

- 原始 workload 中有多少 kernel
- PKA 压缩后剩多少 representative kernels
- 我们再把它们压成多少 family / regime
- 最终减少了多少 simulator 分析对象

---

## 11. 最重要的边界：PKA 不能证明我们的机制“正确”

这里必须写得很清楚。

我们不应该说：

**PKA 证明了我们提出的机制 family 是正确的。**

更稳的说法是：

**PKA 为我们提供 representative kernel anchors，而我们的方法负责检查这些 anchors 是否能被少量共享机制稳定解释，并进一步通过 simulator 扰动验证其解释力。**

也就是说：

- PKA 不能直接给出 mechanism truth
- 但它可以给出 representative objects
- 我们再用这些 objects 检查：
  - 同 family 内是否有相近 simulator sensitivity
  - 不同 family 间是否有可区分行为

因此，我们最终想论证的不是：

**机制绝对正确**

而是：

**机制划分具有解释力、可验证、可服务 simulator 分析**

---

## 12. 当前最建议的实现顺序

如果接下来要真正落地，我建议顺序如下：

### Step 1：先定义 PKA 前端的输入数据格式

明确：

- 原始 workload 从哪里来
- 使用哪类 kernel features
- 输入记录单位是 kernel 还是 invocation

### Step 2：实现最小可用的 representative kernel selection

目标不是完整复现 PKA，而是先得到：

- representative kernels
- cluster membership
- weight / coverage

### Step 3：定义代表 kernel 输出表

把第 8 节中的字段尽可能落成结构化表。

### Step 4：把这张表接到 family / regime 模块

也就是开始真正回答：

- 这些 representative kernels 共享什么机制
- 哪些可以放入同一 family
- 哪些需要单独成为 outlier / regime

### Step 5：再接 simulator

最后才是：

- 对 representative regime 进入 simulator lane
- 做 sensitivity / perturbation / validation

---

## 13. 当前阶段的简短结论

如果把当前共识压成最短形式，可以写成：

1. PKA 解决的是 representative kernel compression，而不是 simulator-side structural organization。
2. 对我们来说，PKA 最重要的价值是提供 representative kernel anchors、membership 和 weight 信息。
3. 我们真正需要复现的不是 PKA 的全部论文负担，而是它对我们有用的输入输出契约。
4. 在 PKA 风格压缩之后，我们的方法继续完成 `representative kernels -> family / regime -> simulator` 的结构化过渡。
5. 因此，PKA 适合作为我们端到端方法链中的前端压缩层，而不是竞争对象或替代对象。
