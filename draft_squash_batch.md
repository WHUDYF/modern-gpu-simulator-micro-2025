# Squash + Batch 方法草稿

日期：2026-04-14

## 目标

这份草稿用于总结当前已经明确的 `squash + batch` 方法设想。

当前的核心目标不是单独评价某一个机制是否“足够强”，而是明确：

**如果我们的主故事是建立从 workload 到 simulator 的端到端视线，那么 `squash` 和 `batch` 分别承担什么角色，它们为什么都有学术价值。**

本文档暂时不展开 `delta`，只聚焦 `squash + batch` 两层。

---

## 一、总问题定义

我们希望解决的问题不是：

- 单独对某个 kernel 做更细的 profiling
- 单独提出一个新的 simulator 参数调整技巧
- 单独展示某个 cluster 结果好不好看

我们想解决的是一个更上游的问题：

**如何把复杂 workload 的执行行为，逐层收缩成后续可进入 simulator 验证的结构化对象。**

从这个角度看，`squash` 和 `batch` 都不是附属分析工具，而是中间结构层。

---

## 二、方法主叙事

当前我们认可的主叙事是：

**这项工作的核心不是“做一个更强的单点分析器”，而是建立一条从 workload 行为到 simulator 假设的端到端流程。**

在这条流程里：

- `squash` 负责组织执行流的时间结构
- `batch` 负责组织 kernel 之间的共享机制结构

因此，这两个模块的价值不主要体现在“单独发现了多少新现象”，而体现在：

**它们作为必要结构层，使得后续的架构解释和 simulator 验证能够成立。**

---

## 三、Squash 的定义与价值

### 1. 核心定义

`squash` 的作用不是简单地“把 kernel 分成几段”，而是：

- 将长执行流压缩为若干行为稳定的 phase
- 提取 workload 在时间维度上的结构
- 为后续代表 trace 选择和阶段级分析提供基础

换句话说，`squash` 处理的是：

**执行流如何沿时间展开，以及哪些时间段具有稳定、可复用的行为特征。**

### 2. 它解决的问题

如果没有 `squash`，复杂 workload 的执行流通常会有几个问题：

- 时间序列太长，难以直接理解
- 不同阶段混在一起，无法稳定映射到后续分析对象
- 很难判断哪些 trace 可以代表某一整段行为

因此，`squash` 的价值在于把“原始执行流”变成“阶段化的执行结构”。

### 3. 当前对其学术价值的判断

如果孤立看，`squash` 的发现性可能并不总是最强，它很多时候更偏确认性。

但如果放到端到端故事里，它的价值会显著上升，因为：

- 它提供了时间结构层
- 它让后续分析不需要直接面对未经组织的长执行流
- 它为 trace 复用、阶段代表选择和成本压缩提供依据

因此，`squash` 的学术价值不主要是“单独发现新架构规律”，而是：

**为端到端流程提供 phase-level 中间表示。**

---

## 四、Batch 的定义与价值

### 1. 核心定义

当前我们对 `batch` 的稳定定义是：

**`batch` 是一个解释层模块，用于识别共享同一架构机制的 kernel family。**

这意味着 `batch` 不应该只是按算子名分组，也不应该只是为了把结果整理得更整齐。

它真正要回答的是：

- 哪些 kernel 虽然名字不同，但受同一类架构机制主导
- 哪些 kernel 虽然名字相同，但行为异质，不能共用解释
- 当前 workload 的共享机制边界在哪里

### 2. 当前明确否定的弱定义

我们认为，`batch` 如果只是：

- 按算子名归组
- 对已有标签做结果整理
- 按语义名称做展示层聚合

那么它的价值是有限的。

因为这种做法默认了：

- 同算子名 = 同架构行为
- 不同算子名 = 不同架构行为

而这在复杂 workload 中并不成立。

### 3. Batch 在实现层面的落地

虽然 `batch` 的定义停留在解释层，但我们已经明确：

**在实现上，`batch` 识别出的 family 会进一步指导 simulator 侧的处方复用与验证分流。**

也就是说，`batch` 的结构化输出将帮助后续决定：

- 哪些 kernel 可以优先共用一类 simulator 处方
- 哪些 case 可以合并验证
- 哪些 kernel 必须作为 outlier 单独进入后续分析

这里需要注意一个边界：

我们不把“验证调度”作为 `batch` 的主定义，而把它视为 `batch` 解释层输出的自然后果。

因此，当前最合适的表述是：

**定义上选解释层，落地上实现验证分流。**

### 4. Batch 的学术价值

在这个定义下，`batch` 的价值已经不再是“我做了一个聚类器”，而是：

- 它识别共享架构机制的 family
- 它建立了 workload 到 simulator 假设之间的共享解释边界
- 它降低了后续逐个 kernel 人工分析和逐个 case 手工验证的复杂度

因此，`batch` 的学术价值可以概括为：

**它在 workload 和 simulator 之间建立了一个以共享架构机制为核心的解释层。**

---

## 五、Squash 与 Batch 的分工

当前最清晰的分工可以写成：

- `squash`：组织时间结构
- `batch`：组织共享架构机制结构

更具体地说：

### `squash` 回答的问题

- 这个 workload 的执行流如何沿时间展开
- 哪些阶段是稳定的
- 哪些阶段可以由代表 trace 近似

### `batch` 回答的问题

- 哪些 kernel 共享同一种架构解释
- 哪些 kernel 属于同一个 family
- 哪些 family 适合共用后续 simulator 处方

所以二者并不重复。

`squash` 面向的是**phase**，  
`batch` 面向的是**family**。

一个偏时间维度，一个偏行为/机制维度。

---

## 六、当前方法价值的统一理解

我们已经达成的关键共识是：

**`squash` 和 `batch` 不是最终贡献本身，而是让最终端到端方法论成立的必要结构层。**

这句话的含义是：

1. 它们不一定各自都要单独产出最强的新发现
2. 但缺少它们，workload 到 simulator 的映射会明显失去结构
3. 因此它们在方法论层面是必要的，而不是可有可无的辅助模块

换一种更适合论文的说法：

**`squash` 和 `batch` 共同把复杂 workload 的原始执行行为，转化为后续架构解释和 simulator 验证可接受的标准化中间表示。**

---

## 七、当前适合论文的表述方式

基于目前的讨论，`squash + batch` 最适合以如下方式进入论文：

### 对 `squash` 的表述

`squash` identifies stable execution phases from long workload traces, providing a phase-level representation for subsequent reasoning and representative trace selection.

可对应理解为：

**`squash` 用于从长执行流中识别稳定 phase，为后续分析和代表 trace 选择提供阶段级表示。**

### 对 `batch` 的表述

`batch` identifies kernel families that share a common architectural regime. These families then guide prescription reuse and validation partitioning on the simulator side.

可对应理解为：

**`batch` 用于识别共享同一架构机制的 kernel family，这些 family 进一步指导 simulator 侧的处方复用与验证分流。**

### 二者合并后的主句子

**`squash` and `batch` serve as two complementary structure layers: one organizes temporal execution phases, and the other organizes kernel families with shared architectural explanations.**

中文可以表述为：

**`squash` 与 `batch` 共同构成两层互补的结构层：前者组织执行流的时间阶段，后者组织共享架构解释的 kernel family。**

---

## 八、当前不在本草稿内展开的内容

为了保持边界清晰，下面这些内容暂时不在本草稿内展开：

- `delta` 如何在 family / phase 上提取候选架构因素
- Stage C 如何基于这些因素做 simulator 参数扰动
- 哪些候选信号最终被归类为可模拟 / 不可模拟 / 模型盲区

这些内容可以在后续单独文档中继续接上。

---

## 九、当前阶段的简短结论

如果把当前共识压缩成最短形式，可以写成：

1. `squash` 的职责是把复杂 workload 的长执行流组织成稳定 phase。
2. `batch` 的职责是识别共享架构机制的 kernel family，而不是仅按算子名分组。
3. `batch` 在实现层面进一步指导 simulator 处方复用与验证分流。
4. `squash + batch` 的共同价值，在于为“从 workload 到 simulator”的端到端流程提供必要的中间结构层。
