# Family Method Positioning And Abstract Drafts

日期：2026-04-17

## 目的

这份文档用于整理当前 `squash + batch family` 方法线的论文定位、与相关工作的区分，以及当前可用的中英文摘要草稿。

它不是最终投稿稿，而是当前阶段的论文主张与摘要工作底稿，方便后续继续收敛主 claim、整理 related work，并逐步演化成正式 paper draft。

---

## 一、当前主 claim 结构

### 核心主张

**我们的核心贡献是一种从 workload 行为到 simulator 的结构化 family 接口。**

这套接口不是简单的 kernel 分类，也不是 representative sampling 中的代表样本选择，而是通过：

- `squash` 提取时间结构
- `batch` 识别共享架构机制的 kernel family
- `boundary cases` 逼出 family 判据
- `analysis cards` 压缩判据证据
- `family cards` 形成结构化解释
- `synthesis` 将这些局部结构提升为方法论表达

把复杂 workload 的原始执行行为，组织成 simulator 可以理解、比较和承接的中间结构层。

### 次级主张

**在此基础上，这种 family 接口为 family-aware simulator tuning 提供了结构基础。**

也就是说，我们当前不把 family 本身当成最终优化结果，而把它视为：

- 组织验证主线的结构基础
- 从单 kernel 调参走向 family-level 调参的桥梁
- 连接 workload 行为与算法级 simulator 优化目标的中间层

### 当前最稳的总表述

**我们提出一种从 workload 行为到 simulator 的结构化 family 接口；这种接口已经在 `mini_transformer_v4` 上形成第一版原型，并进一步为 family-aware simulator tuning 提供组织基础。**

---

## 二、与相关工作的关键区分

### 1. 与 PKA / PKS 的区别

PKA / PKS 的核心是 representative sampling。它们主要回答：

- 哪些 kernel / invocation 可以代表其他样本
- 如何用更少的样本近似整个 workload 的行为

它们压缩的是：

**simulation samples**

而我们的目标不是 representative kernel selection，而是：

- 识别共享执行模板 / 数据通路的 family
- 用这些 family 去组织 simulator 调参目标

因此，我们压缩的是：

**hardware optimization problem**

一句最简区分：

**PKA/PKS 主要解决“模谁、模多久”；我们更想解决“按什么结构去调”。**

### 2. 与 Sieve 的区别

Sieve 比 PKS 更进一步地关注边界稳定性，它通过 instruction-count-based stratification 提高 sampled simulation 的稳定性与精度。

它和我们的相似点在于：

- 都重视“边界不能乱并”
- 都不再盲信“看起来像就能并类”

但它最终仍然服务于：

**sampled simulation**

而我们想服务的是：

**family-aware simulator tuning**

一句最简区分：

**Sieve 关心“怎么分层抽样更准”；我们关心“怎么按共享执行模板构造 family 并组织调参”。**

### 3. 与 proxy benchmark / motif 路线的区别

proxy benchmark 工作更像是在构造更小的代理 workload，用一个替身程序去代表原始 workload。

而我们不是在造替身，而是在原始 workload 内部抽取：

- family
- boundary
- weighting
- validation lanes

所以我们的目标不是 workload replacement，而是：

**simulator-side structural organization**

---

## 三、当前最关键的方法论认识

### 1. 统计相似性不等于结构归属

representative sampling 类工作主要依赖统计相似性；
而我们现在越来越明确地认为，我们的方法核心不应再用“相似度”去描述，而应该用：

- family membership
- boundary validity
- shared data-path
- shared execution template

来描述。

更准确地说：

**我们不是在做“更好的相似度度量”，而是在做“结构化的归属判断”。**

### 2. `batch` 的核心不是分组，而是划边界

当前我们已经基本确认：

- `batch` 的价值不在于“把 kernel 放在一起”
- 而在于“说明为什么不能轻易把谁放在一起”

所以它更像一个：

**解释边界识别器**

而不只是 cluster 生成器。

### 3. family 不是最终优化结果，而是优化组织层

family 的意义不是停留在分类，而是成为：

- 连接 workload 行为与 simulator 的中间结构
- 组织后续 validation lane 的基础
- 未来 family-level tuning 的调参载体

---

## 四、当前最大的脆弱点

我们现在最薄弱的地方，不是原型不够多，而是：

**family 的边界判据为什么是这些，还没有被形式化成一套选择协议。**

当前可以更稳地说的是：

- family 规则不是终极规则
- 它们是从关键 boundary case 中筛出来的第一版最小充分判据

这意味着后续最关键的补强文档应该是：

**criterion selection protocol**

---

## 五、中文摘要草稿

现代 GPU 模拟器已经能够支持越来越复杂的工作负载，但对于算法级 workload，研究者通常仍需在大量 kernel、phase 和潜在架构因素之间手工定位瓶颈与验证对象。现有 representative sampling 方法能够减少需要模拟的样本数量，但它们主要服务于 workload 估计与 sampled simulation，较少回答一个更上游的问题：**如何把复杂 workload 的执行行为结构化地映射为 simulator 可以承接的解释对象。**

我们提出一种从 workload 行为到 simulator 的结构化 family 接口。该接口不以统计相似性为核心，而是通过时间结构、边界 case 与共享执行模板来构造 data-path family，使 kernel 的分组建立在共享硬件工作模式之上，而不是仅仅建立在 profile 空间中的接近性之上。基于 `mini_transformer_v4` 的第一版原型，我们展示了如何从 boundary cases 出发，逐步形成 analysis cards、family cards 和方法级 synthesis，从而把复杂 workload 组织成可解释、可比较、可服务后续验证的中间结构。

进一步地，这种 family 接口并不只是分类结果，而是为 family-aware simulator tuning 提供组织基础。它使后续优化不再停留在单 kernel 层面的局部调参，而能够朝着共享执行模板的 family 级调参与算法整体目标对齐的方向推进。

---

## 六、英文摘要草稿（保守版）

Modern GPU simulators can model complex workloads, but connecting algorithm-level behavior to simulator-side reasoning still requires substantial manual effort. Existing cost-reduction methods mainly focus on representative sampling, reducing the number of kernel instances that need to be simulated. However, they do not explicitly expose a structured interface that organizes workload behavior into hardware-meaningful groups for downstream simulator reasoning.

We propose a structured family interface from workload behavior to GPU simulation. Our method organizes kernels into data-path families based on shared execution templates and resource interaction patterns, and refines these families through boundary-case analysis rather than statistical similarity alone. We instantiate this interface on `mini_transformer_v4` and build a first prototype consisting of boundary cases, analysis cards, family cards, and a family-level synthesis. This prototype shows that workload behavior can be transformed into a small set of interpretable structures that are better aligned with simulator-side analysis and future family-aware tuning.

---

## 七、后续建议

从这份文档往前推进，最自然的下一步是：

1. 先锁定最终主 claim 版本
2. 再写 `criterion-selection-protocol.md`
3. 然后把 family 与 tuning 之间的桥接逻辑单独写出来

在这些工作完成之前，建议把当前 abstract 视为“方向性草稿”，不要过早把它当成最终投稿版。
