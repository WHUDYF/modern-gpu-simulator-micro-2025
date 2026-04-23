# 当前阶段总结：从 Workload 到 Simulator 的结构化 Family 接口

日期：2026-04-19

## 1. 这份文档的目的

这份文档用于总结我们在当前阶段已经达成的关键共识，避免后续继续讨论时反复回到同一组基本问题。

当前最重要的目标不是立刻给出最终可投稿版本，而是先把下面三件事讲清楚：

1. 我们的工作主张到底是什么。
2. 我们为什么不能再沿用“按算子名分组”的思路。
3. 我们下一步应该如何把 `mini_transformer_v4` 上的方法原型继续做硬。

---

## 2. 当前主张：我们要补的是一层结构化接口

当前我们已经明确：

**这项工作的主线不是 sampled simulation，也不是 proxy benchmark，更不是单 kernel 调优技巧。**

我们真正想补上的，是：

**从 workload 行为到 simulator 分析对象之间的一层结构化接口。**

更具体地说，现有流程虽然已经可以：

- 对 workload 做模拟
- 对 kernel 做 profile
- 用 sampling 降低模拟成本

但在 `workload -> simulator reasoning` 之间，仍然缺少一层明确的中间结构。  
这导致很多关键步骤仍然高度依赖经验，例如：

- 哪些 kernel 应该被放在一起看
- 哪些 kernel 应该被拆开
- 哪些对象值得共享验证主线
- 哪些对象应该保留为 outlier

因此，我们当前的方法目标可以写成：

**把原始 workload 行为组织成 simulator 可以承接、比较、验证的结构化对象。**

---

## 3. 与现有工作的当前区分

### 3.1 PKA / Sieve 做了什么

我们当前认可，PKA、Sieve 这类工作是重要的 related work，但它们的核心目标与我们不同。

- `PKA`
  - 通过 representative kernel selection 和 projection 减少进入 simulator 的工作量
  - 主要压缩的是 `simulation samples`

- `Sieve`
  - 通过 stratification 和 representative invocation selection 提高 sampled simulation 的稳定性与精度
  - 主要改善的是 sampling 质量

### 3.2 我们和它们的关键区别

我们当前最稳的表述是：

**PKA / Sieve 主要回答“模哪些样本、模多少样本”；我们更想回答“按什么结构去解释和组织 workload”。**

所以，我们不应把自己包装成：

- 更好的采样方法
- 更好的聚类器
- 更好的 representative-kernel selection

而应该更准确地说：

**我们试图提出一种从 workload 到 simulator 的结构化接口。**

---

## 4. 为什么不能只按算子分类

这是当前阶段最重要的认识之一。

我们已经明确：

**如果 family 只是按算子名分组，那么这件事的学术价值会很弱。**

原因在于，按算子名分组默认了两件事：

1. 同名算子一定共享同一类硬件行为
2. 不同名字的算子一定属于不同类

这两个假设都不稳。

例如：

- `attention_score` 和 `gemm_tiled`
  - 上层语义不同
  - 但实现上可能共享相近的 tiled dense compute 模板

- `softmax_kernel` 和 `context_mul`
  - 同属 attention 主链
  - 但执行骨架不同，不能因为处在同一 attention 子模块就强行并类

因此，当前我们已经放弃如下弱定义：

- 按算子名做 batch / family
- 按上层模块语义直接归类
- 用“都属于 attention”来定义共享 family

---

## 5. 当前最重要的方法转向：引入两层结构

为了避免“按算子语义分类”过于薄弱，我们当前已经形成了一个更稳的两层结构。

### 第一层：Route Primitive

这一层回答：

**这个 kernel 在 workload 计算路线里扮演什么角色？**

例如：

- `Dense Projection/Transform`
- `Pairwise Score`
- `Reduction / Normalize`
- `Weighted Aggregation`
- `Elementwise Fusion`

这一层更靠近 workload 主计算路径。

### 第二层：Implementation Template

这一层回答：

**这个 kernel 在硬件上主要通过什么模板实现？**

例如：

- `Dense Tiled Compute`
- `Reduction Template`
- `Streaming Aggregation Template`
- `Elementwise Template`

这一层更靠近底层硬件执行骨架。

### 为什么要分成这两层

因为有些 kernel：

- 在 **workload 路线角色** 上不同
- 但在 **底层实现模板** 上相近

也有些 kernel：

- 在 **workload 路线** 上相邻
- 但在 **执行骨架** 上完全不同

如果把这两个层次混在一起，family 的定义就会非常不稳定。

---

## 6. 当前对几个关键 kernel 的稳定认识

现阶段，我们最适合先用下面这张表来组织 `mini_transformer_v4` 的几个核心 kernel。

| Kernel | Route Primitive | Implementation Template | 上层路线归属 | 当前理解 |
|---|---|---|---|---|
| `gemm_tiled` | `Dense Projection/Transform` | `Dense Tiled Compute` | `projection / FFN route` | 核心是 tile 化 dense multiply-accumulate |
| `attention_score` | `Pairwise Score` | `Dense Tiled Compute` | `attention readout route` | 路线角色是生成 pairwise score，但实现上和 tiled dense compute 接近 |
| `softmax_kernel` | `Reduction / Normalize` | `Reduction Template` | `attention readout route` | 核心是 row-wise reduction、normalization、同步与归约 |
| `context_mul` | `Weighted Aggregation` | `Streaming Aggregation Template` | `attention readout route` | 核心是读权重与 `V` 并做带权累加 |
| `layernorm_kernel` | `Reduction / Normalize` | `Reduction Template` | `normalization route` | 是通用 normalization 路线的一部分 |
| `residual_add` | `Elementwise Fusion` | `Elementwise Template` | `residual route` | 是轻量但高频的逐元素融合 |

这张表当前的意义不是直接给出最终 family，而是先把下面几个最难的问题拆开：

1. 它在 workload 路线里是什么角色？
2. 它在硬件上通过什么模板执行？
3. 这两个层次是否共享？

---

## 7. `softmax` 和 `context_mul` 当前应该怎么分

这是当前阶段已经基本稳定下来的一个关键判断。

### 7.1 结论

**在 primitive 层，`softmax` 和 `context_mul` 必须分开。**

但在更高一层的 Transformer 主计算路线里，它们可以同属：

**`attention readout route`**

也就是说：

- **primitive 层：拆开**
- **route 层：相邻、串联、同属一条上层路径**

### 7.2 为什么必须拆开

#### `softmax_kernel`

它的执行骨架是：

- row-wise reduction
- normalization
- 多轮同步
- shared memory / warp reduction

所以它更适合作为：

**`Reduction / Normalize primitive`**

#### `context_mul`

它的执行骨架是：

- 读取权重
- 读取 `V`
- 做带权累加
- 形成输出向量

所以它更适合作为：

**`Weighted Aggregation primitive`**

### 7.3 当前最稳的表述

我们现在不应该说：

- `softmax` 和 `context_mul` 属于同一个 family

更好的说法是：

**它们属于同一条上层 Transformer 计算路线，但对应不同的 execution primitive。**

---

## 8. 当前对 Transformer 主链的理解

当前我们最适合把 Transformer 主计算路线写成：

`QKV / projection -> attention_score -> softmax -> context_mul -> output projection -> residual/norm -> FFN`

这条路线的重要性在于：

1. 它本身是现代 AI workload 中非常核心的一条路径
2. 它已经能覆盖多类关键 primitive
3. 它足以作为第一版方法原型的承载对象

在这条主链上，我们当前已经能比较稳定地识别出：

- `projection / FFN` -> `Dense Projection/Transform`
- `attention_score` -> `Pairwise Score`
- `softmax` -> `Reduction / Normalize`
- `context_mul` -> `Weighted Aggregation`
- `residual_add` -> `Elementwise Fusion`
- `layernorm` -> `Reduction / Normalize`

所以，Transformer 主链当前已经不仅仅是一个案例，而是：

**第一版结构化接口的工作台。**

---

## 9. 当前对 family 的认识还没有最终定稿

这是当前阶段需要特别强调的边界。

我们虽然一直在讨论 family，但当前最稳的做法不是立刻给出最终 family 划分，而是先承认：

**family 的最终定义层次还没有完全锁定。**

当前我们已经知道：

- family 不能简单等于算子名
- family 不能简单等于上层模块名
- family 也不能只看一个瓶颈项

更可能的情况是：

**family 位于 `Route Primitive` 与 `Implementation Template` 之间，或者由两者共同决定。**

因此，当前阶段不应该强行把所有 kernel 硬塞进一个固定 family taxonomy。

更好的做法是：

1. 先把 kernel 的两层结构写清楚
2. 再观察哪些对象在两层上同时稳定共享
3. 最后再决定 family 应该收在哪一层最合理

---

## 10. 当前阶段的汇报主线

当前我们已经把导师汇报的主线收束成下面这条：

1. 现有流程缺少一层 `workload -> simulator reasoning` 的结构化接口
2. PKA / Sieve 等工作主要解决 sampled simulation 问题，而不是结构接口问题
3. 我们的方法不是新的采样技巧，而是新的组织接口
4. 这套接口在 Transformer 主链上已经有第一版原型
5. boundary case 说明 primitive 判据必须从执行骨架层长出来

所以，当前阶段最重要的一句话是：

**我们要解决的不是“如何少模拟几个 kernel”，而是“如何把 workload 结构化成 simulator 可以承接的硬件分析对象”。**

---

## 11. 下一步最合理的推进顺序

当前阶段不建议继续无边界扩 workload 名单。  
更合理的下一步是：

### 第一步：做硬 Transformer 主链上的 primitive 判据

目标：

- 让每个关键 kernel 都能稳定地写成
  - `Route Primitive`
  - `Implementation Template`

### 第二步：形成 boundary / selection protocol

目标：

- 回答 family 边界为什么是这些
- 回答什么时候该并类
- 回答什么时候该拆类
- 回答什么时候应保留 outlier

### 第三步：再向 simulator lane / tuning lane 对接

目标：

- 不是立刻给出完整 tuning 结果
- 而是先说明哪些对象值得共享验证主线
- 哪些对象适合共用 simulator reasoning lane

---

## 12. 当前阶段的最简结论

到目前为止，我们已经能比较稳地说：

1. 当前 GPU simulator 相关工作中，确实缺少一层从 workload 到 simulator reasoning 的结构化接口。
2. 我们的方法主线不应再写成“按算子分组”，而应转向“`Route Primitive + Implementation Template`”的两层结构。
3. `softmax` 和 `context_mul` 在 primitive 层必须拆开，但可以同属 Transformer 的 `attention readout route`。
4. Transformer 主计算路线已经足以作为第一版方法原型的承载对象。
5. 当前阶段最重要的任务不是扩大覆盖面，而是先把 primitive 判据和 boundary protocol 做硬。

这就是我们现阶段最稳定的方法论总结。
