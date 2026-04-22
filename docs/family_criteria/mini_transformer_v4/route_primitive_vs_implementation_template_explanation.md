# Route Primitive 与 Implementation Template 解释文档

日期：2026-04-19

## 1. 文档目的

这份文档用于单独解释我们当前讨论中的两个核心概念：

- `Route Primitive`
- `Implementation Template`

并进一步说明为什么下面这些概念：

- `Dense Projection/Transform`
- `Pairwise Score`
- `Reduction / Normalize`
- `Weighted Aggregation`
- `Elementwise Fusion`

更适合作为 **Route Primitive**，也就是更靠近 **workload 主计算路径** 的一层，而不是直接当作底层硬件实现分类。

这份文档的目标不是给出最终 family taxonomy，而是帮助我们先把“算法角色”和“硬件模板”分开。

---

## 2. 两层结构的最短定义

### 2.1 Route Primitive

`Route Primitive` 回答的问题是：

**这个 kernel 在 workload 的主计算路线里，到底扮演什么计算角色。**

它关心的是：

- 这一步在算法里负责做什么
- 这一步和前后步骤是什么关系
- 它为什么会出现在这条主路径上

所以，`Route Primitive` 更像是：

**算法视角的计算骨架。**

### 2.2 Implementation Template

`Implementation Template` 回答的问题是：

**这个 kernel 在 GPU 上主要通过什么执行模板实现出来。**

它关心的是：

- 是否使用 tile 化 dense compute
- 是否是 reduction 驱动
- 是否是 streaming accumulation
- 是否是简单逐元素访问

所以，`Implementation Template` 更像是：

**硬件视角的执行骨架。**

---

## 3. 为什么说 Route Primitive 更靠近 workload 主计算路径

核心原因很简单：

**这一层描述的是“算法主链依次做了哪些计算动作”，而不是“这些动作在 GPU 上怎么实现”。**

以 Transformer 主链为例：

`input -> Q/K/V projection -> attention_score -> softmax -> context_mul -> residual -> norm -> FFN`

如果把这条路径翻译成 `Route Primitive`，可以写成：

- `Dense Projection/Transform`
- `Pairwise Score`
- `Reduction / Normalize`
- `Weighted Aggregation`
- `Elementwise Fusion`
- `Reduction / Normalize`
- `Dense Projection/Transform`

可以看到，这一层几乎是在直接描述：

**workload 主路径上每一步到底在做什么计算动作。**

所以它更靠近：

- 算法结构
- workload 主链
- 各阶段在路径中的功能角色

而不是：

- shared memory 多大
- register 压力多高
- block 怎么排
- warp 如何同步

这些后者，才更接近 `Implementation Template`。

---

## 4. 五个 Route Primitive 的直觉解释

下面这五个概念，当前最适合作为 `Route Primitive`，因为它们描述的是 **计算角色**，而不是具体实现手法。

### 4.1 Dense Projection/Transform

直觉上就是：

**把一批输入表示，通过一个稠密线性变换映射成新的表示。**

典型例子：

- `Q = XWq`
- `K = XWk`
- `V = XWv`
- FFN 中的线性层

这一 primitive 强调的是：

- 输入是表示
- 输出还是表示
- 中间做的是稠密变换

它在 workload 路线中的角色是：

**表示变换。**

这里并没有规定它必须如何用 shared memory，也没有规定它一定如何 tile。  
所以它是计算角色，不是硬件模板。

---

### 4.2 Pairwise Score

直觉上就是：

**拿两组对象做两两关系计算，得到一个 score 矩阵。**

典型例子：

- `QK^T`

这一 primitive 强调的是：

- 不是在生成新的 embedding
- 而是在计算对象与对象之间的关系强弱

它在 workload 路线中的角色是：

**关系评分。**

所以即使 `attention_score` 底层实现上可能很像 GEMM，它在 workload 主链中的角色也不应该直接被写成 `Dense Projection/Transform`。

---

### 4.3 Reduction / Normalize

直觉上就是：

**先把一组值压缩成统计量，再基于这些统计量做归一化。**

典型例子：

- `softmax`
- `layernorm`

这一 primitive 强调的是：

- reduction
- normalization
- 跨元素统计
- 再回写调整

它在 workload 路线中的角色是：

**把一组值规范化成后续可用的形式。**

所以这里强调的是“先归约、再规范化”的计算动作，而不是具体用哪种同步指令。

---

### 4.4 Weighted Aggregation

直觉上就是：

**已经有了权重之后，把多个值按权重聚合成新的输出。**

典型例子：

- `softmax(QK^T) * V`

这一 primitive 强调的是：

- 权重已经存在
- 当前任务不是再算关系
- 当前任务是根据权重把信息聚起来

它在 workload 路线中的角色是：

**按权重汇总信息。**

所以 `context_mul` 更适合作为 `Weighted Aggregation`，而不是和 `softmax` 一起被粗略地视为同一种“attention 子模块操作”。

---

### 4.5 Elementwise Fusion

直觉上就是：

**对已有表示做逐元素层面的融合、修正或叠加。**

典型例子：

- residual add
- bias add
- 某些简单 activation 前后融合

这一 primitive 强调的是：

- 不做复杂关系计算
- 不做大规模归约
- 不做带权聚合
- 只对已有结果进行轻量融合

它在 workload 路线中的角色是：

**轻量级结果融合。**

---

## 5. 为什么这些概念不适合直接当作底层硬件模板

因为这些概念回答的是：

**这一步在算法里做什么。**

而不是：

**这一步在 GPU 上怎么执行。**

例如：

### `attention_score`

它在 `Route Primitive` 层应写成：

`Pairwise Score`

因为它在算法里的角色是生成两两关系分数。

但在 `Implementation Template` 层，它完全可能写成：

`Dense Tiled Compute`

因为它在底层实现上仍然可能通过 tiled dense multiply-accumulate 来完成。

这说明：

- 路线角色可以和 `gemm_tiled` 不同
- 底层模板却可以和 `gemm_tiled` 相近

如果不分层，这件事就很难讲清楚。

---

## 6. `softmax` 和 `context_mul` 为什么能说明这两层必须分开

这是当前阶段最典型的一组边界案例。

### 6.1 从 workload 路线看

它们都属于：

`attention readout route`

也就是说，在上层主计算路径中，它们前后相邻，并共同构成 attention 输出计算的一部分。

### 6.2 但从 Route Primitive 看

它们不能并成同一个 primitive。

#### `softmax`

更适合写成：

`Reduction / Normalize`

因为它的核心动作是：

- row-wise reduction
- normalization
- 同步与统计量生成

#### `context_mul`

更适合写成：

`Weighted Aggregation`

因为它的核心动作是：

- 读取权重
- 读取 `V`
- 做带权累加
- 构造 context vector

### 6.3 再从 Implementation Template 看

它们也不应写成同一个模板：

- `softmax` 更像 `Reduction Template`
- `context_mul` 更像 `Streaming Aggregation Template`

所以这组 case 同时说明了两件事：

1. **同属一条 workload 路线，不等于同一个 primitive**
2. **primitive 不同，template 也可能不同**

---

## 7. 和 Implementation Template 的区别到底是什么

这是当前阶段最重要的区分。

### Route Primitive 关心的是：

**它在算法主链里扮演什么角色。**

例如：

- 是在做表示变换
- 是在做关系评分
- 是在做归约归一化
- 是在做带权聚合
- 是在做逐元素融合

### Implementation Template 关心的是：

**这个角色在 GPU 上主要通过什么模板实现。**

例如：

- 是否是 dense tiled compute
- 是否是 reduction template
- 是否是 streaming accumulation
- 是否是 elementwise access

所以当前最短的区别可以写成：

**Route Primitive 描述算法角色；Implementation Template 描述硬件执行方式。**

---

## 8. 为什么这两层结构对后续 family 讨论有帮助

因为我们当前一直卡住的问题，本质上都是不同层次混在了一起。

例如：

- `attention_score` 和 `gemm_tiled`
  - 在 route 层不同
  - 在 template 层相近

- `softmax` 和 `context_mul`
  - 在 route 层同属 attention 路线
  - 但 primitive 必须拆开
  - template 也不同

所以如果我们不先把这两层拆开，就很容易出现下面这些错误：

1. 因为上层语义接近，就强行并类
2. 因为底层实现相似，就误以为 workload 角色也相同
3. 因为瓶颈项相似，就误以为它们属于同一个 family

而当前这两层结构的作用，就是先把这些问题拆开。

---

## 9. 当前最短结论

到目前为止，我们可以先稳定地说：

1. `Dense Projection/Transform`、`Pairwise Score`、`Reduction / Normalize`、`Weighted Aggregation`、`Elementwise Fusion` 这些概念更适合作为 `Route Primitive`。
2. 它们更靠近 workload 主计算路径，因为它们描述的是算法主链上的计算角色，而不是底层实现模板。
3. `Implementation Template` 则更靠近硬件实现层，因为它描述的是这些角色在 GPU 上主要通过什么执行模板被实现出来。
4. `softmax` 和 `context_mul` 这组边界 case 说明：同属一条 workload 路线，并不等于属于同一个 primitive。
5. 这两层结构不是最终 family taxonomy，而是让后续 family 定义变得更清楚的中间支撑层。
