# 硬件层执行模板解释文档

日期：2026-04-19

## 1. 文档目的

这份文档用于单独解释当前讨论中的几类**硬件层执行模板**，并说明它们为什么属于“硬件层面”的概念，而不是直接属于 workload 主计算路径的概念。

当前这份文档主要覆盖以下几类模板：

- `Dense Tiled Compute`
- `Reduction Template`
- `Streaming Aggregation Template`
- `Elementwise Template`

这份文档的作用不是给出最终全领域执行模板 taxonomy，而是先把 `mini_transformer_v4` 当前最常用、最稳定的几类 GPU 执行骨架讲清楚，方便后续继续做 family 归类与 protocol 收束。

---

## 2. 什么叫“硬件层执行模板”

所谓“硬件层执行模板”，当前最简洁的理解是：

**它描述的是一个 kernel 在 GPU 上主要以什么方式执行。**

它不直接关心：

- 这个 kernel 在算法里属于 attention 还是 FFN
- 这个 kernel 在 workload 路线里承担什么语义角色
- 它是“关系计算”还是“表示变换”

它真正关心的是：

- 数据怎么从 global memory 搬到片上
- 线程之间怎么协作
- 是否依赖 shared memory
- 是否依赖 warp / block reduction
- 是否主要是流式读取和累加
- 是否只是逐元素读写

所以，硬件层执行模板更像是在回答：

**GPU 实际上是怎么把这类 kernel 跑起来的。**

---

## 3. 为什么说这一层更靠近硬件，而不是更靠近 workload

原因很直接：

这一层描述的不是“算法做什么”，而是“GPU 怎么执行”。

例如，一个 kernel 在 workload 路线里可能属于：

- `Pairwise Score`
- `Reduction / Normalize`
- `Weighted Aggregation`

但它在硬件层面上，可能分别体现为：

- `Dense Tiled Compute`
- `Reduction Template`
- `Streaming Aggregation Template`

前者关心的是**计算角色**，后者关心的是**执行骨架**。

因此：

- `Route Primitive` 更靠近算法主链
- `Hardware Execution Template` 更靠近 GPU 运行方式

---

## 4. `Dense Tiled Compute`

### 4.1 直觉解释

`Dense Tiled Compute` 的最直觉理解是：

**把大计算切成小块，搬到片上，再反复做密集乘加。**

这是 GPU 上最典型的一类执行方式之一。

### 4.2 它在硬件上通常长什么样

这类模板通常包含下面这些典型动作：

- 把 global memory 中的数据切成 tile
- 先搬到 shared memory
- 再从 shared memory 提供给寄存器 / ALU / Tensor Core
- 多个线程协同完成同一个 tile 的计算
- 尽量在片上复用数据，减少回 DRAM 的次数

### 4.3 它最关心哪些硬件问题

当一个 kernel 属于 `Dense Tiled Compute` 时，通常会更敏感于：

- shared memory 布局是否合理
- register 压力是否过大
- occupancy 是否被寄存器限制
- tile 尺寸是否合适
- compute pipeline / Tensor Core 是否能吃满

### 4.4 典型 kernel

在当前 `mini_transformer_v4` 里，最典型的例子包括：

- `gemm_tiled`
- `attention_score` 的 tiled dense 实现

### 4.5 为什么它是硬件层概念

因为它强调的是：

**数据如何分块、如何搬运、如何复用、线程如何协同完成 dense multiply-accumulate。**

它并不关心这个 dense compute 是在做 QKV projection，还是在做 pairwise score。

---

## 5. `Reduction Template`

### 5.1 直觉解释

`Reduction Template` 的最直觉理解是：

**很多线程先分别算出局部结果，再把结果一层层收拢成一个统计量。**

这类模板在 GPU 上非常常见，因为很多操作都要：

- 求和
- 求最大值
- 求均值
- 求方差

### 5.2 它在硬件上通常长什么样

典型模式通常包括：

- 每个线程先处理一部分输入
- 形成 partial result
- 在 warp 内或 block 内做 reduction
- 借助 shared memory 或 warp shuffle 合并结果
- 把最后的统计量广播回去，继续后续 normalize 或写回

### 5.3 它最关心哪些硬件问题

当一个 kernel 属于 `Reduction Template` 时，通常更敏感于：

- 线程同步开销
- reduction tree 的组织方式
- warp divergence
- shared memory 与 warp shuffle 的取舍
- reduction 与回写之间的额外访存成本

### 5.4 典型 kernel

当前最典型的例子包括：

- `softmax_kernel`
- `layernorm_kernel`

### 5.5 为什么它是硬件层概念

因为它强调的是：

**跨线程如何把局部结果收拢起来、如何同步、如何完成 reduction tree。**

这已经是非常典型的 GPU 执行组织问题，而不再只是数学上的“归一化”概念。

---

## 6. `Streaming Aggregation Template`

### 6.1 直觉解释

`Streaming Aggregation Template` 的最直觉理解是：

**数据一边读进来，一边按规则累加到输出上。**

它和 `Dense Tiled Compute` 最大的不同在于：

- 它通常不强调把一个 tile 搬上来反复复用
- 它更像顺序读取、逐步累加、形成输出

### 6.2 它在硬件上通常长什么样

典型模式通常包括：

- 线程读取权重
- 线程读取 value / input
- 在线程本地寄存器中维护 accumulator
- 随着数据流入，逐步把结果累加到输出上
- 更依赖 locality、cache 行为和访问顺序

### 6.3 它最关心哪些硬件问题

当一个 kernel 属于 `Streaming Aggregation Template` 时，通常更敏感于：

- 访问是否连续
- L1 / cache 命中情况
- memory latency
- thread-local accumulator 的压力
- 累加链条是否过长

### 6.4 典型 kernel

当前最典型的例子包括：

- `context_mul`

更广义地说，一些 graph aggregation / sequence aggregation 也可能落到这一模板。

### 6.5 为什么它是硬件层概念

因为它强调的是：

**流式读取、逐步累加、依赖 locality 和 accumulator 的执行方式。**

它并不关心上层语义上这是不是 attention，也不关心它是不是在“聚合信息”。

---

## 7. `Elementwise Template`

### 7.1 直觉解释

`Elementwise Template` 的最直觉理解是：

**每个线程独立地读自己的输入，做简单计算，再把结果写回。**

这类模板通常最轻，但在 workload 中往往出现得很多。

### 7.2 它在硬件上通常长什么样

典型模式通常包括：

- 每个线程负责一小段元素
- 几乎不需要复杂线程协作
- 很少使用 shared memory
- 主要是 load -> compute -> store

### 7.3 它最关心哪些硬件问题

当一个 kernel 属于 `Elementwise Template` 时，通常更敏感于：

- memory bandwidth
- global memory coalescing
- cache 命中情况
- 读写是否连续
- 访存相对计算的占比

### 7.4 典型 kernel

当前最典型的例子包括：

- `residual_add`

### 7.5 为什么它是硬件层概念

因为它强调的是：

**线程独立、访存主导、计算轻量的执行方式。**

这是一种执行模式，而不是 workload 路线中的高层语义角色。

---

## 8. 这几类模板之间怎么区分

当前最适合的区分方式不是看算子名字，而是看下面三个问题：

### 8.1 数据是怎么被消费的

- 是先搬上片上再复用？
  - 更像 `Dense Tiled Compute`
- 是边读边收拢成统计量？
  - 更像 `Reduction Template`
- 是边读边加到输出？
  - 更像 `Streaming Aggregation Template`
- 是每线程独立读写？
  - 更像 `Elementwise Template`

### 8.2 线程协作是怎么发生的

- 多线程围绕一个 tile 协同计算？
  - `Dense Tiled Compute`
- 多线程共同完成 reduction？
  - `Reduction Template`
- 线程主要各自维护 accumulator？
  - `Streaming Aggregation Template`
- 线程基本独立？
  - `Elementwise Template`

### 8.3 关键资源压力来自哪里

- shared memory / register / compute pipeline
  - `Dense Tiled Compute`
- synchronization / reduction tree
  - `Reduction Template`
- locality / cache / accumulator
  - `Streaming Aggregation Template`
- bandwidth / load-store
  - `Elementwise Template`

---

## 9. 和 Route Primitive 的关系

这一点非常关键。

### 9.1 Route Primitive 回答什么

它回答的是：

**这个 kernel 在 workload 主计算路径里扮演什么角色。**

例如：

- `Dense Projection/Transform`
- `Pairwise Score`
- `Reduction / Normalize`
- `Weighted Aggregation`
- `Elementwise Fusion`

### 9.2 Hardware Execution Template 回答什么

它回答的是：

**这个角色在 GPU 上主要通过什么执行模板实现。**

例如：

- `Dense Tiled Compute`
- `Reduction Template`
- `Streaming Aggregation Template`
- `Elementwise Template`

### 9.3 为什么这两层必须分开

因为一个 kernel：

- 在 workload 路线里的角色，和
- 在 GPU 上的实现方式

不一定是一一对应的。

最典型的例子是：

#### `attention_score`

- Route Primitive:
  - `Pairwise Score`
- Hardware Template:
  - `Dense Tiled Compute`

也就是说：

- 在算法里，它是在做关系评分
- 在硬件上，它却可能通过 tiled dense compute 跑出来

再例如：

#### `softmax_kernel`

- Route Primitive:
  - `Reduction / Normalize`
- Hardware Template:
  - `Reduction Template`

#### `context_mul`

- Route Primitive:
  - `Weighted Aggregation`
- Hardware Template:
  - `Streaming Aggregation Template`

所以当前最短的区别可以写成：

**Route Primitive 描述算法角色；Hardware Execution Template 描述 GPU 执行方式。**

---

## 10. 当前这组模板的边界

需要明确的是，当前这组模板不是最终全领域版本。

它们目前主要用于支撑：

- `mini_transformer_v4`
- Transformer 主链
- 当前已经讨论过的边界 case

如果后续扩展到更广的主流 AI workload，仍然很可能需要补充：

- `Sparse Gather / Lookup`
- `Sparse Scatter / Routing`
- `Spatial Local Stencil`

因此，更稳的说法不是：

**我们已经得到了主流 GPU workload 的完整硬件模板分类。**

而是：

**我们在当前方法原型上，已经稳定识别出一组最常用的硬件层执行模板。**

---

## 11. 当前最短结论

到目前为止，我们可以比较稳地说：

1. `Dense Tiled Compute`、`Reduction Template`、`Streaming Aggregation Template`、`Elementwise Template` 这些概念属于硬件层执行模板。
2. 它们之所以属于硬件层，是因为它们描述的是 GPU 实际如何搬运数据、组织线程协作和消耗资源。
3. 它们不直接描述 workload 路线中的语义角色；后者应由 `Route Primitive` 来描述。
4. 同一个 workload 路线角色，不一定对应唯一的硬件模板；反过来，不同 workload 角色也可能共享某种底层执行模板。
5. 当前这组模板已经足以支撑 `mini_transformer_v4` 的第一版方法原型，但还不是最终全领域 taxonomy。
