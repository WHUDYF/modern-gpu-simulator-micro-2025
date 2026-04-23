# mini_transformer_v4 Kernel 两层结构对照表

日期：2026-04-19

## 1. 文档目的

这份文档用于把 `mini_transformer_v4` 当前已经分析过的关键 kernel，统一压缩到同一套两层结构里：

1. `Route Primitive`
2. `Implementation Template`

当前阶段的目标不是立刻给出最终 family taxonomy，而是先把讨论对象固定下来，避免后续继续混用：

- 算子语义
- 上层模块语义
- family 名字
- 瓶颈项
- 硬件执行骨架

更直接地说，这份文档要解决的是：

**当我们讨论一个 kernel 应该怎么归类时，我们到底在看它的哪一层属性。**

---

## 2. 两层结构的当前定义

### 2.1 Route Primitive

`Route Primitive` 回答的是：

**这个 kernel 在 workload 主计算路线里扮演什么角色。**

它更靠近 workload 路线本身，而不是直接等于底层实现方式。

当前我们在 `mini_transformer_v4` 上已经实际用到的 primitive 包括：

- `Dense Projection/Transform`
- `Pairwise Score`
- `Reduction / Normalize`
- `Weighted Aggregation`
- `Elementwise Fusion`

### 2.2 Implementation Template

`Implementation Template` 回答的是：

**这个 kernel 在硬件上主要通过什么执行模板实现。**

它更靠近底层硬件工作模式。

当前这几个 kernel 对应的 template 包括：

- `Dense Tiled Compute`
- `Reduction Template`
- `Streaming Aggregation Template`
- `Elementwise Template`

---

## 3. 当前最小对照表

| Kernel | Route Primitive | Implementation Template | 上层路线归属 | 当前判断 |
|---|---|---|---|---|
| `gemm_tiled` | `Dense Projection/Transform` | `Dense Tiled Compute` | `projection / FFN route` | 主计算骨架，代表 tile 化 dense multiply-accumulate 主线 |
| `attention_score` | `Pairwise Score` | `Dense Tiled Compute` | `attention readout route` | 在路线角色上不是普通 GEMM，而是生成两两 score；但实现模板与 dense tiled compute 高度接近 |
| `softmax_kernel` | `Reduction / Normalize` | `Reduction Template` | `attention readout route` | 核心是 row-wise reduction 与 normalization，不应与 `context_mul` 直接并类 |
| `context_mul` | `Weighted Aggregation` | `Streaming Aggregation Template` | `attention readout route` | 核心是读权重与 `V` 后做带权累加，属于 attention readout 的聚合侧 |
| `layernorm_kernel` | `Reduction / Normalize` | `Reduction Template` | `normalization route` | 与 `softmax` 在 primitive / template 上有共享，但 workload 路线角色不同 |
| `residual_add` | `Elementwise Fusion` | `Elementwise Template` | `residual route` | 是轻量但高频的逐元素融合路径，当前是稳定的 memory-side 样本 |

---

## 4. 逐个 kernel 的当前解释

### 4.1 `gemm_tiled`

#### Route Primitive

当前最适合写成：

`Dense Projection/Transform`

原因是：

- 它在 workload 路线里承担的是线性变换主干
- 在 transformer 中更像 projection / FFN 路径的通用计算骨架

#### Implementation Template

当前最适合写成：

`Dense Tiled Compute`

原因是：

- 有明显的 tile 化 dense multiply-accumulate 特征
- shared memory / register 复用模式清楚
- analysis card 中也显示其主解释是 compute-heavy 且 register-limited

#### 当前意义

它是当前两层结构里最稳定的锚点之一。

---

### 4.2 `attention_score`

#### Route Primitive

当前最适合写成：

`Pairwise Score`

原因是：

- 它在 attention 路径中的角色不是一般线性变换，而是生成 `Q-K` 两两关系分数
- 如果直接写成 `Dense Projection/Transform`，会丢失它在 workload 路线中的功能位置

#### Implementation Template

当前最适合写成：

`Dense Tiled Compute`

原因是：

- 它在实现层依然体现出与 `gemm_tiled` 很接近的 tiled dense compute 模板
- 当前 analysis card 也说明它与 `gemm_tiled` 共享主解释，只是在 shared memory / waves 特征上更强

#### 当前意义

`attention_score` 是最能说明“两层结构必要性”的样本：

- 在 route 层，它和 `gemm_tiled` 不同
- 在 template 层，它和 `gemm_tiled` 相近

如果没有这两层结构，它会非常难讲清楚。

---

### 4.3 `softmax_kernel`

#### Route Primitive

当前最适合写成：

`Reduction / Normalize`

原因是：

- 它的 workload 角色是把 score 变成可用权重
- 关键动作是 reduction、normalization 与同步

#### Implementation Template

当前最适合写成：

`Reduction Template`

原因是：

- 核心执行骨架是 row-wise reduction
- 其代价来源不在于 dense compute，而在于 reduction / normalization 组织方式

#### 当前意义

它不能因为同属 attention 路线就和 `context_mul` 并类。

当前最稳的结论是：

**`softmax` 与 `context_mul` 同属 `attention readout route`，但属于不同 primitive。**

---

### 4.4 `context_mul`

#### Route Primitive

当前最适合写成：

`Weighted Aggregation`

原因是：

- 它在 workload 路线中的作用是根据已经得到的权重，对 `V` 做带权聚合
- 更像 context vector 构造，而不是归约归一化

#### Implementation Template

当前最适合写成：

`Streaming Aggregation Template`

原因是：

- analysis card 明确指出它的主侧重点是 locality / L1-resident 行为
- 它的关键骨架是读权重、读值、做 streaming accumulation

#### 当前意义

它是当前 memory-side 边界里最重要的另一侧样本。  
它和 `softmax` 的分离，是当前 primitive 判据能否站住的关键之一。

---

### 4.5 `layernorm_kernel`

#### Route Primitive

当前最适合写成：

`Reduction / Normalize`

原因是：

- 它在 workload 角色上是 normalization 路径的一部分
- 与 `softmax` 一样共享 reduction / normalize 侧的 primitive

#### Implementation Template

当前最适合写成：

`Reduction Template`

原因是：

- analysis card 中它也表现出 mixed reduction / normalization behavior
- 它依然是 reduction 驱动的执行对象

#### 当前意义

`layernorm_kernel` 当前的价值不在于立刻并入哪个 family，而在于：

**它可以检验 `softmax` 所在 primitive 是否真的是一般化的 `Reduction / Normalize`，而不只是 attention 专属标签。**

---

### 4.6 `residual_add`

#### Route Primitive

当前最适合写成：

`Elementwise Fusion`

原因是：

- 它的路线角色是逐元素残差累加
- 不涉及 pairwise score、reduction 或 weighted aggregation

#### Implementation Template

当前最适合写成：

`Elementwise Template`

原因是：

- 其执行模式简单直接
- analysis card 明确显示其主导限制是 DRAM bandwidth 与 streaming access

#### 当前意义

它是当前表里最稳定的 memory-side 样本之一。  
更重要的是，它说明：

**memory-heavy 并不等于 `softmax` 这类 mixed reduction/normalize 行为。**

---

## 5. `softmax` 与 `context_mul` 的当前稳定结论

这是当前阶段最关键的一组边界结论，单独列出。

### 5.1 不能怎么分

当前我们已经明确，下面两种分法都不够稳：

#### 方式 A：按 attention 子模块直接并类

例如：

- `softmax`
- `context_mul`

都属于 attention readout，所以归为同一类

这个分法的问题是：

- 它停留在上层模块语义
- 无法解释为什么二者的执行骨架明显不同

#### 方式 B：按算子名字或 memory-side 标签模糊归类

例如：

- 它们都和 attention 有关
- 它们都带有 memory-side 特征

所以可以放在同一个 family

这个分法的问题是：

- `softmax` 的核心是 reduction / normalize
- `context_mul` 的核心是 weighted aggregation

它们共享的是 attention 路线，不是同一个 primitive。

### 5.2 当前最稳的分法

当前最稳的分法是：

- `softmax_kernel`
  - route primitive: `Reduction / Normalize`
  - implementation template: `Reduction Template`

- `context_mul`
  - route primitive: `Weighted Aggregation`
  - implementation template: `Streaming Aggregation Template`

但在更高一层，它们都属于：

`attention readout route`

### 5.3 当前这组结论的意义

这组结论说明了一个更一般的方法论事实：

**上层路线共享，不等于 primitive 共享。**

这也是我们为什么必须把 Route Primitive 和 Implementation Template 两层分开的核心原因。

---

## 6. 这张表当前能帮助回答什么问题

这份两层对照表目前最重要的作用，不是直接给出 family 定义，而是帮助我们先把下面三个问题拆开：

### 问题 1：一个 kernel 在 workload 路线中的角色是什么

这由 `Route Primitive` 回答。

### 问题 2：一个 kernel 在硬件上主要通过什么模板执行

这由 `Implementation Template` 回答。

### 问题 3：family 最终应该定义在哪一层

当前还没有完全定稿，但这张表已经表明：

- family 不能简单等于算子名
- family 不能简单等于上层模块名
- family 也不应直接等于单一瓶颈项

更可能的情况是：

**family 位于 `Route Primitive` 与 `Implementation Template` 之间，或者由两者共同决定。**

---

## 7. 当前阶段仍未完全解决的问题

这张表虽然能明显提高讨论稳定性，但还没有解决全部问题。

当前仍然保留以下开放点：

### 7.1 `attention_score` 和 `gemm_tiled` 最终是否应该落入同一 family

当前可以稳定说：

- route primitive 不同
- implementation template 相近

但 family 是否以 route 为主、以 template 为主，仍需后续 protocol 来定。

### 7.2 `layernorm_kernel` 是不是只应视为 `Reduction / Normalize` 的一般化样本

当前倾向是：

- 可以作为 general reduction / normalize primitive 的检验样本

但它是否要形成独立 mixed/outlier 讨论线，仍未完全定稿。

### 7.3 `family` 是否需要第三层

如果后续发现：

- route primitive 太抽象
- template 又太底层

那么 family 很可能要成为两者之间的组合层，而不是直接等于其中任一层。

---

## 8. 当前阶段的最简结论

到目前为止，我们已经能比较稳地说：

1. `mini_transformer_v4` 的关键 kernel 不能再用单层分类去理解。
2. 当前最稳的工作结构是：
   - `Route Primitive`
   - `Implementation Template`
3. `attention_score` 和 `gemm_tiled` 说明：
   - route 角色不同
   - template 可相近
4. `softmax_kernel` 和 `context_mul` 说明：
   - 同属 attention 路线
   - 但 primitive 必须拆开
5. 这张表的作用不是给出最终 family，而是给后续 family boundary protocol 提供稳定底稿。

因此，当前阶段最自然的下一步不是继续泛谈 family，而是：

**基于这张两层对照表，继续写出 family selection / boundary protocol。**
