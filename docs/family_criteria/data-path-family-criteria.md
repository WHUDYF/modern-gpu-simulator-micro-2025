# Data-Path Family Criteria

日期：2026-04-19

## 1. 目标

这份文档的目标是形式化说明：

**什么叫同一条 data-path family，以及 family 判据应该如何选择。**

它要解决的问题不是：

- 哪些 kernel 在统计上像
- 哪些 kernel 名字相近
- 哪些 kernel 当前瓶颈类似

而是：

**哪些 kernel 在 GPU 内部共享同一类工作模式，因此应该先被归入同一个 family。**

---

## 2. 为什么需要 data-path family

如果我们直接用：

- 算子名
- profile 相似度
- 当前瓶颈项

来做分组，会有几个问题：

1. 同算子名不一定共享同一执行模板
2. 统计相似性不一定意味着共享同一硬件工作模式
3. 单个瓶颈项往往只描述当前限制，而不是完整工作模式
4. 过早按瓶颈拆分，会把本应属于同一工作模式的 kernel 过早拆散

因此，我们需要一个更底层的分组对象：

**data-path family**

---

## 3. Data-Path Family 的核心定义

一个 kernel 是否属于某个 data-path family，不由“像不像”决定，而由：

**它是否共享同一套硬件工作模式** 决定。

这里的“硬件工作模式”主要包括三层：

### 3.1 数据移动方式

问的是：

- 数据是怎么被搬进计算单元附近的？
- 是 global streaming
- shared-memory tiling
- cache / locality reuse
- 还是 thread-local 顺序扫描

这一层关注的是：

**数据是通过哪种通路进入计算。**

### 3.2 线程协作方式

问的是：

- 线程之间怎么合作？
- 是 block 内 cooperative load
- block 内 reduction
- warp/block 频繁同步
- 还是每线程独立积累

这一层关注的是：

**计算是如何在 SM 内部分工与同步的。**

### 3.3 计算-访存耦合方式

问的是：

- 计算和访存是怎么交织的？
- 是先搬数据再密集算
- 边读边算的 streaming accumulation
- reduction + normalize
- 或其他混合形式

这一层关注的是：

**该 kernel 在执行时的整体工作模板。**

---

## 4. 为什么瓶颈项不直接定义 family

我们明确把下面这些东西放到 family 判据之后：

- register pressure
- occupancy limit
- DRAM bandwidth pressure
- cache-capacity pressure
- shared-memory bank pressure

原因是：

### 4.1 瓶颈项描述的是“当前最紧的限制”

但 family 想定义的是：

**更稳定的工作模式归属。**

### 4.2 同一工作模式下可以有不同次级瓶颈

例如：

- `gemm_tiled`
- `attention_score`

它们可能共享同一执行模板，但在 shared-memory 使用和次级限制项上并不完全相同。  
如果一开始就按瓶颈拆分，它们会被过早拆开。

### 4.3 瓶颈项更适合服务 tuning

瓶颈项不是没用，而是更适合放到下一层：

**intra-family tuning**

---

## 5. Family 与 Tuning 的分工

### 5.1 Family 负责什么

family 负责回答：

**谁共享同一条数据通路 / 执行模板。**

### 5.2 Intra-family tuning 负责什么

intra-family tuning 负责回答：

**在同一 family 内，不同 kernel 的限制项差异该如何平衡。**

这里才轮到：

- register / occupancy
- DRAM bandwidth
- cache-capacity
- locality
- shared memory

进入调参目标。

所以更准确的结构是：

**family 先定义工作模式，tuning 再处理限制项。**

---

## 6. 和统计聚类方法的区别

PKA / Sieve 这类方法的 grouping 更偏：

- profiling similarity
- stratification stability
- representative sample selection

而 data-path family 更偏：

- shared execution template
- shared hardware data path
- shared resource interaction structure

因此：

- 他们的 grouping 主要服务 sampling
- 我们的 family 主要服务 tuning

换句话说：

- 他们压缩的是 sample
- 我们压缩的是 hardware optimization problem

---

## 7. 当前版本的定位

当前版本不声称：

- 这套判据已经最终完备
- 所有 family 都已经稳定
- 这套规则已经可直接泛化到任意 workload

当前更稳的表述是：

**这是第一版 data-path family 判据框架，它通过关键 boundary cases 被逼出来，并允许后续被新的 case 推翻、修订或扩展。**

---

## 8. 当前最关键的使用原则

可以压缩成一句话：

**先按共享工作模式构造 family，再按主导限制项做 family 内部调参与平衡优化。**

---

## 9. 当前阶段的意义

这份文档的价值不在于一次性给出最终规则，而在于把两个原来容易混淆的问题明确拆开：

1. **family 是什么**
2. **family 内部该怎么调**

一旦这两层被分开，后续方法论就会更稳：

- `boundary case` 用于逼出 family 边界
- `analysis card` 用于压缩判据证据
- `family card` 用于形成结构化解释
- `intra-family tuning` 再处理同一 family 内部的平衡点

这也使得从 workload 到 simulator 的端到端视线，不再停留在“谁和谁像”，而上升为：

**谁共享同一工作模式，谁应共享同一类优化组织方式。**
