# mini_transformer_v4 方法线阶段分析报告

日期：2026-04-19

## 1. 本轮迭代的目标

本轮迭代的目标不是继续堆叠零散想法，而是把当前方法线收敛成一套已经能够稳定复述、稳定拆解、并能继续向实验推进的结构。

更具体地说，本轮主要回答了五个问题：

1. 我们这项工作的主问题到底是什么。
2. `squash`、`family`、`representative object` 三者之间是什么关系。
3. family 不能只按算子名分组之后，应该按什么结构重新定义。
4. `softmax` 和 `context_mul` 这类最关键边界 case 应该如何拆开。
5. 后续真正进入 simulator lane 的对象到底应该是什么。

---

## 2. 本轮已经完成的核心收敛

### 2.1 主问题已经稳定

当前最稳的主问题表述是：

**我们希望提出一套从 workload 行为到 simulator 分析对象的结构化接口。**

这意味着我们当前的工作重点已经不再是：

- 单独做 sampled simulation
- 单独做 representative kernel selection
- 单独做 profile 结果归纳

而是要解决：

**复杂 workload 应该如何被压缩成 simulator 可以承接、比较、调参与验证的结构化对象。**

这一定义相比之前“做 family 分类”更强，也更适合作为后续汇报与论文主线。

### 2.2 方法主线已经从散点想法变成了可执行链条

当前方法主线已经可以稳定写成：

`workload -> squash -> family / execution template -> representative execution regime -> simulator lane / tuning`

这条链条的意义在于：

- `squash` 负责时间压缩
- `family` 负责结构压缩
- `representative execution regime` 负责把结构对象真正落到 simulator 可处理的调参单元

也就是说，我们已经不再停留在“先分组看看”的阶段，而是已经有了从 workload 到 simulator 的连续中间层。

### 2.3 family 的定义已经从“标签系统”转为“模拟组织对象”

本轮最关键的认识之一是：

**family 不是算子名标签，也不是模块名标签，而是一组在相同 phase 上下文中能够共享同一条 simulator reasoning lane 的对象集合。**

这一定义比“按算子分类”强很多，因为它把 family 的价值直接绑定到了后续 simulator 组织与调参复用上。

这也意味着：

- family 不能脱离 phase 单独定义
- family 不能只按上层 attention / FFN 语义定义
- family 不能只按单个瓶颈项定义

### 2.4 两层结构已经形成

为了替代“按算子分类”这种过薄的定义，本轮已经把 kernel 的讨论对象固定到两层结构上：

#### 第一层：Route Primitive

回答的是：

**这个 kernel 在 workload 主计算路径中扮演什么角色。**

当前原型已使用的 primitive 包括：

- `Dense Projection/Transform`
- `Pairwise Score`
- `Reduction / Normalize`
- `Weighted Aggregation`
- `Elementwise Fusion`

#### 第二层：Hardware Execution Template

回答的是：

**这个 kernel 在 GPU 上主要通过什么执行骨架实现。**

当前原型已使用的 template 包括：

- `Dense Tiled Compute`
- `Reduction Template`
- `Streaming Aggregation Template`
- `Elementwise Template`

这两层结构的意义非常明确：

- 有些对象 route 不同，但 template 相近
- 有些对象 route 相邻，但 template 完全不同

如果不把这两层拆开，family 判据会持续摇摆。

### 2.5 两个最关键的边界 case 已经获得稳定结论

#### Case A：`gemm_tiled` vs `attention_score`

当前最稳的结论是：

**弱共享 / 边界候选。**

原因是：

- 在 `Route Primitive` 上不同
  - `Dense Projection/Transform`
  - `Pairwise Score`
- 在 `Hardware Execution Template` 上相近
  - 都接近 `Dense Tiled Compute`
- 在 baseline / batch / full evidence 中又同时表现出：
  - 共享寄存器主机制
  - 但 `attention_score` 仍保留更强的 shared-memory / waves 特征

因此，它们可以共享部分 simulator reasoning，但不应被粗暴视为完全同质对象。

#### Case B：`softmax_kernel` vs `context_mul`

当前最稳的结论是：

**应拆开，不建议合并为同一 primitive。**

原因是：

- 同属 attention 上层路线
- 但 `Route Primitive` 不同
  - `Reduction / Normalize`
  - `Weighted Aggregation`
- `Hardware Execution Template` 也不同
  - `Reduction Template`
  - `Streaming Aggregation Template`
- memory-side 的主机制也不同
  - `softmax` 更接近 cache-capacity / DRAM-pressure
  - `context_mul` 更接近 locality-dominated / L1-resident

这说明“都在 attention 里”并不足以支撑 family 合并。

### 2.6 代表对象已经从“representative kernel”升级为“representative execution regime”

这是本轮最重要的新收敛。

当前已经明确：

**后续进入 simulator lane 的单位，不应只是“一个代表 kernel”，而应是“一个代表执行区间”。**

其原因在于：

1. 同名 kernel 不一定属于同一调参对象。
2. 同一 family 内部仍可能存在多个稳定工作区间。
3. 真正能复用 simulator reasoning 的，不是名字，而是相近的 phase / route / template / shape / resource 行为组合。

因此，当前最合理的 simulator 接口对象已经变成：

**representative execution regime**

而不是：

**representative kernel**

这一步很关键，因为它把 family 和 simulator lane 之间最后缺失的那层结构补上了。

---

## 3. 当前已经稳定的工作主张

到本轮结束，可以认为下面几条已经进入“稳定结论”状态。

### 3.1 我们压缩的不是 sample 本身，而是“需要被单独理解和单独调参的对象”

这是当前与 PKA / Sieve 最稳的区分。

- `PKA / Sieve`
  - 更偏向压缩 `simulation samples`
  - 关注的是“模哪些样本、模多少样本”

- 我们当前的方法
  - 压缩的是 `phase`、`family`、`execution regime`
  - 关注的是“按什么结构理解 workload，并减少后续重复模拟与重复调参对象”

因此，我们的方法价值不应再表述成：

**更好的 sampling**

而应表述成：

**更好的 workload-to-simulator structural interface**

### 3.2 `squash` 和 `family` 不是重复功能

它们当前的边界已经比较清楚：

- `squash`
  - 负责时间组织
  - 负责找稳定 phase

- `family`
  - 负责结构组织
  - 负责识别共享执行模式

两者共同构成“先按时间压，再按结构压”的方法线，而不是互相替代。

### 3.3 真正的 simulator lane 单位已经不再模糊

当前最稳的接口分层是：

- phase：时间上下文
- route primitive：算法路径角色
- hardware execution template：GPU 执行骨架
- representative execution regime：最终 simulator 代表对象

这说明我们的方法已经从“如何讨论 kernel”推进到了“如何组织后续模拟单元”。

---

## 4. 当前仍然没有完成的关键缺口

虽然方法线已经成立，但当前距离“可做定量实验”和“可写完整论文方法”仍有几块明显缺口。

### 4.1 phase 还没有和具体 regime ledger 完整打通

当前我们已经有：

- `squash` 的 phase-level 思路
- regime 的提取协议

但还没有把它们彻底打通成一张完整表，例如：

- 每个 phase 里有哪些 regime
- 每个 regime 对应哪些 kernel 实例
- 每个 regime 的覆盖范围与权重

换句话说，我们已经有 protocol，但还没有完整的 regime ledger。

### 4.2 shape / size regime 仍然是概念化的，还没有做定量切分

当前文档里已经明确 shape / size 是 regime 的关键字段，但还缺：

- 如何定义 shape bucket
- 如何判断两个 shape 属于同一 regime
- 不同 M / N / K、sequence length、head dim、batch size 的切分边界

如果这一步不补，representative execution regime 仍然停留在概念层。

### 4.3 权重体系还没有正式落成

当前已经有三层权重想法：

- `coverage weight`
- `time weight`
- `decision weight`

但现在仍然主要是概念澄清，没有形成：

- 可计算的字段
- 统一记录格式
- 后续调参中的实际使用规则

### 4.4 simulator lane 还没有被实例化成具体实验流程

当前我们已经知道“该把什么送进去”，但还没完成：

- 具体如何生成 representative regime 输入
- 如何让 simulator 只跑这些 regime
- 如何把 regime 级结果回写到 phase / workload 级判断

也就是说，结构接口已经成型，但 simulator 对接流程还没落地。

### 4.5 还没有做定量比较来证明能压缩模拟或调参成本

当前最需要补的一类证据是：

- 与 naive per-kernel 分析相比，减少了多少独立对象
- 与只做 representative sampling 相比，减少了多少重复调参对象
- 在保持主要结论稳定的前提下，缩短了多少模拟 / 分析时间

没有这组量化实验，当前方法仍然更像“强方法框架”，还不是“闭环验证完成的系统”。

---

## 5. 当前成熟度判断

如果把当前方法分成三个层次，那么本轮结束后的成熟度大致如下。

### 5.1 概念层：较成熟

已经比较稳定的内容包括：

- 主问题定义
- 与 related work 的区分
- `squash -> family -> regime` 主线
- 两层结构
- 边界协议
- representative execution regime 的必要性

这一层已经足以支撑：

- 组会讲清楚工作价值
- 向导师说明这不是异想天开
- 形成较稳定的论文方法章节骨架

### 5.2 协议层：中等成熟

已经较稳，但还需要补实证绑定的内容包括：

- family boundary protocol
- regime extraction protocol
- 当前 6 个 kernel 的 route / template 对照

这一层已经能支撑方法论讨论，但还需要进一步和实际 trace / shape / weight 表格绑定。

### 5.3 实验层：仍然偏早期

当前最缺的是：

- regime ledger
- shape regime quantization
- weight ledger
- simulator 对接与复原流程
- 定量压缩收益验证

因此，当前状态更准确地说是：

**方法框架已经成立，实验闭环还没有完成。**

---

## 6. 下一轮最值得做的工作

从当前状态出发，下一轮最应该做的不是继续扩张概念，而是把结构接口推进到实验对象层。

### 6.1 先建立一张 `phase -> primitive -> template -> regime` 总表

这是当前最高优先级工作。

目标是把当前已有的协议和六个 kernel 的理解，压成一张能直接驱动后续实验的表。

这张表至少应包含：

- phase id
- kernel / invocation group
- route primitive
- hardware template
- shape regime
- resource signature
- coverage weight
- time weight
- 当前决策备注

这一步完成后，方法就能从“文字协议”进一步变成“实验账本”。

### 6.2 对 Transformer 主链做第一版 regime ledger

优先对象建议仍然是：

`projection -> attention_score -> softmax -> context_mul -> residual / norm -> FFN`

原因是：

- 它已经覆盖了当前最关键 primitive
- 边界问题已经暴露得很充分
- 继续在这条主链内做硬，比盲目扩 workload 更有价值

### 6.3 开始把 `decision weight` 从定性推进到半定量

建议第一版不要追求严格数学化，而是先做“半定量规则”：

- 是否位于主计算路径
- 是否支配某 phase 的主要时间
- 是否对关键硬件参数敏感
- 是否代表一个不可替代的 template

这样可以先得到可操作的权重排序，而不是停留在概念讨论。

### 6.4 设计第一组定量验证问题

建议后续实验不要一开始就追求完整系统，而是先回答三类最核心问题：

1. 从全 trace 到 regime ledger，独立分析对象减少了多少。
2. 用 regime 作为 simulator lane 输入后，需要单独调参的对象减少了多少。
3. 在结论基本一致的前提下，是否能减少模拟时间或分析时间。

这三类问题比一开始就做完整 benchmark 更适合作为第一轮定量验证入口。

---

## 7. 对当前工作的总体判断

从本轮迭代结果看，当前工作已经不是“零散直觉”，而是一条已经可以持续推进的方法线。

最重要的三个收获是：

1. 主问题已经站稳。
2. family 的定义已经摆脱“按算子分类”的薄弱状态。
3. simulator 对接对象已经从模糊的 representative kernel 收敛为更稳的 representative execution regime。

这说明当前工作的价值至少已经可以稳定表述为：

**不是再做一个 sampling trick，而是提出一种从 workload 行为到 simulator 组织对象的结构化接口。**

如果下一轮能够补上：

- regime ledger
- 权重记录
- simulator 对接流程
- 第一组定量验证

那么这条方法线就会从“概念成立”进入“实验闭环成型”阶段。

---

## 8. 本轮新增或关键相关文档

本轮分析主要落在以下文档上：

- [current-goal-and-method-clarification-2026-04-19.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/current-goal-and-method-clarification-2026-04-19.md)
- [family_selection_boundary_protocol.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/family_selection_boundary_protocol.md)
- [representative_execution_regime_protocol.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/representative_execution_regime_protocol.md)
- [mini_transformer_v4_route_primitive_template_table.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/mini_transformer_v4_route_primitive_template_table.md)
- [gemm_tiled-vs-attention_score.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/boundary_cases/gemm_tiled-vs-attention_score.md)
- [softmax_kernel-vs-context_mul.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/family_criteria/mini_transformer_v4/boundary_cases/softmax_kernel-vs-context_mul.md)
- [current-family-interface-summary-2026-04-19.md](/home/dyf/modern-gpu-simulator-micro-2025/docs/current-family-interface-summary-2026-04-19.md)
