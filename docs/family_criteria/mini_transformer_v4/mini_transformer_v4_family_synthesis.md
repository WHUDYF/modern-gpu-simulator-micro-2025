# mini_transformer_v4 Family Synthesis

## Goal

这份文档是第一版 `squash + batch` family 判据原型的主交付物。

它的目标不是再重复列出单个 kernel 指标，而是回答四个更上游的问题：

1. 这套 family 判据框架是如何工作的
2. 为什么第一轮要优先处理边界 case
3. 这套结构为什么能把后续 simulator 验证从“逐 kernel 猜”压缩成“少数验证主线 + 少量例外”
4. 第一版目前稳定得出了什么，哪些地方仍然明确不稳定

## Why Boundary Cases Come First

这套方法没有从“先把所有卡片铺满”开始，而是先从两组最容易混淆的边界 case 开始：

- `gemm_tiled` vs `attention_score`
- `softmax_kernel` vs `context_mul`

原因在于，family 判据在当前阶段还没有完全成熟。  
如果一开始就先写满所有 analysis cards，很容易把不成熟的判据硬编码进卡片里。  
相反，边界 case 更适合作为第一轮方法生长的主舞台，因为它们能最快暴露：

- 哪些共享点其实不足以支撑并类
- 哪些区分点才真正决定 family 边界
- analysis card 里哪些字段必须保留

因此，第一轮采用了：

**边界 case -> analysis cards -> family cards -> synthesis**

而不是：

**analysis cards -> family cards -> synthesis**

## How the Family Criteria Framework Works

第一版框架当前更准确地说，可以拆成两层：

- **data-path family 层**
- **intra-family tuning 层**

前者负责定义“谁属于同一工作模式”，后者负责回答“在同一 family 内应该优先调什么、怎样找平衡点”。

### A. Data-Path Family 层

这一层的目标不是先看单个瓶颈，而是先判断 kernel 是否共享同一类硬件工作模式。  
当前我们把这层进一步拆成三步。

### 1. Execution Mode 粗分

先判断 kernel 更接近：

- `compute-heavy`
- `memory-heavy`
- `mixed`

这一层只负责粗分，不直接决定 family 边界。

### 2. Data Movement / Cooperation / Coupling 边界

真正决定 family 边界的，不是最终瓶颈项本身，而是更靠近硬件工作模式的三类因素：

- 数据移动方式
- 线程协作方式
- 计算-访存耦合方式

例如：

- 是否先搬到 shared memory 再复用
- 是否以 block 内 reduction 为主
- 是否主要是 thread-local streaming accumulation
- 是否属于“先搬数据、再做密集乘加”的模板

这一层回答的是：

**为什么这些 kernel 可以被看成同一条 data path family。**

### 3. Family / Outlier 收束

当共享点和区分点都被边界 case 分析过后：

- 能稳定共享解释的对象，进入 family
- 暂时无法稳定吸收的对象，保留为 outlier

因此，这套方法不是“按算子名分组”，而是：

**按共享执行模板 / 数据通路分组。**

### B. Intra-Family Tuning 层

当 family 已经由工作模式定义出来以后，才进入下一层：

**同一 family 内应该如何调参。**

在这一层，主导限制项才变得重要，例如：

- register / occupancy
- DRAM bandwidth
- cache-capacity / DRAM-pressure
- locality / L1-resident

也就是说，当前方法不再把瓶颈项直接拿来定义 family，而是把它们作为：

**family 内部调参与平衡优化的依据。**

这意味着：

- family 负责“结构归属”
- dominant bottleneck 负责“后续调参”

这个分层能避免一个常见错误：

如果一开始就用单个瓶颈项定义 family，那么像 `gemm_tiled` 与 `attention_score` 这种共享工作模式、但次级限制项不同的 kernel，会被拆得过早，失去作为同一 data-path family 的价值。

## Version-1 Output Structure

当前第一版已经形成了下面几层产物：

### Boundary Cases

- `gemm_tiled vs attention_score`
- `softmax_kernel vs context_mul`

这两份文档给出的不是绝对裁决，而是分级结论：

- `gemm_tiled vs attention_score`: **弱共享**
- `softmax_kernel vs context_mul`: **边界未定（倾向拆分）**

### Analysis Cards

六个代表 kernel 的 analysis cards 都已回填完成，并带有：

- 统一模板
- 明确证据引用
- 边界说明 / 不确定性说明

所以 analysis cards 在这里不是工程底稿，而是：

**由边界 case 反推出来的证据层对象。**

### Family Cards

第一版当前形成的 family / outlier 结构是：

- `compute-heavy -> register-limited`
- `memory-heavy -> dram-dominated`
- `mixed -> cache-capacity-sensitive`
- `mixed -> locality-dominated`
- `outliers`：当前主要保留 `layernorm_kernel`

## What This Changes for Simulator Validation

这套结构最重要的意义，不是“名字更好看”，而是它改变了后续验证组织方式。

在没有 family 结构时，后续 simulator 验证更像：

- 每个 kernel 单独看
- 每个 kernel 单独猜验证主线
- 每个 kernel 单独组织解释

这会导致：

- 工作量高
- 解释不一致
- 相邻 kernel 之间的共享主线难以复用

有了第一版 family 结构之后，后续验证可以变成：

- 先从 family 级别出发考虑共享验证主线
- 再把真正的边界样本或 outlier 单独保留

因此，family 先于处方。

更准确地说：

**family 的作用是先定义“哪些对象值得共享解释”，再决定“哪些对象值得共享验证主线”。**

而且这里有一个新的关键分层：

- family 不是由“当前瓶颈是什么”直接决定的
- family 先由共享数据通路 / 执行模板定义
- 瓶颈项则在下一层用于 family 内部调参

因此，我们现在不再把问题写成：

> 哪些 kernel 的瓶颈相同，所以它们应当归为一类

而是更准确地写成：

> 哪些 kernel 共享同一工作模式，因此应先归为同一 family；随后再利用它们在限制项上的差异去寻找 family 内部最平衡的调参点

这就是它为什么能把后续 simulator 验证从“逐 kernel 猜”压缩成“少数验证主线 + 少量例外”。

## Version-1 Limits

第一版仍然有清晰边界：

- 不展开 `delta`
- 不输出具体 simulator 参数处方
- 不定量声称节省了多少验证时间或成本
- 不声称当前 family 规则已经稳定可迁移

此外，下面这些点仍然明确处于未完全稳定状态：

- `attention_score` 与 `gemm_tiled` 虽共享主解释，但 shared-memory 差异是否足以在后续拆族，仍未完全定案
- `softmax_kernel` 与 `context_mul` 都落在 memory-side 边界，但是否最终形成两个稳定 mixed 子类，还需后续轮次继续验证
- `layernorm_kernel` 目前保留为 outlier，是因为第一版更重视边界保真，而非强行追求结构完整

## Current Methodological Takeaway

第一版最重要的结论不是“已经得到了完美 family”，而是：

1. `batch` 的核心不是按算子名分组，而是识别共享架构解释
2. family 判据不能直接从单卡静态归纳，必须先经过边界 case 的逼近
3. family 由共享工作模式定义，而主导限制项应当留给 family 内调参与平衡优化
4. analysis cards 与 family cards 只是方法生长过程中的支撑层
5. synthesis 才是第一版最重要的主交付物，因为它真正说明了方法是如何工作的

所以，当前这轮原型真正证明的是：

**从 workload 到 simulator 的端到端视线，确实可以先通过“边界 case -> data-path family -> intra-family tuning”这条路径长出来。**
