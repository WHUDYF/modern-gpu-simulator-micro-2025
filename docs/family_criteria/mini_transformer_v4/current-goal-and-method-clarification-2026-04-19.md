# 当前目标与方法澄清

日期：2026-04-19

## 1. 文档目的

这份文档用于把我们当前阶段已经明确的方法目标、核心对象和后续推进路径固定下来。

当前最重要的变化不是又增加了一组零散想法，而是：

**我们已经基本明确，这项工作的主问题、方法主线和落地方向都已经成立，可以进入持续投入阶段。**

因此，这份文档要回答三个问题：

1. 我们现在到底要做什么。
2. 我们的方法相比现有 sampled simulation / representative sampling 工作到底新在哪里。
3. 接下来最合理的落地路径是什么。

---

## 2. 当前目标：提出从 Workload 到 Simulator 的结构化接口

当前最稳的目标，不是再泛泛地说“做 family 分组”，而是更准确地写成：

**我们希望提出一套从 workload 行为到 simulator 分析对象的结构化接口。**

这套接口的作用不是简单地整理结果，而是把原本依赖经验完成的几个步骤显式化：

- workload 的长 trace 如何组织
- 哪些 kernel / primitive 属于同一种工作模式
- 哪些对象值得共享 simulator reasoning lane
- 哪些对象可以作为后续调参与验证的代表样本

所以，我们当前的目标不是：

- 单独提出一个新的 sampling 技巧
- 单独提出一个新的 kernel 聚类器
- 单独展示几个算子的 profile 结论

而是：

**把复杂 workload 逐步压缩为 simulator 可以承接、比较、调参与验证的结构化对象。**

---

## 3. 当前方法主线：`squash -> family -> representative execution regime`

经过当前阶段的收敛，我们的方法主线已经可以稳定写成：

`workload -> squash -> family / execution template -> representative execution regime -> simulator lane / tuning`

这条主线对应三个层次的压缩。

### 3.1 第一层：时间压缩

由 `squash` 完成。

它解决的问题是：

**长 trace 如何沿时间展开，哪些时间段构成稳定 phase。**

`squash` 的作用不是定义 family，而是先把 workload 的原始时间轴整理成少数几个稳定 phase。

这一步压缩的是：

**temporal redundancy**

也就是：

- 不是每个时间片都需要被单独理解
- 不是整条 trace 都必须保持同样粒度
- 后续 family 分析必须建立在稳定 phase 上，而不是建立在一锅混合 trace 上

### 3.2 第二层：结构压缩

由 `family` / `execution template` 分析完成。

它解决的问题是：

**这些 kernel 到底共享哪种工作模式。**

当前我们已经明确，不能再只按算子名分组，也不能只按上层 attention / FFN 模块语义分组。  
更稳的做法是：

- 用 `Route Primitive` 描述算法路径中的计算角色
- 用 `Hardware Execution Template` 描述 GPU 上的执行骨架

这一步压缩的是：

**structural redundancy**

也就是：

- 不再逐 kernel name 独立建模
- 不再把所有 kernel 都看成独立调参对象
- 先识别共享执行模板与共享硬件工作模式

### 3.3 第三层：代表对象压缩

这是当前最重要的新收敛。

我们当前已经明确，后续对接 simulator 时，不应简单地提取“代表 kernel”，而更适合提取：

**representative execution regime**

也就是：

**代表执行区间 / 代表工作区间**

它至少应由下面几个维度共同决定：

1. 属于哪个 phase
2. 属于哪个 route primitive
3. 属于哪个 hardware execution template
4. shape / size 落在哪个 regime

例如在 Transformer 中，多层 GEMM 虽然都叫 GEMM，但它们可能来自：

- Q/K/V projection
- output projection
- FFN up projection
- FFN down projection

这些对象在实现模板上可能都接近 `Dense Tiled Compute`，  
但它们的 shape、上下文角色、权重和调参敏感性不一定相同。

因此，当前最稳的目标不是“挑一个代表 GEMM”，而是：

**在每个 phase 内，按 execution family 与 shape regime 提取少量代表执行对象。**

这一步压缩的是：

**parameter-search redundancy**

也就是：

- 不是每个实例都单独调
- 而是围绕少量代表 execution regimes 建立 simulator lane 和调参复用

---

## 4. 当前最重要的方法认识

### 4.1 我们压缩的不是 sample 本身，而是“需要被单独理解和单独调参的对象”

这是当前阶段已经非常清楚的一个判断。

如果把这项工作与 PKA / Sieve 这类 representative sampling 工作相比，那么最稳的区别是：

- `PKA / Sieve`
  - 压缩的是 `simulation samples`
  - 主要关注“模哪些样本、模多少样本”

- 我们当前的方法
  - 压缩的是 `phase`、`execution family`、`representative regimes`
  - 主要关注“按什么结构组织 workload，并减少后续重复模拟与重复调参对象”

所以我们真正试图压缩的，不只是一次 sampled simulation 的样本数，而是：

**多轮架构验证与参数探索中，需要被单独理解、单独模拟、单独调参的对象数量。**

### 4.2 `squash` 和 `family` 不是重复功能

当前可以稳定地说：

- `squash`
  - 组织时间结构
  - 提供 phase-level 表示

- `family`
  - 组织共享执行模板结构
  - 提供结构化的 simulator 分析对象

一个回答“什么时候发生了什么稳定行为”，  
一个回答“这些行为属于哪种执行模式”。

### 4.3 family 后面不应直接接“所有实例都调”，而应接 representative execution regime

这是当前方法从“分组”走向“可落地调参”的关键一步。

如果只有 family，你仍然会面临：

- 一个 family 内部要跑多少个 kernel
- 哪些 shape 可以合并
- 哪些对象值得做代表

所以后面真正进入 simulator lane 的单位，不应只是 family 名字，而应是：

**带有 phase / primitive / template / shape 信息的代表执行区间。**

---

## 5. 当前对调参逻辑的理解

当前我们已经形成一个更稳的调参理解：

**硬件调参不应再以单个 kernel 的最优为目标，而应以 workload 中多种 execution template 的加权平衡为目标。**

更具体地说，后续在某个 phase 内，应该先判断：

- 哪个 template / family 是主优化对象
- 哪个 template / family 是约束对象
- 哪些对象可以降优先级

也就是说，调参不是“平均照顾所有 family”，而是：

1. 先围绕主导工作模式优化
2. 再保证关键约束模式不被明显伤害

这意味着，后续需要引入 template / family 的权重概念。

当前最稳的理解是把权重分成三层：

### 5.1 Coverage Weight

表示：

**该 template / family 在 workload 主路径中覆盖了多少计算步骤。**

### 5.2 Time Weight

表示：

**该 template / family 实际消耗了多少运行时间。**

### 5.3 Decision Weight

表示：

**该 template / family 对硬件参数决策到底有多大影响。**

当前阶段，前两层可以逐步定量化；  
第三层更适合先用定性规则表达，而不是过早写成严格公式。

---

## 6. 当前最稳的落地点：Transformer 主计算路线

当前我们已经确认，第一版方法原型最适合放在 Transformer 主计算路线中生长。

这条主链可以写成：

`QKV / projection -> attention_score -> softmax -> context_mul -> output projection -> residual / norm -> FFN`

它之所以适合作为第一版原型，有三个原因：

1. 它本身就是当前主流 AI workload 中最重要的路径之一。
2. 它已经能覆盖多种 `Route Primitive` 与 `Hardware Execution Template`。
3. 它已经暴露出最关键的边界 case，例如：
   - `gemm_tiled` vs `attention_score`
   - `softmax_kernel` vs `context_mul`

因此，当前最合理的推进方式不是继续无边界扩 workload 名单，而是先把这条主链做硬。

---

## 7. 当前这项工作的价值判断

到当前阶段，我们已经可以比较稳地认为：

**这项工作无论从工作量还是价值上，都已经构成一条值得持续投入的优质工作线。**

这里的“价值”主要不在于：

- 当前已经拿到了多少最终实验数字
- 当前已经证明了多少绝对降本比例

而在于：

### 7.1 方法价值

它补上了一层以往缺失的结构化接口：

`workload -> structured reasoning object -> simulator lane`

### 7.2 研究价值

它把 GPU simulator 研究从：

- 单 kernel 局部观察
- sampled simulation 样本压缩

推进到：

- workload 级结构组织
- phase-level 与 family-level reasoning
- 面向调参循环的对象压缩

### 7.3 工作量价值

这条线本身具有明显的持续展开空间：

- primitive 判据可以继续做硬
- family boundary protocol 需要独立写清楚
- representative execution regime 的选择规则需要明确
- simulator lane / tuning lane 需要进一步落地

也就是说，它不是一个很快写完的小点子，而是一条具有清晰层次和明确扩展空间的方法线。

---

## 8. 当前最合理的后续推进顺序

如果按当前状态继续往前推进，最自然的顺序应该是：

### 第一步：把 Transformer 主链上的 primitive / template 判据继续做硬

目标：

- 让每个关键 kernel 都能稳定映射到：
  - `Route Primitive`
  - `Hardware Execution Template`

### 第二步：写出 family selection / boundary protocol

目标：

- 回答 family 为什么这样划
- 回答什么时候应该并类
- 回答什么时候必须拆开
- 回答什么时候保留 outlier

### 第三步：定义 representative execution regime 的提取规则

目标：

- 在每个 phase 内，明确：
  - 哪些 shape / size 可以归入同一 regime
  - 每个 family / template 保留几个代表对象
  - 这些代表对象如何进入 simulator lane

### 第四步：再向 simulator lane / tuning lane 对接

目标：

- 不是直接做全量复杂 tuning
- 而是先把“phase -> family -> representative regime -> lane”这条链打通

---

## 9. 当前阶段的最简结论

到目前为止，我们已经可以用比较稳定的话来总结当前进展：

1. 我们的目标已经从“泛泛谈 family”收束成“提出从 workload 到 simulator 的结构化接口”。
2. 当前的方法主线已经明确为：
   - `squash`
   - `family / execution template`
   - `representative execution regime`
   - `simulator lane / tuning`
3. 我们当前压缩的对象，不只是 sample，而是：
   - 时间上的 phase
   - 结构上的 execution family
   - 调参上的代表执行区间
4. 后续调参的核心不再是单个 kernel 最优，而是 workload 中多种执行模板的加权平衡。
5. Transformer 主计算路线已经足以作为第一版方法原型的主要工作台。
6. 当前这项工作已经具备清晰的问题、明确的方法、可展开的工作量和足够高的研究价值。

这就是我们当前阶段最稳定的目标与方法总结。
