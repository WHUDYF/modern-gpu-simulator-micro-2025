# PKA 到 Family 的接口设计

日期：2026-04-20

## 1. 文档目的

这份文档用于把当前方法线里的一个关键接口固定下来：

**当 PKA 风格的前端压缩已经输出 representative kernels 之后，我们的方法到底如何把这些输出接到后续的 family / regime / simulator 分析层。**

当前最重要的问题不是：

- PKA 论文细节是否被 100% 复刻
- 我们是否已经能直接进入 simulator

而是：

**PKA 风格压缩输出和我们的 family 方法之间，究竟应该通过什么字段、什么顺序、什么规则连接起来。**

---

## 2. 当前接口设计的总目标

当前更合理的整体方法线可以写成：

`full workload -> PKA-style compression -> representative kernels + membership + weights -> family organization -> representative execution regime -> simulator lane / validation`

这条链路中，`PKA -> family` 接口要解决的问题是：

1. 如何把 representative kernels 从“压缩结果”变成“结构化分析输入”
2. 如何把 cluster / membership / weight 信息保留下来
3. 如何避免直接把 representative kernel 当作最终 simulator 单位
4. 如何让后续 family / regime 仍然能保留 phase、shape、template 等信息

---

## 3. 当前接口的核心原则

### 原则 1：PKA 输出的是代表对象，不是最终解释对象

PKA 风格压缩后的 representative kernels 只是：

- 压缩后的代表锚点
- 进入后续分析的前端输入

它们不应被直接视为：

- 最终 family
- 最终 simulator lane
- 最终 tuning 单位

因为我们的后续分析还要继续回答：

- 这些 representative kernels 共享什么机制
- 它们是否处于同一 phase
- 它们是否只是同一 family 内的不同 regime

### 原则 2：PKA 的 membership / weight 信息必须贯穿保留

如果只保留代表 kernel 本身，而丢掉：

- cluster membership
- coverage
- time weight

那么后续 family 分析会失去两个能力：

1. 无法知道一个 representative kernel 代表多少 workload
2. 无法把 simulator 结果回写到原始 workload

所以 `PKA -> family` 的接口中，membership 和 weight 不是附属字段，而是主字段。

### 原则 3：phase 信息应在 family 之前补回

PKA 风格压缩本身主要面向 kernel-level representative selection，不天然保留 phase 结构。

但我们的 family 定义明确依赖：

- 同一稳定 phase
- shared route primitive
- shared hardware template

因此，进入 family 分析前，必须先把 representative kernels 补回到某个 phase context 中。

这意味着：

**family 不应直接吃 PKA 原始输出，而应吃“带 phase 上下文的 representative kernels”。**

### 原则 4：family 之后仍然需要 regime 层

即便经过 PKA 压缩后，进入同一 family 的 representative kernels 也可能因为：

- shape
- context
- resource signature

不同，而继续拆成多个 representative execution regimes。

所以接口终点不是 family，而是：

**family-aware representative execution regime construction**

---

## 4. 当前输入输出对象定义

### 4.1 PKA 前端输出对象

当前最理想的 PKA 输出对象可以定义为：

**Representative Kernel Anchor**

每个 anchor 至少包含：

- 一个 representative kernel
- 它所代表的原始 kernel / invocation 集合
- 对应的 coverage / time weight
- 基础执行 metadata

这意味着 PKA 的输出在语义上更接近：

**压缩锚点**

而不是：

**机制解释单元**

### 4.2 family 层输入对象

family 层真正接收的输入对象应该定义为：

**Phase-Annotated Representative Kernel Anchor**

也就是在 representative kernel anchor 的基础上，再补充：

- phase id
- trace position
- optional shape hint
- optional route hint

只有这样，family 层才能在正确上下文中讨论：

- 是否共享同一种工作模式
- 是否值得合并
- 是否应保留边界

### 4.3 regime 层输入对象

regime 层的输入对象应定义为：

**Family-Annotated Representative Anchor Set**

也就是：

- 已经过 phase 组织
- 已经过 family 判据筛选
- 但仍保留细粒度 shape / weight / context 差异

这一步才是后续真正进入 simulator lane 的前置对象。

---

## 5. 当前建议的数据流

当前最稳的实现顺序如下：

### Step 1：从完整 workload 生成 PKA 输入表

输入表的记录单位建议是：

- kernel invocation
或
- kernel-level compressed record

至少包含：

- `kernel_name`
- `kernel_invocation_id`
- `dynamic_inst_count`
- `grid_dim`
- `block_dim`
- `trace_order`
- `feature_vector`

### Step 2：运行 PKA-style compression

输出：

- representative kernel anchors
- cluster membership
- anchor weights

### Step 3：为 representative anchors 补回 phase 信息

这一阶段可通过：

- trace order
- squash 输出
- phase mapping table

把每个 representative anchor 重新映射到：

- `phase_id`
- `phase_type`
- `phase_local_order`

### Step 4：进入 family selection

在同一 phase 内，对 representative anchors 按以下顺序组织：

1. `Route Primitive`
2. `Hardware Execution Template`
3. 边界 case 检查
4. family merge / split / unresolved decision

### Step 5：在 family 内提取 representative execution regime

进一步按：

- shape / size regime
- resource signature
- weights

构造最终进入 simulator lane 的 regime 对象。

---

## 6. 当前推荐的数据表设计

### 6.1 表 A：Raw Kernel Invocation Table

这是 PKA 前端的输入基础表。

建议字段：

| 字段 | 含义 |
|---|---|
| `kernel_invocation_id` | invocation 标识 |
| `kernel_name` | kernel 名称 |
| `trace_order` | 在 trace 中的顺序 |
| `grid_dim` | grid 配置 |
| `block_dim` | block 配置 |
| `dynamic_inst_count` | 动态指令数 |
| `feature_vector` | 用于 PKA 的特征向量 |
| `shape_hint` | 可选，M/N/K / seq / batch 等 |

### 6.2 表 B：Representative Kernel Anchor Table

这是 PKA 输出表。

建议字段：

| 字段 | 含义 |
|---|---|
| `rep_kernel_id` | representative kernel 标识 |
| `kernel_name` | 代表 kernel 名称 |
| `cluster_id` | 所属 cluster |
| `coverage_count` | 覆盖 invocation 数 |
| `coverage_weight` | 覆盖比例 |
| `time_weight` | 时间占比 |
| `member_invocations` | 被代表 invocation 列表 |
| `grid_dim_summary` | 覆盖对象的 grid 概括 |
| `block_dim_summary` | 覆盖对象的 block 概括 |
| `shape_hint_summary` | 覆盖对象的 shape 概括 |

### 6.3 表 C：Phase-Annotated Anchor Table

这是进入 family 层前必须增加的一张表。

建议字段：

| 字段 | 含义 |
|---|---|
| `rep_kernel_id` | representative kernel 标识 |
| `phase_id` | 所属 phase |
| `phase_type` | 主 phase / 过渡 phase |
| `phase_local_order` | phase 内局部顺序 |
| `coverage_weight` | 来自 PKA 的覆盖权重 |
| `time_weight` | 来自 PKA 的时间权重 |
| `shape_hint_summary` | shape 概括 |
| `trace_position_summary` | 位置概括 |

### 6.4 表 D：Family Assignment Table

这是 family 层输出。

建议字段：

| 字段 | 含义 |
|---|---|
| `rep_kernel_id` | representative kernel 标识 |
| `phase_id` | 所属 phase |
| `route_primitive` | 路径角色 |
| `hardware_template` | 执行模板 |
| `family_id` | family 标识 |
| `family_decision` | strong share / weak share / split / unresolved |
| `boundary_notes` | 共享点与区分点 |

### 6.5 表 E：Representative Execution Regime Table

这是最终进入 simulator lane 的对象表。

建议字段：

| 字段 | 含义 |
|---|---|
| `regime_id` | regime 标识 |
| `family_id` | 所属 family |
| `phase_id` | 所属 phase |
| `route_primitive` | 路径角色 |
| `hardware_template` | 执行模板 |
| `shape_regime` | 形状区间 |
| `resource_signature` | 主导资源行为 |
| `coverage_weight` | 覆盖权重 |
| `time_weight` | 时间权重 |
| `decision_weight` | 当前阶段可先定性 |
| `source_rep_kernels` | 来源 representative kernels |

---

## 7. 当前最重要的字段映射

从 `PKA -> family` 的角度看，最关键的映射不是全部字段，而是下面几项。

### 7.1 `rep_kernel_id -> family_id`

这个映射回答：

- 一个 representative kernel 被归到哪个 family

### 7.2 `member_invocations -> family coverage`

这个映射回答：

- 一个 family 实际覆盖原始 workload 中多少 invocation

### 7.3 `trace_order -> phase_id`

这个映射回答：

- 一个 representative kernel 在时间结构上属于哪个 phase

### 7.4 `shape_hint_summary -> shape_regime`

这个映射回答：

- 一个 representative kernel 在 family 内应落到哪个 regime

### 7.5 `coverage_weight / time_weight -> simulator priority`

这个映射回答：

- 哪些 regime 在 simulator 中应该优先验证、优先调参

---

## 8. 当前实现中的关键难点

### 难点 1：PKA 输出天然偏 kernel-centric，而我们后面偏 phase-aware

这是当前接口最大的结构张力。

PKA 更关心：

- 哪个 kernel 有代表性

而我们后面更关心：

- 这个 kernel 在哪个 phase 中
- 它共享什么机制
- 是否值得进入同一 simulator lane

所以 phase 回补是接口中的必要步骤，而不是可选项。

### 难点 2：同一 representative kernel 可能覆盖多个上下文

例如某个 GEMM 类 representative kernel，可能覆盖：

- QKV projection
- output projection
- FFN up projection

这就意味着：

- 单个 representative kernel 不一定直接对应单个 regime
- 甚至不一定直接对应单个 family

因此，membership 回写时必须保留 context 信息，否则会把不同上下文错误揉平。

### 难点 3：family 仍然不能跳过边界判据

引入 PKA 后，不意味着 family 可以直接由 clustering 结果代替。

我们仍然需要：

- Route Primitive
- Hardware Template
- boundary-first

因为 PKA 的 cluster 是“可代表性压缩”视角，
而我们的 family 是“共享机制组织”视角。

这两个视角相关，但不相同。

---

## 9. 我们当前最应该避免的错误

### 错误 1：把 representative kernel 直接当 family

这会使我们的方法退化成：

- `压缩后对象列表`

而不是：

- `structured simulator interface`

### 错误 2：把 PKA 的 cluster 直接当 mechanism family

这会让 workload compression 和 shared mechanism identification 混在一起。

但二者并不等价。

### 错误 3：丢掉 membership / weight

如果没有 membership / weight，后续无法回答：

- 一个 family 实际代表多少 workload
- simulator 结果如何回写

### 错误 4：不补 phase context

如果 phase 信息不补回，family 判据会失去时间上下文基础。

---

## 10. 当前最推荐的最小可用实现

为了控制工作量，我建议第一版只实现下面这条最小链路：

### 第一版目标

`full workload -> PKA-style representative kernel anchors -> phase annotation -> family assignment -> regime extraction`

### 第一版不强求

- 完整复刻 PKA 的所有评价
- 完整自动化 simulator 回写
- 完整一般化到所有 workload

### 第一版必须做到

1. 至少得到一张 representative kernel anchor 表
2. 至少保留 membership 与 weight
3. 至少能补回 phase 信息
4. 至少能完成 `representative kernels -> family -> regime` 的一条完整示例链

如果这一版跑通，你们的方法就已经从概念推进到了真正的接口级原型。

---

## 11. 当前阶段的简短结论

如果把当前设计压成最短形式，可以写成：

1. PKA 前端输出的是 representative kernel anchors，而不是最终 family。
2. 进入 family 层前，必须为 representative anchors 补回 phase context。
3. family 层继续按 route primitive、hardware template 和 boundary case 组织共享机制。
4. family 之后仍需按 shape / resource / weight 提取 representative execution regimes。
5. 因此，`PKA -> family` 的真正接口对象不是“代表 kernel 本身”，而是“带 phase 上下文的 representative anchor”。 
