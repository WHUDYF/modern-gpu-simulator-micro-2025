# A 线与 PKA 行为特征空间对齐校正说明

日期：2026-04-24

## 1. 文档目的

这份文档用于明确当前 A 线前端分组与 `PKA` 行为特征空间之间的差距，
并给出下一次 RLCR 时应如何修正的具体方向。

当前我们已经形成了一个可运行的 A 线 frontend anchor v1，
但从方法论上看，仍存在一个重要问题：

**A 线虽然在定位上声称自己是 `PKA-style frontend anchor`，但主分组结构还没有真正对齐到 PKA 的行为特征空间。**

这会带来两个风险：

1. reviewer 可能认为我们只是在借用 PKA 的叙事，而没有真正继承其前端分组基础；
2. A 线可能逐渐从 `representative compression` 退化成一个“为了后段方便而拼信息做 grouping”的装置。

因此，当前最需要做的不是继续往 A 线塞更多信息，
而是：

**把 A 线的主分组结构收回到 PKA-core behavior feature space。**

---

## 2. 先明确：我们这里说的“对齐”是什么意思

这里的“对齐”不是说：

- 我们必须逐字段、逐公式、逐聚类结果完全复刻 PKA；
- 我们最终得到的 cluster 数量必须和 PKA 完全一致；
- invocation-level 结果必须和 kernel-level PKA 输出逐组一一对应。

这里的“对齐”真正指的是：

1. **主分组基础应建立在 PKA 核心行为特征空间上**
2. **任何非 PKA 特征都不应默认进入主 grouping 逻辑**
3. **非 PKA 信息如果保留，也应只作为 constrained refinement 或 downstream-required metadata**

一句话说：

**先按 PKA-core 分，再按最小必要扩展修。**

---

## 3. PKA 对我们最关键的约束是什么

根据当前仓库中的相关整理，PKA 给我们的最关键启发有两条：

### 3.1 PKA 不按名字分组

PKA 的核心不是：

- 同名 kernel 归一组

而是：

- 在行为特征空间中寻找 representative objects

这说明：

**A 线不能以名字或名字+形状作为真正主分组基础。**

### 3.2 PKA 的 cluster 只回答“谁代表谁”

PKA 主要回答的是：

- 哪些对象可以代表哪些对象

它不负责直接回答：

- family 是什么
- regime 是什么
- lane 是什么

这说明：

**A 线只能负责 representative compression，不能在前端层把后段决策逻辑提前塞进 grouping。**

---

## 4. 当前 A 线实际主分组逻辑

当前 A 线 `selector.py` 中的主分组逻辑可以概括为：

### 4.1 `pka-like-coarse`

当前 coarse key 由下面几类字段组成：

- `kernel_name`
- `grid_dim`
- `block_dim`
- `compute_throughput_pct`
- `dram_throughput_pct`

### 4.2 `hybrid`

在 coarse bucket 内，当前进一步按以下字段细分：

- `total_dynamic_insts`
- `achieved_occupancy_pct`
- `compute_throughput_pct`
- `dram_throughput_pct`
- `ipc_active`
- `cross_tb_offset_coverage`

这说明当前 A 线的 grouping 现实上是：

**名字 + launch shape + 一小组执行/压缩特征**

而不是：

**以 PKA-core behavior feature space 为主干，再附带约束性 refinement**

---

## 5. 当前 A 线与 PKA 的差距

### 5.1 当前 A 线的问题不是“完全错误”

当前 A 线已经做对了几件事：

- 它没有退化成 `name-only`
- 它承认同名 invocation 可能异质
- 它已经使用了部分行为相关特征
- 它保留了 membership / weight / context metadata

所以问题不是“不能用”，而是：

**主特征层级顺序还不对。**

### 5.2 当前最大的偏差

当前最大的偏差是：

**`kernel_name`、`grid_dim`、`block_dim` 被放进了 coarse 主分组逻辑。**

这会导致方法姿态上出现一个风险：

- 我们看起来更像在做“名字/形状引导的 grouping”
- 而不是“行为特征空间引导的 representative compression”

### 5.3 第二个偏差

当前 `cross_tb_offset_coverage` 这类 compression-side 特征进入了 `hybrid` 主分组逻辑。

这会带来一个很明显的问题：

- 这种特征更像 downstream-related extension
- 而不是 PKA 核心行为特征

如果它直接参与前端主 clustering，
reviewer 很容易认为：

**我们在用后段相关信息反向塑造前端 grouping。**

---

## 6. 当前 A 线特征与 PKA-core 对照表

| 当前 A 线字段 | 当前角色 | 与 PKA 的关系 | 建议处理 |
|---|---|---|---|
| `kernel_name` | coarse 主分组字段 | 非 PKA behavior feature | 从主 grouping 移出；仅保留为 identity / 导出字段，或最多作为后处理可读标签 |
| `grid_dim` | coarse 主分组字段 | 非 PKA-native；更像 shape / launch metadata | 不应作为第一主分组轴；可降级为 refinement guardrail 或 metadata |
| `block_dim` | coarse 主分组字段 | 非 PKA-native；更像 launch metadata | 不应作为第一主分组轴；可降级为 refinement guardrail 或 metadata |
| `compute_throughput_pct` | coarse / hybrid 主特征 | 可视为 PKA-proxy，但不是文档中最核心的原生表述 | 可以保留，但建议降为 behavior proxy，而不是和 `kernel_name` 并列主轴 |
| `dram_throughput_pct` | coarse / hybrid 主特征 | 接近 memory behavior proxy | 建议保留，作为 PKA-core memory-behavior 代理项之一 |
| `total_dynamic_insts` | hybrid 主特征 | 对齐 PKA 中的 instruction count / work-size proxy | 应保留，并提升为更核心的主分组轴 |
| `achieved_occupancy_pct` | hybrid 主特征 | 不是 PKA 文档中最直接的核心字段，但可作 execution-efficiency proxy | 可保留为 PKA-proxy，建议放在主 clustering 第二层而非附会 downstream |
| `ipc_active` | hybrid 主特征 | 属于 performance/efficiency proxy，非 PKA 原生主表述 | 可保留为 proxy，但不应压过 memory + inst 行为特征 |
| `cross_tb_offset_coverage` | hybrid 主特征 | 非 PKA-native，更接近 compression/downstream 特征 | 应从主 grouping 特征中移出，降级为 guardrail / analysis note |
| `exec_time` / `cycle_proxy` | 当前在 grouping 中没有直接作为主轴 | 文档中已建议为 PKA-style 主特征之一 | 建议补回主 grouping 特征空间 |
| global/local/shared memory stats | 当前没有系统进入 selector 主逻辑 | PKA-core behavior feature | 应补进主 grouping 特征空间，至少作为 memory behavior 主轴 |
| divergence efficiency | 当前没有进入 selector 主逻辑 | PKA-core behavior feature | 应评估是否可从现有数据稳定恢复；若可恢复，应进入 PKA-core feature set |
| thread block 数量 / work-size proxy | 当前主要被 `grid_dim` 近似代替 | PKA-core feature | 建议显式化为行为/规模特征，不要只通过 `grid_dim` 间接体现 |
| `trace_order` | metadata | 非 PKA 主 grouping 特征 | 保留为 context metadata，不参与主 grouping |
| `shape_hint` | metadata | 非 PKA 主 grouping 特征 | 保留为 context metadata，不参与主 grouping |
| squash boundary / segment 信息 | guardrail | 非 PKA 主 grouping 特征 | 只保留为 heterogeneity/refinement guardrail，不参与主 grouping |

---

## 7. 建议的新结构：两层式 A 线分组

下一次 RLCR 时，建议把 A 线分组明确改写为两层结构。

### 7.1 Stage 1：PKA-core grouping

这一步只允许使用：

- instruction / work-size 相关特征
- global / local / shared memory behavior
- divergence / efficiency proxy
- thread-block/work-size proxy
- `exec_time` / `cycle_proxy`（若当前设计确认保留）

这一步的目标是：

**用行为特征空间做真正主分组。**

### 7.2 Stage 2：Constrained refinement

这一步只允许在 Stage 1 结果之上，
用最小必要的信息做 refinement 或 guardrail。

可保留的信息包括：

- `grid_dim`
- `block_dim`
- `trace_order`
- `shape_hint`
- squash boundary
- 其他 downstream-required 但非 PKA-native 的字段

这一步的目标不是重新聚类，
而是：

- 防止明显 invocation heterogeneity 被压平
- 为后段 family / regime 保留必要输入

---

## 8. 下一次 RLCR 应做的具体修改

建议下一次 RLCR 不要一开始就重写整套前端，
而是按下面顺序推进。

### 8.1 第一步：先做特征审计

输出一张正式对照表，把当前 A 线特征分成三类：

1. `PKA-native`
2. `PKA-proxy`
3. `Non-PKA downstream-required`

目标是：

先把“哪些特征应参与主 grouping”这件事说清楚。

### 8.2 第二步：重写 selector 的分层逻辑

把当前：

- `kernel_name + grid_dim + block_dim + throughput`

主导 coarse grouping 的方式，
改成：

- `PKA-core behavior features`

主导 coarse grouping。

同时把：

- `kernel_name`
- `grid_dim`
- `block_dim`
- `cross_tb_offset_coverage`

降级出主分组逻辑。

### 8.3 第三步：保留 constrained refinement

不要把所有非 PKA 信息删除。

更稳的做法是：

- 保留它们
- 但只让它们参与 refinement / guardrail / metadata
- 不让它们决定主 grouping

### 8.4 第四步：做三组对照

建议至少保留三种模式：

- `name-only`
- `PKA-core only`
- `PKA-core + constrained refinement`

这会让你们后续能明确证明：

1. 只按名字不够
2. 纯 PKA-core 能提供可信前端基础
3. 最小 refinement 是必要但不过界的

---

## 9. 这次校正的真正目标

这次校正的目标不是：

- 把 A 线彻底改成“严格复刻 PKA”

而是：

**让 A 线在方法姿态上真正站回 PKA-compatible frontend anchor，而不是继续滑向 downstream-aware information integrator。**

也就是说，校正后的 A 线应该满足：

1. 主 grouping 建立在行为特征空间之上
2. 非 PKA 特征只在 downstream necessity 成立时保留
3. 非 PKA 保留字段不提前产出 family / regime 结论
4. invocation-level 细化仍然存在，但作为 constrained refinement，而不是前端主 grouping 的替代品

---

## 10. 最终结论

当前对 A 线最准确的判断是：

**它已经不是 name-only baseline，但也还没有真正做到“主分组结构与 PKA 行为特征空间对齐”。**

因此，下一次 RLCR 的重点不应是继续往 A 线加更多信息，
而应是：

**把 A 线的主 grouping 收回到 PKA-core behavior feature space，并把其余信息降级为 constrained refinement。**

如果这一步完成，
那么 A 线的方法定位会明显更稳，
也会显著降低“信息总结器”风险。
