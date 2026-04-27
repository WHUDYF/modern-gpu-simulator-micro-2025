# A 线 PKA Baseline 复现契约

日期：2026-04-26

## 1. 文档目的

这份文档用于把 A 线下一阶段的首要任务正式固定下来：

**A 线当前首先要完成的，不是继续扩展 frontend anchor v1，而是先复现一个可运行、可检查、可比较的 `PKA baseline`。**

这样做的原因很直接：

1. `PKA` 已经被我们明确视为 A 线前端压缩层的直接 baseline；
2. 如果 baseline 本身没有先站稳，后续任何 A 线“优化”都无法判断：
   - 是真的改进；
   - 还是只是换了一种分组方式；
   - 还是 baseline 本身就没有对齐正确；
3. 如果没有 baseline，A 线后续也无法稳定暴露问题来源。

因此，当前 A 线的优先级应重排为：

`PKA baseline reproduction -> baseline evaluation -> constrained extension`

---

## 2. 当前判断：为什么要先复现 PKA

当前仓库中已经有一版可运行的 A 线 frontend anchor v1，
但这版实现更适合作为：

- 原型实现
- 接口验证
- 比较路径

而不应继续充当：

- A 线正式 baseline
- 后续所有优化的参照系

原因在于：

### 2.1 当前 v1 不是严格意义上的 PKA baseline

当前 `selector.py` 的主分组逻辑仍然包含：

- `kernel_name`
- `grid_dim`
- `block_dim`

这意味着它的主 grouping 结构并没有真正建立在 `PKA core behavior feature space` 上。

### 2.2 如果 baseline 不对齐，后续 extension 无法解释

如果我们在一个未对齐的 baseline 上继续加入：

- heterogeneity guardrail
- trace / shape context
- downstream-required metadata

那么最后即使结果变好，也无法清楚回答：

- 改进来自 PKA baseline 本身被修正；
- 还是来自我们新增的 constrained extension；
- 还是来自某些非 PKA 字段改变了主 grouping。

### 2.3 先复现 baseline 才能建立后续论证顺序

我们后续真正想讲的不是：

**我们做了一个比当前 v1 更复杂的前端。**

而是：

**我们先复现了一个 PKA-compatible baseline，然后证明在不破坏前端边界的前提下，最小扩展能为后段 family / regime / priority 提供额外价值。**

---

## 3. 本 contract 对 A 线的重定义

从本文件开始，A 线的当前主任务定义为：

**实现并固定一个可运行的 `PKA baseline reproduction path`，作为后续 A 线所有扩展与比较的唯一正式前端基线。**

因此，A 线当前不再以“继续增强 frontend anchor v1”为主目标，
而是改为以下两阶段：

### 阶段 A：PKA baseline reproduction

目标：

- 先得到一个行为特征空间对齐的 representative compression baseline
- 明确输出 representative objects、membership、coverage、weight
- 为后续 comparison 提供稳定参照系

### 阶段 B：Constrained extension

目标：

- 只在 baseline 已经稳定的前提下
- 引入最小必要扩展
- 检查这些扩展是否真的为后段提供必要输入

只有阶段 A 完成之后，阶段 B 才允许继续推进。

---

## 4. 我们这里说的“复现 PKA”是什么意思

这里的“复现”不是指：

- 完整复刻 PKA 论文的全部实验；
- 完整复刻其所有公式、参数和评测；
- 逐个 kernel 与原论文结果严格一一对齐。

这里的“复现”具体指的是：

### 4.1 复现其前端压缩主逻辑

即：

- 以行为特征空间组织对象
- 在该空间中完成 representative compression
- 输出 representative objects 与 membership / weight 信息

### 4.2 复现其可供我们使用的输入输出契约

即至少要稳定得到：

- representative anchor / representative kernel object
- cluster / membership
- coverage / time weight
- 最小 metadata

### 4.3 复现其“baseline 角色”

也就是说，它必须能作为我们后续 extension 的正式对照组，
而不是只是一个“灵感来源”。

---

## 5. PKA baseline 必须复现的内容

### 5.1 输入对象

当前 baseline 允许使用统一后的 `KernelInvocationRecord`，
但要满足两个约束：

1. 对象必须是可比较的 invocation-level 或 kernel-level record；
2. record 中必须包含足以构成 PKA-style behavior feature space 的字段。

当前最低要求字段：

- `kernel_invocation_id`
- `kernel_name`
- `trace_order`
- `exec_time` 或 `cycle_proxy`
- `dynamic_inst_count`
- `feature_vector`

允许额外保留：

- `grid_dim`
- `block_dim`
- `shape_hint`

但这些字段不应默认进入 baseline 主 grouping。

### 5.2 特征空间

PKA baseline 的主 grouping 只允许使用下列两类特征：

#### A. PKA-core features

- instruction count / work-size proxy
- global / local / shared memory behavior
- divergence / efficiency related behavior
- thread block count / workload scale proxy

#### B. PKA-proxy features

当当前数据无法完整恢复 PKA 原生字段时，
允许使用可解释的 proxy，
但必须显式标注。

例如当前仓库中可优先考虑：

- `total_dynamic_insts`
- `num_blocks`
- `threads_per_block`
- `dram_throughput_pct`
- `l1_hit_rate_pct`
- `l2_hit_rate_pct`
- `ipc_active`
- `achieved_occupancy_pct`

### 5.3 分组流程

PKA baseline 的分组流程必须满足：

1. 主 grouping 先建立在 behavior feature space 上；
2. `kernel_name` 不得作为第一主分组轴；
3. `grid_dim` / `block_dim` 不得作为第一主分组轴；
4. compression-side / downstream-side 特征不得进入 baseline 主 grouping。

### 5.4 输出对象

PKA baseline 至少应输出：

- representative objects
- cluster membership
- coverage count
- coverage weight
- time weight
- 最小 metadata

建议最小输出字段：

- `rep_kernel_id`
- `cluster_id`
- `representative_kernel_name`
- `member_invocations`
- `member_kernel_names`
- `coverage_count`
- `coverage_weight`
- `time_weight`
- `dynamic_inst_summary`
- `memory_summary_optional`
- `trace_order_summary`

---

## 6. PKA baseline 明确不要求复现的内容

为了防止范围失控，当前阶段明确不要求：

### 6.1 不要求完整论文级实验复现

当前不是要做一个独立的 PKA reproduction project。

### 6.2 不要求立刻证明 PKA 最优

当前阶段只要求它成为：

- 可运行 baseline
- 可比较 baseline
- 可诊断 baseline

### 6.3 不要求 baseline 直接输出 family / regime

PKA baseline 只能回答：

- 谁代表谁
- 覆盖多少
- 哪些对象值得留下

它不能直接回答：

- family 是什么
- regime 是什么
- lane 是什么

### 6.4 不要求第一版就完整恢复 heterogeneity refinement

当前阶段 heterogeneity 只允许作为：

- error diagnosis clue
- later extension entry

而不是 baseline 主逻辑。

---

## 7. 与 constrained extension 的边界

从本 contract 开始，A 线代码与文档中必须显式区分下面两层：

### 7.1 `pka_baseline`

职责：

- 做 PKA-style representative compression
- 输出 representative objects + membership + weights
- 作为正式 baseline

### 7.2 `pka_extension`

职责：

- 在 baseline 之上引入最小必要扩展
- 仅增加 downstream necessary 的字段或 guardrail
- 不得反向重写 baseline 的主逻辑

也就是说，后续任何 A 线新增字段、规则或 guardrail，
都必须先回答一个问题：

**这是 baseline 本身的一部分，还是 extension 的一部分？**

如果回答不清楚，就不应直接进入主线实现。

---

## 8. 对当前仓库实现的直接影响

### 8.1 当前 `frontend anchor v1` 的角色调整

当前基于：

- `name-only`
- `PKA-like coarse`
- `hybrid`

的实现路径，应从主 baseline 角色调整为：

- 原型路径
- 比较路径
- 过渡实现

它不应继续被称为：

- 正式 PKA baseline

### 8.2 必须新增独立的 baseline 模式

当前 A 线实现中应显式增加一个独立模式：

- `pka_baseline`

并与下列模式区分：

- `name_only_baseline`
- `current_v1_prototype`
- `pka_extension`（后续）

### 8.3 文档口径必须同步

后续所有 A 线文档都应避免继续混用下面两种表述：

- “PKA-style”
- “已经对齐 PKA”

除非对应实现已经明确走的是 `pka_baseline` 路径。

---

## 9. 当前建议的工程改动范围

当前建议优先修改以下位置：

### 9.1 `selector.py`

需要做的不是继续微调当前 `hybrid`，
而是新增或重写：

- `pka_baseline` selector path

其要求是：

- 行为特征主导 grouping
- 非 PKA 字段退出主 grouping
- 保留后续 extension 接口

### 9.2 `exporter.py`

需要把输出对象从“默认同名 kernel anchor”改成：

**representative object with explicit membership**

建议新增：

- `representative_kernel_name`
- `member_kernel_names`
- `mixed_kernel_name_flag`

### 9.3 测试

需要新增 baseline-level tests，检查：

- 主 grouping 不依赖 `kernel_name`
- 主 grouping 不依赖 `grid_dim` / `block_dim`
- baseline 输出有稳定 membership / weight
- baseline 与 current v1 路径可以并行比较

### 9.4 文档

需要新增或更新：

- PKA baseline reproduction contract
- A 线 feature audit
- A 线 baseline evaluation note

---

## 10. baseline 验收标准

只有满足下面这些条件，A 线才可以声称“PKA baseline 已经建立”。

### 10.1 基线独立存在

仓库中已经存在一个明确可调用的 `pka_baseline` 路径，
并且它与 `current_v1` 原型路径分离。

### 10.2 特征空间可解释

baseline 主 grouping 使用的每个字段都能明确归入：

- `PKA-core`
或
- `PKA-proxy`

### 10.3 非 PKA 字段退出主 grouping

下面这些字段不得再作为 baseline 主 grouping 的必要部分：

- `kernel_name`
- `grid_dim`
- `block_dim`
- `cross_tb_offset_coverage`
- squash boundary fields
- trace-order-only context fields

### 10.4 输出对象稳定

baseline 至少能稳定导出：

- representative object
- membership
- coverage / time weights
- 最小 metadata

### 10.5 comparison 能成立

后续 extension 可以直接与 baseline 比较：

- anchor 数量
- time coverage
- cluster 内方差
- split / merge 差异

如果 baseline 本身不能被稳定比较，则视为未完成。

---

## 11. A 线下一阶段的建议执行顺序

建议顺序固定为：

### Step 1：写 feature audit

把现有字段分成三类：

1. `PKA-core`
2. `PKA-proxy`
3. `Non-PKA extension-only`

### Step 2：实现 `pka_baseline`

在当前 selector 体系中新增独立 baseline 路径。

### Step 3：补 baseline 输出与测试

确保 representative object、membership、weight 都能稳定导出与验证。

### Step 4：建立 baseline evaluation

至少比较：

- baseline vs name-only
- baseline vs current_v1 prototype

### Step 5：只有 baseline 稳定后，才恢复 extension 工作

此时才允许重新讨论：

- heterogeneity guardrail
- trace / phase context
- downstream-required constrained extension

---

## 12. 当前阶段的简短结论

如果把本 contract 压成最短形式，可以写成：

1. `PKA` 已经被确定为 A 线前端压缩层的正式 baseline。
2. 因此，A 线当前首要任务不是继续增强 prototype，而是先建立 `pka_baseline`。
3. 这个 baseline 必须以行为特征空间为主分组基础，并稳定输出 representative objects、membership 和 weights。
4. 当前 `frontend anchor v1` 应降级为原型 / 比较路径，而不是正式 baseline。
5. 只有在 baseline 建立之后，A 线后续优化与 extension 才具备可验证性与可诊断性。
