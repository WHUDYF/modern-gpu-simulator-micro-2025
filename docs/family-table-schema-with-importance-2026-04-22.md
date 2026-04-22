# Family 与 Regime 表结构（含 Importance 字段）

日期：2026-04-22

## 1. 文档目的

这份文档用于把当前方法线里最关键的结构化对象表正式固定下来。

当前我们已经有：

- `PKA` 前端压缩锚点
- `squash` 的 phase 结构
- `batch / family` 的共享机制组织
- `importance ratio` 的第一版定义

但如果这些对象没有统一 schema，那么后续：

- 实现会反复返工
- importance ratio 难以真正落地
- simulator side 的 tuning priority 也无法稳定串起来

因此，这份文档的目标是：

1. 固定 `family table` 和 `regime table` 的字段
2. 说明每个字段的角色
3. 把 `importance ratio` 明确嵌入 schema

---

## 2. 当前对象层级

当前方法线中的结构化对象层级可以写成：

`representative kernel anchors -> family -> representative execution regime -> simulator lane`

这意味着我们至少需要三张表：

1. `Representative Anchor Table`
2. `Family Table`
3. `Regime Table`

本文件重点放在后两张表，但会先给出 Anchor Table 的最小接口，保证链路完整。

---

## 3. Anchor Table（最小接口）

### 3.1 作用

这张表承接前端 PKA-style compression 输出，作为后续 family 构建的输入。

### 3.2 建议字段

| 字段 | 类型建议 | 含义 |
|---|---|---|
| `rep_kernel_id` | string | representative kernel 标识 |
| `kernel_name` | string | kernel 名称 |
| `cluster_id` | string | 所属 compression cluster |
| `member_invocations` | list/string | 被代表 invocation 列表 |
| `coverage_count` | int | 覆盖 invocation 数 |
| `coverage_weight` | float | 覆盖比例 |
| `time_weight` | float | 时间占比 |
| `trace_order_summary` | string/int | 在 trace 中的位置摘要 |
| `phase_id` | string | 所属 phase |
| `grid_dim_summary` | string | grid 信息摘要 |
| `block_dim_summary` | string | block 信息摘要 |
| `shape_hint_summary` | string | 形状提示摘要 |

### 3.3 说明

这张表不是最终调参对象表，只是：

**family 层的输入锚点表**

---

## 4. Family Table

### 4.1 Family Table 的角色

Family Table 的作用不是重复记录 representative kernels，而是把它们提升成：

**共享机制层的组织对象。**

它回答的是：

- 哪些代表对象共享同一种机制
- 哪个 family 覆盖多大 workload
- 哪个 family 在后续调参中更重要

### 4.2 建议字段

| 字段 | 类型建议 | 含义 |
|---|---|---|
| `family_id` | string | family 标识 |
| `phase_scope` | string/list | 主要覆盖的 phase |
| `route_primitive` | string | 路径角色 |
| `hardware_template` | string | 执行模板 |
| `member_rep_kernels` | list/string | 所包含 representative anchors |
| `member_count` | int | 包含的 representative anchors 数量 |
| `boundary_status` | string | strong share / weak share / split / unresolved |
| `boundary_notes` | string | 共享点 / 区分点说明 |
| `shape_regime_summary` | string | family 内 shape 区间摘要 |
| `resource_signature_summary` | string | 主导资源行为摘要 |
| `coverage_weight` | float | family 覆盖权重 |
| `time_weight` | float | family 时间权重 |
| `decision_weight` | float/string | family 决策权重 |
| `importance_score` | float | 综合 importance 分数 |
| `priority_class` | string | High / Medium / Low |
| `recommended_tuning_target` | string | 优先关注的调参方向 |
| `notes` | string | 额外备注 |

### 4.3 核心字段解释

#### `route_primitive`

表示这个 family 在 workload 主路径中的功能角色，例如：

- `Dense Projection/Transform`
- `Pairwise Score`
- `Reduction / Normalize`
- `Weighted Aggregation`
- `Elementwise Fusion`

#### `hardware_template`

表示这个 family 在 GPU 上的执行骨架，例如：

- `Dense Tiled Compute`
- `Reduction Template`
- `Streaming Aggregation Template`
- `Elementwise Template`

#### `boundary_status`

表示该 family 是否稳定、是否仍处于边界候选状态。

#### `importance_score`

表示这个 family 在后续调参与验证中的综合优先级。

### 4.4 Family Table 的输出作用

这张表是：

**importance ratio 生成的第一层载体**

后续：

- tuning priority
- simulator validation lane
- regime selection

都会从这张表继续展开。

---

## 5. Regime Table

### 5.1 Regime Table 的角色

Regime Table 是最终进入 simulator lane 的对象表。

它回答的是：

- 同一 family 内哪些执行区间值得被单独看待
- 哪些 shape / context / resource 条件下构成稳定 regime
- 后续实际进入 simulator 的单位是什么

### 5.2 建议字段

| 字段 | 类型建议 | 含义 |
|---|---|---|
| `regime_id` | string | regime 标识 |
| `family_id` | string | 所属 family |
| `phase_id` | string | 所属 phase |
| `route_primitive` | string | 路径角色 |
| `hardware_template` | string | 执行模板 |
| `source_rep_kernels` | list/string | 来源 representative anchors |
| `shape_regime` | string | shape 区间定义 |
| `context_scope` | string | 所属 layer / route / trace context |
| `resource_signature` | string | 主导资源机制 |
| `coverage_weight` | float | regime 覆盖权重 |
| `time_weight` | float | regime 时间权重 |
| `family_importance_score` | float | 上层 family 的 importance |
| `local_decision_weight` | float/string | regime 局部调参权重 |
| `regime_priority_score` | float | regime 综合优先级 |
| `simulator_lane_id` | string | 进入哪个 simulator lane |
| `validation_status` | string | pending / selected / validated |
| `notes` | string | 额外备注 |

### 5.3 核心字段解释

#### `shape_regime`

这是 regime 与 family 最大的差别来源之一。

例如：

- 小序列长度 attention score
- 大序列长度 attention score
- 小 M/N/K dense tiled compute
- 大 M/N/K dense tiled compute

#### `context_scope`

表示该 regime 所属的工作负载上下文，不只是 shape。

#### `regime_priority_score`

这是后续 simulator lane 中真正用于排序的字段。

### 5.4 Regime Table 的输出作用

Regime Table 是：

**后续 simulator perturbation / validation 的直接入口表**

---

## 6. importance 字段如何落入 schema

### 6.1 Family 层

family 层至少应记录：

- `coverage_weight`
- `time_weight`
- `decision_weight`
- `importance_score`
- `priority_class`

其中：

- `importance_score` 是综合分
- `priority_class` 是便于后续操作的离散标签

### 6.2 Regime 层

regime 层至少应记录：

- `family_importance_score`
- `local_decision_weight`
- `regime_priority_score`

这意味着：

- family importance 是上层约束
- regime priority 是最终 simulator 入口排序

---

## 7. family 与 regime 的区别必须体现在 schema 中

这是最重要的地方之一。

### 7.1 family 更像“共享机制层”

family 的主要作用是：

- 聚合同类
- 定义机制边界
- 给出重要性排序

### 7.2 regime 更像“执行区间层”

regime 的主要作用是：

- 在 family 内进一步按 shape / context / resource 拆分
- 形成实际进入 simulator lane 的对象

所以 schema 上必须明确：

- family 不能直接替代 regime
- importance score 不能只停留在 family 层

---

## 8. 当前推荐的最小可实现版本

为了避免第一版 schema 过重，建议：

### Family Table 最小版必须有

- `family_id`
- `route_primitive`
- `hardware_template`
- `member_rep_kernels`
- `coverage_weight`
- `time_weight`
- `decision_weight`
- `importance_score`
- `priority_class`

### Regime Table 最小版必须有

- `regime_id`
- `family_id`
- `phase_id`
- `shape_regime`
- `resource_signature`
- `family_importance_score`
- `regime_priority_score`
- `simulator_lane_id`

只要这一版能跑通，后续就能开始真正做：

- lane selection
- validation ordering
- tuning priority

---

## 9. 当前阶段的简短结论

如果把这份文档压成最短形式，可以写成：

1. Anchor Table 承接前端 compression 输出，Family Table 承接共享机制组织，Regime Table 承接最终 simulator lane 输入。
2. Family Table 必须显式加入 `coverage_weight`、`time_weight`、`decision_weight` 和 `importance_score`。
3. Regime Table 必须显式加入 `family_importance_score` 和 `regime_priority_score`，保证后续进入 simulator lane 的对象可排序。
4. family 与 regime 的区别必须在 schema 中被明确保留，否则 importance ratio 无法真正服务后续调参与验证。
