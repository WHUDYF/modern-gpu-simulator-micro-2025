# mini_transformer_v4 Family Table（第一版正式对象表）

日期：2026-04-22

## 1. 文档目的

这份文档用于把 `mini_transformer_v4` 的共享机制层正式固定成：

**Family Table**

Family 的角色不是重复记录 anchors，而是把它们提升成：

**共享 data path / execution template 的组织对象**

---

## 2. 当前 Family 定义原则

当前第一版 Family 主要依据：

- `phase-aware anchors`
- `route primitive`
- `hardware execution template`
- `boundary-first` 判据

当前不直接用下面这些因素决定 Family：

- 单个 bottleneck 名称
- kernel 算子名
- 某一项孤立 hotspot 指标

---

## 3. 当前正式 Family Table

| family_id | input_anchor_ids | phase_scope | route_primitive | hardware_template | boundary_status | boundary_notes | shape_regime_summary | resource_signature_summary | coverage_weight | time_weight | decision_weight | importance_score | priority_class | recommended_tuning_target | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `F1_dense_tiled_backbone` | `A1,A2,A5,A8,A9` | `Phase A + Phase B + Phase C` | `Dense Projection/Transform + Pairwise Score` | `Dense Tiled Compute` | `weak_share_but_keep_together` | `attention_score` 与 projection/FFN dense objects 在 route 上不同，但共享 dense tiled compute 主模板；当前在 family 层合并，在 regime 层拆开 | `48x32 projection/contract, 192x32 expansion, 32x32x12 pairwise score` | `register / occupancy primary; attention_score has stronger shmem coupling` | `High` | `High` | `High` | `0.90` | `High` | `register-sensitive, occupancy-sensitive, dense tiled path` | 当前整个 workload 的主计算 backbone |
| `F2_reduction_normalize` | `A3,A7` | `Phase B + Phase C` | `Reduction / Normalize` | `Reduction Template` | `strong_share_with_context_split` | `softmax` 与 `layernorm` 在 context 上不同，但共享 reduction / normalize 主模板；必须在 regime 层拆开 | `attention normalization vs residual normalization` | `reduction / synchronization; softmax has stronger cache-capacity and DRAM-pressure signature` | `Medium` | `Medium-High` | `High` | `0.78` | `High` | `cache-capacity, reduction behavior, normalization path sensitivity` | 当前第二优先的结构对象 |
| `F3_streaming_aggregation` | `A4` | `Phase B` | `Weighted Aggregation` | `Streaming Aggregation Template` | `stable_singleton` | `context_mul` 当前没有稳定的同类对象进入合并 | `attention aggregation region` | `locality-dominated, L1-resident streaming accumulation` | `Medium` | `Medium` | `Medium-High` | `0.68` | `Medium` | `locality-sensitive, aggregation path validation` | 当前作为单锚点 family 保留 |
| `F4_elementwise_residual` | `A6` | `Phase C` | `Elementwise Fusion` | `Elementwise Template` | `stable_singleton` | `residual_add` 当前是轻量稳定的 elementwise 样本 | `residual elementwise region` | `lightweight memory-side streaming / regression constraint` | `High` | `Low` | `Low-Medium` | `0.46` | `Low` | `lightweight regression, constraint checking` | 高频但不应占据主调参预算 |

---

## 4. 当前 Family 层判断

### 4.1 `F1_dense_tiled_backbone`

这是当前最关键的 family。

它覆盖了：

- 前段 Q/K/V projection
- attention score
- output projection
- FFN expansion
- FFN contraction

当前不把这些对象直接拆成多个 family，是因为它们共享：

- dense tiled compute 主模板
- register / occupancy 主导解释

但它们不能直接作为单一 backend 对象进入 simulator，因此 regime 层仍然必须继续拆。

### 4.2 `F2_reduction_normalize`

这个 family 的核心价值在于：

- 保留 reduction / normalize 的共享结构
- 同时允许 attention-side 与 normalization-side 在 regime 层继续分开

### 4.3 `F3_streaming_aggregation`

当前作为 singleton 保留是合理的，因为：

- `context_mul` 的 locality / L1-resident 行为很稳定
- 它与 `softmax` 虽同属 attention 路径，但共享机制不够强

### 4.4 `F4_elementwise_residual`

当前主要作为：

- regression constraint object
- low-cost validation object

而不是主调参 family。

---

## 5. Family 层的当前输出作用

这张表当前承担三件事：

1. 把 phase-aware anchors 提升成共享机制对象
2. 给后续 regime 提取提供稳定上层约束
3. 给 importance ratio 提供 family-level 承载表

---

## 6. 当前阶段的简短结论

如果压成最短形式，可以写成：

1. `mini_transformer_v4` 当前可稳定形成 4 个 Family。
2. 最重要的 Family 是 `F1_dense_tiled_backbone`，但它不能直接充当 backend 入口对象。
3. 因此 Family 的任务是组织共享机制，而不是取代 Regime。
