# mini_transformer_v4 Representative Anchor Table（第一版）

日期：2026-04-22

## 1. 文档目的

这份文档用于生成当前方法线中缺失的一张关键表：

**Representative Anchor Table**

这张表的作用是：

1. 把前端 compression 输出显式化
2. 让 `PKA-style frontend -> family -> regime` 链路真正接起来
3. 为后续替换成真实前端输出留下稳定接口

当前第一版不要求：

- 完整复现 PKA
- 真实跑出 clustering 结果

但要求：

**先把最小可运行的 anchor 对象定义出来。**

---

## 2. 当前版本的定位

当前这份 Anchor Table 是：

**前端锚点占位版（frontend-anchor placeholder version）**

这意味着：

- 它先用当前已知的 representative kernels 构成一版 anchor table
- 后续可以逐步替换成真实的 `PKA-style compression output`

当前这么做的理由是：

- 我们不能等前端完全实现后才定义输入对象
- 否则 family / regime / importance ratio 无法真正落地

因此：

**当前版本的目标是先让后段方法链有输入接口。**

---

## 3. 当前 Anchor Table 的设计原则

### 原则 1：Anchor 不是最终 family

每个 anchor 只是：

- compression 之后的代表锚点

而不是：

- 最终机制解释对象

### 原则 2：Anchor 必须保留 membership / weight 接口

即便当前第一版还没有真实 PKA 输出，也必须在 schema 中预留：

- `cluster_id`
- `member_invocations`
- `coverage_count`
- `coverage_weight`
- `time_weight`

否则后续 family / regime 的重要性无法回写到 workload。

### 原则 3：Anchor 进入 family 前必须带 phase context

当前 family 层的定义依赖：

- phase
- route primitive
- hardware template

因此当前 anchor table 中必须保留：

- `phase_id`
- `trace_order_summary`
- `shape_hint_summary`

---

## 4. 当前 representative anchors 的来源

在 `mini_transformer_v4` 上，当前已明确的 representative kernel-level objects 包括：

- `gemm_tiled`
- `attention_score`
- `softmax_kernel`
- `context_mul`
- `layernorm_kernel`
- `residual_add`

这 6 个对象当前已经在：

- route primitive / template 对照表
- analysis cards
- family / regime table

中被固定下来，因此它们可以先构成第一版 representative anchors。

需要强调：

**这 6 个对象当前是“方法原型 representative anchors”，而不是正式 PKA 输出。**

---

## 5. 当前 Anchor Table

| rep_kernel_id | kernel_name | cluster_id | member_invocations | coverage_count | coverage_weight | time_weight | trace_order_summary | phase_id | grid_dim_summary | block_dim_summary | shape_hint_summary | route_hint | template_hint | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A1` | `gemm_tiled` | `C_dense_main` | `placeholder_gemm_invocations` | `TBD` | `High` | `High` | `projection / FFN route repeated across layers` | `Phase A` | `TBD` | `TBD` | `projection-like dense M/N/K region` | `Dense Projection/Transform` | `Dense Tiled Compute` | 当前作为 dense projection 主锚点 |
| `A2` | `attention_score` | `C_dense_attention` | `placeholder_attention_score_invocations` | `TBD` | `Medium` | `Medium-High` | `attention score repeated in readout stages` | `Phase B` | `TBD` | `TBD` | `attention-score dense region` | `Pairwise Score` | `Dense Tiled Compute` | 当前作为 dense tiled compute 边界锚点 |
| `A3` | `softmax_kernel` | `C_reduce_attention` | `placeholder_softmax_invocations` | `TBD` | `Medium` | `Medium-High` | `attention normalization stage` | `Phase B` | `TBD` | `TBD` | `softmax row-wise normalize region` | `Reduction / Normalize` | `Reduction Template` | 当前作为 memory-side / reduction 边界锚点 |
| `A4` | `context_mul` | `C_streaming_attention` | `placeholder_context_mul_invocations` | `TBD` | `Medium` | `Medium` | `attention aggregation stage` | `Phase B` | `TBD` | `TBD` | `attention aggregation region` | `Weighted Aggregation` | `Streaming Aggregation Template` | 当前作为 streaming aggregation 主锚点 |
| `A5` | `layernorm_kernel` | `C_reduce_norm` | `placeholder_layernorm_invocations` | `TBD` | `Medium` | `Medium` | `normalization path repeated after residuals` | `Phase C` | `TBD` | `TBD` | `layernorm reduction region` | `Reduction / Normalize` | `Reduction Template` | 当前作为 normalization-path reduction 锚点 |
| `A6` | `residual_add` | `C_elementwise_residual` | `placeholder_residual_add_invocations` | `TBD` | `High` | `Low` | `residual path repeated frequently` | `Phase C` | `TBD` | `TBD` | `residual elementwise region` | `Elementwise Fusion` | `Elementwise Template` | 当前作为轻量 elementwise 锚点 |

---

## 6. 当前字段解释

### 6.1 `rep_kernel_id`

这是 anchor 的稳定标识，后续 family / regime / simulator lane 都应通过它引用输入对象。

### 6.2 `cluster_id`

当前这列是：

**前端 compression cluster 的占位字段**

它现在不是正式 PKA 结果，而是：

- 预留后续真实前端输出接口

### 6.3 `member_invocations`

当前也是占位字段。

后续一旦前端 PKA-style compression 跑通，这里应替换成：

- 被该 representative anchor 代表的 invocation 列表

### 6.4 `coverage_weight`

当前第一版是半定量：

- `High / Medium`

其作用是先给后续 family 层提供一个 provisional coverage signal。

### 6.5 `time_weight`

当前也是半定量：

- 基于 analysis cards 和当前 family / regime 判断给出

后续应替换成：

- silicon measurement
或
- representative time aggregation

### 6.6 `phase_id`

这是当前最关键的字段之一。

它保证 representative anchors 进入 family 之前，已经具备：

- phase context

也就是说，family 层不是直接吃“裸 kernel”，而是吃：

**phase-annotated representative anchors**

### 6.7 `route_hint` 与 `template_hint`

这两列不是为了直接替代 family 判定，而是：

- 给 family 层提供初始结构提示

真正 family 形成时，仍需结合：

- boundary cases
- family protocol

做最终判定。

---

## 7. 为什么当前 6 个 representative anchors 是合理的

当前之所以先用这 6 个对象，不是因为它们已经被 PKA 正式压缩出来，而是因为：

### 7.1 它们已经覆盖了当前最重要的四类机制

- Dense Tiled Compute
- Reduction / Normalize
- Streaming Aggregation
- Elementwise Fusion

### 7.2 它们已经形成了关键边界 case

例如：

- `gemm_tiled` vs `attention_score`
- `softmax_kernel` vs `context_mul`

### 7.3 它们已经足以支撑第一版 family / regime / importance 结构

也就是说，当前第一版 anchor table 的目标不是：

- 完整代表所有 workload

而是：

- 为方法原型提供一组可运行 anchors

---

## 8. 当前 Anchor Table 如何接到后续方法线

当前这张表后续应这样使用：

### Step 1：Anchor -> Family

根据：

- `phase_id`
- `route_hint`
- `template_hint`
- boundary notes

把 anchors 映射到：

- `Family Table`

### Step 2：Family -> Regime

根据：

- `shape_hint_summary`
- `context_scope`
- `resource_signature`

把同一 family 内的 anchors 映射到：

- `Regime Table`

### Step 3：Importance Ratio

使用：

- `coverage_weight`
- `time_weight`
- 后续补充的 `decision_weight`

计算：

- family importance
- regime priority

---

## 9. 当前最需要后续真实替换的字段

当前第一版表里最需要后续由真实前端替换的字段包括：

### 9.1 `cluster_id`

后续应由 PKA-style compression 真正生成。

### 9.2 `member_invocations`

后续应由 compression membership 真正生成。

### 9.3 `coverage_count`

后续应由 membership 或 representative coverage 统计得到。

### 9.4 `coverage_weight`

后续应由真实 invocation coverage 替换当前 `High / Medium`。

### 9.5 `time_weight`

后续应由 measured / simulated weighted time 替换当前 provisional label。

---

## 10. 当前阶段的简短结论

如果把这份文档压成最短形式，可以写成：

1. Representative Anchor Table 是连接前端 compression 与后续 family / regime 的关键输入表。
2. 当前 `mini_transformer_v4` 已经可以先用 6 个 representative kernel-level objects 构成第一版 anchors。
3. 这张表的当前定位是“前端锚点占位版”，目的是先让后段方法有明确输入对象。
4. 后续只要把 `cluster_id / membership / coverage / time` 四类字段替换成真实前端输出，这张表就能自然演化成正式的 compression input interface。
