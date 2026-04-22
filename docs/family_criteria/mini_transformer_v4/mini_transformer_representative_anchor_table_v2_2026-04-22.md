# mini_transformer_v4 Representative Anchor Table（第二版）

日期：2026-04-22

## 1. 文档目的

这份文档用于把第一版 kernel-name-aware anchor table 升级为：

**phase-aware / context-aware / shape-aware 的 operational anchor table**

第二版最关键的变化是：

- 不再把 `kernel_name` 当成 anchor 主键
- Anchor 主键提升到 `kernel + phase/context + shape`
- 让后续 `Anchor -> Family -> Regime` 的中段结构层真正可执行

---

## 2. 当前版本的定位

这份表仍然不是正式的 PKA 输出。

它的定位是：

**middle layer operational placeholder version**

也就是说：

- 它仍然允许 membership / coverage 是 placeholder
- 但对象粒度已经足够支撑后续 family / regime / lane 设计

---

## 3. 设计原则

### 3.1 Anchor 主键不再等于 kernel 名

当前 anchor 主键至少要包含：

- `kernel_name`
- `phase_id`
- `context_scope`
- `shape_hint_summary`

### 3.2 同名 kernel 可以拆成多个 anchors

例如 `gemm_tiled` 在当前 workload 中并不只代表一个执行对象。

它至少出现在：

- Q/K/V projection 路径
- output projection 路径
- FFN expansion / contraction 路径

因此不能继续只保留一个 `gemm_tiled` anchor。

### 3.3 仍然保留真实前端替换接口

即便当前是 placeholder version，也必须保留：

- `cluster_id`
- `member_invocations`
- `coverage_count`
- `coverage_weight`
- `time_weight`

后续才能平滑替换为真实 compression 输出。

---

## 4. 当前推荐的 Operational Anchors

| anchor_id | kernel_name | phase_id | context_scope | cluster_id | member_invocations | coverage_count | coverage_weight | time_weight | trace_order_summary | grid_dim_summary | block_dim_summary | shape_hint_summary | route_hint | template_hint | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A1_qkv_projection_dense_48x32` | `gemm_tiled` | `Phase A` | `Q/K/V projection path` | `C_dense_proj_qkv` | `placeholder_kernel_1_4` | `TBD` | `High` | `High` | `trace front; kernels 1..4` | `(48, 32, 1)` | `(16, 16, 1)` | `small dense projection region` | `Dense Projection/Transform` | `Dense Tiled Compute` | 当前 dense 主干的前段锚点 |
| `A2_attention_score_dense_32x32x12` | `attention_score` | `Phase B` | `attention score path` | `C_dense_attention_score` | `placeholder_kernel_5` | `TBD` | `Medium` | `Medium-High` | `attention score stage` | `(32, 32, 12)` | `(16, 16, 1)` | `pairwise score dense region` | `Pairwise Score` | `Dense Tiled Compute` | Dense family 的边界锚点 |
| `A3_softmax_reduce_24x1` | `softmax_kernel` | `Phase B` | `attention normalization path` | `C_reduce_attention_softmax` | `placeholder_kernel_6` | `TBD` | `Medium` | `Medium-High` | `attention normalize stage` | `(24, 1, 1)` | `(256, 1, 1)` | `row-wise reduction normalize region` | `Reduction / Normalize` | `Reduction Template` | Reduction family 的 attention 侧锚点 |
| `A4_context_stream_4x32x12` | `context_mul` | `Phase B` | `attention aggregation path` | `C_stream_attention_context` | `placeholder_kernel_7` | `TBD` | `Medium` | `Medium` | `attention aggregation stage` | `(4, 32, 12)` | `(16, 16, 1)` | `streaming weighted aggregation region` | `Weighted Aggregation` | `Streaming Aggregation Template` | Streaming family 主锚点 |
| `A5_output_projection_dense_48x32` | `gemm_tiled` | `Phase B_to_C` | `attention output projection path` | `C_dense_output_proj` | `placeholder_kernel_8` | `TBD` | `Medium` | `Medium` | `post-attention dense projection` | `(48, 32, 1)` | `(16, 16, 1)` | `post-attention dense projection region` | `Dense Projection/Transform` | `Dense Tiled Compute` | 与 `A1` 同 shape，但 context 不同 |
| `A6_residual_elementwise_1536` | `residual_add` | `Phase C` | `residual path` | `C_elementwise_residual` | `placeholder_kernel_9_13` | `TBD` | `High` | `Low` | `repeated after attention and FFN` | `(1536, 1, 1)` | `(256, 1, 1)` | `elementwise residual region` | `Elementwise Fusion` | `Elementwise Template` | 当前轻量但高频的约束锚点 |
| `A7_layernorm_reduce_512` | `layernorm_kernel` | `Phase C` | `normalization path` | `C_reduce_layernorm` | `placeholder_kernel_10_14` | `TBD` | `Medium` | `Medium` | `repeated after residuals` | `(512, 1, 1)` | `(256, 1, 1)` | `layernorm reduction region` | `Reduction / Normalize` | `Reduction Template` | Reduction family的 normalization 侧锚点 |
| `A8_ffn_expand_dense_192x32` | `gemm_tiled` | `Phase C` | `FFN expansion path` | `C_dense_ffn_expand` | `placeholder_kernel_11` | `TBD` | `Medium` | `High` | `FFN expansion stage` | `(192, 32, 1)` | `(16, 16, 1)` | `large dense transform region` | `Dense Projection/Transform` | `Dense Tiled Compute` | Dense family的后段重计算锚点 |
| `A9_ffn_contract_dense_48x32` | `gemm_tiled` | `Phase C` | `FFN contraction path` | `C_dense_ffn_contract` | `placeholder_kernel_12` | `TBD` | `Medium` | `Medium` | `FFN contraction stage` | `(48, 32, 1)` | `(16, 16, 1)` | `dense contraction region` | `Dense Projection/Transform` | `Dense Tiled Compute` | 与 `A1/A5` 同 shape，但 route context 更靠近 FFN |

---

## 5. 当前版本相对第一版的升级点

### 5.1 从 6 个 kernel-name anchors 升级到 9 个 operational anchors

当前最重要的升级是：

- `gemm_tiled` 被拆成多个 context-aware anchors
- `residual_add` 和 `layernorm_kernel` 保留重复路径但暂时合并

### 5.2 主键从 name-aware 升级为 phase/context/shape-aware

这让后续：

- Family 不必面对过于粗糙的输入
- Regime 可以直接继承更稳定的 shape/context 信息

### 5.3 为真实 frontend 替换保留接口

当前 placeholder 字段包括：

- `cluster_id`
- `member_invocations`
- `coverage_count`

后续只需替换这些字段，不必推翻中段结构层对象本身。

---

## 6. 当前使用方式

这张表当前应这样进入后续方法线：

### Step 1：Anchor -> Family

优先依据：

- `phase_id`
- `route_hint`
- `template_hint`
- `context_scope`

### Step 2：Family -> Regime

优先依据：

- `shape_hint_summary`
- `context_scope`
- `resource_signature`

### Step 3：Regime -> Lane

用 regime 作为 backend 的直接入口，而不是让 lane 直接挂在 anchor 上。

---

## 7. 当前阶段的简短结论

如果压成最短形式，可以写成：

1. 第一版 anchor table 解决了“有无输入接口”的问题。
2. 第二版 anchor table 解决了“输入对象粒度过粗”的问题。
3. 对 `mini_transformer_v4`，当前更合理的 middle-layer 输入是 9 个 phase/context/shape-aware anchors，而不是 6 个 kernel-name-aware anchors。
