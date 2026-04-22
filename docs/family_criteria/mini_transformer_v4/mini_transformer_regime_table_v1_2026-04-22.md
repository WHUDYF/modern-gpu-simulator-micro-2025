# mini_transformer_v4 Representative Regime Table（第一版正式对象表）

日期：2026-04-22

## 1. 文档目的

这份文档用于把 `mini_transformer_v4` 的 backend 入口对象正式固定成：

**Representative Regime Table**

当前最重要的判断是：

**Regime 才是中段结构层送进 simulator lane 的直接对象。**

---

## 2. 当前 Regime 定义原则

当前第一版 regime 主要依据：

- `family_id`
- `phase_id`
- `context_scope`
- `shape_regime`
- `resource_signature`

只要这五类信息里有明显差异，就不应强行合并为同一 regime。

---

## 3. 当前正式 Regime Table

| regime_id | family_id | source_anchor_ids | phase_id | route_primitive | hardware_template | shape_regime | context_scope | resource_signature | coverage_weight | time_weight | family_importance_score | local_decision_weight | regime_priority_score | validation_status | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `R1_qkv_projection_dense` | `F1_dense_tiled_backbone` | `A1` | `Phase A` | `Dense Projection/Transform` | `Dense Tiled Compute` | `48x32 projection-like dense region` | `Q/K/V projection path` | `register-limited dense backbone` | `High` | `High` | `0.90` | `High` | `0.93` | `pending` | 当前 dense family 的首要 regime |
| `R2_attention_score_dense` | `F1_dense_tiled_backbone` | `A2` | `Phase B` | `Pairwise Score` | `Dense Tiled Compute` | `32x32x12 attention-score region` | `attention score path` | `register + shmem coupled dense compute` | `Medium` | `Medium-High` | `0.90` | `High` | `0.85` | `pending` | route 与 shmem 特征使其必须单独保留 |
| `R3_output_projection_dense` | `F1_dense_tiled_backbone` | `A5` | `Phase B_to_C` | `Dense Projection/Transform` | `Dense Tiled Compute` | `48x32 post-attention projection region` | `attention output projection path` | `dense transform with post-attention context` | `Medium` | `Medium` | `0.90` | `Medium` | `0.72` | `pending` | 与 `R1` 同 shape，但 trace context 不同 |
| `R4_ffn_expand_dense` | `F1_dense_tiled_backbone` | `A8` | `Phase C` | `Dense Projection/Transform` | `Dense Tiled Compute` | `192x32 FFN expansion region` | `FFN expansion path` | `large-shape dense compute, register-sensitive` | `Medium` | `High` | `0.90` | `High` | `0.88` | `pending` | 当前 dense family 的后段重计算对象 |
| `R5_ffn_contract_dense` | `F1_dense_tiled_backbone` | `A9` | `Phase C` | `Dense Projection/Transform` | `Dense Tiled Compute` | `48x32 FFN contraction region` | `FFN contraction path` | `dense contraction with lower local leverage than expansion` | `Medium` | `Medium` | `0.90` | `Medium` | `0.70` | `pending` | 适合作为 dense family 的补充验证对象 |
| `R6_softmax_reduction` | `F2_reduction_normalize` | `A3` | `Phase B` | `Reduction / Normalize` | `Reduction Template` | `24x1 row-wise normalization region` | `attention normalization path` | `cache-capacity-sensitive, DRAM-pressure` | `Medium` | `Medium-High` | `0.78` | `High` | `0.81` | `pending` | 当前 reduction family 中最值得先验证的对象 |
| `R7_layernorm_reduction` | `F2_reduction_normalize` | `A7` | `Phase C` | `Reduction / Normalize` | `Reduction Template` | `512x1 layernorm reduction region` | `residual normalization path` | `reduction / normalization dominated` | `Medium` | `Medium` | `0.78` | `Medium` | `0.66` | `pending` | 更适合作为 normalization path 约束对象 |
| `R8_context_streaming` | `F3_streaming_aggregation` | `A4` | `Phase B` | `Weighted Aggregation` | `Streaming Aggregation Template` | `4x32x12 weighted aggregation region` | `attention aggregation path` | `locality-dominated, L1-resident streaming` | `Medium` | `Medium` | `0.68` | `Medium-High` | `0.71` | `pending` | 当前 attention readout 中独立的非-reduction regime |
| `R9_residual_elementwise` | `F4_elementwise_residual` | `A6` | `Phase C` | `Elementwise Fusion` | `Elementwise Template` | `1536-wide residual elementwise region` | `residual path` | `lightweight elementwise memory-side` | `High` | `Low` | `0.46` | `Low` | `0.39` | `pending` | 当前优先级最低，更适合作为回归检查对象 |

---

## 4. 当前 Regime 层排序

当前第一版 regime priority 顺序建议为：

1. `R1_qkv_projection_dense`
2. `R4_ffn_expand_dense`
3. `R2_attention_score_dense`
4. `R6_softmax_reduction`
5. `R3_output_projection_dense`
6. `R8_context_streaming`
7. `R5_ffn_contract_dense`
8. `R7_layernorm_reduction`
9. `R9_residual_elementwise`

这个顺序表达的是：

- 先看 dense backbone 中最重、最敏感的部分
- 再看 attention 路径里的关键边界对象
- 最后看 constraint / regression objects

---

## 5. 为什么 Family 之后还必须保留 Regime

即使在 `F1_dense_tiled_backbone` 内，下面这些对象也不应直接混成一个 backend 单元：

- `A1` QKV projection
- `A2` attention score
- `A5` output projection
- `A8` FFN expansion
- `A9` FFN contraction

原因在于它们至少在下面三方面不同：

- `shape`
- `context`
- `resource signature`

这正是 Regime 存在的意义：

**Family 负责共享机制，Regime 负责 backend 入口。**

---

## 6. 当前阶段的简短结论

如果压成最短形式，可以写成：

1. 当前 `mini_transformer_v4` 上更合理的 backend 入口对象是 9 个 representative regimes。
2. Dense family 不能直接作为单个 backend 单元，必须继续拆成 projection / score / FFN 等 regimes。
3. 因此 Regime Table 是 middle layer 到 backend 的真正桥梁。
