# mini_transformer_v4 Simulator Lane Mapping（第一版）

日期：2026-04-22

## 1. 文档目的

这份文档用于把 `mini_transformer_v4` 的 representative regimes 接到 backend validation / tuning workflow。

它的目标不是直接给出完整参数处方，而是先固定：

- 每个 regime 去哪条 lane
- 该 lane 主要验证什么
- 该 lane 优先看什么参数方向
- 结果如何回写

---

## 2. 当前 Lane Mapping 原则

### 2.1 Lane 直接对接 Regime，而不是直接对接 Family

Family 是共享机制层。

Lane 需要对接的对象应是：

**具有稳定 shape/context/resource signature 的 regime**

### 2.2 Lane 先固定参数方向，不一开始追求完整参数表

第一版 lane mapping 只需要固定：

- `parameter_direction`
- `baseline_type`
- `validation_metric`
- `writeback_target`

### 2.3 Lane 必须支持结果回写

每条 lane 至少应支持：

- `lane result -> regime`
- `regime conclusion -> family`
- `family conclusion -> workload explanation`

---

## 3. 当前 Simulator Lane Mapping Table

| lane_id | target_regime_id | target_family_id | lane_goal | parameter_direction | baseline_type | validation_metric | writeback_target | notes |
|---|---|---|---|---|---|---|---|---|
| `L1_dense_projection` | `R1_qkv_projection_dense` | `F1_dense_tiled_backbone` | `验证 dense backbone 前段 projection 是否主导 register / occupancy 敏感性` | `register-sensitive, occupancy-sensitive` | `importance-guided vs time-only` | `cycles delta, occupancy response, top-k coverage gain` | `R1 -> F1 -> dense backbone summary` | 当前第一优先 lane |
| `L2_attention_score` | `R2_attention_score_dense` | `F1_dense_tiled_backbone` | `验证 attention score 是否需要与一般 dense projection 分开处理` | `shared-memory-coupled, register-sensitive` | `importance-guided vs manual` | `cycles delta, l1/l2 behavior shift, shmem-coupled response` | `R2 -> F1 boundary refinement` | 主要服务 dense family 边界验证 |
| `L3_output_projection` | `R3_output_projection_dense` | `F1_dense_tiled_backbone` | `验证 post-attention dense projection 是否可复用 projection lane 结论` | `register-sensitive, projection-path reuse` | `importance-guided vs family-shared baseline` | `cycles delta, reuse consistency, lane overlap` | `R3 -> F1 dense reuse note` | 当前作为 dense reuse 检查 lane |
| `L4_ffn_expand` | `R4_ffn_expand_dense` | `F1_dense_tiled_backbone` | `验证 FFN expansion 是否是 dense family 中最重的后段 regime` | `register-sensitive, large-shape dense compute` | `importance-guided vs time-only` | `cycles delta, priority rank gain, sensitivity concentration` | `R4 -> F1 FFN summary` | 当前 dense family 的第二主 lane |
| `L5_ffn_contract` | `R5_ffn_contract_dense` | `F1_dense_tiled_backbone` | `验证 FFN contraction 是否只需轻量补充验证` | `dense contraction reuse, occupancy-sensitive` | `importance-guided vs no-priority` | `cycles delta, marginal gain` | `R5 -> F1 secondary regime note` | 当前低于 `L4` 的后段 lane |
| `L6_softmax` | `R6_softmax_reduction` | `F2_reduction_normalize` | `验证 softmax 的 cache-capacity / DRAM-pressure 解释是否稳定` | `cache-sensitive, reduction-sensitive` | `importance-guided vs time-only` | `cycles delta, dram throughput response, cache behavior response` | `R6 -> F2 reduction summary` | 当前 reduction family 的主 lane |
| `L7_layernorm` | `R7_layernorm_reduction` | `F2_reduction_normalize` | `验证 layernorm 主要作为 normalization-path constraint object 是否合理` | `reduction-sensitive, normalization-path validation` | `importance-guided vs family-shared baseline` | `cycles delta, normalization consistency` | `R7 -> F2 normalization note` | 更偏约束与一致性检查 |
| `L8_context_streaming` | `R8_context_streaming` | `F3_streaming_aggregation` | `验证 locality-dominated streaming aggregation 是否需要独立 lane` | `locality-sensitive, L1-sensitive` | `importance-guided vs manual` | `cycles delta, l1 hit response, locality concentration` | `R8 -> F3 streaming summary` | 当前 attention aggregation 的独立 lane |
| `L9_residual_regression` | `R9_residual_elementwise` | `F4_elementwise_residual` | `把 residual elementwise 作为轻量 regression / constraint lane` | `lightweight memory-side, regression-check` | `no-priority baseline` | `correctness-preserving delta, regression stability` | `R9 -> F4 residual constraint note` | 不作为主调参 lane |

---

## 4. 当前 Lane 层的最小验证指标

第一版每条 lane 至少应记录以下一组指标：

1. `cycles delta`
2. `对应资源指标变化`
3. `importance-guided 与 baseline 的排序收益`
4. `结果回写摘要`

---

## 5. 当前 Lane 层的最小 Baseline 集合

第一版建议统一保留下面三种 baseline：

1. `No Priority`
2. `Time-Only Priority`
3. `Importance-Guided Priority`

如果某条 lane 的目标是边界验证，可额外加入：

4. `Manual / Family-Shared Baseline`

---

## 6. 当前阶段的简短结论

如果压成最短形式，可以写成：

1. Lane Mapping 的作用是把 regime 从静态分类对象变成 backend 可验证对象。
2. `mini_transformer_v4` 当前已经可以形成 9 条第一版 lanes。
3. 其中优先级最高的 lanes 是 `L1_dense_projection`、`L4_ffn_expand`、`L2_attention_score`、`L6_softmax`。
