# mini_transformer_v4 Family / Regime 表（第一版）

日期：2026-04-22

## 1. 文档目的

这份文档用于把当前 `mini_transformer_v4` 上已经讨论清楚的结构对象，真正落成第一版可用的：

- `Family Table`
- `Representative Execution Regime Table`

当前这不是最终定量版，而是：

**半定量、可运行、可继续被真实数据替换的第一版。**

它的作用是：

1. 先把当前方法线里的对象层级固定下来
2. 给后续 importance ratio 和 simulator lane 提供真实落点
3. 让“family / regime / priority”不再只停留在概念层

---

## 2. 当前使用说明

这张表的当前状态有三个特点：

### 2.1 它依赖现有 mini_transformer_v4 证据

当前表格主要基于：

- route primitive / hardware template 对照表
- boundary case 文档
- analysis cards
- current-goal / protocol 文档

### 2.2 它是第一版半定量表

目前：

- `coverage_weight`
- `time_weight`
- `decision_weight`

更多是：

- `High / Medium / Low`
- 配合一个 provisional score

而不是最终测量值。

### 2.3 它的目标是先形成稳定排序

当前更重要的是先把：

- family 边界
- regime 粒度
- priority 顺序

钉住，而不是一开始追求每个分数绝对准确。

---

## 3. 当前 Family Table

| family_id | phase_scope | route_primitive | hardware_template | member_rep_kernels | boundary_status | shape_regime_summary | resource_signature_summary | coverage_weight | time_weight | decision_weight | importance_score | priority_class | recommended_tuning_target | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `F1_dense_tiled` | `Phase A + Phase B` | `Dense Projection/Transform + Pairwise Score` | `Dense Tiled Compute` | `gemm_tiled`, `attention_score` | `weak_share` | `dense matmul-like shapes; projection and score contexts need split at regime level` | `register / occupancy primary; attention_score carries stronger shmem / waves signature` | `High` | `High` | `High` | `0.90` | `High` | `register-sensitive, occupancy-sensitive, tiled compute path` | 当前是最重要的 compute 主干 family，但内部不能直接揉平为单一 regime |
| `F2_reduction_normalize` | `Phase B + Phase C` | `Reduction / Normalize` | `Reduction Template` | `softmax_kernel`, `layernorm_kernel` | `strong_share_with_context_split` | `attention-normalize vs normalization-path should be split at regime level` | `reduction / synchronization; softmax more cache-capacity / DRAM-pressure sensitive` | `Medium` | `Medium-High` | `High` | `0.78` | `High` | `cache-capacity, reduction behavior, normalization path sensitivity` | family 层共享 reduction 机制，但 `softmax` 与 `layernorm` 上下文不同 |
| `F3_streaming_aggregation` | `Phase B` | `Weighted Aggregation` | `Streaming Aggregation Template` | `context_mul` | `stable_singleton` | `attention readout aggregation shapes` | `locality-dominated / L1-resident / streaming accumulation` | `Medium` | `Medium` | `Medium-High` | `0.68` | `Medium` | `locality-sensitive, aggregation path validation` | 当前以单代表对象存在，未来可能因 shape 再分多个 regime |
| `F4_elementwise_fusion` | `Phase C` | `Elementwise Fusion` | `Elementwise Template` | `residual_add` | `stable_singleton` | `residual elementwise paths` | `lightweight memory-side fusion` | `High` | `Low` | `Low-Medium` | `0.46` | `Low` | `lightweight regression / constraint checking` | 高频但不是主时间贡献，更多适合作为约束 family |

---

## 4. Family Table 解释

### 4.1 `F1_dense_tiled`

这是当前最重要的 family。

原因：

- 它覆盖了 projection / score 两条核心路径
- `gemm_tiled` 是主计算骨架
- `attention_score` 虽然 route 不同，但共享 dense tiled compute 模板
- 对 register / occupancy / tiled compute 调参高度敏感

它的重要性高，不是因为所有成员完全同质，而是因为：

**它代表了 workload 中最关键的 dense compute 主干。**

### 4.2 `F2_reduction_normalize`

这是当前第二重要的 family。

原因：

- `softmax` 与 `layernorm` 共享 reduction / normalize 机制
- 这类对象虽然不是最重的 compute 核，但常常会形成重要瓶颈或约束
- 尤其 `softmax` 对 cache-capacity / DRAM-pressure 有特殊价值

### 4.3 `F3_streaming_aggregation`

这是 attention readout 中独立的 aggregation family。

它当前的重要性低于 `F1 / F2`，但仍然不可忽略，因为：

- 它代表了 attention 路线里与 reduction 完全不同的一类机制
- 它有明显 locality / streaming signature

### 4.4 `F4_elementwise_fusion`

这是当前最轻量的 family。

它的重要性较低，但仍应保留，因为：

- 它高频出现
- 更适合作为调参约束或回归检查对象

---

## 5. 当前 Representative Execution Regime Table

| regime_id | family_id | phase_id | route_primitive | hardware_template | source_rep_kernels | shape_regime | context_scope | resource_signature | coverage_weight | time_weight | family_importance_score | local_decision_weight | regime_priority_score | simulator_lane_id | validation_status | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `R1_projection_dense` | `F1_dense_tiled` | `Phase A` | `Dense Projection/Transform` | `Dense Tiled Compute` | `gemm_tiled` | `projection-like dense M/N/K region` | `QKV / output projection / FFN projection context` | `register-limited compute-heavy` | `High` | `High` | `0.90` | `High` | `0.92` | `L1_dense_projection` | `pending` | 当前最优先进入 simulator lane 的 regime |
| `R2_attention_score_dense` | `F1_dense_tiled` | `Phase B` | `Pairwise Score` | `Dense Tiled Compute` | `attention_score` | `attention-score dense region` | `attention readout` | `register + shmem coupled dense compute` | `Medium` | `Medium-High` | `0.90` | `High` | `0.84` | `L2_attention_score` | `pending` | 与 `R1` 同 family，但因 route / shmem 特征不同单独成 regime |
| `R3_softmax_reduction` | `F2_reduction_normalize` | `Phase B` | `Reduction / Normalize` | `Reduction Template` | `softmax_kernel` | `softmax row-wise normalize region` | `attention readout` | `cache-capacity-sensitive / DRAM-pressure` | `Medium` | `Medium-High` | `0.78` | `High` | `0.81` | `L3_softmax` | `pending` | 重点验证 cache / DRAM 相关参数 |
| `R4_layernorm_reduction` | `F2_reduction_normalize` | `Phase C` | `Reduction / Normalize` | `Reduction Template` | `layernorm_kernel` | `layernorm reduction region` | `normalization path` | `reduction / normalization dominated` | `Medium` | `Medium` | `0.78` | `Medium` | `0.67` | `L4_layernorm` | `pending` | 与 `R3` 同 family，但主要起 normalization-path 验证作用 |
| `R5_context_streaming` | `F3_streaming_aggregation` | `Phase B` | `Weighted Aggregation` | `Streaming Aggregation Template` | `context_mul` | `attention aggregation region` | `attention readout` | `L1-resident / locality-dominated streaming` | `Medium` | `Medium` | `0.68` | `Medium-High` | `0.71` | `L5_context_mul` | `pending` | 重点验证 locality / aggregation 路径 |
| `R6_residual_elementwise` | `F4_elementwise_fusion` | `Phase C` | `Elementwise Fusion` | `Elementwise Template` | `residual_add` | `residual fusion region` | `residual path` | `lightweight elementwise memory-side` | `High` | `Low` | `0.46` | `Low` | `0.39` | `L6_residual_add` | `pending` | 优先级最低，更适合作为约束与回归检查 |

---

## 6. 当前排序的直观解释

### 6.1 当前 family 层排序

当前第一版排序为：

1. `F1_dense_tiled`
2. `F2_reduction_normalize`
3. `F3_streaming_aggregation`
4. `F4_elementwise_fusion`

这表示：

- 先看 dense compute 主干
- 再看 reduction / normalize
- 再看 aggregation
- 最后看 elementwise fusion

### 6.2 当前 regime 层排序

当前第一版排序为：

1. `R1_projection_dense`
2. `R2_attention_score_dense`
3. `R3_softmax_reduction`
4. `R5_context_streaming`
5. `R4_layernorm_reduction`
6. `R6_residual_elementwise`

这个顺序表达的是：

- 先压 dense compute 主干
- 再补 attention route 中的重要分支
- 再看 normalize 和约束对象

---

## 7. 为什么当前先做半定量是合理的

当前最重要的不是马上得到绝对精确的分数，而是先把：

- 哪些对象存在
- 哪些对象属于同一 family
- 哪些需要继续拆成不同 regime
- 大致优先顺序是什么

这些问题固定下来。

如果没有这一版表，后续：

- importance ratio 只是抽象定义
- validation plan 无法真正接对象
- simulator lane 也无法具体进入

因此，这份文档的价值在于：

**先把方法对象实体化。**

---

## 8. 下一步应该如何用真实数据替换

后续建议按下面顺序替换当前半定量字段：

### Step 1

用 representative anchor 表替换：

- `coverage_weight`

### Step 2

用 profiling / measured time / simulator baseline 替换：

- `time_weight`

### Step 3

用 sensitivity experiment 替换：

- `decision_weight`

### Step 4

重新计算：

- `importance_score`
- `regime_priority_score`

这样就可以把这份第一版表逐步升级为正式实验表。

---

## 9. 当前阶段的简短结论

如果把这份文档压成最短形式，可以写成：

1. 当前已经可以在 `mini_transformer_v4` 上落下第一版 Family Table 和 Regime Table。
2. Family 层当前可稳定形成四个对象：Dense Tiled、Reduction/Normalize、Streaming Aggregation、Elementwise Fusion。
3. Regime 层当前可稳定形成六个对象，对应六个关键 representative kernels 的执行区间。
4. 当前分数仍是半定量，但已经足以支撑后续 importance ratio、validation plan 和 simulator lane 的继续实现。
