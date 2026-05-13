# PKA 与 B/C 线 Contract Method 设计

日期：2026-05-13

## 1. 目标

本 spec 定义 PKA representative compression 与 B/C 线决策层之间的接口方法。

核心目标不是继续优化 PKA selector，也不是立即实现完整 GCL，而是固定一条 representation-measured-only 的方法链：

```text
PKA representative anchors
  + 12D measured representation weights
  + anchor membership / coverage
  + measured timing and network-structure prior
  -> B-line simulator-side decision objects
  -> C-line backend validation planning
```

PKA 保持 A 线职责：压缩 kernel records，输出 representative anchors、membership、coverage 和压缩质量信息。

B 线承担方法贡献：把 PKA anchors 提升为 simulator-side decision objects，即 family、regime、lane 和 importance。

C 线承担验证贡献：检查 B 线给出的对象排序和 lane 是否真的改善 backend planning，而不是只证明 artifact pipeline 能跑。

这里的 measured-only gate 指的是 12D representation 必须来自 measured features。时间重要性可以由 measured timing、cycle proxy、coverage weight 和带 provenance 的 network-structure prior 共同组成；network prior 可以进入 formal importance，但不能伪装成 measured timing。

## 2. 非目标

本 spec 不做以下事情：

- 不重新设计 PKA selector；
- 不把 PKA cluster 直接定义为 family；
- 不允许 B 线直接读取 PKA 内部临时 JSON 作为正式输入；
- 不允许没有完整 12D measured representation 的对象进入正式 B/C 决策层；
- 不把 smoke run 当成 formal validation；
- 不用 kernel name、cluster id 或 workload label 作为正式 family 主依据；
- 不声明 fixture 产物是 claim-bearing evidence。

## 3. 总体数据流

```text
PKA selector output
  -> RepresentativeAnchorTable

PKA 12D measured feature rows
  -> RepresentationWeightSummary

RepresentativeAnchorTable
  + RepresentationWeightSummary
  + optional NetworkStructureContext
  -> B-line Anchor Evidence View
  -> Family Table
  -> Regime Table
  -> Lane Table
  -> Importance Scoring Sheet

B-line tables
  -> Backend Scenario Matrix
  -> Backend Run Manifest
  -> Baseline Plan
  -> Result Summary
  -> Writeback Map
```

B 线只能消费稳定 contract，不直接依赖 PKA selector 的内部实现细节。C 线只能消费 B 线对象，不反向读取 A 线 artifacts 来重算 priority。

## 4. A 线输出契约

### 4.1 RepresentativeAnchorTable

每个 PKA anchor 至少需要以下字段：

| 字段 | 要求 |
|---|---|
| `anchor_id` | 稳定、可复现的 anchor id |
| `workload_id` | 来源 workload |
| `source_cluster_id` | PKA selector 的 cluster id，只能作为 provenance |
| `representative_record_id` | 代表 record id |
| `representative_kernel_invocation` | 代表 kernel invocation 名称 |
| `member_record_ids` | 被该 anchor 覆盖的原始 records |
| `coverage_count` | anchor 覆盖的 record 数 |
| `coverage_weight` | anchor 在 workload 或 corpus 中的覆盖权重 |
| `duration_weight` | duration / cycle / time proxy 权重，允许为空但必须显式标记 |
| `compression_scope` | `per_workload` 或 `cross_workload` |
| `feature_mode` | 例如 `pka_m1_measured` |
| `provenance` | 说明字段来源和生成命令 |

`source_cluster_id` 不得作为 family、regime 或 lane 的主分类依据。它只能用于追踪 PKA 输出来源。

### 4.2 RepresentationWeightSummary

正式 B/C 决策层要求每个进入对象的 record 具备完整 12D measured representation。

最低字段：

| 字段 | 要求 |
|---|---|
| `record_id` | 与 anchor membership 可 join |
| `feature_vector_12d` | 按 canonical order 保存的 12D measured features |
| `hardware_axis_weights` | per-record normalized hardware-axis weights |
| `dominant_hardware_axis` | 主要硬件差异轴 |
| `bc_lane_hint` | 由 dominant axis 得到的 lane hints |
| `algorithmic_weight` | timing / cycle / weight input 归一化权重 |
| `algorithmic_weight_basis` | `duration_ns`、`elapsed_cycles`、`weight_input` 或 fallback |
| `tuning_priority_score` | 初始 representation priority |

12D features 必须全部存在、状态为 `measured`、值为有限数值。任何缺失都应产生 `blocked_missing_12d_representation`，不能进入 claim-bearing B/C artifacts。

如果 `algorithmic_weight_basis` 只是 member-count fallback，它只能作为接口兼容字段。Claim-bearing B/C importance 还需要 measured timing / cycle proxy，或带 provenance 的 `network_structure_prior`。

## 5. NetworkStructureContext

B 线不应只看硬件轴。对于 AI workload，kernel 在网络结构中的位置可以提供时间重要性和算法重要性先验。

因此 B 线允许引入 `NetworkStructureContext`，作为 formal importance 的组成部分，但必须携带 provenance 和 confidence。

最低字段：

| 字段 | 含义 |
|---|---|
| `record_id` 或 `anchor_id` | 与 PKA anchor 或 record join |
| `network_scope` | 例如 `transformer_layer`、`attention_block`、`ffn_block` |
| `network_role` | 例如 `attention_score`、`softmax`、`context_aggregation`、`ffn_expand` |
| `expected_temporal_importance` | `low`、`medium`、`high` 或数值 |
| `confidence` | `low`、`medium`、`high` |
| `source` | 例如 model annotation、kernel card、manual phase map |
| `provenance` | 生成方式、版本、输入文件 |

`network_structure_prior` 可以进入正式 importance score，但不能伪装成 measured timing。所有 score 必须标记 `importance_provenance`：

- `measured_only`
- `network_prior_only`
- `hybrid_measured_and_prior`

## 6. B 线对象定义

### 6.1 Family

`family` 表示共享硬件执行机制的对象组。Family 不等于 PKA cluster，也不等于网络模块名称。

第一版 family 可以包含：

| Family | 主依据 |
|---|---|
| `dense_compute` | dense / tiled compute，compute 或 occupancy 相关 |
| `reduction_normalization` | reduction、normalization、softmax path |
| `streaming_memory` | streaming load/store、memory aggregation |
| `irregular_traversal` | graph / sparse / pointer chasing / atomics |
| `elementwise_or_bookkeeping` | elementwise、control、constraint kernels |
| `boundary_or_unresolved` | 证据不足，不能安全归类 |

Family 的主证据应来自 hardware-axis weights、resource signature、membership 汇总和可解释的 execution pattern。Kernel name 和 workload label 只能作为辅助解释，不能作为主判据。

### 6.2 Regime

`regime` 是 B 线的 simulator-side decision regime，不只是执行区间。

Regime 定义为：

```text
regime =
  hardware execution pattern
  + algorithmic / network role
  + temporal importance context
  + validation role
  + boundary policy
```

Regime 可以在同一 family 内因为以下原因拆分：

- dominant hardware axis 不同；
- launch scale、shape、membership 覆盖差异显著；
- measured timing weight 差异显著；
- network_structure_prior 表明算法位置不同；
- validation role 不同，例如 main-object、review-object、constraint-object。

Regime 必须记录：

| 字段 | 要求 |
|---|---|
| `regime_id` | 稳定 id |
| `family_id` | 所属 family |
| `source_anchor_ids` | 来源 anchors |
| `dominant_hardware_axis_summary` | 主要硬件轴摘要 |
| `network_role_summary` | 网络结构角色摘要，可为空但必须显式标记 |
| `measured_timing_weight` | 实测或准实测 timing 权重 |
| `network_structure_prior_weight` | 网络结构先验权重 |
| `importance_provenance` | score 来源 |
| `validation_role` | `main-object`、`review-object` 或 `constraint-object` |
| `boundary_status` | `stable`、`provisional`、`boundary` 或 `blocked` |

### 6.3 Lane

`lane` 是 C 线能执行和验证的 backend 方向。

第一版 lane：

| Lane | 后端含义 |
|---|---|
| `occupancy_sensitive` | occupancy、register、block resource |
| `cache_sensitive` | cache capacity / hit behavior |
| `memory_coalescing_sensitive` | memory transaction / coalescing |
| `reduction_path_sensitive` | reduction / synchronization path |
| `irregular_control_sensitive` | divergence / graph traversal / atomics |
| `constraint_regression` | 只做 regression constraint，不作为主调参对象 |

Lane 必须从 regime 反查 family 和 anchors。Lane 不得成为网络模块标签，也不得只是 family 的重命名。

## 7. Importance Score

第一版 importance score 使用可解释线性组合，不引入学习模型。

建议结构：

```text
importance_score =
  w_coverage * coverage_weight
  + w_measured_time * measured_timing_weight
  + w_axis * dominant_axis_weight
  + w_network_prior * network_structure_prior_weight
  - w_boundary * boundary_penalty
```

要求：

- `coverage_weight` 来自 RepresentativeAnchorTable；
- `measured_timing_weight` 来自 duration / cycle / timing proxy；
- `dominant_axis_weight` 来自 RepresentationWeightSummary；
- `network_structure_prior_weight` 来自 NetworkStructureContext；
- `boundary_penalty` 用于降低 evidence 不足对象的优先级；
- 每个 score 必须输出 component breakdown 和 `importance_provenance`。

如果 `network_structure_prior` 是唯一时间重要性来源，则该对象可以进入 formal B 线排序，但必须标记 `network_prior_only`，并在 C 线 baseline 中单独审计。

## 8. Measured-Only Gate 和 Blocker 状态

正式 B/C 决策层必须通过以下 gate：

1. Anchor 有 membership 和 coverage；
2. Anchor members 能 join 到 12D measured rows；
3. 12D rows 全部 measured；
4. 有 hardware-axis weights；
5. 有 algorithmic/timing weight 或明确的 network_structure_prior；
6. 有 lane mapping；
7. 有 boundary policy。

失败状态：

| 状态 | 含义 |
|---|---|
| `blocked_missing_anchor_membership` | anchor 无法确认成员 |
| `blocked_missing_12d_representation` | 缺完整 12D measured features |
| `blocked_missing_weight` | 没有 timing / algorithmic weight，也没有 network prior |
| `pending_network_context` | 硬件证据完整，但网络结构先验未接入 |
| `pending_c_line_scenario` | B 线对象存在，但 C 线 lane 没有 scenario |
| `fixture_non_claim_bearing` | fixture 输出仅用于接口测试 |

当前 worktree 中真实 `pka_feature_table_l1.json` 为空数组时，只能输出 blocker 或 fixture artifacts，不能输出 claim-bearing B/C artifacts。

## 9. C 线验证设计

C 线消费以下 B 线 artifacts：

- `family_table`
- `regime_table`
- `lane_table`
- `importance_scoring_sheet`

C 线输出：

- `backend_scenario_matrix`
- `backend_run_manifest`
- `backend_baseline_plan`
- `backend_result_summary`
- `writeback_map`

Baseline 至少包含：

| Baseline | 含义 |
|---|---|
| `no-priority` | 不使用 B 线 priority |
| `time-only` | 只按 measured timing 排序 |
| `importance-guided` | 使用 B 线 full importance score |

如果存在 `network_prior_only` 对象，C 线还应在 result summary 中单独记录其表现，避免把 network prior 的有效性和 measured timing 的有效性混在一起。

Smoke 和 formal validation 必须分离：

- `smoke` 只证明 command、parser、artifact pipeline 可以运行；
- `formal_validation` 才能证明 importance-guided 是否优于 baseline；
- smoke 成功不得提升 regime、family 或 anchor 的 validation status。

## 10. Acceptance Criteria

### AC-1: A/B/C 职责边界清晰

PKA 只负责 representative compression。B 线只消费 stable contracts。C 线只消费 B 线对象并执行 validation planning。

### AC-2: RepresentativeAnchorTable 定义完整

Spec 必须定义 anchor id、membership、coverage、duration weight、compression scope、feature mode 和 provenance 的最低字段。

### AC-3: RepresentationWeightSummary 是正式 B/C 的强前置条件

缺完整 12D measured features 时，正式 B/C artifacts 必须 blocked，不能用 name / cluster fallback 生成 claim-bearing objects。

### AC-4: Regime 承接硬件机制和网络结构上下文

Regime 不只是执行区间。它必须能表达 hardware execution pattern、network role、temporal importance context、validation role 和 boundary policy。

### AC-5: Network structure prior 可以进入 formal importance

`network_structure_prior` 可以进入 importance score，但必须携带 source、confidence、scope、role 和 `importance_provenance`。

### AC-6: Importance score 可解释

每个 importance score 必须输出 component breakdown，不允许只输出不可解释总分。

### AC-7: C 线验证 smoke/formal 分离

Run manifest、baseline plan 和 writeback map 必须区分 smoke 与 formal validation。Smoke 不得提升 validation status。

### AC-8: Fixture artifacts 明确非 claim-bearing

Fixture 可以用于接口测试和 schema 回归，但必须标记 `fixture_non_claim_bearing`。

## 11. 第一版实现边界

第一版实现应优先固定 contract 和 schema：

1. Anchor contract loader；
2. Representation weight join checker；
3. B-line Anchor Evidence View；
4. Family / regime / lane / importance builder；
5. C-line backend planning adapter；
6. focused tests 和 fixture artifacts。

真实 workload 的 claim-bearing B/C artifacts 需要等真实 PKA 12D measured table 可用后再生成。
