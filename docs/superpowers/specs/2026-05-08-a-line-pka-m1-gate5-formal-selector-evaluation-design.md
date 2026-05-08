# A 线 PKA-M1 Gate 5 Formal Selector + Evaluation Design Spec

日期：2026-05-08

## 1. 背景

Gate 1 到 Gate 3 负责把 L1 P0 workload 转成真实 measured 12D PKA feature records。

Gate 4 负责检查 selector eligibility，并在满足条件时输出 selector input projection。

Gate 5 是 M1 第一版 measured loop 的最后一站。它消费 Gate 4 已批准的 selector input projection，运行正式 M1 selector 和 structural compression evaluation。

Gate 5 不再处理 workload、NCU capture、feature extraction 或 backward repair。

## 2. 目标

Gate 5 的目标是：

```text
Gate 4 selector input projection
  -> shared selector core
  -> preprocessing
  -> numpy SVD PCA
  -> deterministic farthest-first k-means
  -> nearest-centroid representative anchors
  -> structural compression evaluation
  -> formal M1 artifacts
```

输入：

- `artifacts/a_line/l1/m1_selector_eligibility_l1.json`
- `artifacts/a_line/l1/m1_selector_input_l1.json`

输出：

- `artifacts/a_line/l1/pka_pca_projection_l1.json`
- `artifacts/a_line/l1/pka_kmeans_clusters_l1.json`
- `artifacts/a_line/l1/representative_anchor_table_l1.json`
- `artifacts/a_line/l1/pka_compression_evaluation_l1.json`

## 3. 非目标

Gate 5 不做：

- workload resolution。
- smoke run。
- NCU capture。
- NCU CSV parsing。
- measured feature extraction。
- acquisition gap repair。
- simulator execution。
- simulator accuracy evaluation。
- measured speedup conclusion。
- B-line consumption。

B-line consumption 后续应作为独立 Gate / integration spec。

## 4. 前置条件

Gate 5 只能在 Gate 4 明确允许时运行。

必须满足：

```text
gate5_allowed == true
selector_eligibility_state in [
  "selector_ready",
  "selector_ready_with_remaining_gaps"
]
```

如果 Gate 4 状态是：

- `selector_blocked_insufficient_measured_records`
- `selector_blocked_invalid_feature_table`
- `selector_blocked_mixed_timing_unit`

Gate 5 必须 abort。

## 5. Selector Input Contract

Gate 5 只能读取：

```text
artifacts/a_line/l1/m1_selector_input_l1.json
```

禁止直接读取完整 `pka_feature_table_l1.json` metadata 来决定 clustering。

每条 selector input record 只允许包含：

- `record_id`
- `kernel_invocation_id`
- `features`
- `feature_mode`
- `weight_input`

`weight_input` 只能包含 Gate 4 已批准的 timing / member-count weight 信息。

禁止 selector input 包含：

- `kernel_name`
- `source_path`
- `expected_behavior_axis`
- `family`
- `regime`
- `shape_hint`
- `trace_order`
- B-line semantic metadata

如果 Gate 5 发现 forbidden field，必须 abort。

## 6. Shared Selector Core

M0 和 M1 必须共用 selector core。

推荐新增公共模块：

```text
experiments/baseline_diagnosis/pka_selector_core.py
```

该模块包含纯算法逻辑：

- fixed feature order。
- selector record validation。
- raw matrix extraction。
- `log1p` / clip / z-score preprocessing。
- `numpy SVD` PCA。
- deterministic farthest-first k-means。
- cluster label generation。
- nearest-centroid representative selection。
- structural evaluation helpers。
- deterministic hash helpers。

M0 wrapper：

```text
experiments/baseline_diagnosis/pka_m0_pipeline.py
```

负责：

- fixture loading。
- M0 mode / artifact naming。
- 调用 shared selector core。

M1 wrapper：

```text
experiments/baseline_diagnosis/pka_m1_selector.py
```

负责：

- Gate 4 projection loading。
- M1 mode / artifact naming。
- 调用 shared selector core。

禁止复制一份独立 M1 PCA/k-means 实现导致 M0/M1 算法漂移。

## 7. Algorithm Requirements

Gate 5 必须继承 M0 已验证算法语义。

### 7.1 Preprocessing

固定 feature order。

Count-like features：

```text
log1p(value) -> z-score
```

Ratio feature：

```text
divergence_efficiency clip to [0, 1] -> z-score
```

zero-std feature：

```text
normalized value = 0
```

并记录 zero-std feature list。

### 7.2 PCA

PCA 必须使用：

```python
np.linalg.svd(X_centered, full_matrices=False)
```

禁止使用：

```text
sklearn.decomposition.PCA
```

如果输入退化导致 explained variance 总和为 0，Gate 5 必须 abort，并输出：

```text
pca_degenerate_input
```

### 7.3 K-means

k-means 必须使用 deterministic farthest-first。

要求：

- `k = ceil(sqrt(n_records))`，并 clamp 到 `[2, n_records]`。
- 第一个 centroid 是 `record_id` 字典序最小的 record。
- 后续 centroid 使用 farthest-first。
- tie-breaker 使用 `record_id` 字典序。
- assignment 使用 PCA 空间 squared Euclidean distance。
- distance metadata 使用 `squared_euclidean_in_pca_space`。
- `max_iter = 300`。
- 不使用 random initialization。
- 不使用 `sklearn.cluster.KMeans`。

### 7.4 Representative Selection

每个 cluster 输出一个 representative anchor。

Representative 必须是 cluster 内到 centroid squared distance 最小的真实 record。

如果并列，选择 `record_id` 字典序最小的 record。

禁止使用 centroid 平均点作为 representative。

## 8. Formal M1 Artifacts

Gate 5 输出正式 M1 artifacts：

- `artifacts/a_line/l1/pka_pca_projection_l1.json`
- `artifacts/a_line/l1/pka_kmeans_clusters_l1.json`
- `artifacts/a_line/l1/representative_anchor_table_l1.json`
- `artifacts/a_line/l1/pka_compression_evaluation_l1.json`

禁止写入或覆盖：

- `artifacts/a_line/l1/pka_m0_pca_projection_l1.json`
- `artifacts/a_line/l1/pka_m0_kmeans_clusters_l1.json`
- `artifacts/a_line/l1/pka_m0_representative_anchor_table_l1.json`
- `artifacts/a_line/l1/pka_m0_compression_evaluation_l1.json`

所有 M1 artifacts 必须包含：

- `mode: pka_m1_measured`
- `feature_mode: pka_m1_measured`，如适用。
- input selector projection hash。
- Gate 4 eligibility artifact hash。
- deterministic replay hash。

## 9. PCA Artifact

`pka_pca_projection_l1.json` 至少包含：

- `artifact_name`
- `mode`
- `method: numpy_svd`
- `input_selector_projection_path`
- `input_selector_projection_hash`
- `gate4_eligibility_path`
- `gate4_eligibility_hash`
- `feature_order`
- `normalization_config`
- `components`
- `explained_variance`
- `explained_variance_ratio`
- `transformed_coordinates`
- `record_ids`
- `deterministic_replay_hash`

## 10. K-means Artifact

`pka_kmeans_clusters_l1.json` 至少包含：

- `artifact_name`
- `mode`
- `method: deterministic_farthest_first_kmeans`
- `input_pca_artifact_path`
- `input_pca_artifact_hash`
- `kmeans_config`
- `k`
- `initial_centroid_record_ids`
- `initialization_trace`
- `centroids`
- `assignments`
- `members_by_cluster`
- `distance_to_centroid`
- `inertia`
- `iterations_run`
- `converged`
- `empty_cluster_events`
- `deterministic_replay_hash`

## 11. Representative Anchor Table

`representative_anchor_table_l1.json` 至少包含：

- `artifact_name`
- `mode`
- `feature_mode`
- `selector_name`
- `selected_features`
- `normalization_config`
- `dimensionality_reduction_config`
- `clustering_config`
- `selection_rule: nearest_centroid_record`
- `forbidden_field_audit`
- `anchors`
- `deterministic_replay_hash`

每个 anchor 至少包含：

- `anchor_id`
- `cluster_id`
- `representative_record_id`
- `members`
- `weight`
- `representative_distance_to_centroid`
- `cluster_label`

Anchor membership 必须覆盖所有 measured selector input records。

每个 measured record 必须属于且只属于一个 anchor。

## 12. Weight Handling

Gate 5 不自行决定 weight mode。

Gate 5 必须读取 Gate 4 输出的：

- `weight_mode`
- `timing_unit`

允许：

### 12.1 `member_count_fallback`

当 Gate 4 声明：

```text
weight_mode = member_count_fallback
```

Gate 5 使用：

```text
每个 member weight = 1
```

### 12.2 `timing_weight`

当 Gate 4 声明：

```text
weight_mode = timing_weight
```

Gate 5 使用 Gate 4 指定的统一 timing unit。

如果 Gate 5 发现输入 timing 与 Gate 4 声明不一致，必须 abort。

### 12.3 禁止情况

如果 Gate 4 未提供 `weight_mode`，Gate 5 必须 abort。

如果 Gate 4 允许通过但 selector input 混用 timing unit，Gate 5 必须 abort，并报告 Gate 4 contract violation。

## 13. Structural Evaluation

Gate 5 evaluation 只计算 structural compression metrics。

允许输出：

- compression ratio。
- coverage count。
- weighted coverage。
- cluster feature variance。
- top-k coverage。
- anchor membership。
- PCA diagnostics。
- k-means diagnostics。
- deterministic replay hash。

禁止输出或声称：

- simulator accuracy。
- measured speedup。
- final PKA reproduction conclusion。

`pka_compression_evaluation_l1.json` 必须包含：

- `metric_scope: structural_only_not_simulator_accuracy`
- `compression_ratio`
- `coverage_count`
- `weighted_coverage`
- `weight_mode`
- `timing_unit`
- `cluster_feature_variance`
- `top_k_coverage`
- `pca_diagnostics`
- `kmeans_summary`
- `cluster_anchor_context`
- artifact hashes for selector input、PCA、k-means、anchor table
- `deterministic_replay_hash`

## 14. Forbidden-field Audit

Gate 5 必须记录 forbidden-field audit。

Audit 至少包含：

- allowed input fields。
- forbidden fields。
- actual read fields。
- status。

如果 actual read fields 包含 forbidden fields，Gate 5 必须 abort。

Allowed fields：

- `record_id`
- `kernel_invocation_id`
- `features`
- `feature_mode`
- `weight_input`

Forbidden fields：

- `kernel_name`
- `source_path`
- `expected_behavior_axis`
- `family`
- `regime`
- `shape_hint`
- `trace_order`
- B-line semantic metadata

## 15. Completion Rules

Gate 5 完成，当且仅当：

- Gate 4 `gate5_allowed == true`。
- Gate 4 state 是 `selector_ready` 或 `selector_ready_with_remaining_gaps`。
- selector input projection schema valid。
- forbidden-field audit passed。
- PCA artifact 存在。
- k-means artifact 存在。
- representative anchor table 存在。
- compression evaluation artifact 存在。
- anchor membership 覆盖所有 measured selector input records。
- 每个 measured record 只属于一个 anchor。
- deterministic replay hash 存在且稳定。
- M0 artifacts 未被覆盖。
- regression tests 通过。

Gate 5 不要求 B-line consumption。

## 16. Stop / Failure Rules

以下任一情况必须 abort：

- Gate 4 不允许 Gate 5。
- selector input projection 缺失。
- selector input projection 包含 forbidden metadata。
- measured selector input records 少于 3。
- feature mode 不是 `pka_m1_measured`。
- PCA degenerate input。
- k-means assignment 丢失 record 或重复覆盖 record。
- anchor membership 不完整。
- Gate 4 weight contract 缺失或冲突。
- 试图写入 `pka_m0_*` artifact。

## 17. Determinism

Gate 5 output 必须 deterministic：

- selector input row order stable。
- PCA output stable。
- k-means initialization stable。
- k-means tie-breaker stable。
- anchor representative tie-breaker stable。
- JSON output stable key order。
- replay hash stable。

## 18. 测试要求

Gate 5 测试至少覆盖：

- Gate 4 blocked -> Gate 5 abort。
- valid selector input -> 生成四个 formal M1 artifacts。
- M0 artifacts 不被覆盖。
- PCA method 是 `numpy_svd`。
- 禁止 sklearn PCA。
- k-means method 是 deterministic farthest-first。
- 禁止 sklearn KMeans / random init。
- representative 是 nearest centroid real record。
- anchor membership 覆盖所有 records 且不重复。
- `member_count_fallback` weight 正确。
- `timing_weight` 使用统一 timing unit。
- Gate 4 weight contract 缺失 -> abort。
- forbidden metadata 出现在 selector input -> abort。
- structural evaluation 不声称 simulator accuracy / speedup。
- repeated run replay hash 稳定。

## 19. 与后续工作的关系

Gate 5 完成后，M1 第一版 measured selector loop 完成。

后续工作可以单独定义：

- Gate 6 / B-line consumption。
- simulator validation。
- L2 measured expansion。
- Photon / PKA / other baseline ablation。

这些不属于 Gate 5。

## 20. 简短结论

Gate 5 的核心是：

```text
Gate 4 approved selector input
  -> shared selector core
  -> PCA
  -> k-means
  -> anchors
  -> structural evaluation
```

它把 M1 从 measured feature table 推进到正式 representative anchors，但不扩大到 simulator accuracy 或 B-line integration。
