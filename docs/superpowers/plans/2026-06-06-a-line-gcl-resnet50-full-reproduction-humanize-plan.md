# A 线 GCL ResNet-50 Full Reproduction Humanize Plan

## Goal Description

实现真实 ResNet-50 输入上的 GCL-Sampler 核心复现，并把复现结果推进到可量化验证的 cluster correctness evidence。

完整 formal 路径是：

```text
真实 ResNet-50 NVBit trace
  -> Gate 0 trace acquisition
  -> Gate 1 trace adapter
  -> Gate 2 representative SM manifest
  -> Gate 3 canonical graph
  -> Gate 4 graph tensorization
  -> Gate 5 RGCN contrastive embedding
  -> Gate 6 silhouette-K / deterministic K-Means selector
  -> Gate 7 cluster correctness evaluation
```

Gate 8 / Gate 9 是我们在 GCL-Sampler 复现之后的扩展：

```text
Gate 8: cluster / family -> simulator tuning vector proposal
Gate 9: sampled-vs-full simulator evaluation
```

本 plan 的核心边界是：**formal GCL reproduction 不能使用人工 trace、ResNet-like fixture、mini-transformer trace 或 simulator replay 替代真实 ResNet-50 NVBit trace**。Fixture 只能用于 unit / smoke / debug。

## Current Gate0 Status

Gate0 formal ResNet-50 NVBit trace acquisition 已经打通，本轮 RLCR 不应再把“是否能采集真实 ResNet-50 trace”作为当前阻塞项。Gate0 现在是后续 Gate1-7 的 formal input baseline 和 regression guard。

Formal trace root:

```text
artifacts/gcl_resnet50_gate0_formal_trace/traces
```

已确认的 Gate0 artifact:

- `dynamic_trace.pb`
- `threadblocks/`
- `extra_info/enhanced_execution_info.json`
- `scheduler_metadata.json`
- `stats.csv`
- `gate0_trace_acquisition_manifest.json`
- `nvbit_collection_evidence.json`
- `nvbit_collector_attestation.json`
- `.nvbit_collector_session.json`

Formal manifest 必须记录：

- `artifact_status = formal`
- `formal_input_eligible = true`
- `workload_id = resnet50`
- `execution_mode = real_trace`
- `trace_source = nvbit`
- `input_scope = full_resnet50_inference_trace`
- `scheduler_metadata_source = real_nvbit_smid`

当前 baseline 规模：

- trace root size: `2.4G`
- threadblock protobuf files: `124876`
- scheduler metadata kernel invocations: `265`
- scheduler metadata CTA records: `124876`

当前已通过的 regression validation:

```bash
pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py
```

Expected: `73 passed`。

## Acceptance Criteria

- AC-1: Gate 0 formal ResNet-50 NVBit trace acquisition 已完成，并由 regression tests 保护
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py::test_gate0_acquisition_runner_sets_nvbit_trace_folder_environment`
    - `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py::test_gate0_acquisition_runner_records_evidence_from_real_artifact_contract`
    - Formal trace root 必须是 `artifacts/gcl_resnet50_gate0_formal_trace/traces`。
    - Manifest 必须包含 `artifact_status = formal`、`formal_input_eligible = true`、`workload_id = resnet50`、`execution_mode = real_trace`、`trace_source = nvbit`、`input_scope = full_resnet50_inference_trace`。
    - `scheduler_metadata.json` 必须来自 `scheduler_metadata_source = real_nvbit_smid`，并包含真实 `sm_id`、`cta_id`、`warp_ids`、`first_seen_order`、`last_seen_order` 和 `trace_entry_count`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py::test_gate0_rejects_missing_real_smid_metadata`
    - `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py::test_gate0_rejects_synthetic_helper_even_if_scope_claims_real_collection`
    - `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py::test_gate0_rejects_handwritten_session_attestation_triplet_on_synthetic_root`
    - 缺少真实 `%smid` 采集来源时不得生成 formal acquisition manifest。
    - synthetic / simulator replay source 必须标记为 `debug_not_formal` 或被 formal validation 拒绝。

- AC-2: Gate 1 formal adapter 只接受真实 ResNet-50 trace artifacts
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_builds_formal_adapter_from_real_resnet50_trace`
    - Gate 1 从 Gate 0 输出读取 `dynamic_trace.pb`、`threadblocks/`、`enhanced_execution_info.json`、`scheduler_metadata.json` 和 `stats.csv`。
    - 输出 `resnet50_trace_adapter_bundle.json`，并记录稳定 `kernel_invocation_id`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_rejects_fixture_as_formal_input`
    - ResNet-like fixture、hand-written opcode sequence、mini-transformer trace、simulator replay trace 不能输出 `adapter_validation_report.status = passed`。

- AC-3: Gate 1 能解析 kernel / CTA / warp / instruction 输入结构
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_emits_kernel_cta_warp_trace_records`
    - Adapter bundle 包含 `kernel_invocation_table`、`cta_scheduler_records`、`per_warp_trace_records` 和 static instruction metadata。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_reports_missing_static_instruction_metadata`
    - 缺失 static instruction metadata 时必须记录 missing count 或 `unknown_opcode`，不得静默生成不可审计 row。

- AC-4: Gate 2 使用 deterministic representative SM policy 生成 formal manifest
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate2_representative_sm.py::test_gate2_selects_scheduler_signature_medoid_sm`
    - Gate 2 使用 `scheduler_signature_medoid_sm`，输出 `selected_sm_policy_report.json` 和 `representative_sm_trace_manifest.json`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate2_representative_sm.py::test_gate2_rejects_debug_adapter_bundle`
    - `explicit_sm_id`、random SM、file-order fallback 或 debug adapter 不得进入 formal manifest。

- AC-5: Gate 2 manifest scope 必须是 selected SM 上全部 CTA
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate2_representative_sm.py::test_gate2_manifest_includes_all_ctas_on_selected_sm`
    - `included_cta_ids` 只来自 selected SM，且覆盖该 SM 上所有 CTA。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate2_representative_sm.py::test_gate2_rejects_partial_selected_sm_scope`
    - selected SM 上 CTA 被遗漏、混入其他 SM CTA、或 `collection_scope` 不是 `single_representative_sm_all_ctas` 时必须失败。

- AC-6: Gate 3 从 formal representative-SM manifest 生成 canonical graph
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate3_canonical_graph.py::test_gate3_builds_canonical_graph_from_real_manifest`
    - 输出 `phase_b_trace_records.json`、`canonical_graph_bundle.json`、`graph_size_audit.json` 和 `graph_construction_report.json`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate3_canonical_graph.py::test_gate3_rejects_debug_or_fixture_manifest`
    - 缺少真实 ResNet-50 provenance 的 manifest 不得生成 formal `canonical_graph_bundle.json`。

- AC-7: Gate 3 graph schema 保持 GCL-compatible typed graph
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate3_canonical_graph.py::test_gate3_uses_allowed_node_and_edge_types`
    - Graph 包含 instruction node、variable node、可选 `mem_ref` pseudo node。
    - Edge relation 只允许 `control_flow`、`data_source`、`data_destination`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate3_canonical_graph.py::test_gate3_rejects_unapproved_relation_or_pseudo_node`
    - 未经 spec 批准的 edge relation 或 pseudo node 类型必须失败。

- AC-8: Gate 4 tensorization 输出 RGCN 可消费的 tensor bundle
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate4_tensorization.py::test_gate4_outputs_rgcn_tensor_bundle`
    - 输出 `node_features.shape = [node_count, 64]`、`edge_index.shape = [2, edge_count]`、`edge_type.shape = [edge_count]`、warp / CTA partition tensors。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate4_tensorization.py::test_gate4_rejects_invalid_feature_width_or_partition_metadata`
    - feature width 不是 64、edge shape 不匹配或 partition metadata 缺失时必须失败。

- AC-9: Gate 4 继承 formal provenance 和 representation metadata
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate4_tensorization.py::test_gate4_records_representation_and_real_trace_provenance`
    - Tensor bundle 记录 `representation_mode`、`pseudo_node_mode`、`paper_reproduction_mode` 和真实 ResNet-50 provenance。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate4_tensorization.py::test_gate4_rejects_fixture_graph_as_formal_tensor`
    - fixture / debug graph bundle 不能进入 Gate 5 formal training。

- AC-10: Gate 5 训练只在 training views 上做 augmentation
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate5_rgcn_training.py::test_gate5_augmentation_does_not_overwrite_canonical_tensor`
    - `augmentation_manifest.json` 记录 node dropping、edge dropping、feature noise 的 seed、rate 和 source tensor hash。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate5_rgcn_training.py::test_gate5_rejects_augmented_tensor_as_selector_embedding_source`
    - selector embedding 来自 augmented view 时必须失败。

- AC-11: Gate 5 使用 GCL-compatible RGCN / readout / projection contract
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate5_rgcn_training.py::test_gate5_exports_256d_canonical_kernel_embeddings`
    - RGCN 结构为 64 -> 128 -> 128 -> 256。
    - Readout 使用 `node -> warp -> CTA -> selected SM -> kernel`。
    - InfoNCE 使用 projection head output，selector 使用 projection head 之前的 256D kernel embedding。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate5_rgcn_training.py::test_gate5_rejects_projection_head_output_for_selector`
    - 64D projection output 不得写入 formal `kernel_embedding_table.json`。

- AC-12: Gate 6 selector 只消费 Gate 5 formal embedding table
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate6_selector.py::test_gate6_accepts_real_resnet50_gate5_embedding_table`
    - 输入 `kernel_embedding_table.json` 包含 `embedding_dim = 256`、canonical non-augmented source、real ResNet-50 provenance 和可复现 hash。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate6_selector.py::test_gate6_rejects_fixture_projection_or_augmented_embeddings`
    - fixture embedding、projection head output、augmented-view embedding 或缺少 provenance 的 embedding table 必须失败。

- AC-13: Gate 6 执行 silhouette-K 和 deterministic K-Means
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate6_selector.py::test_gate6_runs_silhouette_k_and_deterministic_kmeans`
    - 输出 `embedding_normalization_report.json`、`k_selection_report.json`、`kmeans_cluster_assignment_table.json` 和 `representative_anchor_table.json`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate6_selector.py::test_gate6_rejects_forbidden_fields_in_clustering_path`
    - `kernel_name`、family label、runtime、graph size、weight input 不得参与 normalization、K selection、K-Means 或 centroid 计算。

- AC-14: Gate 6 family evidence 只用于 post-clustering evaluation
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate6_selector.py::test_gate6_family_evidence_is_post_clustering_only`
    - `cluster_family_evidence_report.json` 记录 `family_labels_used_for_clustering = false`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate6_selector.py::test_gate6_rejects_family_label_guided_clustering`
    - family label 参与 assignment 或 K selection 时必须失败。

- AC-15: Gate 7 输出 embedding geometry correctness metrics
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_records_embedding_geometry_metrics`
    - 输出 silhouette、Davies-Bouldin、Calinski-Harabasz、intra/inter distance 和 inter/intra ratio。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_rejects_debug_selector_artifacts`
    - debug / fixture selector artifacts 不得进入 Gate7 formal evaluation。

- AC-16: Gate 7 输出 family alignment 和 representative quality metrics
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_records_family_and_representative_quality`
    - 输出 cluster purity、weighted purity、ARI、NMI、representative p95 distance、outlier ratio 和 high-weight outlier count。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_does_not_modify_gate6_assignments`
    - Gate7 不得重新选择 K、重新运行 formal K-Means 或修改 Gate6 assignment。

- AC-17: Gate 7 输出 measured / simulator metric consistency
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_records_metric_error_reports`
    - 输出 cluster weighted MAPE、cluster p95 relative error、global weighted MAPE 和 high-weight bad cluster count。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_reports_metric_unit_conflict`
    - mixed metric units 无法归一化时必须标记 `metric_unit_conflict`，不得伪装可比较。

- AC-18: Gate 7 输出 multi-run stability 或明确 single-run limitation
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_records_stability_or_single_run_status`
    - 多 run 时输出 assignment stability ARI、K stability、centroid drift、representative stability rate。
    - single run 时输出 `stability_status = single_run_not_evaluated`。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_rejects_stability_claim_from_single_run`
    - single-run evaluation 不得声明 cluster assignment stable。

- AC-19: Gate 8 只在 Gate7 evidence 支持时生成 tuning vector proposal
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate8_tuning.py::test_gate8_generates_tuning_vectors_from_trusted_clusters`
    - Gate8 读取 Gate7 correctness manifest、representative anchors、metric error report 和 simulator tunable component schema。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate8_tuning.py::test_gate8_rejects_high_weight_mixed_or_high_error_clusters`
    - 高权重 mixed-family cluster、高 metric error cluster 或 weak representative cluster 不得直接进入 tuning vector proposal。

- AC-20: Gate 9 执行 sampled-vs-full simulator evaluation
  - Positive Tests (expected to PASS):
    - `pytest -q tests/gcl_resnet50/test_gate9_simulator_evaluation.py::test_gate9_compares_sampled_against_full_baseline`
    - Gate9 输出 full-vs-sampled simulation report、sampled speedup report、sampled error report 和 tuning effect report。
  - Negative Tests (expected to FAIL):
    - `pytest -q tests/gcl_resnet50/test_gate9_simulator_evaluation.py::test_gate9_rejects_speedup_claim_without_full_baseline`
    - 没有 full baseline 或 measured baseline 时不得声明 speedup / accuracy。

## Path Boundaries

### Upper Bound (Maximum Scope)

本 plan 的最大可接受范围是完成 Gate0-9 的 end-to-end formal pipeline：

```text
真实 ResNet-50 NVBit trace
  -> formal GCL reproduction
  -> quantified cluster correctness evidence
  -> tuning vector proposal
  -> sampled-vs-full simulator evaluation
```

其中 Gate8 / Gate9 是我们的扩展，不得写成 GCL-Sampler 原始论文复现的一部分。

### Lower Bound (Minimum Scope)

最低可接受实现是完成 Gate0-7：

```text
已验证 Gate0 formal ResNet-50 trace root
  -> Gate1-5 formal embedding table
  -> Gate6 selector artifacts
  -> Gate7 report-only correctness evaluation
```

本轮实现必须从已验证 Gate0 formal trace root 出发。Gate0 blocker report 只保留为 regression path，用来证明缺少真实 trace 时 pipeline 会停止；它不再是当前主线的预期状态。

### Allowed Choices

Can use:

- Python standard library；
- repo-local JSON artifact helpers；
- NumPy；
- PyTorch；
- pytest；
- existing `experiments.gcl_phase_a` / `experiments.gcl_phase_b` modules；
- repo-local deterministic K-Means / silhouette implementation；
- NCU measured metrics 或 simulator metrics，前提是 metric source 和 unit 可审计。

Cannot use:

- fixture / synthetic trace 作为 formal input；
- projection head output 作为 selector embedding；
- family label、kernel name、runtime、graph size 参与 Gate6 clustering；
- Gate7 修改 Gate6 cluster assignment；
- Gate8 在 Gate7 high-risk cluster 上直接生成 tuning proposal；
- Gate9 在没有 full / measured baseline 时声明 speedup 或 accuracy。

## Dependencies and Sequence

### Milestone 1: Gate0 Closed / Formal Trace Root Baseline

- Phase A: 以 `artifacts/gcl_resnet50_gate0_formal_trace/traces` 作为 Gate1-7 formal baseline。
- Phase B: 固化 `gate0_trace_acquisition_manifest.json`、`nvbit_collection_evidence.json`、`nvbit_collector_attestation.json` 和 `.nvbit_collector_session.json` 的 provenance contract。
- Phase C: 保留 blocked / synthetic / fixture tests 作为 regression guard，确保它们不能被提升为 formal。

Required before:

```text
Gate1 formal adapter
```

### Milestone 2: Gate 1 Formal Adapter From Real Trace Root

- Phase A: Gate1 读取已验证 Gate0 trace root 中的 `dynamic_trace.pb`、`threadblocks/`、`extra_info/enhanced_execution_info.json`、`scheduler_metadata.json` 和 `stats.csv`。
- Phase B: Gate1 输出 formal `resnet50_trace_adapter_bundle.json`。
- Phase C: Gate1 生成稳定 `kernel_invocation_id`，并保留真实 scheduler metadata lineage。
- Phase D: Gate1 拒绝缺少 Gate0 formal manifest、缺少 real NVBit attestation、或来自 fixture / synthetic / debug replay 的输入。

Required before:

```text
Gate2 representative SM selection
```

### Milestone 3: Gate 2-4 Trace To Graph

- Phase A: Gate2 从真实 `scheduler_metadata.json` 运行 `scheduler_signature_medoid_sm`，生成 representative-SM manifest。
- Phase B: Gate3 从 representative-SM manifest 生成 canonical graph。
- Phase C: Gate4 将 canonical graph tensorize 为 RGCN 输入。
- Phase D: Gate2-4 的每个 artifact 都必须继承 Gate0 / Gate1 formal provenance。

Required before:

```text
Gate5 RGCN training
```

### Milestone 4: Gate 5 RGCN Embedding

- Phase A: training-only augmentation。
- Phase B: RGCN contrastive training。
- Phase C: hierarchical readout。
- Phase D: canonical non-augmented embedding export。

Required before:

```text
Gate6 selector
```

### Milestone 5: Gate 6 Selector

- Phase A: embedding table validation。
- Phase B: engineering default z-score normalization。
- Phase C: silhouette-K。
- Phase D: deterministic K-Means。
- Phase E: representative anchor selection。
- Phase F: post-clustering family evidence report。

Required before:

```text
Gate7 correctness evaluation
```

### Milestone 6: Gate 7 Correctness Evaluation

- Phase A: embedding geometry metrics。
- Phase B: family alignment metrics。
- Phase C: representative quality metrics。
- Phase D: measured / simulator metric error。
- Phase E: multi-run stability or single-run limitation。
- Phase F: `threshold_policy = report_only_v1` manifest。

Required before:

```text
Gate8 tuning proposal
```

### Milestone 7: Gate 8 / Gate 9 Extension

- Phase A: Gate8 consumes Gate7 evidence and simulator tunable schema。
- Phase B: Gate8 emits cluster tuning vector proposals。
- Phase C: Gate9 compares sampled representative run against full / measured baseline。
- Phase D: Gate9 emits speedup / error reports only when baseline is present。

## Implementation Notes

- Code should not contain plan terminology such as `Milestone 1` or `AC-7`; tests and artifact names can reference gate names when they are part of the domain contract.
- All formal artifacts must carry source artifact hashes.
- All debug artifacts must be visibly non-formal.
- Gate6 normalization must record `normalization_policy = engineering_default_z_score` and `paper_defined = false`.
- Gate7 first version must record `threshold_policy = report_only_v1`; quantitative metrics are required as reports, not as enforced thresholds.
- Any claim about cluster correctness, tuning quality, speedup, or simulator accuracy must be tied to the latest gate that actually proves it.
- Next RLCR focus should be Gate1 / Gate2 consumption of real Gate0 artifacts, representative-SM selection from real `scheduler_metadata.json`, and formal provenance propagation into Gate3+.
- Gate0 acquisition should be reviewed as a closed baseline with regression protection, not as the primary open blocker.

## Review Checklist

- [ ] No fixture / synthetic trace can enter formal Gate1-7.
- [ ] Gate0 provenance is required before any formal GCL reproduction claim.
- [ ] Gate6 clustering uses only 256D canonical embeddings.
- [ ] Gate7 does not mutate Gate6 results.
- [ ] Gate8 and Gate9 are labeled as our extension, not original GCL-Sampler reproduction.
- [ ] All AC tests have positive and negative cases.
- [ ] All quantitative metrics are marked as report-only until real ResNet-50 baseline thresholds are established.
