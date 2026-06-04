# A 线 GCL-M1/M2 Phase B 实施计划

## 目标描述

实现 GCL Phase B：在 Phase A 已验证的 graph / tensorization / RGCN / readout / embedding export / M0 selector contract 之上，把输入从 controlled fixture 升级为 GCL-Sampler 论文默认的 representative-SM trace scope。

Phase B 的目标路径是：

```text
Phase A verified implementation
  -> representative-SM trace manifest
  -> scheduler_signature_medoid_sm selected SM
  -> selected SM 上全部 CTA trace
  -> per-warp graph construction
  -> kernel graph union with warp_partitions
  -> audit-only graph size report
  -> tensorization with Phase A strict node feature schema
  -> augmentation manifests for training views
  -> node -> warp -> kernel readout
  -> 256-dimensional kernel embedding table
  -> M0 silhouette K-Means selector
  -> representative anchor artifacts
```

本计划不证明 GNN embedding quality、sampling accuracy 或 simulator speedup。Phase B 的完成标准是：真实 representative-SM trace scope 可以按论文默认路径进入 Phase A 已打通的语义闭环，并且所有 artifacts 可 replay、可 audit。

## 参考文档

必须遵循：

- `docs/superpowers/specs/2026-06-01-a-line-gcl-m1-m2-phase-b-scalable-graph-design.md`
- `docs/superpowers/specs/2026-06-04-a-line-gcl-phase-b-representative-sm-selection-design.md`
- `docs/superpowers/plans/2026-06-02-a-line-gcl-m1-m2-phase-a-semantic-e2e-plan.md`

实现细节参考：

- `docs/superpowers/specs/2026-06-02-a-line-gcl-rgcn-message-passing-readout-explainer-design.md`
- `docs/superpowers/specs/2026-06-02-a-line-gcl-sampler-rgcn-contrastive-learning-explainer-design.md`
- `docs/superpowers/specs/2026-06-02-a-line-gcl-phase-a-normalized-pc-positional-encoding-design.md`

后续阶段参考：

- `docs/superpowers/specs/2026-06-01-a-line-gcl-m1-m2-phase-c-compression-abstraction-design.md`
- `docs/superpowers/specs/2026-05-13-a-line-gcl-sampler-overall-architecture-design.md`

如果上述文档之间存在冲突，本计划采用以下最新决策：

- Phase B 默认 SM selection 使用 `scheduler_signature_medoid_sm`，不使用 `max_cta_count_sm` 作为完成路径。
- Graph size audit 在 Phase B 第一版只作为 audit-only guardrail，不因为 size class 自动 blocked。
- 只有实际 tensorization / batching / training 资源失败时，才输出 `resource_blocked_artifact`。
- Phase B 默认 readout 保持 GCL-compatible `node -> warp -> kernel`；CTA / selected SM 信息进入 trace manifest 和 provenance，不作为默认 pooling 层。

## 验收标准

- AC-1: Phase A 实现已承接并可作为 Phase B 依赖
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_a`
    - `python -m experiments.gcl_phase_a.pipeline --out artifacts/gcl_phase_a`
    - 当前实现分支必须包含 `experiments/gcl_phase_a` 和 `tests/gcl_phase_a`。
    - Phase A graph / tensorizer / RGCN / embedding export / selector contract 必须保持通过。
  - 负向测试（预期失败）:
    - 缺少 `experiments/gcl_phase_a` 时，Phase B pipeline import 必须失败并说明缺少 Phase A dependency。
    - Phase A node feature schema 被修改为非 `gcl_m2_phase_a_paper_node_feature_v1` 时，Phase B tensorizer validator 必须拒绝。

- AC-2: Phase B trace manifest 明确记录 representative-SM scope
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_trace_scope.py::test_trace_manifest_records_single_representative_sm_all_ctas`
    - 输入 trace manifest 必须包含 `collection_scope = single_representative_sm_all_ctas`。
    - 每个 kernel invocation 必须记录 `selected_sm`、`selected_sm_policy`、`selected_sm_reason`、`candidate_sm_count`、`included_cta_ids`、`instruction_count`、`warp_count`、`selected_sm_policy_report_hash` 和 `trace_hash`。
    - `included_cta_ids` 必须只来自 selected SM，且覆盖 selected SM 上所有可见 CTA。
  - 负向测试（预期失败）:
    - `collection_scope` 为 `selected_warps_fixture`、`bounded_instruction_window` 或 `full_gpu_full_kernel_dynamic_trace` 时，manifest validator 必须拒绝。
    - 缺少 `selected_sm_reason`、`included_cta_ids` 或 `selected_sm_policy_report_hash` 时，validator 必须拒绝。

- AC-3: `scheduler_signature_medoid_sm` 可以 deterministic 地选择 representative SM
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_sm_selection.py::test_scheduler_signature_medoid_selects_nearest_sm`
    - selection 必须从 per-SM scheduler metadata 构造 raw signature。
    - selection 必须执行 per-kernel min-max normalization。
    - selection 必须计算 equal-weight L2 distance。
    - distance 最小的 SM 必须成为 `selected_sm`。
    - distance 相同的 tie 必须按 `lowest_sm_id` 处理。
  - 负向测试（预期失败）:
    - candidate SM 为空时，selection 必须失败并返回 `no_candidate_sm_for_kernel_invocation`。
    - 缺少 `cta_start_order` 或 `cta_end_order` 时，selection 必须失败。
    - 随机 tie-break 或 first-observed SM 不能进入 `phase_b_complete` artifact。

- AC-4: `selected_sm_policy_report` 完整记录 SM selection evidence
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_sm_selection.py::test_selected_sm_policy_report_is_hashable_and_complete`
    - report 必须包含 `selected_sm_policy = scheduler_signature_medoid_sm`、`selected_sm_policy_version`、`candidate_sm_ids`、`signature_fields`、`signature_field_weights`、`raw_signature_by_sm`、`normalized_signature_by_sm`、`global_sm_signature`、`distance_metric`、`distance_to_global_signature_by_sm`、`tie_break_rule`、`instruction_count_proxy_source` 和 `selection_hash`。
    - `selection_hash` 必须由 canonical JSON 计算，重复运行可复现。
  - 负向测试（预期失败）:
    - 用 `0` 填充缺失 signature 字段后继续 selection 时，validator 必须拒绝。
    - report 中 `selected_sm` 与最小 distance SM 不一致时，validator 必须拒绝。

- AC-5: scope audit 正确记录 before-scope / after-scope 信息
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_trace_scope.py::test_scope_audit_records_before_and_after_counts`
    - audit 必须记录 `scope_policy`、`scope_reason`、`selected_sm`、`included_cta_ids`、`instruction_count_before_scope`、`instruction_count_after_scope`、`warp_count_before_scope`、`warp_count_after_scope` 和 `trace_scope_hash`。
    - 如果 acquisition layer 无法提供 full-GPU candidate counts，必须记录 `before_scope_counts_available = false` 和 `missing_before_scope_reason`。
  - 负向测试（预期失败）:
    - 用空值或零值伪装 unavailable before-scope count 时，validator 必须拒绝。
    - after-scope count 与 included CTA trace 实际条目数不一致时，validator 必须拒绝。

- AC-6: M1 按 warp 构建小图，然后合并为 kernel canonical graph
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_graph_builder.py::test_builds_per_warp_graphs_then_kernel_union`
    - trace entries 必须先按 `warp_id` 分组，再按每个 warp 内的 `trace_index` 排序。
    - 每个 warp graph 必须包含 instruction nodes、variable nodes、pseudo nodes、control-flow edges 和 data-flow edges。
    - 合并后的 canonical graph 必须包含 `graph_id`、`kernel_invocation_id`、`nodes`、`edges`、`warp_partitions`、`graph_summary` 和 `graph_hash`。
  - 负向测试（预期失败）:
    - 直接把不同 warp 的 instruction 串成一条 control-flow 主链时，graph validator 必须拒绝。
    - 删除任意非空 warp partition 后，graph validator 必须拒绝。

- AC-7: `warp_partitions` 支持 node -> warp -> kernel readout
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_graph_builder.py::test_warp_partitions_are_complete_and_replayable`
    - 每个 partition 必须记录 `warp_id`、`node_ids`、`edge_ids`、`instruction_count`、`node_count`、`edge_count`、`first_trace_index` 和 `last_trace_index`。
    - 所有 graph node 必须属于且只属于一个 warp partition。
    - CTA / selected SM provenance 必须保留在 trace manifest 或 graph metadata 中。
  - 负向测试（预期失败）:
    - 同一个 node 出现在多个 warp partition 中时，validator 必须拒绝。
    - 非空 warp partition 缺少 instruction node 时，validator 必须拒绝。

- AC-8: graph size audit 只输出 audit-only size class
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_graph_audit.py::test_graph_size_audit_records_size_class_without_auto_blocking`
    - audit 必须记录 `instruction_count`、`warp_count`、`node_count`、`edge_count`、各类 node / edge count、`max_warp_instruction_count`、`max_warp_node_count`、`max_warp_edge_count`、`graph_size_class`、`size_policy_version` 和 `training_resource_status`。
    - `small`、`medium`、`large` 和 `oversized` 都只表示 audit class，不得自动改变 trace scope。
    - `size_policy_version` 必须为 `phase_b_audit_guardrail_v1`。
  - 负向测试（预期失败）:
    - large / oversized graph 被自动截断后继续标记为 Phase B complete 时，validator 必须拒绝。
    - audit count 与 canonical graph 实际 count 不一致时，validator 必须拒绝。

- AC-9: 资源失败时输出 resource-blocked artifact
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_pipeline.py::test_training_resource_failure_writes_resource_blocked_artifact`
    - 当 per-warp graph batching / tensorization / training 因真实资源限制失败时，pipeline 必须写出 `resource_blocked_artifact`。
    - `resource_blocked_artifact` 必须包含 graph hash、size audit、failed stage、resource failure reason、attempted batch config 和 suggested next spec boundary。
    - size class 本身不能直接触发 resource-blocked。
  - 负向测试（预期失败）:
    - resource-blocked 后继续生成 embedding table 或 selector artifacts 时，pipeline test 必须失败。
    - resource-blocked 后改用 bounded window、selected warps 或 sampled graph path 时，validator 必须拒绝。

- AC-10: M2 tensorization 继承 Phase A strict paper schema
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_tensorizer.py::test_phase_b_tensorization_reuses_phase_a_strict_schema`
    - tensor artifact 必须记录 `tensorizer_version`、`input_graph_hash`、`node_feature_schema`、`edge_relation_schema`、`feature_width`、`padding_policy`、`missing_value_policy` 和 `tensor_hash`。
    - `node_feature_schema` 必须为 `gcl_m2_phase_a_paper_node_feature_v1`。
    - `feature_width` 必须为 64。
    - `paper_reproduction_mode` 必须为 `strict_gcl_sampler_node_features`。
  - 负向测试（预期失败）:
    - 在 Phase B 中引入新的 node feature schema 或改变 feature width 时，validator 必须拒绝。
    - tensorization 修改 canonical graph artifact 时，replay test 必须失败。

- AC-11: M2 输出 warp partition tensors 和 graph batch metadata
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_tensorizer.py::test_tensor_bundle_contains_warp_partition_tensors`
    - tensor bundle 必须包含 `node_features`、`edge_index`、`edge_type`、`warp_partitions`、`warp_partition_tensors` 和 `graph_batch_metadata`。
    - `edge_index.shape = [2, edge_count]`，`edge_type.shape = [edge_count]`。
    - 每个 warp partition tensor 必须引用 canonical graph 中的 node index 范围或 explicit node index list。
  - 负向测试（预期失败）:
    - warp partition tensor 中引用不存在的 node index 时，validator 必须拒绝。
    - `edge_index`、`edge_type` 或 partition tensors 长度不一致时，validator 必须拒绝。

- AC-12: training augmentation 只作用于 M2 training view，不覆盖 canonical graph
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_augmentation.py::test_augmentation_manifests_reference_canonical_graph_without_overwrite`
    - 从同一个 canonical graph 派生两个 augmented views。
    - augmentation manifest 必须记录 `augmentation_manifest_hash`、`input_graph_hash`、`random_seed`、`view_id`、`augmentation_types`、`rates`、`dropped_node_count`、`dropped_edge_count`、`feature_noise_std`、`retry_count` 和 `view_hash`。
    - canonical graph hash 在 augmentation 前后必须保持不变。
  - 负向测试（预期失败）:
    - augmentation 直接覆盖 canonical graph artifact 时，test 必须失败。
    - augmentation 删除整个 warp partition 且没有 retry / reject 记录时，validator 必须拒绝。

- AC-13: hierarchical readout 使用 node -> warp -> kernel pooling
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_readout.py::test_hierarchical_readout_pools_nodes_to_warps_to_kernel`
    - 每个 warp 内使用 mean pooling 得到 warp embedding。
    - kernel embedding 由所有 warp embeddings average pooling 得到。
    - readout manifest 必须记录每个 warp 的 `node_count_used`、`pooling_method = mean`、`warp_embedding_dim`，以及 kernel 层的 `warp_count_used`、`pooling_method = average`、`kernel_embedding_dim`。
  - 负向测试（预期失败）:
    - 使用 all-node global pooling 跳过 warp 层时，readout validator 必须拒绝。
    - 空 warp partition 被生成随机 embedding 时，validator 必须拒绝。

- AC-14: embedding export 使用 canonical non-augmented graph 的 256 维 kernel embedding
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_embedding_export.py::test_phase_b_exports_m0_compatible_256_dim_embeddings`
    - 每个 embedding row 必须包含 `record_id`、`kernel_invocation_id`、`representation_mode`、`embedding_dim`、`embedding`、`source_graph_hash`、`encoder_manifest_hash`、`embedding_hash` 和 `weight_input`。
    - `embedding_dim = 256`。
    - selector 使用 projection head 之前的 kernel embedding，contrastive loss 使用 projection output。
  - 负向测试（预期失败）:
    - 导出 projection head 的 64 维 output 作为 selector embedding 时，validator 必须拒绝。
    - 导出 augmented view embedding 作为 selector embedding 时，validator 必须拒绝。

- AC-15: M0 selector 可以消费 Phase B embedding table
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_selector_integration.py::test_m0_selector_consumes_phase_b_embedding_table`
    - M0 selector 必须执行 z-score normalization、silhouette-selected K、deterministic K-Means 和 representative anchor selection。
    - 输出必须包含 cluster assignments、silhouette report、representative anchor table 和 structural evaluation artifacts。
  - 负向测试（预期失败）:
    - embedding table 中 mixed `representation_mode` 或 mixed `embedding_dim` 时，selector validator 必须拒绝。
    - resource-blocked invocation 出现在 formal selector input 中时，validator 必须拒绝。

- AC-16: Phase B pipeline 可以一条命令跑通 eligible trace batch
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_pipeline.py::test_phase_b_pipeline_e2e_on_eligible_trace_batch`
    - `python -m experiments.gcl_phase_b.pipeline --input tests/fixtures/gcl_phase_b/representative_sm_trace_manifest.json --out artifacts/gcl_phase_b`
    - 输出目录必须包含 trace manifest、selected SM policy report、scope audit、graph bundle、graph size audit、tensor bundle、augmentation manifests、training report、checkpoint manifest、readout manifest、embedding table、selector artifacts 和 pipeline manifest。
  - 负向测试（预期失败）:
    - 删除 selected SM policy report 后继续运行 graph construction，pipeline 必须失败。
    - 删除 graph size audit 后继续运行 tensorization，pipeline 必须失败。
    - 删除 tensor bundle 后运行 embedding export，pipeline 必须失败。

- AC-17: Phase B artifacts 可 replay、可 audit
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_b/test_replay.py::test_phase_b_artifacts_are_replayable`
    - 固定 seed 后，`selection_hash`、`trace_scope_hash`、`graph_hash`、`graph_size_audit_hash`、`tensor_hash`、`augmentation_manifest_hash`、`encoder_manifest_hash`、`embedding_hash` 和 `selector_manifest_hash` 必须可复现。
    - pipeline manifest 必须引用所有关键 artifact hash。
  - 负向测试（预期失败）:
    - 修改 selected SM、included CTA、canonical graph、tensorizer schema 或 checkpoint bytes 后，相关 hash 必须变化，replay validator 必须拒绝 stale artifact。

## 路径边界

### 上界

允许的最大范围：

- 新增 `experiments/gcl_phase_b` Python package，作为 Phase A implementation 的 trace-scope 扩展层；
- 新增 representative-SM trace manifest parser / validator；
- 新增 `scheduler_signature_medoid_sm` selected SM policy resolver；
- 新增 selected SM policy report、scope audit、graph size audit 和 resource-blocked artifact；
- 复用或扩展 Phase A graph builder，使其支持 representative-SM trace manifest 输入和 per-warp graph union；
- 复用 Phase A tensorizer、RGCN、augmentation、embedding export 和 M0 selector contract；
- 新增 `tests/gcl_phase_b` 下的 unit tests、contract tests 和 end-to-end smoke tests；
- 新增 `tests/fixtures/gcl_phase_b` 小规模 representative-SM trace manifest fixture；
- 新增 `artifacts/gcl_phase_b` 作为默认本地输出目录。

### 下界

最低可接受实现：

- 当前实现分支已经承接 Phase A RLCR 实现；
- 能读取一个小规模 representative-SM trace manifest fixture；
- 能根据 `explicit_sm_id` 或 `scheduler_signature_medoid_sm` 得到 selected SM all-CTA scope；
- 能按 warp 构建 canonical graph 并生成 `warp_partitions`；
- 能输出 audit-only graph size audit；
- 对未发生 resource-blocked 的 graph，能完成 tensorization、augmentation manifest、hierarchical readout、embedding export 和 M0 selector；
- 所有关键 artifacts 支持 hash replay。

### 允许与禁止

可以使用：

- Python standard library、`numpy`、`pytest`；
- Phase A 已实现的 `experiments.gcl_phase_a` modules；
- `torch` 继续执行 Phase A RGCN training；
- JSON artifacts、manifest hashes 和 deterministic validators；
- 小规模 representative-SM fixture 作为 Phase B E2E smoke input。

不可以使用：

- `selected_warps_fixture` 作为 Phase B complete path；
- bounded instruction window；
- selected-warps fallback；
- random SM、first-observed SM、max-instruction-only SM 或 max-CTA-only SM 作为完成路径；
- attention pooling；
- full-GPU full-kernel dynamic trace 作为 Phase B 默认输入；
- 改变 Phase A strict node feature schema；
- projection head output 作为 selector embedding；
- augmented graph 覆盖 canonical graph；
- graph 被静默截断后继续标记为 Phase B complete；
- size class 自动触发 blocked 或 compression。

## 依赖与顺序

### 里程碑

1. 里程碑 0：承接 Phase A RLCR 实现
   - 确认当前实现分支包含 `experiments/gcl_phase_a` 和 `tests/gcl_phase_a`。
   - 跑通 Phase A 基础验证。
   - 写 AC-1 tests。

2. 里程碑 1：Representative-SM trace manifest 和 SM selection
   - 实现 representative-SM trace manifest parser / validator。
   - 实现 `explicit_sm_id` 和 `scheduler_signature_medoid_sm`。
   - 实现 scheduler signature normalization、global signature、distance 计算和 deterministic tie-break。
   - 输出 `selected_sm_policy_report` 和 `selection_hash`。
   - 写 AC-2 / AC-3 / AC-4 tests。

3. 里程碑 2：Scope audit
   - 实现 selected SM all-CTA scope extraction。
   - 实现 before-scope / after-scope count 记录。
   - 输出 `trace_scope_hash`。
   - 写 AC-5 tests。

4. 里程碑 3：Per-warp graph construction 和 kernel graph union
   - 将 selected SM all-CTA trace entries 按 warp 分组。
   - 对每个 warp 复用 Phase A graph construction semantics。
   - 合并为 kernel canonical graph，并保留完整 `warp_partitions`。
   - 写 AC-6 / AC-7 tests。

5. 里程碑 4：Graph size audit 和 resource guard
   - 实现 `small` / `medium` / `large` / `oversized` audit-only size class。
   - size class 不自动阻止 M2 training。
   - 实现真实资源失败时的 `resource_blocked_artifact`。
   - 写 AC-8 / AC-9 tests。

6. 里程碑 5：Phase B tensorization boundary
   - 复用 Phase A strict node feature schema。
   - 增加 warp partition tensors 和 graph batch metadata。
   - 保证 tensorization 只生成派生产物，不修改 canonical graph。
   - 写 AC-10 / AC-11 tests。

7. 里程碑 6：Training augmentation 和 hierarchical readout
   - 从 canonical graph 派生两个 training views。
   - 输出 augmentation manifests。
   - 实现 node -> warp -> kernel readout manifest。
   - 写 AC-12 / AC-13 tests。

8. 里程碑 7：Embedding export 和 M0 selector integration
   - 导出 256 维 canonical kernel embedding。
   - 接入 M0 selector，输出 representative anchors。
   - 写 AC-14 / AC-15 tests。

9. 里程碑 8：End-to-end pipeline 和 replay
   - 实现 `python -m experiments.gcl_phase_b.pipeline --input tests/fixtures/gcl_phase_b/representative_sm_trace_manifest.json --out artifacts/gcl_phase_b`。
   - 保存所有 Phase B artifacts 和 pipeline manifest。
   - 实现 disk-backed repair / replay validators。
   - 写 AC-16 / AC-17 tests。

## 实施说明

- 代码中不要写入 `AC-1` 或 `Milestone 1` 这类 plan 术语。
- Artifact 名称使用稳定英文标识符。
- 面向人的文档可以使用中文，JSON keys、module names、manifest keys 和 CLI flags 保持英文。
- Phase B 必须写出 graph size audit，但不能只因 size class 改变 trace scope 或跳过 tensorization / training。
- Phase B 的 resource-blocked path 是正式结果，不是测试失败；size class 本身不是 blocked reason。
- 如果当前环境没有 `torch`，training entrypoint 必须给出清晰依赖错误，不能静默替换成非 RGCN 路径。
- 所有随机行为必须显式记录 seed，包括 selected SM tie-break、augmentation、RGCN initialization、training 和 selector。
- 所有 disk-backed repair stage 必须既返回 artifact，也写回对应 JSON 文件。
- Phase C compression 只作为后续阶段；Phase B 不默认实现 instruction stream compression。

## 建议验证命令

```bash
pytest -q tests/gcl_phase_a
pytest -q tests/gcl_phase_b
python -m experiments.gcl_phase_b.pipeline --input tests/fixtures/gcl_phase_b/representative_sm_trace_manifest.json --out artifacts/gcl_phase_b
python -m pytest -q tests/gcl_phase_b/test_trace_scope.py::test_trace_manifest_records_single_representative_sm_all_ctas
python -m pytest -q tests/gcl_phase_b/test_sm_selection.py::test_scheduler_signature_medoid_selects_nearest_sm
python -m pytest -q tests/gcl_phase_b/test_graph_audit.py::test_graph_size_audit_records_size_class_without_auto_blocking
python -m pytest -q tests/gcl_phase_b/test_pipeline.py::test_training_resource_failure_writes_resource_blocked_artifact
python -m pytest -q tests/gcl_phase_b/test_readout.py::test_hierarchical_readout_pools_nodes_to_warps_to_kernel
```
