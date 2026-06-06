# A 线 GCL ResNet-50 Full Reproduction Plan

日期：2026-06-06

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. This plan is a roadmap-level implementation plan; each gate may still need its own focused implementation plan before code changes.

## 1. 总目标

完整复现 GCL-Sampler 在 ResNet-50 上的核心路径，并在复现之后增加我们自己的 correctness evaluation 和调参扩展。

总路径：

```text
真实 ResNet-50 NVBit trace
  -> Gate 0 real trace acquisition
  -> Gate 1 trace adapter
  -> Gate 2 representative SM manifest
  -> Gate 3 canonical graph construction
  -> Gate 4 graph tensorization
  -> Gate 5 RGCN contrastive embedding
  -> Gate 6 silhouette-K / K-Means selector
  -> Gate 7 cluster correctness evaluation
  -> Gate 8 tuning vector proposal
  -> Gate 9 sampled-vs-full simulator evaluation
```

## 2. 硬约束

正式路径必须使用真实完整 ResNet-50 NVBit trace。

formal artifacts 必须记录：

```text
workload_id = resnet50
model = torchvision.models.resnet50
execution_mode = real_trace
trace_source = nvbit
scheduler_metadata_source = real_nvbit_smid
input_scope = full_resnet50_inference_trace
```

以下输入只能用于 unit / smoke / debug，不得作为 formal pass：

```text
synthetic trace
ResNet-like fixture
hand-written opcode sequence
mini-transformer trace
simulator replay trace
file_order_fallback scheduler metadata
partial manually selected kernel-only trace
```

当前人工 trace 已经证明工程连通性，但不能作为 GCL formal reproduction 的正确性证据。

## 3. 参考文档

必须遵循：

- `docs/superpowers/specs/2026-06-05-a-line-gcl-resnet50-real-trace-acquisition-design.md`
- `docs/superpowers/specs/2026-06-05-a-line-gcl-resnet50-gate1-trace-adapter-design.md`
- `docs/superpowers/specs/2026-06-05-a-line-gcl-resnet50-gate2-representative-sm-manifest-design.md`
- `docs/superpowers/specs/2026-06-05-a-line-gcl-resnet50-gate3-canonical-graph-construction-design.md`
- `docs/superpowers/specs/2026-06-06-a-line-gcl-resnet50-gate4-tensorization-design.md`
- `docs/superpowers/specs/2026-06-06-a-line-gcl-resnet50-gate5-rgcn-contrastive-training-design.md`
- `docs/superpowers/specs/2026-06-06-a-line-gcl-resnet50-gate6-selector-family-evidence-design.md`
- `docs/superpowers/specs/2026-06-06-a-line-gcl-resnet50-gate7-cluster-correctness-evaluation-design.md`

已有实现计划参考：

- `docs/superpowers/plans/2026-06-06-a-line-gcl-resnet50-gate1-5-implementation-plan.md`
- `docs/superpowers/plans/2026-06-02-a-line-gcl-m1-m2-phase-a-semantic-e2e-plan.md`
- `docs/superpowers/plans/2026-06-04-a-line-gcl-m1-m2-phase-b-implementation-plan.md`

## 4. 当前状态

已完成：

```text
人工 / ResNet-like fixture
  -> Gate 1 adapter
  -> Gate 2 representative SM
  -> Gate 3 canonical graph
  -> Gate 4 tensorization
  -> Gate 5 RGCN / embedding export
```

这说明 Gate1-5 的工程链路、artifact hash、replay、resource-blocked recovery 和 readout 层级已经有 smoke validation。

尚未完成：

```text
真实完整 ResNet-50 NVBit trace
  -> formal Gate 1-5
  -> Gate 6 selector
  -> Gate 7 correctness evaluation
  -> Gate 8 tuning vector
  -> Gate 9 simulator evaluation
```

因此当前不能声称：

```text
真实 ResNet-50 trace 已接入
GCL embedding 语义正确
GCL clusters 正确
representative kernel 可代表 cluster members
simulator speedup / accuracy 成立
```

## 5. Phase 0: Real Trace Acquisition

### Gate 0: ResNet-50 NVBit Trace Acquisition

目标：

```text
torchvision ResNet-50 inference
  -> NVBit trace acquisition
  -> real scheduler metadata
```

输入：

```text
model = torchvision.models.resnet50
weights = torchvision.models.ResNet50_Weights.DEFAULT
precision = fp16_autocast
batch_size = 1
input_shape = [1, 3, 224, 224]
```

输出：

```text
dynamic_trace.pb
threadblocks/
extra_info/enhanced_execution_info.json
extra_info/scheduler_metadata.json
stats.csv
gate0_trace_acquisition_manifest.json
```

必须记录：

```text
kernel_invocation_id
kernel_id
kernel_name
function_unique_id
cta_id
sm_id
warp_id
trace_index
first_seen_order
last_seen_order
trace_entry_count
```

验收：

- [ ] 可以运行 ResNet-50 inference 并产生 NVBit trace artifacts。
- [ ] `scheduler_metadata.json` 来自真实 `%smid` 采集路径。
- [ ] 每个 `cta_id` 在同一 kernel invocation 中只有一个 `sm_id`。
- [ ] `trace_entry_count` 与 threadblock trace records 对齐。
- [ ] 缺少真实 SM metadata 时输出 debug / failure artifact，不进入 formal path。

## 6. Phase 1: Trace To Graph

### Gate 1: Trace Adapter

目标：

```text
real ResNet-50 trace artifacts
  -> resnet50_trace_adapter_bundle.json
```

输出：

```text
resnet50_trace_adapter_bundle.json
adapter_validation_report.json
```

验收：

- [ ] formal adapter bundle 必须来自真实完整 ResNet-50 NVBit trace。
- [ ] synthetic / fixture / replay 输入必须被拒绝 formal pass。
- [ ] 可以建立稳定 `kernel_invocation_id`。
- [ ] 可以解析 static instruction metadata。
- [ ] 可以展开 per-warp dynamic trace entries。
- [ ] bundle hash 可复现。

### Gate 2: Representative SM Manifest

目标：

```text
resnet50_trace_adapter_bundle.json
  -> scheduler_signature_medoid_sm
  -> representative_sm_trace_manifest.json
```

输出：

```text
representative_sm_trace_manifest.json
selected_sm_policy_report.json
scope_preview_report.json
```

验收：

- [ ] 输入必须是 Gate1 formal passed bundle。
- [ ] 选择策略默认 `scheduler_signature_medoid_sm`。
- [ ] `included_cta_ids` 只来自 selected SM，且覆盖该 SM 上所有 CTA。
- [ ] manifest 继承真实 ResNet-50 provenance。
- [ ] Phase B validator 接受该 manifest。

### Gate 3: Canonical Graph Construction

目标：

```text
representative_sm_trace_manifest.json
  -> selected-SM per-warp trace records
  -> canonical_graph_bundle.json
```

输出：

```text
phase_b_trace_records.json
canonical_graph_bundle.json
graph_size_audit.json
graph_construction_report.json
```

验收：

- [ ] 按 warp 独立建小图，再合并成一个 typed canonical graph。
- [ ] 保留 `warp_partitions` 和 CTA 层级。
- [ ] relation type 只允许 `control_flow`、`data_source`、`data_destination`。
- [ ] strict path 第一版只允许 `mem_ref` pseudo node。
- [ ] graph size audit 只用于审计，不自动截断正式输入。

### Gate 4: Tensorization

目标：

```text
canonical_graph_bundle.json
  -> graph_tensor_bundle.json
```

输出：

```text
graph_tensor_bundle.json
tensorization_report.json
```

验收：

- [ ] 输出 `node_features.shape = [node_count, 64]`。
- [ ] 输出 `edge_index.shape = [2, edge_count]`。
- [ ] 输出 `edge_type.shape = [edge_count]`。
- [ ] 输出 warp / CTA partition tensors。
- [ ] 记录 `representation_mode`、`pseudo_node_mode`、`paper_reproduction_mode`。

## 7. Phase 2: GCL Embedding Reproduction

### Gate 5: RGCN Contrastive Training / Embedding Export

目标：

```text
graph_tensor_bundle.json
  -> training-only graph augmentation
  -> RGCN contrastive training
  -> canonical non-augmented inference
  -> kernel_embedding_table.json
```

输出：

```text
augmentation_manifest.json
rgcn_training_run_manifest.json
rgcn_checkpoint_manifest.json
kernel_embedding_table.json
embedding_export_report.json
```

验收：

- [ ] augmentation 只属于 training，不覆盖 canonical tensor。
- [ ] RGCN 使用三层结构：64 -> 128 -> 128 -> 256。
- [ ] readout 使用 `node -> warp -> CTA -> selected SM -> kernel`。
- [ ] InfoNCE 使用 projection head output。
- [ ] selector 使用 projection head 之前的 256 维 `kernel_embedding`。
- [ ] `kernel_embedding_table.json` 只来自 canonical non-augmented graph。

## 8. Phase 3: GCL Selector

### Gate 6: Selector / Family Evidence

目标：

```text
kernel_embedding_table.json
  -> embedding validation
  -> z-score normalization
  -> silhouette-K
  -> deterministic K-Means
  -> representative anchor table
  -> cluster-family evidence report
```

输出：

```text
embedding_validation_report.json
embedding_normalization_report.json
k_selection_report.json
kmeans_cluster_assignment_table.json
representative_anchor_table.json
cluster_family_evidence_report.json
gate6_selector_manifest.json
```

验收：

- [ ] 只接受真实完整 ResNet-50 Gate5 formal embedding table。
- [ ] clustering 只使用 256 维 canonical kernel embedding。
- [ ] `kernel_name`、family label、runtime、graph size 等字段不得参与 clustering。
- [ ] 默认使用 `silhouette_k`。
- [ ] K-Means assignment deterministic 且可复现。
- [ ] representative 必须是真实 kernel invocation。
- [ ] family evidence 只用于 post-clustering evaluation。

说明：

`z-score normalization` 是工程默认，不应声明为 GCL-Sampler 论文强制步骤。Gate6 artifacts 必须记录：

```text
normalization_policy = engineering_default_z_score
paper_defined = false
```

## 9. Phase 4: Correctness Evaluation

### Gate 7: Cluster Correctness Evaluation

目标：

```text
Gate6 selector artifacts
  + family labels
  + graph / trace structural summaries
  + measured NCU or simulator metrics
  + multi-run artifacts
  -> quantified correctness evidence
```

输出：

```text
cluster_embedding_quality_report.json
cluster_family_alignment_report.json
representative_quality_report.json
cluster_metric_error_report.json
cluster_stability_report.json
gate7_cluster_correctness_manifest.json
```

必须量化：

```text
embedding geometry:
  silhouette_score
  davies_bouldin_index
  calinski_harabasz_index
  inter_intra_distance_ratio

family alignment:
  cluster_purity
  weighted_cluster_purity
  adjusted_rand_index
  normalized_mutual_information

representative quality:
  p95_distance_to_representative
  outlier_member_ratio
  high_weight_outlier_count

metric consistency:
  cluster_weighted_mape
  cluster_p95_relative_error
  global_weighted_mape

stability:
  assignment_stability_ari
  k_stability
  representative_stability_rate
```

验收：

- [ ] 不重新选择 K，不修改 Gate6 cluster assignment。
- [ ] family labels 只用于 post-clustering evaluation。
- [ ] measured / simulator metric 缺失时显式标记，不伪装成功。
- [ ] 第一版使用 `threshold_policy = report_only_v1`。
- [ ] 第一版默认 `claim_status = quantified_no_correctness_claim`。

## 10. Phase 5: Our Extension

这一阶段不是 GCL-Sampler 原始复现，而是我们的扩展工作。

### Gate 8: Tuning Vector Proposal

目标：

```text
Gate7 correctness evidence
  + trusted cluster / family mapping
  -> simulator tuning vector proposal
```

输入：

```text
gate7_cluster_correctness_manifest.json
representative_anchor_table.json
cluster_metric_error_report.json
cluster_family_alignment_report.json
simulator_tunable_component_schema.json
```

输出：

```text
cluster_tuning_vector_table.json
tuning_vector_provenance_report.json
tuning_safety_report.json
gate8_tuning_manifest.json
```

验收：

- [ ] Gate8 必须拒绝高权重 mixed-family cluster 直接进入调参。
- [ ] Gate8 必须拒绝 high metric error cluster 直接进入调参。
- [ ] 每个 tuning vector 必须绑定 cluster / representative / evidence hash。
- [ ] 第一版只输出 proposal，不声明 simulator accuracy。

### Gate 9: Sampled-vs-Full Simulator Evaluation

目标：

```text
full simulation / measured baseline
  vs
representative sampled simulation + tuning proposal
  -> speedup / error / stability evaluation
```

输出：

```text
full_vs_sampled_simulation_report.json
sampled_speedup_report.json
sampled_error_report.json
tuning_effect_report.json
gate9_simulator_evaluation_manifest.json
```

验收：

- [ ] 有 full baseline 或明确的 measured baseline。
- [ ] sampled run 使用 Gate6 representative anchors。
- [ ] tuning proposal 来自 Gate8。
- [ ] 输出 speedup、error、p95 error 和 high-weight bad case。
- [ ] 只有 Gate9 通过后，才能声明 sampled simulation 或 tuning 有效。

## 11. 实施顺序

推荐按以下顺序实施：

```text
1. 修正现有 Gate1-5 实现，使 formal path 拒绝 fixture。
2. 实现 Gate0 真实 ResNet-50 NVBit trace acquisition。
3. 用真实 ResNet-50 trace 跑通 Gate1-5 formal path。
4. 实现 Gate6 selector / representative anchor。
5. 实现 Gate7 correctness evaluation。
6. 根据 Gate7 evidence 决定是否进入 Gate8。
7. 实现 Gate8 tuning vector proposal。
8. 实现 Gate9 sampled-vs-full simulator evaluation。
```

## 12. 阶段性判断标准

### 可以称为 GCL-Sampler 核心复现成立

必须至少完成：

```text
Gate0 -> Gate1 -> Gate2 -> Gate3 -> Gate4 -> Gate5 -> Gate6
```

且 Gate6 formal selector artifacts 来自真实完整 ResNet-50 trace。

### 可以称为 GCL 分组可信

必须至少完成：

```text
Gate7
```

并且真实 ResNet-50 baseline 上的 correctness metrics 达到后续定义的 enforced thresholds。

### 可以称为我们的方法产生调参收益

必须至少完成：

```text
Gate8 -> Gate9
```

并且 sampled-vs-full simulator evaluation 支持 speedup / error claim。

## 13. 风险与处理

### 风险 1: 真实 ResNet-50 NVBit trace 采集失败

处理：

- 只允许输出 debug / failure artifact；
- 不允许 fixture 替代 formal path；
- 回到 Gate0 修复 trace acquisition。

### 风险 2: Graph 规模过大导致 Gate5 资源失败

处理：

- 输出 resource-blocked artifact；
- 不自动截断 formal trace；
- 后续单独设计 resource strategy，不混入 strict reproduction。

### 风险 3: Gate6 cluster 和 family label 对齐差

处理：

- Gate7 报告 mixed-family clusters；
- 不训练 classifier head 固化错误；
- 回看 Gate3 graph schema、Gate5 encoder、Gate6 K policy。

### 风险 4: Representative metric error 高

处理：

- Gate7 标记 weak representative / high-error cluster；
- Gate8 不消费这些 cluster；
- 必要时重新审计 embedding quality 或 representative selection。

### 风险 5: 多 run 稳定性差

处理：

- Gate7 标记 unstable assignment；
- 不进入 Gate8；
- 回看 RGCN training seed、augmentation、dataset split 和 embedding export。

## 14. 当前下一步

下一步不应该继续扩展 Gate8，而应该先修复 formal 输入边界：

```text
修正 Gate1-5 实现
  -> fixture 只能 debug
  -> formal path 必须读取真实 ResNet-50 trace
```

然后进入 Gate0：

```text
采集真实 ResNet-50 NVBit trace
```

只有 Gate0 成立后，后续 Gate1-7 的 formal evidence 才有可信输入来源。
