# A 线 GCL GNN Trustworthiness Acceptance Design Spec

日期：2026-06-11

## 1. 目标

本 spec 定义 GCL 复现中 GNN 是否可信的验收标准。它不替代 Gate 5 的 RGCN training spec，也不替代 Gate 7 的 cluster correctness evaluation spec；它把这些证据合并成一个更高层的验收判断：

```text
真实 trace
  -> canonical graph
  -> graph tensor
  -> RGCN contrastive training
  -> kernel embedding
  -> selector clustering
  -> cluster evidence
  -> GNN trustworthiness acceptance
```

本验收要回答的问题是：

```text
这个 GNN 是否真的利用 graph structure 学到了稳定、有语义、对下游有用的 kernel representation？
```

## 2. 非目标

本 spec 不做：

- 新的 trace acquisition；
- 新的 graph construction；
- 新的 tensor schema 设计；
- 新的 GNN architecture 设计；
- 新的 K-Means 或 silhouette-K 算法；
- supervised cluster head 训练；
- 调参比例预测；
- simulator speedup claim。

它只定义如何判断已有 GNN 结果是否可信，以及何时可以升级 claim status。

## 3. 验收状态

GNN acceptance report 必须输出一个总状态：

```text
gnn_acceptance_status
```

允许值：

```text
accepted
weak_acceptance_structure_valid_but_correctness_unproven
rejected_training_insufficient
rejected_unstable_embedding
rejected_no_graph_signal
rejected_downstream_unproven
not_evaluable_missing_artifacts
```

含义：

```text
accepted
  输入真实，结构正确，训练充分，embedding 优于 baseline，
  clustering 稳定，cluster 有语义，下游 representative 有效。

weak_acceptance_structure_valid_but_correctness_unproven
  GNN 结构和端到端链路成立，embedding 有初步信号，
  但缺少训练充分性、稳定性、baseline 或下游证据。

rejected_training_insufficient
  训练规模或训练过程不足以支撑 representation claim。

rejected_unstable_embedding
  多 seed 下 selected_k、assignment、representative 或 embedding geometry 不稳定。

rejected_no_graph_signal
  完整 RGCN 不优于 random / histogram / no-edge baseline。

rejected_downstream_unproven
  embedding / cluster 几何成立，但 representative 无法代表下游 simulator 或 measured metric。

not_evaluable_missing_artifacts
  必要 artifact 缺失，无法验收。
```

## 4. 验收等级

每个验收项必须使用以下等级：

```text
PASS
WEAK_PASS
FAIL
NOT_AVAILABLE
NOT_APPLICABLE
UNPROVEN
```

要求：

- `PASS` 必须有 artifact 证据支持；
- `WEAK_PASS` 表示有正向信号但证据不足；
- `FAIL` 表示证据存在且不满足要求；
- `NOT_AVAILABLE` 表示该证据尚未生成；
- `NOT_APPLICABLE` 表示该验收项对当前 run 不适用。
- `UNPROVEN` 表示有相关证据，但该证据不能支撑语义正确性或下游正确性声明。

报告不得把 `WEAK_PASS` 解释为最终正确性通过。

## 5. 输入 Provenance 验收

GNN 可信性验收首先检查输入是否真实。

必须满足：

```text
workload_id = resnet50 或正式 workload id
execution_mode = real_trace
trace_source = nvbit
input_scope = full workload trace
formal_full_trace_run = true
synthetic_fixture = false
debug_slice = false
```

必须记录：

```text
input_root
kernel_invocation_count
cta_record_count
trace_acquisition_manifest_hash
adapter_bundle_hash
```

如果输入来自 synthetic / hand-written / bounded fixture，则 GNN 只能做 smoke validation，不能进入可信性验收。

## 6. Graph 与 Tensor 验收

Graph / tensor 输入必须证明 GNN 看到的是 canonical non-augmented graph，而不是 training augmentation view。

必须满足：

```text
canonical_graph_bundle exists
graph_tensor_bundle exists
node_features.shape = [node_count, 64]
edge_index.shape = [2, edge_count]
edge_type.shape = [edge_count]
warp / CTA / selected-SM hierarchy is preserved
tensor_hash reproducible
```

必须记录：

```text
representation_mode
pseudo_node_mode
edge_relation_schema
readout_hierarchy
source_graph_hash
source_graph_tensor_bundle_hash
```

如果 augmentation 覆盖 canonical artifact，验收必须失败。

## 7. RGCN 结构验收

RGCN 结构必须是真正的 relation-aware message passing。

当前 GCL reproduction 默认接受的结构是：

```text
layers = 3
input_dim = 64
hidden_dim = 128
kernel_embedding_dim = 256
relation_count = 3
projection_hidden_dim = 128
projection_output_dim = 64
```

relation schema 必须至少包含：

```text
control_flow
data_source
data_destination
```

readout 必须记录：

```text
node -> warp -> CTA -> selected SM -> kernel
```

验收判断：

- 如果 relation type 只被保存但没有进入 message passing 参数选择，结构验收失败；
- 如果 GNN 输出只是手工 feature pooling，不经过 message passing，结构验收失败；
- 如果 projection head output 被错误用于 selector，结构验收失败；
- 如果 selector 使用 canonical encoder readout embedding，结构验收可通过。

## 8. Training Adequacy 验收

训练充分性不能只看 training run 是否结束。

必须记录：

```text
train_graph_count
export_graph_count
epoch_count
optimizer_step_count
batch_size
loss_curve
positive_pair_count
negative_pair_count
augmentation_config
checkpoint_hash
checkpoint_reuse
random_seed
```

最低验收要求：

```text
train_graph_count 必须覆盖足够多的 kernel graph；
optimizer_step_count 必须大于 smoke-test 单步训练；
loss_curve 必须可解释，不能只有单点；
checkpoint 必须来自本次 formal training 或明确记录 reuse reason；
training subset policy 必须说明是否会造成 selector bias。
```

如果当前 run 只有极小训练子集或单个 optimizer step，则必须标记：

```text
training_adequacy = FAIL
```

即使 embedding geometry 有正向信号，也不能升级为可信 GNN。

## 9. Embedding Geometry 验收

Embedding geometry 是必要证据，但不是充分证据。

必须计算：

```text
silhouette_score
davies_bouldin_index
calinski_harabasz_index
mean_intra_cluster_distance
mean_inter_cluster_distance
inter_intra_distance_ratio
cluster_radius_by_cluster
cluster_separation_by_cluster
```

验收判断：

- 指标可计算且方向合理，可以给 `WEAK_PASS`；
- 指标显著差于 baseline，必须失败；
- 没有 baseline 时，不得给最终 `PASS`；
- cluster 极不均衡时，必须附带 outlier-discovery 风险说明。

## 10. Baseline Ablation 验收

完整 RGCN 必须优于简单 baseline，才能证明 graph message passing 有贡献。

必须至少比较：

```text
random_embedding_baseline
opcode_histogram_baseline
node_feature_pooling_no_edge_baseline
control_flow_only_rgcn
data_flow_only_rgcn
full_rgcn
```

比较指标：

```text
silhouette
davies_bouldin
inter_intra_ratio
assignment_stability_ari
representative_metric_error
```

验收规则：

```text
full_rgcn 必须在多个指标上优于 random 和 histogram baseline；
full_rgcn 必须证明 edge relation 对结果有贡献；
如果 no-edge baseline 与 full_rgcn 接近，则不能声明 graph learning 有效。
```

## 11. Multi-Seed Stability 验收

可信 GNN 必须稳定。

必须运行多 seed：

```text
training_seed_count >= 3
kmeans_seed_count >= 5
```

必须记录：

```text
k_stability
assignment_stability_ari
assignment_stability_nmi
centroid_drift
representative_stability_rate
metric_mean_and_std
```

验收规则：

- selected_k 大幅波动时，不能通过；
- assignment ARI 低时，不能通过；
- representative anchor 不稳定时，不能声称 sampled representative 可靠；
- 单次运行必须标记 `single_run_not_evaluated`。

## 12. Semantic Cluster Alignment 验收

Cluster 必须能与可解释的 kernel 语义对齐。

可使用的 post-clustering labels：

```text
kernel_name
operator_type
layer_id
opcode_mix
memory_instruction_ratio
control_flow_edge_ratio
data_flow_edge_ratio
node_count
edge_count
warp_count
cta_count
runtime_weight
```

必须计算：

```text
cluster_purity
weighted_cluster_purity
adjusted_rand_index
normalized_mutual_information
homogeneity
completeness
v_measure
mixed_family_cluster_count
high_weight_mixed_family_cluster_count
```

如果只有一个粗粒度 label，例如所有 kernel 都是 `resnet50_real_trace`，则 purity 不能证明分类正确。报告必须说明 label 粒度不足。

## 13. Representative Downstream Usefulness 验收

GCL 的最终价值不是只得到 cluster，而是让 representative kernel 能代表 cluster members。

必须计算：

```text
mean_distance_to_representative
p95_distance_to_representative
max_distance_to_representative
representative_rank_to_centroid
outlier_member_ratio
high_weight_outlier_count
```

如果有 measured / simulator metrics，必须计算：

```text
cluster_weighted_mape
cluster_p95_relative_error
global_weighted_mape
global_p95_relative_error
cluster_metric_correlation
cluster_metric_rank_correlation
```

没有 simulator 或 measured metric 时，必须标记：

```text
downstream_representative_usefulness = NOT_AVAILABLE
```

## 14. Claim Status 升级规则

默认 claim status 必须保守。

允许状态：

```text
quantified_no_correctness_claim
structure_valid_embedding_signal_only
cluster_stability_supported
semantic_cluster_supported
representative_downstream_supported
gnn_trustworthiness_accepted
```

升级条件：

```text
structure_valid_embedding_signal_only
  RGCN 结构通过，embedding geometry 有正向信号。

cluster_stability_supported
  多 seed 下 selected_k、assignment、representative 稳定。

semantic_cluster_supported
  cluster 与细粒度 semantic labels 对齐。

representative_downstream_supported
  representative 能近似 cluster members 的 measured / simulator metrics。

gnn_trustworthiness_accepted
  baseline ablation、多 seed stability、semantic alignment、
  downstream representative usefulness 全部通过。
```

如果任一必要证据缺失，必须保持或降级 claim status。

## 15. 输出 Artifact

GNN acceptance report 必须输出：

```text
gnn_acceptance_report.md
gnn_acceptance_manifest.json
gnn_acceptance_summary.json
```

manifest 至少包含：

```text
artifact_type
artifact_version
workload_id
input_artifact_hashes
acceptance_items
gnn_acceptance_status
claim_status
blocking_gaps
recommended_next_gates
report_hash
```

## 16. 当前 ResNet-50 run 的预期判断

基于当前 artifacts，ResNet-50 full trace run 应被判断为：

```text
gnn_acceptance_status =
  weak_acceptance_structure_valid_but_correctness_unproven

claim_status =
  quantified_no_correctness_claim
```

原因：

- 输入是真实 ResNet-50 full trace；
- RGCN 结构和 readout hierarchy 成立；
- embedding geometry 有初步分离信号；
- 训练只有小规模 smoke 强度；
- baseline ablation 尚未完成；
- multi-seed stability 尚未评估；
- semantic label 粒度不足；
- downstream simulator / measured metric representative validation 尚未完成。
