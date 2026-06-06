# A 线 GCL ResNet-50 Gate 6 Selector / Family Evidence Design Spec

日期：2026-06-06

## 1. Gate 6 定位

Gate 6 的目标是消费 Gate 5 导出的 formal `kernel_embedding_table.json`，在真实 ResNet-50 kernel embedding space 中执行 GCL-Sampler 风格的无监督 selector，并输出 cluster、representative anchor 和 cluster-to-family evidence。

Gate 6 只负责：

```text
kernel_embedding_table.json
  -> validate real ResNet-50 provenance
  -> embedding row validation
  -> z-score normalization
  -> silhouette-K selection
  -> deterministic K-Means
  -> representative kernel anchor selection
  -> cluster-to-family evidence report
```

Gate 6 不训练 GNN，不修改 RGCN encoder，不引入 supervised classification head，不预测 simulator 调参比例，也不声明 speedup / accuracy。

这些后续任务属于 Gate 7 或更晚阶段。

## 2. Formal Input 硬约束

Gate 6 formal path 的唯一输入是 Gate 5 formal 输出：

```text
kernel_embedding_table.json
```

该输入必须满足：

```text
artifact_type = gcl_resnet50_kernel_embedding_table
artifact_version = gate5_kernel_embedding_table_v1
workload_id = resnet50
model = torchvision.models.resnet50
execution_mode = real_trace
trace_source = nvbit
scheduler_metadata_source = real_nvbit_smid
input_scope = full_resnet50_inference_trace
embedding_source = canonical_non_augmented_graph
embedding_dim = 256
projection_head_used_for_selector = false
source_graph_tensor_bundle_hash 可复现
rgcn_checkpoint_manifest_hash 可复现
kernel_embedding_table_hash 可复现
```

Gate 6 必须拒绝：

```text
synthetic embedding table
ResNet-like fixture embedding table
hand-written opcode embedding table
mini-transformer embedding table
simulator replay embedding table
projection head output embedding
augmented-view embedding
missing real ResNet-50 provenance
missing embedding hash
non-reproducible embedding table hash
```

fixture / debug embedding 可以用于 unit test，但必须标记：

```text
artifact_status = debug_not_formal
formal_input_eligible = false
```

debug run 不得输出可被后续 formal path 消费的 selector artifact。

## 3. 输入 Row Schema

每个 embedding row 至少包含：

```text
record_id
kernel_invocation_id
kernel_name
kernel_launch_order
embedding
embedding_dim
embedding_hash
source_graph_hash
source_tensor_hash
source_readout_manifest_hash
representation_mode
paper_reproduction_mode
weight_input
provenance
```

其中：

- `embedding` 是 Gate 5 canonical non-augmented graph 经过 RGCN encoder readout 得到的 256 维 `z_k`。
- `embedding` 不能是 projection head 输出 `z'_k`。
- `kernel_name` 可以用于后续 family evidence / report，但不能参与 clustering。
- `weight_input` 用于 representative coverage 和后续 weighted evaluation，不能改变 K-Means 距离。

Gate 6 用于 clustering 的唯一 numeric input 是：

```text
embedding[0:256]
```

## 4. Forbidden Fields

Gate 6 在 normalization、K selection、K-Means assignment 和 centroid 计算中禁止读取：

```text
kernel_name
family_label
operator_name
layer_name
grid_dim
block_dim
runtime_ms
weight_input
trace_order
source_path
opcode_histogram
node_count
edge_count
graph_size_class
```

这些字段只能用于：

```text
post-clustering evidence
representative reporting
coverage weighting
debug explanation
```

如果 selector 在 clustering path 中读取 forbidden field，Gate 6 必须失败，并输出 `forbidden_field_used_for_clustering`。

## 5. Embedding Validation

Gate 6 必须先验证 embedding table：

```text
row_count >= 2
embedding_dim = 256
all embeddings finite
no missing embedding_hash
embedding_hash matches embedding payload
record_id unique
kernel_invocation_id unique or explicitly versioned
source_graph_hash present
source_tensor_hash present
source_readout_manifest_hash present
```

如果 `row_count < 2`，Gate 6 不得运行 clustering，必须输出：

```text
selector_status = insufficient_records
minimum_required_records = 2
```

## 6. Normalization

Gate 6 必须在 clustering 前对 256 维 embedding 做 z-score normalization：

```text
normalized_embedding[d] =
  (embedding[d] - mean[d]) / std[d]
```

如果某一维 `std[d] = 0`：

```text
normalized_embedding[d] = 0
zero_variance_dimensions += d
```

normalization report 必须记录：

```text
normalization_mode = z_score
mean_vector_hash
std_vector_hash
zero_variance_dimensions
input_embedding_table_hash
normalized_embedding_table_hash
```

Raw embedding 和 normalized embedding 的 hash 都必须可复现。

## 7. K Selection

Gate 6 默认使用 `silhouette_k`：

```text
candidate_k_min = 2
candidate_k_max = min(10, row_count)
```

对每个候选 K：

```text
run deterministic K-Means
compute silhouette_score on normalized embeddings
record score
```

选择规则：

```text
choose K with highest silhouette_score
if scores tie within epsilon = 1e-6:
  choose smaller K
```

如果 `row_count = 2`：

```text
selected_k = 2
silhouette_score = not_applicable_two_records
```

Gate 6 允许 `deterministic_fixed_k` 作为 ablation mode，但默认必须是 `silhouette_k`。fixed-K run 必须显式记录：

```text
k_selection_mode = deterministic_fixed_k
ablation_mode = true
```

## 8. Deterministic K-Means

K-Means 必须满足：

```text
distance_metric = squared_l2
initialization = deterministic
max_iterations = 300
tolerance = 1e-6
empty_cluster_policy = fail_and_report
input = normalized_embedding
```

Deterministic initialization 推荐使用：

```text
first centroid = lowest stable record_id
next centroid = farthest point from existing centroids
tie_break = stable record_id
```

K-Means metadata 必须记录：

```text
selected_k
candidate_k_values
initial_centroid_record_ids
iterations
converged
inertia
centroid_hash
assignment_hash
random_seed_policy
```

即使初始化不使用随机数，也必须记录 `random_seed_policy = deterministic_no_rng`。

## 9. Representative Anchor Selection

每个 cluster 必须选择一个真实 kernel invocation 作为 representative anchor。

选择规则：

```text
for each cluster:
  compute squared_l2 distance from each member to cluster centroid
  choose nearest member
  tie_break = stable record_id
```

Representative 必须是真实 input row，不能生成 synthetic representative。

anchor table 至少记录：

```text
cluster_id
representative_record_id
representative_kernel_invocation_id
representative_kernel_name
member_record_ids
member_kernel_invocation_ids
member_count
coverage_weight_sum
distance_to_centroid
centroid_hash
selection_reason
```

`coverage_weight_sum` 来自 `weight_input`，用于解释 representative 覆盖量；它不得参与 K-Means assignment。

## 10. Cluster-to-Family Evidence

Gate 6 可以在 clustering 完成后读取 family evidence metadata，用于评估 cluster 是否能稳定对应 kernel family。

允许的 post-clustering family evidence 来源：

```text
kernel_name pattern
operator label from ResNet runtime metadata
layer id
manual family annotation file
```

这些 family evidence 不能参与 clustering，只能在 assignment 完成后计算：

```text
cluster_family_distribution
cluster_majority_family
cluster_purity
weighted_cluster_purity
family_to_cluster_distribution
ambiguous_clusters
unlabeled_records
```

family evidence report 必须明确：

```text
family_labels_used_for_clustering = false
evidence_claim_status = post_clustering_evaluation_only
```

如果某个 cluster 的 majority family 占比低于阈值：

```text
family_purity_threshold = 0.8
cluster_evidence_status = mixed_family_cluster
```

Gate 6 可以报告 mixed cluster，但不能因此修改 K-Means assignment。

## 11. 输出 Artifact

Gate 6 至少输出：

```text
embedding_validation_report.json
embedding_normalization_report.json
k_selection_report.json
kmeans_cluster_assignment_table.json
representative_anchor_table.json
cluster_family_evidence_report.json
gate6_selector_manifest.json
```

`gate6_selector_manifest.json` 至少记录：

```text
artifact_type = gcl_resnet50_gate6_selector_manifest
artifact_version = gate6_selector_manifest_v1
source_kernel_embedding_table_hash
normalization_report_hash
k_selection_report_hash
cluster_assignment_table_hash
representative_anchor_table_hash
cluster_family_evidence_report_hash
selector_status
formal_input_eligible
forbidden_field_audit_status
```

## 12. Failure Modes

Gate 6 必须显式处理：

```text
missing_real_resnet50_provenance
debug_or_fixture_embedding_rejected
projection_head_embedding_rejected
augmented_view_embedding_rejected
insufficient_records
non_finite_embedding
embedding_dim_mismatch
embedding_hash_mismatch
duplicate_record_id
missing_source_hash
forbidden_field_used_for_clustering
empty_cluster
non_deterministic_assignment
family_evidence_missing
```

其中 `family_evidence_missing` 不阻塞 clustering / anchor selection，但必须让 `cluster_family_evidence_report.json` 标记：

```text
family_evidence_status = unavailable
family_claim_status = no_family_claim
```

## 13. 验收标准

Gate 6 完成时必须证明：

1. 只接受真实完整 ResNet-50 Gate 5 formal `kernel_embedding_table.json`。
2. 拒绝 fixture / synthetic / debug embedding 进入 formal selector path。
3. 拒绝 projection head output 和 augmented-view embedding。
4. clustering 只使用 256 维 canonical kernel embedding。
5. forbidden fields 不参与 normalization、K selection、K-Means 或 centroid 计算。
6. 默认 `silhouette_k` 可运行，并记录每个候选 K 的 score。
7. deterministic K-Means assignment 可复现。
8. 每个 cluster 都选择真实 representative anchor。
9. anchor table 可以回答 representative 覆盖哪些 members。
10. family label 只用于 post-clustering evidence，不影响 clustering。
11. mixed-family cluster 会被报告，不会被静默当作成功 family mapping。
12. Gate 7 可以只读取 Gate 6 selector artifacts，不需要读取 Gate 5 checkpoint 或原始 trace。

## 14. 非目标

Gate 6 不做：

- RGCN retraining；
- graph augmentation；
- embedding generation；
- supervised classification head；
- cluster head training；
- simulator 参数调优比例预测；
- speedup claim；
- sampled-vs-full simulator accuracy evaluation；
- 修改 representative SM selection；
- graph compression。

## 15. 结论

Gate 6 是真实 ResNet-50 GCL pipeline 中从 learned kernel embedding 走向 selector / family evidence 的第一层。它保持 GCL-Sampler 的无监督 clustering 思路，用 silhouette-K 和 deterministic K-Means 选择 representative kernels，同时把 family label 限制在 post-clustering evidence 层。

这样可以先回答一个关键问题：

```text
真实 ResNet-50 kernel embeddings 是否能在不偷看 family label 的情况下形成可解释、可复现的 clusters？
```

只有这个问题成立，后续 Gate 7 才适合讨论 supervised cluster head、kernel family classifier 或调参比例预测。
