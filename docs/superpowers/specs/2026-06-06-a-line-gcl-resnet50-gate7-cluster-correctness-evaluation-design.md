# A 线 GCL ResNet-50 Gate 7 Cluster Correctness Evaluation Design Spec

日期：2026-06-06

## 1. Gate 7 定位

Gate 7 的目标是验证 Gate 6 生成的 GCL clusters 是否可信。它不重新聚类，不训练 classifier head，也不预测调参比例；它只对已经生成的 clusters、representative anchors 和 family evidence 做定量正确性评估。

Gate 7 只负责：

```text
Gate 6 selector artifacts
  + post-clustering family labels
  + graph / trace structural summaries
  + measured NCU or simulator metrics
  + multi-run stability artifacts
  -> cluster correctness evaluation reports
```

Gate 7 要回答：

```text
同一 cluster 内的 kernels 是否真的相似？
不同 cluster 的 kernels 是否真的不同？
representative kernel 是否能代表 cluster members？
这种代表关系在 measured / simulator metrics 上是否成立？
cluster assignment 是否稳定？
```

Gate 7 的结论不能直接声明 simulator speedup，也不能输出调参策略。它只能给出 cluster correctness evidence。

## 2. 输入

Gate 7 的正式输入分为五类。

### 2.1 Gate 6 Selector Artifacts

必须读取：

```text
gate6_selector_manifest.json
kmeans_cluster_assignment_table.json
representative_anchor_table.json
k_selection_report.json
embedding_normalization_report.json
cluster_family_evidence_report.json
```

这些 artifact 必须来自真实完整 ResNet-50 Gate 6 formal path：

```text
workload_id = resnet50
execution_mode = real_trace
trace_source = nvbit
input_scope = full_resnet50_inference_trace
formal_input_eligible = true
selector_status = passed
```

Gate 7 必须拒绝 debug / fixture / synthetic selector artifacts。

### 2.2 Embedding Table

Gate 7 可以读取 Gate 5 的 `kernel_embedding_table.json`，但只能用于重新计算 evaluation metrics：

```text
embedding distance
intra-cluster distance
inter-cluster distance
representative distance
stability comparison
```

Gate 7 不得修改 Gate 6 assignment，不得重新选择 K，不得重新运行 K-Means 作为 formal output。

### 2.3 Family Labels

Family labels 只能作为 post-clustering evaluation input：

```text
kernel_name pattern
operator label
layer id
manual family annotation
```

family labels 不得参与 Gate 6 clustering，也不得在 Gate 7 中反向修改 cluster assignment。

### 2.4 Structural Summaries

Gate 7 可以读取 graph / trace structural summaries：

```text
opcode distribution
memory instruction ratio
control-flow edge ratio
data-flow edge ratio
node_count
edge_count
warp_count
cta_count
graph_size_class
```

这些字段只用于 evaluation，不得反向改变 clustering。

### 2.5 Measured / Simulator Metrics

Gate 7 可以读取 measured NCU 或 simulator metrics：

```text
runtime
sm_cycles
instruction_count
memory_throughput
l2_hit_rate
dram_bytes
achieved_occupancy
simulator_predicted_time
```

metrics 必须记录来源：

```text
metric_source = ncu | simulator | mixed
metric_capture_hash
metric_unit
metric_timestamp
```

如果同一 report 混用不同 unit 且无法归一化，Gate 7 必须标记 `metric_unit_conflict`。

## 3. 非目标

Gate 7 不做：

- trace acquisition；
- graph construction；
- RGCN training；
- embedding generation；
- K selection；
- K-Means clustering；
- representative anchor reselection；
- supervised cluster head；
- family classifier training；
- tuning vector prediction；
- simulator speedup claim；
- sampled-vs-full final accuracy claim。

## 4. Evaluation Layer 1: Embedding 内部质量

Gate 7 必须计算无标签 embedding / cluster quality metrics：

```text
silhouette_score
davies_bouldin_index
calinski_harabasz_index
mean_intra_cluster_distance
p95_intra_cluster_distance
mean_inter_cluster_distance
inter_intra_distance_ratio
cluster_radius_by_cluster
cluster_separation_by_cluster
```

方向：

```text
silhouette_score: higher is better
davies_bouldin_index: lower is better
calinski_harabasz_index: higher is better
inter_intra_distance_ratio: higher is better
```

第一版不强制阈值，只要求所有指标可计算、可记录、可 replay。

如果 `row_count < 3` 导致某些指标不可计算，report 必须记录：

```text
metric_status = not_applicable_due_to_record_count
```

## 5. Evaluation Layer 2: Family Label 对齐质量

Gate 7 必须在 clustering 完成后计算 label alignment metrics：

```text
cluster_purity
weighted_cluster_purity
adjusted_rand_index
normalized_mutual_information
homogeneity
completeness
v_measure
family_to_cluster_coverage
mixed_family_cluster_count
high_weight_mixed_family_cluster_count
unlabeled_record_count
```

`weighted_cluster_purity` 使用 Gate 6 row 的 `weight_input` 或 measured runtime weight。

family labels 缺失时：

```text
family_evidence_status = unavailable
family_alignment_claim_status = no_family_claim
```

缺失 family label 不阻塞 embedding 内部质量评估和 representative 评估。

## 6. Evaluation Layer 3: Representative 代表性

Gate 7 必须验证 Gate 6 选出的 representative anchor 是否能代表 cluster members。

每个 cluster 计算：

```text
representative_record_id
member_count
coverage_weight_sum
mean_distance_to_representative
p95_distance_to_representative
max_distance_to_representative
representative_rank_to_centroid
outlier_member_ratio
high_weight_outlier_count
```

outlier 判定第一版使用：

```text
member_distance_to_representative >
  cluster_mean_distance_to_representative + 2 * cluster_std_distance_to_representative
```

Gate 7 不重新选择 representative。即使 representative 质量差，也只能报告：

```text
representative_quality_status = weak_representative
```

## 7. Evaluation Layer 4: 性能 / 模拟指标一致性

Gate 7 必须比较 cluster representative 与 cluster members 的 measured / simulator metrics。

对每个 metric：

```text
relative_error(member, representative) =
  abs(metric(member) - metric(representative)) / max(abs(metric(member)), epsilon)
```

其中：

```text
epsilon = 1e-9
```

每个 cluster 计算：

```text
cluster_weighted_mape
cluster_p95_relative_error
cluster_max_relative_error
cluster_metric_correlation
cluster_metric_rank_correlation
high_error_member_count
high_weight_high_error_member_count
```

全局计算：

```text
global_weighted_mape
global_p95_relative_error
global_max_relative_error
bad_cluster_count
high_weight_bad_cluster_count
```

第一版建议记录以下 metrics：

```text
runtime
sm_cycles
instruction_count
memory_throughput
dram_bytes
simulator_predicted_time
```

如果某个 metric 缺失，Gate 7 必须记录 `metric_missing`，但不阻塞其他 metrics 的评估。

## 8. Evaluation Layer 5: 稳定性

Gate 7 必须支持多 run 稳定性评估。

输入可以来自：

```text
different RGCN training seeds
different augmentation seeds
different train / validation splits
same embedding table replay
```

稳定性指标：

```text
assignment_stability_ari
assignment_stability_nmi
k_stability
centroid_drift
representative_stability_rate
family_purity_stability
```

如果只有一个 run：

```text
stability_status = single_run_not_evaluated
```

single-run 不阻塞 Gate 7 第一版完成，但不得声明 cluster assignment stable。

## 9. Threshold Policy

Gate 7 第一版不强制通过 / 失败阈值，因为真实 ResNet-50 数据还没有形成统计基线。

第一版必须输出：

```text
threshold_policy = report_only_v1
```

并记录建议阈值占位字段：

```text
suggested_min_silhouette_score
suggested_min_weighted_cluster_purity
suggested_max_global_weighted_mape
suggested_min_assignment_stability_ari
```

这些字段可以为 `null`，但必须解释：

```text
threshold_claim_status = not_set_until_real_resnet50_baseline
```

后续当真实 ResNet-50 baseline 数据充足后，才能升级为：

```text
threshold_policy = enforced_v2
```

## 10. 输出 Artifact

Gate 7 至少输出：

```text
cluster_embedding_quality_report.json
cluster_family_alignment_report.json
representative_quality_report.json
cluster_metric_error_report.json
cluster_stability_report.json
gate7_cluster_correctness_manifest.json
```

`gate7_cluster_correctness_manifest.json` 至少记录：

```text
artifact_type = gcl_resnet50_gate7_cluster_correctness_manifest
artifact_version = gate7_cluster_correctness_manifest_v1
source_gate6_selector_manifest_hash
source_cluster_assignment_table_hash
source_representative_anchor_table_hash
source_embedding_table_hash
metric_source_manifest_hash
family_label_source_hash
structural_summary_source_hash
embedding_quality_report_hash
family_alignment_report_hash
representative_quality_report_hash
metric_error_report_hash
stability_report_hash
threshold_policy
claim_status
```

`claim_status` 允许：

```text
quantified_no_correctness_claim
partial_evidence_available
correctness_thresholds_not_enforced
blocked_missing_formal_inputs
```

第一版默认：

```text
claim_status = quantified_no_correctness_claim
```

含义是：指标已经量化，但尚未根据真实 ResNet-50 baseline 设置强验收阈值。

## 11. Failure Modes

Gate 7 必须显式处理：

```text
missing_gate6_selector_manifest
debug_or_fixture_selector_rejected
missing_embedding_table
cluster_assignment_hash_mismatch
representative_anchor_hash_mismatch
family_labels_missing
structural_summary_missing
metric_source_missing
metric_unit_conflict
metric_value_non_finite
insufficient_records_for_metric
single_run_stability_not_evaluated
```

其中：

- `family_labels_missing` 不阻塞 embedding quality 和 representative quality。
- `metric_source_missing` 不阻塞 embedding quality / family alignment，但必须让 metric error report 标记 `metric_claim_status = unavailable`。
- `single_run_stability_not_evaluated` 不阻塞第一版 Gate 7，但不得声明稳定性。

## 12. 验收标准

Gate 7 完成时必须证明：

1. 只接受真实 ResNet-50 Gate 6 formal selector artifacts。
2. 不重新选择 K，不重新运行 formal K-Means，不修改 cluster assignment。
3. 能输出 embedding 内部质量指标。
4. 能输出 family label 对齐指标，并保证 label 只用于 post-clustering evaluation。
5. 能输出 representative 代表性指标。
6. 能输出 measured / simulator metric relative error 指标。
7. 能处理 metric missing / family label missing / single-run stability 这些非阻塞缺口。
8. 能输出 multi-run stability 指标，或明确标记 single-run 未评估。
9. 所有报告都带 source artifact hash。
10. 第一版使用 `threshold_policy = report_only_v1`，不伪装成已证明正确。

## 13. 与 Gate 8 的关系

Gate 7 是 Gate 8 的前置质量闸门。

Gate 8 如果要做：

```text
cluster / family -> simulator tuning vector
```

必须先读取 Gate 7 的 correctness evidence。高权重 mixed-family cluster、高 metric error cluster 或 weak representative cluster 不应该直接进入调参策略生成。

## 14. 结论

Gate 6 负责产生 GCL 分组，Gate 7 负责定量证明这些分组是否可信。

Gate 7 不把“聚类已经生成”当成“聚类正确”。它要求每个 cluster 都能被多层指标审计：

```text
embedding geometry
family alignment
representative quality
metric consistency
run stability
```

只有这些 evidence 足够稳定后，后续 Gate 8 才适合讨论基于 cluster / family 的 simulator 调参策略。
