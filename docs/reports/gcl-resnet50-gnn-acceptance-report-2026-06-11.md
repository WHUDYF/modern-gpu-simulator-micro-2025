# GCL ResNet-50 GNN Acceptance Report

日期：2026-06-11

## 1. 结论

当前 ResNet-50 GCL run 的 GNN 验收结论是：

```text
gnn_acceptance_status =
  weak_acceptance_structure_valid_but_correctness_unproven

claim_status =
  quantified_no_correctness_claim
```

含义：

```text
当前 GNN 通过了输入真实性、RGCN 结构真实性、端到端 embedding export 的验收；
embedding space 出现了可量化分离信号；
但训练充分性、多 seed 稳定性、baseline ablation、语义分类正确性、
representative 下游有效性均不足或尚未评估。
```

因此，本报告不能声明 ResNet-50 kernel family 分类已经正确，也不能声明 sampled representative 已经能替代 full trace simulator evaluation。

## 2. 证据来源

本报告读取当前 artifact：

```text
artifacts/gcl_resnet50_full_trace_reproduction/resnet50_full_trace_reproduction_manifest.json
artifacts/gcl_resnet50_full_trace_reproduction/rgcn_training_run_manifest.json
artifacts/gcl_resnet50_full_trace_reproduction/selector_artifacts.json
artifacts/gcl_resnet50_full_trace_reproduction/gate7_cluster_correctness_manifest.json
artifacts/gcl_resnet50_full_trace_reproduction/cluster_stability_report.json
```

本报告不重新训练 GNN，不重新聚类，不重新选择 representative anchor。

## 3. Acceptance Summary

| Item | Status | Evidence | Interpretation |
| ---- | ------ | -------- | -------------- |
| Input provenance | PASS | `formal_full_trace_run = true`, `run_scope = real_resnet50_full_trace` | 输入已切换到真实 ResNet-50 full trace。 |
| Full trace scope | PASS | `input_kernel_invocation_count = 265`, `input_cta_record_count = 124876` | 当前不是 synthetic fixture，也不是 bounded slice。 |
| RGCN architecture | PASS | `layers = 3`, `input_dim = 64`, `hidden_dim = 128`, `kernel_embedding_dim = 256` | 当前是实际 RGCN encoder，不是空壳模型。 |
| Relation-aware message passing contract | PASS | `relation_count = 3`, `control_flow / data_source / data_destination` | edge type 已进入 GNN 结构契约。 |
| Hierarchical readout | PASS | `node_to_warp_to_cta_to_selected_sm_to_kernel` | readout 与 representative-SM trace scope 对齐。 |
| Embedding export coverage | PASS | `export_graph_count = 265` | 265 个 kernel invocation 均进入 embedding export。 |
| Training adequacy | FAIL | `train_graph_count = 4`, `optimizer_step_count = 1` | 只能说明 smoke training 跑通，不能说明 representation 学充分。 |
| Embedding geometry | WEAK_PASS | `silhouette = 0.48186617`, `inter_intra_ratio = 2.01633922` | embedding 有分离信号，但缺少 baseline 对照。 |
| Selector result | WEAK_PASS | `selected_k = 2`, cluster count = `263 / 2` | 更像 outlier discovery，不能直接解释为稳定 family 分类。 |
| Baseline ablation | NOT_AVAILABLE | 当前无 random / histogram / no-edge baseline | 无法证明 graph message passing 相比简单特征有贡献。 |
| Multi-seed stability | NOT_AVAILABLE | `stability_status = single_run_not_evaluated` | 无法判断 selected_k、assignment、representative 是否稳定。 |
| Semantic cluster correctness | UNPROVEN | `ari = 0.0`, `nmi = 0.0`, coarse family label | 现有 family label 粒度不足以证明 cluster 语义正确。 |
| Downstream representative usefulness | NOT_AVAILABLE | `metric_claim_status = unavailable` | 缺少 measured / simulator metric 代表性验证。 |

## 4. 输入验收

输入验收通过。

artifact 显示：

```text
artifact_type = gcl_resnet50_full_trace_reproduction_manifest
run_scope = real_resnet50_full_trace
formal_full_trace_run = true
input_kernel_invocation_count = 265
input_cta_record_count = 124876
final_gate = gate9_report_only
input_root = artifacts/gcl_resnet50_gate0_formal_trace/traces
```

判断：

```text
input_provenance = PASS
```

这说明当前 GCL pipeline 的输入已经不是早期手工构造 trace，而是真实 ResNet-50 full trace acquisition 后的正式 artifact。

## 5. RGCN 结构验收

RGCN 结构验收通过。

当前 training manifest 记录：

```text
layers = 3
input_dim = 64
hidden_dim = 128
kernel_embedding_dim = 256
projection_hidden_dim = 128
projection_output_dim = 64
relation_count = 3
representation_mode = gcl_resnet50_mem_ref_only
pseudo_node_mode = mem_ref_only
readout_hierarchy = node_to_warp_to_cta_to_selected_sm_to_kernel
```

edge relation schema 为：

```text
control_flow = 0
data_source = 1
data_destination = 2
```

判断：

```text
rgcn_structure = PASS
```

这说明当前模型不是只做 feature pooling 的空模型，而是具备 relation-aware message passing 契约的三层 RGCN。

## 6. 训练充分性验收

训练充分性未通过。

当前 training manifest 记录：

```text
train_graph_count = 4
export_graph_count = 265
optimizer = Adam
learning_rate = 0.005
optimizer_step_count = 1
final_loss = 0.0
checkpoint_reuse = true
training_subset_policy = deterministic_prefix_for_full_trace_scalability
```

判断：

```text
training_adequacy = FAIL
```

原因：

- 训练 graph 数量只有 4；
- optimizer 只执行 1 step；
- 没有可解释的 multi-epoch loss curve；
- checkpoint reuse 虽然被记录，但不能替代正式训练充分性；
- `final_loss = 0.0` 在单步小样本设置下不能解释为模型已经收敛。

因此，当前训练只能支撑 smoke-level GNN path validation，不能支撑“GNN 已经学到可信 kernel representation”的声明。

## 7. Embedding Geometry 验收

embedding geometry 给出弱通过。

Gate 7 记录：

```text
silhouette = 0.48186617
davies_bouldin = 0.78974237
calinski_harabasz = 10.30193812
intra_distance_mean = 0.9376443638426335
inter_distance_mean = 1.8906091086825012
inter_intra_ratio = 2.01633922
```

判断：

```text
embedding_geometry_signal = WEAK_PASS
```

这些指标说明 embedding space 不是完全退化的随机云，cluster 之间有一定分离。但是当前没有 random embedding、opcode histogram、no-edge pooling、control-flow-only、data-flow-only 等 baseline，因此还不能证明完整 RGCN 的 graph message passing 是分离信号的来源。

## 8. Selector 与 Cluster 结果验收

selector 结果给出弱通过，但不能证明语义分类正确。

Gate 6 selector artifact 记录：

```text
mode = silhouette_k
selected_k = 2
selected_score = 0.53412531
candidate k scores:
  k=2: 0.53412531
  k=3: 0.32226622
  k=4: 0.40687670
  k=5: 0.42345686
  k=6: 0.43066267
```

cluster assignment 分布：

```text
cluster 0: 263 kernels
cluster 1: 2 kernels
```

representative anchors：

```text
cluster 0 -> d_0_s_0_k_333
cluster 1 -> d_0_s_0_k_269
```

判断：

```text
selector_cluster_result = WEAK_PASS
```

原因：

- silhouette-K 给出了可重复的 k=2 选择；
- cluster 分布极不均衡；
- cluster 1 只有 2 个 kernel，更像 outlier discovery；
- 当前不能把 `263 / 2` 直接解释为 ResNet-50 kernel family 分类。

## 9. Baseline Ablation 验收

baseline ablation 证据缺失。

当前 artifacts 未提供以下对照：

```text
random_embedding_baseline
opcode_histogram_baseline
node_feature_pooling_no_edge_baseline
control_flow_only_rgcn
data_flow_only_rgcn
```

判断：

```text
baseline_ablation = NOT_AVAILABLE
```

影响：

```text
无法证明完整 RGCN 优于简单统计特征；
无法证明 edge relation 或 message passing 对最终 cluster 有实际贡献；
无法排除 cluster 主要来自 graph size、instruction count 或 opcode count 的可能。
```

## 10. Multi-Seed Stability 验收

multi-seed stability 证据缺失。

Gate 7 stability report 记录：

```text
assignment_stability_ari = null
centroid_drift = null
k_stability = null
representative_stability_rate = null
stability_status = single_run_not_evaluated
```

判断：

```text
multi_seed_stability = NOT_AVAILABLE
```

影响：

```text
无法判断 selected_k 是否稳定；
无法判断 cluster assignment 是否稳定；
无法判断 representative anchor 是否稳定；
无法判断当前 k=2 是否只是单次 seed 偶然结果。
```

## 11. Semantic Cluster Correctness 验收

semantic cluster correctness 尚未证明。

Gate 7 family alignment metrics 记录：

```text
cluster_purity = 1.0
weighted_purity = 1.0
ari = 0.0
nmi = 0.0
homogeneity = 1.0
completeness = 0.0
v_measure = 0.0
family_evidence_status = available
family_alignment_claim_status = reported
```

判断：

```text
semantic_cluster_correctness = UNPROVEN
```

解释：

`purity = 1.0` 在这里不能单独证明分类正确，因为当前 family evidence 更像粗粒度 workload label，例如 `resnet50_real_trace`，而不是细粒度 operator family label。`ari = 0.0`、`nmi = 0.0`、`completeness = 0.0` 表明当前 label 对齐还不能支撑语义 family claim。

## 12. Representative Downstream Usefulness 验收

representative 下游有效性证据缺失。

Gate 7 metric error report 记录：

```text
metric_claim_status = unavailable
status = not_provided
complete_row_count = 0
global_weighted_mape = null
global_p95_relative_error = null
```

判断：

```text
downstream_representative_usefulness = NOT_AVAILABLE
```

影响：

```text
当前不能证明 cluster representative 可以近似 cluster members 的 runtime、simulator time、
memory throughput、SM cycles 或其他 measured / simulator metrics。
```

因此，当前不能声明 sampled-vs-full simulator speedup 或 accuracy。

## 13. Blocking Gaps

当前阻塞 GNN 可信性升级的缺口是：

1. 训练不足：`train_graph_count = 4` 且 `optimizer_step_count = 1`。
2. 缺少 baseline ablation：无法证明 RGCN graph message passing 的贡献。
3. 缺少 multi-seed stability：无法证明 selected_k、assignment 和 representative 稳定。
4. 语义 label 粒度不足：无法证明 cluster 对应真实 kernel family。
5. 缺少 measured / simulator metric：无法证明 representative 的下游代表性。

## 14. Recommended Gate10

建议新增 Gate10：GNN Trustworthiness Evaluation。

Gate10 至少实现：

```text
1. full RGCN vs random / histogram / no-edge / edge-ablation baselines
2. multi-seed RGCN training and KMeans clustering
3. cluster semantic profiling using kernel structure and operator metadata
4. representative downstream metric validation
5. claim status upgrade / downgrade manifest
```

Gate10 完成前，当前 run 必须保持：

```text
claim_status = quantified_no_correctness_claim
```

## 15. Final Acceptance Record

最终验收记录：

```text
input_provenance = PASS
rgcn_structure = PASS
end_to_end_embedding_export = PASS
embedding_geometry_signal = WEAK_PASS
selector_cluster_result = WEAK_PASS
training_adequacy = FAIL
baseline_ablation = NOT_AVAILABLE
multi_seed_stability = NOT_AVAILABLE
semantic_cluster_correctness = UNPROVEN
downstream_representative_usefulness = NOT_AVAILABLE

gnn_acceptance_status =
  weak_acceptance_structure_valid_but_correctness_unproven

claim_status =
  quantified_no_correctness_claim
```
