# Artifact Contracts

这篇文档汇总 GCL-M0 到 GCL-M3 的关键 artifacts，以及它们之间的引用关系。

## M0 Artifacts

[[gcl-m0-offline-embedding-selector]] 输出：

```text
gcl_embedding_table_l1.json
gcl_kmeans_clusters_l1.json
gcl_representative_anchor_table_l1.json
gcl_compression_evaluation_l1.json
```

`gcl_embedding_table_l1.json` 是 selector input table。每条 row 必须包含：

```text
record_id
kernel_invocation_id
representation_mode
embedding_dim
embedding
source_graph_hash
encoder_manifest_hash
embedding_hash
weight_input
```

`gcl_kmeans_clusters_l1.json` 记录 cluster assignment、K selection metadata、normalization metadata 和 replay hash。

`gcl_representative_anchor_table_l1.json` 记录每个 cluster 的 real representative record、members、coverage 和 representative distance。

`gcl_compression_evaluation_l1.json` 记录 structural compression metrics。

## M1 Artifacts

[[gcl-m1-trace-graph-construction]] 输出：

```text
gcl_trace_manifest_l1.json
gcl_trace_graphs_l1.jsonl
gcl_graph_construction_audit_l1.json
```

`gcl_trace_manifest_l1.json` 是 trace 输入总账本，记录 trace rows、status、gap、trace hash 和 replay hash。

`gcl_trace_graphs_l1.jsonl` 是 graph 本体。每行是一个 kernel invocation graph，包含 nodes、edges、warp partitions、graph summary 和 graph hash。

`gcl_graph_construction_audit_l1.json` 记录 graph builder 的运行质量、drop reasons、node/edge type counts、dataflow coverage 和 determinism checks。

## M2 Artifacts

[[gcl-m2-rgcn-embedding-and-selector]] 输出：

```text
gcl_rgcn_training_report_l1.json
gcl_rgcn_model_manifest_l1.json
gcl_embedding_table_l1.json
gcl_kmeans_clusters_l1.json
gcl_representative_anchor_table_l1.json
gcl_compression_evaluation_l1.json
```

M2 的 `gcl_embedding_table_l1.json` 应该使用 M1 的 `graph_hash` 作为 `source_graph_hash`。

M2 的 selector-side artifacts 应保持 M0 的结构语义，方便和 M0、PKA baseline 比较。

## M3 Artifacts

[[gcl-m3-simulator-evaluation]] 输出：

```text
gcl_simulator_accuracy_l1.json
gcl_microarchitectural_metric_error_l1.json
gcl_cross_architecture_transfer_l1.json
```

这些 artifacts 才能承载 simulator accuracy、metric error 和 cross-architecture transfer 相关 claim。

## Cross-Stage Hash Chain

理想 hash chain：

```text
trace_hash
  -> graph_hash
  -> source_graph_hash
  -> embedding_hash
  -> selector replay hash
  -> evaluation replay hash
```

这个链条让每个代表点都能追溯到其 trace、graph、encoder 和 selector 输入。

