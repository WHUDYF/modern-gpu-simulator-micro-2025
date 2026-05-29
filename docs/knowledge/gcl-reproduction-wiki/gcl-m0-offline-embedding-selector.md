# GCL-M0 Offline Embedding Selector

GCL-M0 是 GCL reproduction 的 selector interface validation stage。

它只验证一条最小闭环：

```text
fixture/offline embedding table
  -> embedding row validation
  -> stable record ordering
  -> z-score embedding normalization
  -> K selection
  -> deterministic K-Means
  -> nearest-centroid representative selection
  -> structural compression evaluation
  -> replayable GCL artifacts
```

M0 不构建 trace graph，也不训练 RGCN。它的任务是证明 embedding-based selector 可以产出与 PKA baseline 可比较的 representative compression artifacts。

## 输入

M0 的输入是 embedding table。每条 row 至少包含：

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

第一版固定：

```text
representation_mode = gcl_m0_embedding_fixture
```

`source_graph_hash` 在 M0 中可以是 fixture source hash；到了 [[gcl-m1-trace-graph-construction]] 和 [[gcl-m2-rgcn-embedding-and-selector]] 后，它应逐步指向真实 graph artifact hash。

## K Selection

M0 默认使用 `silhouette_k`，让 embedding space 的距离结构决定 cluster count。

同时保留 `deterministic_fixed_k` 作为 ablation mode：

```text
k = ceil(sqrt(n_records)), clamped to [2, n_records]
```

## 输出

M0 输出：

```text
gcl_embedding_table_l1.json
gcl_kmeans_clusters_l1.json
gcl_representative_anchor_table_l1.json
gcl_compression_evaluation_l1.json
```

这些 artifact 的语义在 [[artifact-contracts]] 中汇总。

## 不应声称

M0 不证明 learned representation quality，不证明 simulator accuracy，也不证明 GCL 比 PKA 更准。

M0 的结论只能是：

```text
embedding-based selector interface 已经跑通，并且输出 artifacts 可以和 PKA baseline 对齐比较。
```

相关边界见 [[stage-boundaries]]。

