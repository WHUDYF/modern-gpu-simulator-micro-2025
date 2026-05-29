# GCL-M2 RGCN Embedding And Selector

GCL-M2 是把 [[gcl-m1-trace-graph-construction]] 的 canonical graph artifacts 转换为 learned kernel embeddings，并重新接入 [[gcl-m0-offline-embedding-selector]] 已验证过的 selector contract。

M2 的核心路径是：

```text
M1 graph bundle
  -> tensorization
  -> graph augmentation
  -> RGCN contrastive learning
  -> canonical graph embedding generation
  -> selector-compatible embedding table
  -> M0-style clustering / anchor export
```

## 输入

M2 输入：

```text
gcl_trace_graphs_l1.jsonl
training configuration
augmentation configuration
```

M2 必须消费 canonical graph，而不是直接消费 augmented graph。Augmented views 只用于 contrastive training。

## Tensorization

M2 把 M1 的 semantic graph records 转换为 model tensors：

```text
node feature tensors
relation-indexed edge tensors
warp partitions
graph batch metadata
```

M1 保存的是 semantic graph artifact；M2 才负责 node feature embedding lookup、padding、relation-index packing 和 graph batch 组织。

## Contrastive Learning

默认学习方式是 RGCN contrastive learning：

```text
canonical graph
  -> two augmented views
  -> RGCN encoder
  -> projection head
  -> symmetric InfoNCE
```

Selector 使用 projection head 之前的 kernel embedding，而不是 InfoNCE projection output。

## 输出

M2 输出：

```text
gcl_rgcn_training_report_l1.json
gcl_rgcn_model_manifest_l1.json
gcl_embedding_table_l1.json
gcl_kmeans_clusters_l1.json
gcl_representative_anchor_table_l1.json
gcl_compression_evaluation_l1.json
```

`gcl_embedding_table_l1.json` 必须满足 M0 的 embedding table contract。因此 M2 后半段应该复用 M0 selector 的语义，而不是发明另一套 clustering / anchor artifact。

## 成功标准

M2 完成时，learned embeddings 能驱动与 M0 相同结构的 compression outputs。

M2 可以说明 embedding 已经来自 graph encoder，但不能单独声称 simulator accuracy。 simulator 相关 claim 属于 [[gcl-m3-simulator-evaluation]]。

