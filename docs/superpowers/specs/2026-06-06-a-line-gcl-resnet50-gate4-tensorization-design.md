# A 线 GCL ResNet-50 Gate 4 Tensorization Design Spec

日期：2026-06-06

## 1. Gate 4 定位

Gate 4 的目标是消费 Gate 3 输出的 `canonical_graph_bundle.json`，把 GCL canonical graph 转换为 RGCN / GNN 可以直接消费的 tensor bundle。

Gate 4 只负责：

```text
canonical_graph_bundle.json
  -> validate canonical graph
  -> stable node index assignment
  -> node_features [node_count, 64]
  -> edge_index [2, edge_count]
  -> edge_type [edge_count]
  -> warp_partition_tensors
  -> graph_batch_metadata
```

Gate 4 不做 graph augmentation，不训练 RGCN，不做 contrastive loss，不做 readout，不导出 embedding，不做 clustering，也不做 kernel family classification。

## 2. 输入

Gate 4 的正式输入是：

```text
canonical_graph_bundle.json
```

Gate 4 可以读取以下 Gate 3 artifact 做审计：

```text
graph_size_audit.json
graph_construction_report.json
```

但 tensorization 的正式结构来源只能是 `canonical_graph_bundle.json`。Gate 4 不得读取 Gate 2 manifest、Gate 1 adapter bundle 或原始 ResNet trace。

输入 graph bundle 必须满足：

```text
artifact_type = gcl_resnet50_canonical_graph_bundle
artifact_version = gate3_canonical_graph_bundle_v1
graphs 非空
canonical_graph_bundle_hash 可复现
```

每个 graph 必须满足 Phase B canonical graph schema：

```text
artifact_type = phase_b_canonical_graph
collection_scope = single_representative_sm_all_ctas
nodes
edges
edge_relation_schema
warp_partitions
graph_summary
graph_hash
```

## 3. 输出

Gate 4 至少输出：

```text
graph_tensor_bundle.json
tensorization_report.json
```

`graph_tensor_bundle.json` 是 Gate 5 augmentation / RGCN training 的正式输入。

## 4. Tensor Bundle Schema

`graph_tensor_bundle.json` 至少包含：

```json
{
  "artifact_type": "gcl_resnet50_graph_tensor_bundle",
  "artifact_version": "gate4_graph_tensor_bundle_v1",
  "source_canonical_graph_bundle_hash": "...",
  "tensors": [],
  "graph_tensor_bundle_hash": "..."
}
```

每个 tensor artifact 必须包含：

```text
artifact_type = phase_b_graph_tensor
graph_id
kernel_invocation_id
input_graph_hash
tensorizer_version
phase_b_tensorizer_version
node_feature_schema
edge_relation_schema
feature_width
padding_policy
missing_value_policy
node_ids
node_types
node_features
edge_index
edge_type
warp_partitions
warp_partition_tensors
graph_batch_metadata
tensor_hash
```

`graph_tensor_bundle_hash` 和每个 `tensor_hash` 必须由 canonical JSON 计算并可复现。

## 5. Stable Node Index

Gate 4 必须为每个 graph 生成稳定 node index：

```text
node_index[node_id] = position in graph.nodes
```

`node_ids` 必须按 graph artifact 中 `nodes` 的顺序记录。Gate 4 不得为了 tensorization 重排 nodes，除非后续 spec 明确引入新的 deterministic ordering policy。

`edge_index` 必须使用该 node index：

```text
edge_index[0, e] = source node index
edge_index[1, e] = target node index
```

## 6. Edge Tensor

Gate 4 必须把 graph edges 转为：

```text
edge_index.shape = [2, edge_count]
edge_type.shape = [edge_count]
```

`edge_type` 来自 graph 的 `edge_relation_schema`：

```json
{
  "control_flow": 0,
  "data_source": 1,
  "data_destination": 2
}
```

Gate 4 不得新增 edge relation，也不得把 data flow 合并成无方向单一 edge。`data_source` 与 `data_destination` 仍然属于 data-flow 高层语义，但在 RGCN tensor 中保留方向性 edge type。

## 7. Node Feature Schema

Gate 4 必须复用 Phase A strict GCL-Sampler node feature schema：

```text
schema_name = gcl_m2_phase_a_paper_node_feature_v1
feature_width = 64
paper_reproduction_mode = strict_gcl_sampler_node_features
padding_policy = strict_zero_padding
missing_value_policy = missing numeric values become 0.0
```

Gate 4 不得引入新的 feature width，不得加入 Phase A strict reproduction 之外的扩展 feature。

## 8. Instruction Node Features

Instruction node feature layout：

```text
[0:63)  opcode_token_embedding
[63:64) normalized_pc
```

字段含义：

- `opcode_token_embedding`：由 opcode token 生成的 63 维 learned-embedding input block。
- `normalized_pc`：graph 内 min-max normalized PC，范围为 `[0, 1]`；如果 graph 内所有 instruction PC 相同，则为 `0.0`。

Instruction node 的 control bits、predicate、active mask 等控制信息在 Gate 3 作为 node metadata 保留。Gate 4 strict path 不把它们额外扩展进 64 维 feature。若后续要加入这些控制字段，必须作为单独 ablation spec，不得混入 Gate 4。

## 9. Variable Node Features

Variable node feature layout：

```text
[0:32)   variable_token_embedding
[32:40)  dynamic_value_statistics
[40:64)  zero_padding
```

`dynamic_value_statistics` 共 8 维：

```text
mean
standard_deviation
median
minimum
maximum
percentile_25
percentile_75
skewness
```

这些统计来自 variable node 的 `observed_dynamic_values`。缺失时使用全 0，并受 `missing_value_policy` 约束。

`[40:64)` 必须保持 zero-padding。Gate 4 不得在该区域加入 variable kind、producer-consumer context 或其他扩展。

## 10. Pseudo Node Features

Pseudo node feature layout：

```text
[0:16)   pseudo_token_embedding
[16:64)  zero_padding
```

Gate 4 strict path 第一版只支持：

```text
pseudo_kind = mem_ref
```

`[16:64)` 必须保持 zero-padding。Gate 4 不得加入 memory address class、cache hint、load/store flag 或其他扩展 feature。

## 11. Warp Partition Tensors

Gate 4 必须保留 Gate 3 的 warp partition 结构，并转为 tensor index 形式。

每个 tensor 必须包含：

```text
warp_partitions[partition_id] = [node_index, ...]
```

以及：

```json
{
  "partition_id": "1:0",
  "cta_id": "12,0,0",
  "warp_id": 0,
  "node_indices": [],
  "edge_indices": [],
  "instruction_count": 0
}
```

验证规则：

- 每个 `node_indices` 非空；
- 每个 node index 必须在 `[0, node_count)`；
- 每个 edge index 必须在 `[0, edge_count)`；
- partition tensor 中的 `node_indices` 必须对应 canonical graph 中同一 partition 的 `node_ids`；
- partition tensor 中的 `edge_indices` 必须对应 canonical graph 中同一 partition 的 `edge_ids`。

`warp_partition_tensors` 是 Gate 5 readout 执行 node -> warp -> kernel pooling 的必要输入。

## 12. Graph Batch Metadata

每个 tensor 必须记录：

```json
{
  "graph_id": "...",
  "kernel_invocation_id": "...",
  "node_count": 0,
  "edge_count": 0,
  "warp_count": 0,
  "source_graph_hash": "..."
}
```

这些 metadata 用于后续 batching、training report、readout manifest 和 replay validation。

## 13. Tensorization Report

Gate 4 必须输出 `tensorization_report.json`：

```json
{
  "artifact_type": "gcl_resnet50_tensorization_report",
  "artifact_version": "gate4_tensorization_report_v1",
  "source_canonical_graph_bundle_hash": "...",
  "graph_count": 0,
  "tensor_count": 0,
  "failed_graphs": [],
  "warnings": [],
  "tensorization_report_hash": "..."
}
```

如果某个 graph tensorization 失败，必须记录：

```text
graph_id
kernel_invocation_id
input_graph_hash
failure_stage
failure_reason
```

失败 graph 不进入 formal `graph_tensor_bundle.json`。

## 14. Gate 4 通过标准

Gate 4 通过时必须满足：

1. 输入 canonical graph bundle hash 可复现。
2. 每个 formal graph 都通过 Phase B graph validator。
3. 每个 tensor 的 `node_features.shape = [node_count, 64]`。
4. 每个 tensor 的 `edge_index.shape = [2, edge_count]`。
5. 每个 tensor 的 `edge_type.shape = [edge_count]`。
6. `edge_index`、`edge_type` 与 canonical graph edge 数量一致。
7. `node_feature_schema.schema_name = gcl_m2_phase_a_paper_node_feature_v1`。
8. `feature_width = 64`。
9. variable node `[40:64)` 必须 zero-padding。
10. pseudo node `[16:64)` 必须 zero-padding。
11. warp partition tensor 中所有 node / edge index 有效。
12. tensorization 不修改 canonical graph artifact。
13. `tensor_hash` 和 `graph_tensor_bundle_hash` 可复现。
14. Gate 5 可以只读取 `graph_tensor_bundle.json`，不需要读取 Gate 3 graph bundle 或原始 trace。

## 15. Failure Handling

Gate 4 必须拒绝：

```text
空 canonical graph bundle
graph_hash 不可复现
node_id 重复
edge 引用不存在 node
edge relation 不在 edge_relation_schema
node_features 不是二维矩阵
node_features 第二维不是 64
edge_index 第一维不是 2
edge_index 与 edge_type 长度不一致
node_features 中存在 NaN 或 inf
variable zero padding 非零
pseudo zero padding 非零
warp partition tensor 引用不存在 node / edge index
tensor_hash 不可复现
```

如果所有 graph 都失败，不得生成可供 Gate 5 formal path 消费的 `graph_tensor_bundle.json`。

## 16. 非目标

Gate 4 不做：

- graph augmentation；
- node dropping / edge dropping / feature noise；
- RGCN training；
- contrastive loss；
- readout；
- projection head；
- embedding export；
- GCL selector clustering；
- kernel family classification；
- graph compression；
- resource-blocked decision。

## 17. 结论

Gate 4 是 canonical graph 到 GNN tensor 的转换层。它把 Gate 3 的 graph artifact 转换为严格 GCL-Sampler reproduction 所需的 64 维 node feature、typed edge tensor 和 warp partition tensor，为 Gate 5 的 augmentation、RGCN contrastive training 和 kernel embedding export 提供正式输入。
