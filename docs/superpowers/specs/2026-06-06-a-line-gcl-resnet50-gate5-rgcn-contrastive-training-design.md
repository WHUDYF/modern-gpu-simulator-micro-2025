# A 线 GCL ResNet-50 Gate 5 RGCN Contrastive Training Design Spec

日期：2026-06-06

## 1. Gate 5 定位

Gate 5 的目标是消费 Gate 4 输出的 `graph_tensor_bundle.json`，完成 GCL-Sampler 风格的 RGCN contrastive training，并导出后续 selector / classifier 可以使用的 canonical kernel embedding table。

Gate 5 只负责：

```text
graph_tensor_bundle.json
  -> validate tensor bundle
  -> training-only graph augmentation
  -> RGCN encoder training
  -> projection head InfoNCE loss
  -> canonical non-augmented inference
  -> kernel_embedding_table.json
```

Gate 5 不做 K-Means，不做 silhouette 选 K，不做 representative kernel selection，不做 kernel family classification，不做调参比例预测。

这些后续任务属于 Gate 6。

## 2. 输入

Gate 5 的正式输入是 Gate 4 输出的：

```text
graph_tensor_bundle.json
```

Gate 5 可以读取以下 Gate 4 artifact 做审计：

```text
tensorization_report.json
```

但 training / embedding export 的正式结构来源只能是 `graph_tensor_bundle.json`。Gate 5 不得读取 Gate 3 canonical graph bundle、Gate 2 representative-SM manifest、Gate 1 adapter bundle 或原始 ResNet trace。

输入 tensor bundle 必须满足：

```text
artifact_type = gcl_resnet50_graph_tensor_bundle
artifact_version = gate4_graph_tensor_bundle_v1
workload_id = resnet50
execution_mode = real_trace
trace_source = nvbit
input_scope = full_resnet50_inference_trace
graph_tensor_bundle_hash 可复现
tensors 非空
```

Gate 5 formal training / embedding export 必须拒绝 synthetic / ResNet-like / hand-written fixture tensor bundle。fixture tensor 只能用于 unit test、smoke test 或 debug run，输出不得作为 Gate 6 clustering / family classification 的正式输入。

每个 tensor 必须满足：

```text
artifact_type = phase_b_graph_tensor
feature_width = 64
node_features.shape = [node_count, 64]
edge_index.shape = [2, edge_count]
edge_type.shape = [edge_count]
warp_partition_tensors 非空
tensor_hash 可复现
```

## 3. 输出

Gate 5 至少输出：

```text
augmentation_manifest.json
rgcn_training_run_manifest.json
rgcn_checkpoint_manifest.json
kernel_embedding_table.json
embedding_export_report.json
```

其中 `kernel_embedding_table.json` 是 Gate 6 clustering / classification 的正式输入。

## 4. Representation Mode

Gate 5 必须显式记录输入 graph tensor 的 representation mode，不能假设 pseudo node 一定存在。

允许的模式：

```text
gcl_resnet50_mem_ref_only
gcl_resnet50_no_pseudo_node
```

含义：

```text
gcl_resnet50_mem_ref_only
  使用 instruction node、variable node、mem_ref pseudo node；
  这是当前最接近 GCL-Sampler pseudo-node 描述的模式。

gcl_resnet50_no_pseudo_node
  只使用 instruction node 和 variable node；
  保留 control_flow / data_source / data_destination edges；
  这是 functional-first pipeline 模式，不应标记为 strict paper reproduction。
```

Gate 5 可以训练这两种模式，但必须在所有输出 artifact 中记录：

```text
representation_mode
pseudo_node_mode
paper_reproduction_mode
```

如果输入 artifact 缺少这些字段，Gate 5 必须拒绝 formal training。

如果输入 artifact 缺少真实 ResNet-50 NVBit provenance，Gate 5 必须拒绝 formal training，即使 tensor shape、feature width 和 hash 都合法。

## 5. Training Augmentation

Gate 5 的 augmentation 只属于 training view 生成，不得覆盖 canonical tensor artifact。

对同一个 canonical graph tensor：

```text
G
  -> augmented view G_a
  -> augmented view G_b
```

`G_a` 和 `G_b` 来自同一个 kernel invocation，构成 positive pair。Batch 中其他 kernel invocation 的 augmented views 构成 negative pairs。

Gate 5 使用的 augmentation pool：

```text
node dropping:
  randomly remove 15% nodes and incident edges

edge dropping:
  randomly remove 15% edges

feature noise injection:
  add Gaussian noise with sigma = 0.01 to node_features
```

工程约束：

- augmentation 必须使用 deterministic seed policy；
- augmentation 后不得出现空 graph；
- augmentation 后每个仍参与 pooling 的 warp partition 至少保留一个 node；
- node dropping 后必须重新映射 `edge_index`；
- edge dropping 后 `edge_index` 与 `edge_type` 长度必须一致；
- feature noise 不得写回 canonical `node_features`；
- augmentation manifest 必须记录每个 view 使用的 augmentation type、drop rate、noise sigma、seed 和 source tensor hash。

如果某次 augmentation 产生非法 view，Gate 5 必须按同一 seed policy 重新采样或将该 graph 记录为 failed training sample。非法 view 不能进入 formal training batch。

## 6. RGCN Encoder

Gate 5 使用三层 RGCN encoder：

```text
layer 1:
  input_dim = 64
  output_dim = 128

layer 2:
  input_dim = 128
  output_dim = 128

layer 3:
  input_dim = 128
  output_dim = 256
```

每层必须使用 relation-specific message passing。Relation type 来自 Gate 4 的 `edge_type`：

```json
{
  "control_flow": 0,
  "data_source": 1,
  "data_destination": 2
}
```

Gate 5 不得新增 relation type。`data_source` 与 `data_destination` 都属于 data-flow 高层语义，但在 RGCN 中必须保留为两个方向明确的 relation。

RGCN layer 配置：

```text
basis_decomposition = enabled
layer_norm = enabled
activation = ReLU
dropout = enabled for hidden layers
dropout = disabled for final layer
```

最终 RGCN 输出：

```text
node_embeddings.shape = [node_count, 256]
```

## 7. Hierarchical Readout

Gate 5 的 ResNet-50 Phase B readout 必须使用 selected-SM hierarchy：

```text
node -> warp -> CTA -> selected SM -> kernel
```

具体计算：

```text
warp_embedding =
  mean(node_embeddings in warp_partition)

cta_embedding =
  mean(warp_embeddings grouped by cta_id)

selected_sm_embedding =
  mean(cta_embeddings from selected_sm)

kernel_embedding =
  selected_sm_embedding
```

`kernel_embedding = selected_sm_embedding` 的原因是 Gate 2 已经把 representative SM 作为该 kernel invocation 的 trace scope。Gate 5 必须在 export manifest 中记录：

```text
kernel_embedding_source = selected_sm_embedding
collection_scope = single_representative_sm_all_ctas
readout_hierarchy = node_to_warp_to_cta_to_selected_sm_to_kernel
```

如果输入 tensor 只有 `warp_partition_tensors`，Gate 5 必须通过每个 warp partition 中的 `cta_id` deterministic group-by 生成 CTA partition。不得把不同 CTA 的 warp 直接混合后再 pooling。

最终 kernel embedding 维度：

```text
kernel_embedding.shape = [256]
```

## 8. Projection Head

Gate 5 训练时必须使用 projection head：

```text
kernel_embedding z_k, 256
  -> MLP hidden layer, 128
  -> projection_output z'_k, 64
```

InfoNCE loss 使用：

```text
projection_output z'_k
```

Gate 6 和 selector 使用：

```text
kernel_embedding z_k
```

不是 projection output。

因此 `kernel_embedding_table.json` 必须导出 projection head 之前的 256 维 canonical kernel embedding，不得导出 64 维 projection output 作为正式 embedding。

## 9. Contrastive Training

Gate 5 使用 self-supervised contrastive learning。

每个 training step 中，对 batch 内每个 canonical tensor 生成两个 augmented views：

```text
view_a = augment(G)
view_b = augment(G)
```

同一个 `kernel_invocation_id` 的 `(view_a, view_b)` 是 positive pair。

不同 `kernel_invocation_id` 之间的 views 是 negative pairs。

训练目标：

```text
increase similarity between positive pair projection outputs
decrease similarity between negative pair projection outputs
```

Gate 5 使用 InfoNCE loss，并必须记录：

```text
loss_name
temperature
batch_size
epoch_count
optimizer
learning_rate
random_seed
train_graph_count
failed_training_sample_count
```

Formal contrastive training 至少需要：

```text
train_graph_count >= 2
```

如果只有一个 graph，Gate 5 可以运行 debug smoke path，但输出 artifact 必须标记为：

```text
training_status = debug_single_graph_not_formal
```

该结果不得进入 Gate 6 formal clustering / classification。

## 10. Canonical Embedding Export

训练完成后，Gate 5 必须用训练好的 encoder 对 canonical non-augmented tensors 重新做 inference。

Export 规则：

- 使用 canonical `node_features`、`edge_index`、`edge_type`、`warp_partition_tensors`；
- 不使用 node dropping；
- 不使用 edge dropping；
- 不使用 feature noise；
- RGCN encoder 使用 eval mode；
- Dropout disabled；
- projection head 不参与正式 export；
- 每个 kernel invocation 输出一个 256 维 `kernel_embedding`。

`kernel_embedding_table.json` 至少包含：

```json
{
  "artifact_type": "gcl_resnet50_kernel_embedding_table",
  "artifact_version": "gate5_kernel_embedding_table_v1",
  "source_graph_tensor_bundle_hash": "...",
  "encoder_manifest_hash": "...",
  "checkpoint_hash": "...",
  "embedding_dim": 256,
  "readout_hierarchy": "node_to_warp_to_cta_to_selected_sm_to_kernel",
  "embeddings": [],
  "kernel_embedding_table_hash": "..."
}
```

每条 embedding record 至少包含：

```text
kernel_invocation_id
graph_id
source_tensor_hash
source_graph_hash
representation_mode
pseudo_node_mode
collection_scope
selected_sm
embedding_dim
kernel_embedding
embedding_hash
```

`embedding_hash` 必须由 canonical JSON 中的 embedding record 计算并可复现。

## 11. Training Run Manifest

`rgcn_training_run_manifest.json` 必须记录：

```text
artifact_type = gcl_resnet50_rgcn_training_run_manifest
artifact_version = gate5_rgcn_training_run_manifest_v1
source_graph_tensor_bundle_hash
representation_mode
model_architecture
edge_relation_schema
readout_hierarchy
augmentation_config
contrastive_loss_config
optimizer_config
random_seed
train_graph_count
training_status
final_loss
best_checkpoint_hash
training_run_manifest_hash
```

`rgcn_checkpoint_manifest.json` 必须记录：

```text
encoder_architecture
encoder_state_hash
projection_head_state_hash
optimizer_state_hash
encoder_manifest_hash
checkpoint_hash
checkpoint_created_from_training_run_manifest_hash
```

## 12. Gate 5 通过标准

Gate 5 通过时必须满足：

1. 输入 `graph_tensor_bundle_hash` 可复现。
2. 每个 tensor 的 `node_features.shape = [node_count, 64]`。
3. 每个 tensor 的 `edge_index.shape = [2, edge_count]`。
4. 每个 tensor 的 `edge_type.shape = [edge_count]`。
5. `representation_mode` 和 `pseudo_node_mode` 明确记录。
6. Augmentation 只写入 training view，不覆盖 canonical tensor。
7. Positive pair 来自同一个 `kernel_invocation_id` 的两个 augmented views。
8. Negative pairs 来自 batch 中不同 `kernel_invocation_id`。
9. RGCN encoder 输出 `node_embeddings.shape = [node_count, 256]`。
10. Readout 使用 `node -> warp -> CTA -> selected SM -> kernel`。
11. Projection head 只用于 InfoNCE training。
12. `kernel_embedding_table.json` 导出 256 维 canonical kernel embedding。
13. Export embedding 来自 non-augmented canonical tensor。
14. `embedding_hash`、`encoder_manifest_hash`、`checkpoint_hash` 和 `kernel_embedding_table_hash` 可复现。
15. Gate 6 可以只读取 `kernel_embedding_table.json`，不需要读取 Gate 4 tensor bundle 或原始 trace。

## 13. Failure Handling

Gate 5 必须拒绝：

```text
空 graph_tensor_bundle
tensor_hash 不可复现
node_features 含 NaN 或 inf
edge_index 引用不存在 node
edge_type 不在 edge_relation_schema
warp_partition_tensors 为空
representation_mode 缺失
pseudo_node_mode 缺失
train_graph_count < 2 但被标记为 formal training
augmentation 覆盖 canonical tensor
projection output 被写作正式 kernel embedding
export 时使用 augmented view
kernel embedding 维度不是 256
embedding_hash 不可复现
```

失败 graph 可以从 formal training set 中剔除，但必须进入 `embedding_export_report.json` 的 `failed_graphs`。如果所有 graph 都失败，不得生成可供 Gate 6 formal path 消费的 `kernel_embedding_table.json`。

## 14. 非目标

Gate 5 不做：

- representative-SM selection；
- canonical graph construction；
- tensorization；
- K-Means clustering；
- silhouette coefficient 选 K；
- representative anchor selection；
- kernel family label assignment；
- GNN + fully connected family classifier；
- 调参比例预测；
- graph compression；
- pseudo node 扩展设计。

## 15. 结论

Gate 5 是从 GNN tensor 到 kernel behavioral embedding 的学习层。它把 Gate 4 的 canonical tensor 用 training-only augmentation 生成 positive / negative contrastive views，通过三层 RGCN 和 projection head 训练 encoder，最后只用 canonical non-augmented graph 导出 256 维 kernel embedding。

这一步闭合的是：

```text
real ResNet trace tensor
  -> learned graph representation
  -> canonical kernel embedding table
```

下一步 Gate 6 才把这些 kernel embedding 用于 cluster、family classification 和后续调参策略。
