# A 线 GCL-M1/M2 Phase A Semantic End-to-End GCL Design Spec

日期：2026-06-01

## 1. 定位

Phase A 是 GCL-M1/M2 的第一步。它的目标不是解决真实 kernel trace 的规模问题，而是先证明最小语义通路可以闭合：

```text
small controlled trace
  -> canonical graph
  -> tensorization
  -> minimal RGCN contrastive training
  -> kernel embedding table
  -> M0 selector
  -> cluster / representative anchor / evaluation artifacts
```

这里的“闭合”表示所有关键组件都能通过真实 artifact 串联，而不是每个组件只在单元测试里独立工作。

## 2. 输入范围

Phase A 只允许使用小规模、可审计的 trace input：

```text
single_warp_fixture
selected_warps_fixture
small synthetic trace
small real trace subset
```

推荐默认输入是：

```text
selected_warps_fixture
```

它比 `single_warp_fixture` 更适合验证 `warp_partitions` 和 node-to-warp-to-kernel pooling，同时仍然足够小。

Phase A graph 规模建议控制在：

```text
tens to hundreds of nodes
```

Phase A 不处理：

- full-kernel dynamic trace；
- full-GPU trace；
- representative SM selection；
- oversized graph sampling；
- instruction stream dedup；
- simulator accuracy。

## 3. M1 输出：Canonical Graph

M1 必须从 controlled trace 生成 canonical graph artifact。

必要字段：

```text
graph_id
kernel_invocation_id
collection_scope
nodes
edges
warp_partitions
graph_summary
graph_hash
```

Phase A 的节点类型至少包含：

```text
instruction
register_version
input_variable
unknown_variable
```

Phase A 的边类型至少包含：

```text
control_flow
data_source
data_destination
```

如果 operand ordering 不可靠，M1 必须使用通用的 `data_source`，并记录：

```text
operand_position_known = false
```

## 4. Tensorization

M2 负责把 canonical graph artifact 转成训练张量：

```text
semantic graph records
  -> node feature tensors
  -> relation-indexed edge tensors
  -> warp partition tensors
  -> graph batch metadata
```

Tensorization 必须记录：

```text
tensorizer_version
input_graph_hash
node_feature_schema
edge_relation_schema
feature_width
padding_policy
missing_value_policy
tensor_hash
```

Phase A 默认：

```text
feature_width = 64
```

M2 不得修改 canonical graph artifact。任何 tensorization result 都是派生产物，必须引用 `input_graph_hash`。

## 5. Minimal RGCN Contrastive Training

Phase A 使用最小 RGCN encoder，目的是验证训练路径和 embedding export，不是追求质量。

默认 encoder：

```text
3 RGCN layers
input dimension = 64
hidden dimension = 128
kernel embedding dimension = 256
basis decomposition enabled
layer normalization after convolution
ReLU activation
dropout except final RGCN layer
```

Training projection head：

```text
MLP hidden dimension = 128
projection output dimension = 64
```

Loss：

```text
symmetric InfoNCE
cosine similarity on L2-normalized projection outputs
temperature = 0.05
```

Training metadata 必须记录：

```text
training_config_hash
graph_bundle_hash
tensorizer_version
augmentation_manifest_hash
model_config
optimizer
learning_rate
scheduler
batch_size
epoch_count
random_seeds
train_validation_split
checkpoint_hash
```

## 6. Phase A Augmentation

Phase A 可以使用最小 graph augmentation 生成 contrastive views：

```text
canonical graph
  -> augmented view A
  -> augmented view B
```

默认 augmentation：

```text
node_dropping_rate = 0.15
edge_dropping_rate = 0.15
feature_noise_std = 0.01
views_per_graph = 2
```

Augmentation 不得覆盖 canonical graph。若 node dropping 导致某个 warp partition 为空，M2 必须 reject 或 regenerate，并记录 retry count。

## 7. Kernel Embedding Export

Selector 使用 RGCN encoder 输出的 kernel embedding。

如果 training 使用 projection head：

```text
contrastive loss 使用 projection output
selector 使用 projection head 之前的 kernel embedding
```

Embedding export 必须使用 canonical、non-augmented graph：

```text
canonical graph
  -> tensorization
  -> trained encoder
  -> kernel embedding
```

M2 最终导出 M0-compatible embedding table。每条 row 至少包含：

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

Phase A 推荐：

```text
representation_mode = gcl_m2_rgcn_kernel_embedding
embedding_dim = 256
source_graph_hash = graph_hash from M1
encoder_manifest_hash = hash(model config + checkpoint + tensorizer + augmentation config)
weight_input = 1.0
```

## 8. 接入 M0 Selector

Phase A 必须复用 M0 selector：

```text
gcl_embedding_table_l1.json
  -> z-score normalization
  -> silhouette_k / deterministic_fixed_k
  -> deterministic K-Means
  -> representative anchors
  -> structural evaluation artifacts
```

默认 K selection 仍然是：

```text
silhouette_k
```

`deterministic_fixed_k` 只作为 ablation 或 debug 模式。

## 9. Controlled Encoder Path

`controlled encoder path` 只能作为 debug / ablation 路径，用来定位：

- tensorization 是否错误；
- RGCN training 是否错误；
- embedding export 是否错误；
- M0 selector 是否错误。

它不能替代 Phase A 的主验收路径。Phase A 的主验收必须经过：

```text
trace -> graph -> tensorization -> RGCN -> embedding -> M0 selector
```

## 10. 成功标准

Phase A 完成标准：

1. 能读取 small controlled trace；
2. 能生成 canonical graph artifact；
3. canonical graph artifact 包含 `warp_partitions`；
4. 能完成 tensorization，并记录 `input_graph_hash`；
5. 能通过 minimal RGCN contrastive training 生成 kernel embedding；
6. embedding table 满足 M0 输入契约；
7. 能调用 M0 selector 输出 cluster / representative anchor / structural evaluation artifacts；
8. 不声称 learned embedding quality；
9. 不声称 simulator accuracy；
10. 不引入 instruction stream compression 作为前置条件。
