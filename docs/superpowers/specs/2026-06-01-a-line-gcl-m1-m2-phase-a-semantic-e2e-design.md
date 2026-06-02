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

`feature_width = 64` 对齐 GCL-Sampler 论文中的 RGCN input dimension。它不是 64 个固定人工语义维度，而是一个统一 node feature vector 宽度。不同 node type 使用不同 feature block layout，不足部分 zero-pad 到 64 维。

M2 不得修改 canonical graph artifact。任何 tensorization result 都是派生产物，必须引用 `input_graph_hash`。

## 5. Node Feature Schema

Phase A 必须把每个 graph node 编码成 64 维向量：

```text
node_features.shape = [node_count, 64]
```

这些 64 维由 feature blocks 组成。Block 可以是：

```text
learned embedding block
fixed numeric feature block
zero padding / reserved block
```

`learned embedding block` 中的每一维不是固定语义；它们是训练参数，会随着 contrastive learning 更新。`fixed numeric feature block` 才对应明确的数值统计或 flag。

### 5.1 Instruction Node Feature

Instruction node 表示一条动态 SASS instruction。

Phase A 推荐 layout：

```text
[0:16)   opcode_token_embedding
[16:24)  normalized_pc_positional_encoding
[24:32)  instruction_class_embedding
[32:40)  operand_shape_embedding
[40:48)  memory_access_embedding
[48:56)  predicate_active_mask_features
[56:64)  numeric_flags_or_reserved
```

字段含义：

`opcode_token_embedding`：

- learned embedding；
- 输入来自 normalized opcode token，例如 `LDG`、`STG`、`IMAD`、`FADD`、`BRA`；
- 不直接把 opcode id 当连续数值使用。

`normalized_pc_positional_encoding`：

- fixed numeric feature 或 deterministic encoding；
- 输入来自 normalized PC 或 instruction position；
- 用来保留 instruction 在 kernel code / trace ordering 中的位置线索。

`instruction_class_embedding`：

- learned embedding；
- 表示 load、store、integer arithmetic、floating arithmetic、control、barrier、special function、unknown 等粗粒度类别。

`operand_shape_embedding`：

- learned embedding；
- 表示 source operand count、destination operand count、register / predicate / immediate / memory operand pattern。

`memory_access_embedding`：

- learned embedding 或 fixed flags；
- 表示 no memory、global load、global store、shared load、shared store、local memory、constant memory、unknown memory。

`predicate_active_mask_features`：

- fixed numeric features；
- 表示是否 predicated、active lane count、active mask density 等。

`numeric_flags_or_reserved`：

- 第一版可以用于 is_branch、is_barrier、is_atomic、has_immediate、is_vectorized 等 flags；
- 如果信息不可用，zero-pad，并在 `missing_value_policy` 中记录。

### 5.2 Variable Node Feature

Variable node 表示 register version、predicate、memory reference value、input variable 或 unknown variable。

Phase A 推荐 layout：

```text
[0:32)   variable_token_embedding
[32:40)  dynamic_value_statistics
[40:48)  variable_kind_embedding
[48:56)  producer_consumer_context
[56:64)  zero_padding_or_reserved
```

字段含义：

`variable_token_embedding`：

- learned embedding；
- 输入来自 normalized variable token；
- register 不建议直接使用 raw register id 作为连续数值，而应先归一化成 token。

`dynamic_value_statistics`：

- fixed numeric feature；
- 对齐 GCL-Sampler 论文中的 8 维 dynamic value summary；
- 默认顺序固定为：

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

如果某个 variable 没有可用动态值，必须使用 `missing_value_policy` 明确处理，例如 zero-fill plus missing flag，不能静默写入随机值。

`variable_kind_embedding`：

- learned embedding；
- 表示 register_version、predicate_version、memory_value、memory_address、input_variable、unknown_variable。

`producer_consumer_context`：

- fixed numeric features 或 small learned embedding；
- 可记录 producer instruction class、consumer count bucket、last-use distance bucket 等。

`zero_padding_or_reserved`：

- 第一版默认 zero-pad；
- 后续可以扩展，但必须更新 `node_feature_schema` 和 `tensorizer_version`。

### 5.3 Pseudo Node Feature

Pseudo node 表示不是单条 SASS instruction、但对 graph learning 有意义的中间概念，例如 `mem_ref`。

Phase A 推荐 layout：

```text
[0:16)   pseudo_token_embedding
[16:24)  pseudo_kind_embedding
[24:32)  memory_access_class
[32:40)  fan_in_fan_out_summary
[40:64)  zero_padding_or_reserved
```

字段含义：

`pseudo_token_embedding`：

- learned embedding；
- 输入来自 mem_ref、address_calc、predicate_context、unknown_pseudo 等 token。

`pseudo_kind_embedding`：

- learned embedding；
- 表示 pseudo node 的粗粒度类别。

`memory_access_class`：

- learned embedding 或 fixed flags；
- 对 `mem_ref` 记录 global、shared、local、constant、unknown 等类别。

`fan_in_fan_out_summary`：

- fixed numeric features；
- 可记录 incoming edge count、outgoing edge count、data_source count、data_destination count 等归一化统计。

`zero_padding_or_reserved`：

- 第一版默认 zero-pad；
- 后续扩展必须保持 schema version 可追踪。

### 5.4 Schema Manifest

Tensorization 必须输出 `node_feature_schema`，至少记录：

```text
schema_name
schema_version
feature_width
node_type_layouts
embedding_blocks
numeric_feature_blocks
padding_blocks
normalization_policy
missing_value_policy
```

每个 block 至少记录：

```text
block_name
start_index
end_index
block_kind
source_fields
normalization
default_value
trainable
```

`block_kind` 允许：

```text
learned_embedding
fixed_numeric
zero_padding
reserved
```

Phase A 默认 schema 名称：

```text
node_feature_schema = gcl_m2_phase_a_node_feature_v1
```

任何后续修改都必须递增 schema version，并改变 `tensor_hash` / `encoder_manifest_hash`。

## 6. Minimal RGCN Contrastive Training

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

## 7. Phase A Augmentation

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

## 8. Kernel Embedding Export

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

## 9. 接入 M0 Selector

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

## 10. Controlled Encoder Path

`controlled encoder path` 只能作为 debug / ablation 路径，用来定位：

- tensorization 是否错误；
- RGCN training 是否错误；
- embedding export 是否错误；
- M0 selector 是否错误。

它不能替代 Phase A 的主验收路径。Phase A 的主验收必须经过：

```text
trace -> graph -> tensorization -> RGCN -> embedding -> M0 selector
```

## 11. 成功标准

Phase A 完成标准：

1. 能读取 small controlled trace；
2. 能生成 canonical graph artifact；
3. canonical graph artifact 包含 `warp_partitions`；
4. 能完成 tensorization，并记录 `input_graph_hash`；
5. `node_features.shape = [node_count, 64]`；
6. `node_feature_schema` 记录每个 feature block 的来源、范围和 trainable 状态；
7. 能通过 minimal RGCN contrastive training 生成 kernel embedding；
8. embedding table 满足 M0 输入契约；
9. 能调用 M0 selector 输出 cluster / representative anchor / structural evaluation artifacts；
10. 不声称 learned embedding quality；
11. 不声称 simulator accuracy；
12. 不引入 instruction stream compression 作为前置条件。
