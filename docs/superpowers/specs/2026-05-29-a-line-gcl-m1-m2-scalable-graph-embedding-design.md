# A 线 GCL-M1/M2 Scalable Graph Embedding Design Spec

日期：2026-05-29

## 1. 定位

这份 spec 定义 GCL-M1 到 GCL-M2 之间的可训练 graph embedding 方案。

它专门回答一个问题：

```text
当一个 kernel invocation 的动态 trace 可能包含上万条指令时，
GCL 如何把 trace 转换成可训练的 graph，
并最终导出 M0 selector 可以消费的 kernel embedding？
```

这份 spec 横跨 M1/M2，但不替代已有阶段定义：

```text
GCL-M1: trace records -> canonical graph artifacts
GCL-M2: canonical graph artifacts -> RGCN embedding -> M0-style selector input
```

它补充的是规模控制和训练路径：

```text
trace scope
  -> per-warp graph construction
  -> kernel graph with warp partitions
  -> graph size audit
  -> tensorization
  -> graph augmentation
  -> RGCN contrastive learning
  -> node-to-warp pooling
  -> warp-to-kernel pooling
  -> M0-compatible embedding table
```

---

## 2. 设计问题

朴素做法是把一个 kernel invocation 的所有动态 instruction 都展开成一个巨型 graph：

```text
full kernel dynamic trace
  -> one huge instruction/data-flow graph
  -> RGCN
```

这个做法不可作为默认方案。

原因是：

- 一个 kernel 内部动态指令数可能达到上万甚至更多；
- instruction nodes、variable nodes 和 data-flow edges 会进一步放大 graph size；
- RGCN 训练显存和 batch 组织会变得不可控；
- 如果 M1 不记录 graph size 和 scope，M2 无法判断某个 graph 是否适合训练；
- 如果直接对 full graph 做 pooling，warp-level SIMT 结构会被过早抹平。

因此，这份 spec 采用三条核心策略：

```text
1. 通过 trace scope 限制输入规模；
2. 按 warp 构建小图，并保留 warp_partitions；
3. 在 M2 中通过 augmentation + hierarchical readout 训练 kernel embedding。
```

---

## 3. 总体路径

完整路径如下：

```text
workload kernel invocation
  -> scoped trace acquisition
  -> normalized trace entries
  -> per-kernel partitioning
  -> per-warp trace ordering
  -> per-warp graph construction
  -> kernel graph union with warp_partitions
  -> graph size audit
  -> tensorization
  -> augmented graph views
  -> RGCN encoder
  -> node embeddings
  -> warp embeddings
  -> kernel embedding
  -> M0-compatible embedding table
```

M1 负责到：

```text
kernel graph union with warp_partitions
graph size audit
canonical graph artifact
```

M2 负责：

```text
tensorization
augmentation
RGCN training
hierarchical pooling
embedding export
```

M0 只消费最终 embedding table，不关心 embedding 如何生成。

---

## 4. Trace Scope Strategy

GCL 不默认使用 full-GPU full-kernel dynamic trace。

每个 kernel invocation 的 trace input 必须显式声明：

```text
collection_scope
selected_sm
selected_warp_ids
max_instruction_count_per_warp
trace_window_policy
instruction_count
warp_count
trace_hash
```

### 4.1 允许的 scope

第一版允许以下 scope：

```text
single_warp_fixture
selected_warps_fixture
fixture_selected_warps
single_sm_all_ctas
bounded_instruction_window
```

字段含义：

`single_warp_fixture`：

- 只包含一个 fixture warp 的 trace，用于验证 graph schema。

`selected_warps_fixture`：

- 包含多个 fixture warps，用于验证 kernel graph union 和 warp partitions。

`fixture_selected_warps`：

- 与 `selected_warps_fixture` 等价，保留作为更直观命名；实现中应规范化成单一 canonical value，避免 artifact 分裂。

`single_sm_all_ctas`：

- 对齐 GCL-Sampler 的长期目标：选择一个 representative SM，并 trace 该 SM 上执行的 CTAs。

`bounded_instruction_window`：

- 对每个 warp 或 kernel invocation 设置最大 instruction window，避免 graph size 失控。

### 4.2 默认策略

M1 第一版默认使用：

```text
selected_warps_fixture
```

真实 trace 接入后，推荐目标是：

```text
single_sm_all_ctas
```

如果 trace 规模超过 M2 训练上限，则必须转入 bounded policy：

```text
single_sm_all_ctas
  -> bounded_instruction_window
  -> selected_warps
```

### 4.3 Scope Audit

M1 必须在 trace manifest 或 graph audit 中记录：

```text
scope_policy
scope_reason
selected_sm
selected_warp_ids
included_cta_ids
instruction_count_before_scope
instruction_count_after_scope
warp_count_before_scope
warp_count_after_scope
trace_scope_hash
```

如果使用 bounded window，必须记录：

```text
window_start_policy
window_length
window_selection_reason
truncated_instruction_count
```

禁止静默截断 trace。

---

## 5. Per-Warp Graph Construction

M1 必须先按 warp 构建小图。

构图路径：

```text
kernel trace entries
  -> group by warp_id
  -> sort by trace_index inside each warp
  -> construct one directed graph per warp
```

每个 warp graph 包含：

```text
instruction nodes
variable nodes
pseudo nodes
control-flow edges
data-flow edges
```

### 5.1 Instruction Nodes

每条动态 SASS instruction 生成一个 instruction node。

必要字段：

```text
node_id
node_type = "instruction"
warp_id
trace_index
sequence_index
pc
opcode
active_mask
source_entry_hash
```

### 5.2 Variable Nodes

Variable node 表示 register、predicate、memory address/value 等动态值。

第一版至少支持：

```text
register_version
input_variable
unknown_variable
```

每次 destination register write 创建新的 `register_version` node。

source register read 应连接到同一 warp 内最近可见的 producer。如果找不到 producer，则连接到 `input_variable`。

### 5.3 Pseudo Nodes

Pseudo node 表示不是单条 SASS instruction、但对 graph learning 有意义的中间概念。

第一版可以只支持：

```text
mem_ref
```

后续可扩展：

```text
address_calc
predicate
```

### 5.4 Edge Types

第一版必须支持：

```text
control_flow
data_source
data_destination
```

可选支持：

```text
data_left_source
data_right_source
memory_address_source
memory_value_source
predicate_source
```

如果 operand ordering 不可靠，必须使用 `data_source`，并记录：

```text
operand_position_known = false
```

---

## 6. Kernel Graph Union and Warp Partitions

M1 不应把所有 trace entries 直接混成一个无边界大图。

它必须保留层次结构：

```text
kernel graph
  contains warp graph 0
  contains warp graph 1
  contains warp graph 2
  ...
```

Canonical graph artifact 中必须包含：

```text
graph_id
kernel_invocation_id
nodes
edges
warp_partitions
graph_summary
graph_hash
```

`warp_partitions` 至少记录：

```text
warp_id
node_ids
edge_ids
instruction_count
node_count
edge_count
first_trace_index
last_trace_index
```

`warp_partitions` 的作用是让 M2 可以执行：

```text
node embeddings -> warp embeddings -> kernel embedding
```

而不是只能做：

```text
all node embeddings -> kernel embedding
```

---

## 7. Graph Size Audit and Eligibility

M1 必须输出 graph size audit。它是 M2 判断能否训练的前置条件。

每个 graph 至少记录：

```text
instruction_count
warp_count
node_count
edge_count
instruction_node_count
variable_node_count
pseudo_node_count
control_flow_edge_count
data_flow_edge_count
max_warp_instruction_count
max_warp_node_count
max_warp_edge_count
graph_size_class
training_eligibility
recommended_training_policy
```

### 7.1 Size Class

第一版定义：

```text
small
medium
large
oversized
```

建议默认阈值：

```text
small: node_count <= 2,000
medium: 2,000 < node_count <= 10,000
large: 10,000 < node_count <= 50,000
oversized: node_count > 50,000
```

这些阈值是工程默认值，不是论文 claim。后续可以根据实际 GPU memory 调整，但 artifact 必须记录实际阈值。

### 7.2 Training Eligibility

`training_eligibility` 允许：

```text
eligible_full_graph
eligible_warp_batched
eligible_sampled
ineligible_oversized
```

推荐策略：

```text
small -> eligible_full_graph
medium -> eligible_warp_batched
large -> eligible_sampled
oversized -> ineligible_oversized
```

如果 graph 被标记为 `eligible_sampled` 或 `ineligible_oversized`，M2 不得静默按 full graph 训练。

---

## 8. Tensorization Boundary

M1 输出 semantic graph artifact，不直接输出最终 training tensors。

M2 负责 tensorization：

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

第一版默认：

```text
feature_width = 64
```

M2 不得改变 canonical graph artifact。任何 tensorization result 必须作为派生产物保存，并引用 `input_graph_hash`。

---

## 9. Hierarchical Readout / Pooling

M2 必须支持 hierarchical readout。

默认 readout 路径：

```text
node embeddings
  -> mean pooling within each warp partition
  -> warp embeddings
  -> average pooling across warps
  -> kernel embedding
```

也就是：

```text
node -> warp -> kernel
```

### 9.1 Node To Warp Pooling

对每个 `warp_id`：

```text
warp_embedding = mean(node_embeddings in warp_partition)
```

M2 必须记录：

```text
warp_id
node_count_used
pooling_method = "mean"
warp_embedding_dim
```

如果某个 warp partition 为空，必须报错或记录 explicit gap，不得生成随机 embedding。

### 9.2 Warp To Kernel Pooling

对同一个 kernel invocation 的所有 warp embeddings：

```text
kernel_embedding = average(warp_embeddings)
```

M2 必须记录：

```text
warp_count_used
pooling_method = "average"
kernel_embedding_dim
```

第一版不使用 attention pooling。原因是 mean/average pooling 更可 replay，也更容易和论文默认架构对齐。

### 9.3 Embedding Position

Selector 使用 RGCN encoder 输出的 kernel embedding。

如果 training 使用 projection head，则：

```text
contrastive loss 使用 projection output
selector 使用 projection head 之前的 kernel embedding
```

---

## 10. Graph Augmentation for Contrastive Training

Graph augmentation 只属于 M2 training。

M1 的 canonical graph artifact 不得被 augmentation 覆盖。

训练时，M2 从同一个 canonical graph 派生两个 augmented views：

```text
canonical graph
  -> augmented view A
  -> augmented view B
```

Contrastive learning 的目标是：

```text
embedding(view A of graph X) 接近 embedding(view B of graph X)
embedding(graph X) 远离 embedding(graph Y)
```

### 10.1 允许的 Augmentations

第一版允许：

```text
node dropping
edge dropping
feature noise injection
```

默认参数：

```text
node_dropping_rate = 0.15
edge_dropping_rate = 0.15
feature_noise_std = 0.01
views_per_graph = 2
```

### 10.2 Augmentation Safety

Augmentation 不得破坏以下字段：

```text
graph_id
kernel_invocation_id
source graph_hash
warp_partitions metadata
canonical graph artifact
```

如果 node dropping 导致某个 warp partition 为空，M2 必须：

```text
reject that augmented view
or regenerate with recorded retry count
```

不得 silently train on invalid view。

### 10.3 Augmentation Manifest

M2 必须输出 augmentation manifest：

```text
augmentation_manifest_hash
input_graph_hash
random_seed
view_id
augmentation_types
rates
dropped_node_count
dropped_edge_count
feature_noise_std
retry_count
view_hash
```

---

## 11. RGCN Contrastive Training

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

---

## 12. Embedding Export to M0 Contract

M2 最终必须导出 M0-compatible embedding table。

每条 row 至少包含：

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

M2 第一版推荐：

```text
representation_mode = gcl_m2_rgcn_kernel_embedding
embedding_dim = 256
source_graph_hash = graph_hash from M1
encoder_manifest_hash = hash(model config + checkpoint + tensorizer + augmentation config)
```

Embedding export 必须使用 canonical、non-augmented graph：

```text
canonical graph
  -> tensorization
  -> trained encoder
  -> kernel embedding
```

不得使用 augmented view 导出 selector embedding。

输出 embedding table 后，M2 可以复用 M0 selector：

```text
gcl_embedding_table_l1.json
  -> z-score normalization
  -> silhouette_k / deterministic_fixed_k
  -> deterministic K-Means
  -> representative anchors
```

---

## 13. 非目标

这份 spec 不做：

- 重新定义 M0 selector；
- 替代 M1 trace graph construction spec；
- 替代未来 M2 implementation plan；
- 证明 RGCN embedding 质量；
- 证明 GCL 比 PKA 更准；
- 证明 sampled simulation accuracy；
- 定义真实 NVBit deployment 权限和集群 orchestration；
- 要求第一版支持 full-GPU full-kernel trace。

这份 spec 也不允许：

- 静默截断 trace；
- 静默丢弃 oversized graph；
- 用 augmented graph 覆盖 canonical graph；
- 用 projection head output 作为 selector embedding；
- 跳过 M0 embedding table contract。

---

## 14. 成功标准

这份 M1/M2 scalable graph embedding path 的完成标准：

1. M1 trace manifest 记录 `collection_scope`、`selected_sm`、`selected_warp_ids` 和 instruction/warp counts；
2. M1 能按 warp 构建 graph，并输出 `warp_partitions`；
3. M1 graph audit 记录 graph size class 和 training eligibility；
4. M2 tensorization 引用 canonical `graph_hash`；
5. M2 支持 node/edge relation tensors 和 warp partition tensors；
6. M2 生成两个 augmented views，并记录 augmentation manifest；
7. M2 使用 RGCN contrastive training；
8. M2 使用 node-to-warp pooling 和 warp-to-kernel pooling；
9. M2 从 canonical non-augmented graph 导出 kernel embedding；
10. M2 embedding table 满足 M0 输入契约；
11. oversized graph 不会被静默送入 full-graph training；
12. 所有 artifacts 可 replay、可 audit。

---

## 15. 与现有 Spec 的关系

这份 spec 与现有文档的关系：

```text
GCL-M0 selector interface spec
  定义 embedding table -> selector -> anchors/evaluation

GCL-M1 trace graph construction spec
  定义 trace records -> canonical graph artifacts

本 spec
  定义 M1/M2 之间如何让 graph representation 可训练、可扩展

未来 GCL-M2 implementation spec
  应把本 spec 转成具体任务、模块、测试和 artifacts
```

因此，本 spec 的核心边界是：

```text
scoped trace
  -> per-warp graph
  -> hierarchical RGCN embedding
  -> M0-compatible embedding table
```

它不改变 M0 的 selector semantics，也不改变 M1 canonical graph 的 artifact-first 原则。

