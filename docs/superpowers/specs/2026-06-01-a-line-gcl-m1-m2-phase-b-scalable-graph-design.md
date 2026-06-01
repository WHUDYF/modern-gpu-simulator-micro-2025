# A 线 GCL-M1/M2 Phase B Scalable Graph Embedding Design Spec

日期：2026-06-01

## 1. 定位

Phase B 在 Phase A 语义闭环成立后，引入真实 trace 的规模约束。它的目标是让 GCL 可以处理更接近真实 kernel invocation 的输入，同时不默认 full-kernel dynamic graph 一定能直接送入 RGCN。

Phase B 路径：

```text
scoped trace acquisition
  -> representative SM policy audit
  -> selected warps / bounded instruction windows
  -> per-warp graph construction
  -> kernel graph union with warp_partitions
  -> graph size audit
  -> tensorization
  -> augmentation
  -> hierarchical readout
  -> M0-compatible embedding table
```

## 2. Trace Scope Strategy

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

允许的 scope：

```text
single_warp_fixture
selected_warps_fixture
single_sm_all_ctas
bounded_instruction_window
```

Phase B 的推荐目标是：

```text
single_sm_all_ctas
```

如果 trace 规模超过 M2 训练上限，则必须转入 bounded policy：

```text
single_sm_all_ctas
  -> bounded_instruction_window
  -> selected_warps
```

禁止静默截断 trace。

## 3. Representative SM Policy Audit

Phase B 不应把 `selected_sm` 视为随机默认值。

如果使用 `single_sm_all_ctas`，M1 必须记录：

```text
selected_sm_policy
selected_sm
selected_sm_reason
candidate_sm_count
included_cta_ids
cta_count_by_sm
instruction_count_by_sm
```

第一版允许的 policy：

```text
fixture_selected_sm
explicit_sm_id
first_observed_sm
max_cta_count_sm
max_instruction_count_sm
```

默认推荐：

```text
explicit_sm_id for controlled replay
max_cta_count_sm for trace-driven batch runs
```

`first_observed_sm` 只能作为 debug fallback，并必须在 manifest 中显式标记。不得把它包装成 representative policy。

## 4. Scope Audit

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

如果使用 selected warps，必须记录：

```text
selected_warp_policy
selected_warp_ids
candidate_warp_count
selected_warp_count
```

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

Instruction node 必要字段：

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

Variable node 第一版至少支持：

```text
register_version
input_variable
unknown_variable
```

Pseudo node 第一版可以只支持：

```text
mem_ref
```

Edge types 第一版必须支持：

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

## 6. Kernel Graph Union and Warp Partitions

M1 不应把所有 trace entries 直接混成一个无边界大图。

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

第一版 size class：

```text
small: node_count <= 2,000
medium: 2,000 < node_count <= 10,000
large: 10,000 < node_count <= 50,000
oversized: node_count > 50,000
```

Training eligibility：

```text
small -> eligible_full_graph
medium -> eligible_warp_batched
large -> eligible_sampled
oversized -> ineligible_oversized
```

如果 graph 被标记为 `eligible_sampled` 或 `ineligible_oversized`，M2 不得静默按 full graph 训练。

## 8. Tensorization Boundary

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

M2 不得改变 canonical graph artifact。任何 tensorization result 必须作为派生产物保存，并引用 `input_graph_hash`。

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

对每个 `warp_id`，M2 必须记录：

```text
warp_id
node_count_used
pooling_method = "mean"
warp_embedding_dim
```

对每个 kernel invocation，M2 必须记录：

```text
warp_count_used
pooling_method = "average"
kernel_embedding_dim
```

如果某个 warp partition 为空，必须报错或记录 explicit gap，不得生成随机 embedding。

第一版不使用 attention pooling。原因是 mean/average pooling 更可 replay，也更容易和论文默认架构对齐。

## 10. Graph Augmentation Safety

Graph augmentation 只属于 M2 training。

M1 的 canonical graph artifact 不得被 augmentation 覆盖。

训练时，M2 从同一个 canonical graph 派生两个 augmented views：

```text
canonical graph
  -> augmented view A
  -> augmented view B
```

第一版允许：

```text
node dropping
edge dropping
feature noise injection
```

Augmentation 不得破坏以下字段：

```text
graph_id
kernel_invocation_id
source graph_hash
warp_partitions metadata
canonical graph artifact
```

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

## 11. 成功标准

Phase B 完成标准：

1. M1 trace manifest 记录 `collection_scope`、`selected_sm`、`selected_warp_ids` 和 instruction/warp counts；
2. M1 记录 `selected_sm_policy`，且不把随机或 first-observed 行为伪装成 representative policy；
3. M1 能按 warp 构建 graph，并输出 `warp_partitions`；
4. M1 graph audit 记录 graph size class 和 training eligibility；
5. M2 tensorization 引用 canonical `graph_hash`；
6. M2 支持 node/edge relation tensors 和 warp partition tensors；
7. M2 生成两个 augmented views，并记录 augmentation manifest；
8. M2 使用 node-to-warp pooling 和 warp-to-kernel pooling；
9. M2 从 canonical non-augmented graph 导出 kernel embedding；
10. M2 embedding table 满足 M0 输入契约；
11. oversized graph 不会被静默送入 full-graph training；
12. 所有 artifacts 可 replay、可 audit。
