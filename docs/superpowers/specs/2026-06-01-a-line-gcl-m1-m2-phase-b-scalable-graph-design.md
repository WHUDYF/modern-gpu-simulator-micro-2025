# A 线 GCL-M1/M2 Phase B Scalable Graph Embedding Design Spec

日期：2026-06-01

## 1. 定位

Phase B 在 Phase A 语义闭环成立后，引入 GCL-Sampler 论文中的真实 trace scope。它的目标不是设计一套可调采样方案，而是把 Phase A 已验证的 graph / tensorization / RGCN / readout / M0 selector contract 迁移到论文默认路径：

```text
one representative SM per kernel invocation
  -> complete traces for all CTAs executed on that SM
  -> per-warp graph construction
  -> node -> warp -> kernel readout
  -> kernel embedding
  -> K-Means / silhouette representative selection
```

Phase B 不默认使用 full-GPU full-kernel dynamic trace，也不把 bounded window、selected-warps fallback 或 attention pooling 作为第一版可选策略。任何偏离论文默认 trace scope 的路径都必须作为后续扩展 spec，而不是 Phase B strict GCL reproduction 的一部分。

Phase B 路径：

```text
Phase A verified artifacts
  -> GCL-Sampler representative-SM trace acquisition
  -> all CTAs on selected SM
  -> per-warp graph construction
  -> kernel graph union with warp_partitions
  -> graph size audit
  -> tensorization
  -> augmentation
  -> hierarchical readout
  -> M0-compatible embedding table
```

## 2. Phase A Output Handoff

Phase B 必须复用 Phase A 已经验证过的 contract，而不是重新定义一套 graph encoder。

Phase A 输出中，Phase B 必须继承：

```text
canonical graph artifact schema
warp_partitions contract
node_feature_schema = gcl_m2_phase_a_paper_node_feature_v1
paper_reproduction_mode = strict_gcl_sampler_node_features
tensorization contract
3-layer RGCN encoder config
projection-head-before-selector rule
node -> warp -> kernel readout rule
M0-compatible embedding table schema
M0 selector input contract
```

Phase B 可以替换的是 trace acquisition scope：

```text
Phase A:
selected_warps_fixture

Phase B:
single_representative_sm_all_ctas
```

Phase B 不得替换：

```text
node feature schema
edge relation schema
RGCN readout semantics
selector embedding source
M0 selector contract
```

Phase B 的输出继续进入 Phase A 已打通的后续规划链路：

```text
canonical non-augmented graph
  -> trained RGCN encoder
  -> 256-dimensional kernel embedding
  -> M0-compatible embedding table
  -> K-Means with silhouette-selected K
  -> representative anchor table
  -> sampled simulation planning / downstream evaluation
```

因此，Phase B 的新增价值只在于把输入从 small controlled fixture 升级为论文对齐的 representative-SM trace，而不是改变后面的 embedding、clustering 或 representative planning 方法。

## 3. GCL-Sampler Trace Scope Strategy

GCL-Sampler 不使用 full-GPU full-kernel dynamic trace。Phase B strict reproduction 只允许下面这个 scope：

```text
collection_scope = single_representative_sm_all_ctas
```

每个 kernel invocation 的 trace input 必须显式声明：

```text
collection_scope
selected_sm
selected_sm_policy
selected_sm_reason
included_cta_ids
instruction_count
warp_count
trace_hash
```

Phase B 不允许以下 scope 作为 strict GCL reproduction：

```text
single_warp_fixture
selected_warps_fixture
bounded_instruction_window
full_gpu_full_kernel_dynamic_trace
```

`selected_warps_fixture` 只属于 Phase A fixture。`bounded_instruction_window` 和 `selected_warps` 只能在后续 scalability extension spec 中讨论，不能混入 Phase B 的完成标准。

如果 representative-SM trace 规模超过当前 M2 训练能力，Phase B 必须输出 explicit blocked status：

```text
status = graph_scale_blocked_for_strict_gcl_phase_b
```

不得把 oversized trace 静默截断成 bounded window，也不得只选部分 warps 后继续声称完成 Phase B。

## 4. Representative SM Policy Audit

Phase B 不应把 `selected_sm` 视为随机默认值。M1 必须记录：

```text
selected_sm_policy
selected_sm
selected_sm_reason
candidate_sm_count
included_cta_ids
cta_count_by_sm
instruction_count_by_sm
```

Phase B strict reproduction 允许的 policy：

```text
explicit_sm_id
max_cta_count_sm
```

默认推荐：

```text
explicit_sm_id for controlled replay
max_cta_count_sm for trace-driven batch runs
```

`explicit_sm_id` 用于固定复现实验；`max_cta_count_sm` 用于 trace-driven batch runs，并接近论文中 representative SM 的实际意图。

`first_observed_sm`、随机 SM、max-instruction-only SM 都不属于 Phase B strict GCL reproduction。如果临时 debug 使用这些策略，artifact 必须标记为 `debug_not_phase_b_complete`，不得进入 Phase B representative selection。

## 5. Scope Audit

M1 必须在 trace manifest 或 graph audit 中记录：

```text
scope_policy
scope_reason
selected_sm
included_cta_ids
instruction_count_before_scope
instruction_count_after_scope
warp_count_before_scope
warp_count_after_scope
trace_scope_hash
```

其中：

```text
instruction_count_before_scope = full captured candidate trace count, if available
instruction_count_after_scope = selected SM all-CTA trace count
warp_count_before_scope = full candidate warp count, if available
warp_count_after_scope = selected SM all-CTA warp count
```

如果 acquisition layer 无法提供 before-scope full-GPU candidate counts，必须显式记录：

```text
before_scope_counts_available = false
missing_before_scope_reason
```

不能用空值或零值伪装成 full-GPU count。

## 6. Per-Warp Graph Construction

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

## 7. Kernel Graph Union and Warp Partitions

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

## 8. Graph Size Audit and Eligibility

M1 必须输出 graph size audit。它是 M2 判断 strict GCL Phase B 能否继续训练的前置条件。

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
```

第一版 size class 只用于审计，不用于改变 Phase B 的 trace scope：

```text
small: node_count <= 2,000
medium: 2,000 < node_count <= 10,000
large: 10,000 < node_count <= 50,000
oversized: node_count > 50,000
```

Training eligibility：

```text
small -> eligible_strict_gcl_training
medium -> eligible_strict_gcl_training
large -> graph_scale_blocked_for_strict_gcl_phase_b
oversized -> ineligible_oversized
```

如果 graph 被标记为 `graph_scale_blocked_for_strict_gcl_phase_b` 或 `ineligible_oversized`，M2 不得改用 bounded window、selected warps 或 other sampled graph path。Phase B 必须停止在 blocked artifact，并把 scalability extension 留给后续 spec。

## 9. Tensorization Boundary

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

Tensorization 必须沿用 Phase A strict paper schema：

```text
node_feature_schema = gcl_m2_phase_a_paper_node_feature_v1
feature_width = 64
paper_reproduction_mode = strict_gcl_sampler_node_features
```

M2 不得改变 canonical graph artifact。任何 tensorization result 必须作为派生产物保存，并引用 `input_graph_hash`。

## 10. Hierarchical Readout / Pooling

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

Readout 输出必须和 Phase A export 规则一致：

```text
selector 使用 projection head 之前的 256-dimensional kernel embedding
contrastive loss 使用 projection output
embedding export 使用 canonical non-augmented graph
```

## 11. Graph Augmentation Safety

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

## 12. 成功标准

Phase B 完成标准：

1. Phase B 明确继承 Phase A 的 canonical graph、tensorization、RGCN、readout、embedding export 和 M0 selector contract；
2. M1 trace manifest 记录 `collection_scope = single_representative_sm_all_ctas`；
3. M1 trace manifest 记录 `selected_sm`、`selected_sm_policy`、`selected_sm_reason`、`included_cta_ids` 和 instruction/warp counts；
4. M1 不允许随机 SM、first-observed SM、selected-warps fallback 或 bounded instruction window 进入 Phase B complete path；
5. M1 能按 warp 构建 graph，并输出 `warp_partitions`；
6. M1 graph audit 记录 graph size class 和 strict Phase B training eligibility；
7. M2 tensorization 引用 canonical `graph_hash`，并沿用 Phase A strict paper node feature schema；
8. M2 支持 node/edge relation tensors 和 warp partition tensors；
9. M2 生成两个 augmented views，并记录 augmentation manifest；
10. M2 使用 node-to-warp mean pooling 和 warp-to-kernel average pooling；
11. M2 从 canonical non-augmented graph 导出 projection-head-before-selector 的 256 维 kernel embedding；
12. M2 embedding table 满足 M0 输入契约；
13. M0 selector 使用 K-Means 和 silhouette-selected K 生成 representative anchor table；
14. graph 超出 strict GCL Phase B 训练能力时，输出 blocked artifact，而不是静默截断或改用 selected-warps / bounded-window path；
15. 所有 artifacts 可 replay、可 audit。
