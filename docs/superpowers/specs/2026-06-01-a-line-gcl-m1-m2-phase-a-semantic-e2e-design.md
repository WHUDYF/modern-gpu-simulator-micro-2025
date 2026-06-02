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

## 3. 默认验证数据集

Phase A 默认验证数据集为：

```text
gcl_phase_a_controlled_trace_fixture_v1
```

它是一个 small controlled trace fixture，只用于验证语义通路闭合，不作为性能 benchmark 或 sampling accuracy 证据。

### 3.1 数据集规模

默认规模：

```text
kernel_invocation_count = 12
trace_family_count = 3
kernel_invocations_per_family = 4
warp_count_per_invocation = 2
dynamic_instruction_count_per_warp = 6
total_dynamic_instruction_entries = 144
collection_scope = selected_warps_fixture
```

每个 kernel invocation 生成一个 canonical graph，因此：

```text
expected_graph_count = 12
expected_embedding_table_rows = 12
```

该规模足够覆盖：

- 多个 kernel invocation 的 contrastive batch；
- 每个 graph 的两个 augmented views；
- `warp_partitions`；
- node -> warp -> kernel readout；
- M0 selector 的 embedding table 输入；
- silhouette / K-Means / representative anchor artifact 生成。

同时它仍然足够小，便于人工检查 trace、graph、tensor 和 selector 输出。

### 3.2 Trace Families

Fixture 必须包含 3 个 trace family。`trace_family` 只用于 fixture coverage 和 debug，不作为训练标签输入 RGCN。

```text
trace_family = mem_load_fadd_store
kernel_invocation_ids = gcl_pa_k000 .. gcl_pa_k003
per_warp_opcode_sequence =
  MOV
  IMAD.WIDE
  LDG.E.64.SYS
  FADD
  STG.E.64.SYS
  EXIT
required_pseudo_nodes =
  mem_ref for LDG
  mem_ref for STG
```

```text
trace_family = integer_imad_store
kernel_invocation_ids = gcl_pa_k004 .. gcl_pa_k007
per_warp_opcode_sequence =
  MOV
  IMAD
  IMAD
  IADD3
  STG.E.64.SYS
  EXIT
required_pseudo_nodes =
  mem_ref for STG
```

```text
trace_family = load_branch_store
kernel_invocation_ids = gcl_pa_k008 .. gcl_pa_k011
per_warp_opcode_sequence =
  LDG.E.64.SYS
  ISETP
  BRA
  FADD
  STG.E.64.SYS
  EXIT
required_pseudo_nodes =
  mem_ref for LDG
  mem_ref for STG
```

每个 invocation 内部使用两个 fixture warp：

```text
warp_id = 0
warp_id = 1
```

两个 warp 的 opcode sequence 可以相同，但必须使用不同 `trace_index`、`pc`、register version 和 observed dynamic values，以验证 `warp_partitions` 和 variable statistics 不会被错误合并。

### 3.3 Trace Entry 最小字段

每条 dynamic instruction entry 至少包含：

```text
kernel_invocation_id
trace_family
collection_scope
warp_id
trace_index
pc
opcode
active_mask
destination_operands
source_operands
observed_dynamic_values
source_entry_hash
```

其中：

```text
collection_scope = selected_warps_fixture
```

`observed_dynamic_values` 用于 variable node 的 8 维 dynamic value statistics：

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

每个 variable node 至少应有 4 个 observed values。缺失 observed values 时必须走 `missing_value_policy`，不得生成 random values。

### 3.4 预期图结构覆盖

该 fixture 必须覆盖以下 canonical graph 结构：

```text
instruction node:
  MOV / IMAD.WIDE / LDG / FADD / STG / ISETP / BRA / EXIT

variable node:
  register_version
  input_variable
  unknown_variable

pseudo node:
  mem_ref

edge type:
  control_flow
  data_source
  data_destination
```

控制流覆盖：

```text
consecutive instruction -> consecutive instruction
```

数据流覆盖：

```text
input variable -> instruction
instruction -> register_version
register_version -> consumer instruction
input address variable -> mem_ref
mem_ref -> memory instruction
```

`mem_ref` pseudo node 不进入 control-flow 主链。它只作为 data-flow 中的 instruction-internal semantic node 出现。

### 3.5 验证边界

该 fixture 只验证：

```text
trace -> graph -> tensorization -> minimal RGCN training -> embedding table -> M0 selector
```

它不验证：

- learned embedding quality；
- cluster semantic correctness；
- sampling accuracy；
- simulator speedup；
- real workload generalization。

## 4. M1 输出：Canonical Graph

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

## 5. Tensorization

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

`feature_width = 64` 对齐 GCL-Sampler 论文中的 RGCN input dimension。它不是 64 个固定人工语义维度，而是一个统一 node feature vector 宽度。Phase A 第一版必须优先复现论文描述的 node feature initialization，不引入论文没有明确给出的 instruction sub-block 拆分。

M2 不得修改 canonical graph artifact。任何 tensorization result 都是派生产物，必须引用 `input_graph_hash`。

## 6. Node Feature Schema

Phase A 必须把每个 graph node 编码成 64 维向量：

```text
node_features.shape = [node_count, 64]
```

这些 64 维由论文定义的 node-type-specific initialization 产生。第一版只固定论文明确给出的组成关系：

```text
instruction node:
  dense embedding(opcode token ID)
  + positional encoding(normalized PC)
  -> 64-dimensional vector

variable node:
  32-dimensional token ID embedding
  + 8-dimensional dynamic value statistics
  = 40-dimensional vector
  -> zero-pad to 64

pseudo node:
  16-dimensional token ID embedding
  -> zero-pad to 64
```

其中 dense embedding 内部的每一维不是固定人工语义；它们是训练参数，会随着 contrastive learning 更新。Dynamic value statistics 和 normalized PC derived positional encoding 属于固定数值输入或确定性编码。

### 6.1 Instruction Node Feature

Instruction node 表示一条动态 SASS instruction。

Phase A 默认严格按 GCL-Sampler 论文描述：

```text
dense embedding(opcode token ID)
  + positional encoding(normalized PC)
  -> 64-dimensional instruction node feature
```

论文没有规定 instruction node 内部必须使用 `[0:16)`、`[16:24)` 这类固定 block range。因此 Phase A 第一版不得把 instruction feature 默认拆成：

```text
instruction_class_embedding
operand_shape_embedding
memory_access_embedding
predicate_active_mask_features
numeric_flags_or_reserved
```

Phase A 不讨论这些额外字段。当前首要目标是让 node feature 与 GCL-Sampler 论文版本对应。

Phase A 必须记录：

```text
instruction_feature_mode = paper_opcode_embedding_plus_normalized_pc_position
opcode_token_source = opcode token ID
position_source = normalized PC
instruction_feature_dim = 64
```

Instruction token ID 的 dense embedding 是 learned embedding。Normalized PC positional encoding 是 deterministic input feature。Phase A 默认 combine 方式固定为：

```text
concat_opcode63_normalized_pc1
```

具体配置：

```text
instruction_feature_combine = concat_opcode63_normalized_pc1
opcode_embedding_dim = 63
normalized_pc_dim = 1
position_encoding_method = normalized_pc_scalar
```

### 6.2 Variable Node Feature

Variable node 表示 register version、predicate、memory reference value、input variable 或 unknown variable。

Phase A 默认严格按 GCL-Sampler 论文描述：

```text
[0:32)   variable_token_embedding
[32:40)  dynamic_value_statistics
[40:64)  zero_padding
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

`zero_padding`：

- `[40:64)` 必须 zero-pad；
- Phase A strict reproduction 不在该区域加入任何额外 feature；
- 该区域在 Phase A 必须保持 zero-padding。

### 6.3 Pseudo Node Feature

Pseudo node 表示不是单条 SASS instruction、但对 graph learning 有意义的中间概念，例如 `mem_ref`。

Phase A 默认严格按 GCL-Sampler 论文描述：

```text
[0:16)   pseudo_token_embedding
[16:64)  zero_padding
```

字段含义：

`pseudo_token_embedding`：

- learned embedding；
- 输入来自 mem_ref、address_calc、predicate_context、unknown_pseudo 等 token。

`zero_padding`：

- `[16:64)` 必须 zero-pad；
- Phase A strict reproduction 不在该区域加入任何额外 feature；
- 该区域在 Phase A 必须保持 zero-padding。

### 6.4 Schema Manifest

Tensorization 必须输出 `node_feature_schema`，至少记录：

```text
schema_name
schema_version
feature_width
node_type_layouts
embedding_blocks
numeric_feature_blocks
padding_blocks
instruction_feature_mode
instruction_feature_combine
normalization_policy
missing_value_policy
paper_reproduction_mode
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
paper_defined
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
node_feature_schema = gcl_m2_phase_a_paper_node_feature_v1
paper_reproduction_mode = strict_gcl_sampler_node_features
```

任何偏离本 schema 的实现都不属于 Phase A strict reproduction，并且不得与本 schema 的产物混用。

## 7. Minimal RGCN Contrastive Training

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

## 8. Phase A Augmentation

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

## 9. Kernel Embedding Export

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

## 10. 接入 M0 Selector

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

## 11. Controlled Encoder Path

`controlled encoder path` 只能作为 debug / ablation 路径，用来定位：

- tensorization 是否错误；
- RGCN training 是否错误；
- embedding export 是否错误；
- M0 selector 是否错误。

它不能替代 Phase A 的主验收路径。Phase A 的主验收必须经过：

```text
trace -> graph -> tensorization -> RGCN -> embedding -> M0 selector
```

## 12. 成功标准

Phase A 完成标准：

1. 能读取 `gcl_phase_a_controlled_trace_fixture_v1`；
2. fixture 包含 12 个 kernel invocations、3 个 trace families、每个 invocation 2 个 fixture warps；
3. 能生成 12 个 canonical graph artifacts；
4. 每个 canonical graph artifact 包含 `warp_partitions`；
5. 能完成 tensorization，并记录 `input_graph_hash`；
6. `node_features.shape = [node_count, 64]`；
7. `node_feature_schema` 记录 strict paper reproduction mode；
8. variable node 使用 32 维 token embedding + 8 维 dynamic value statistics + `[40:64)` zero padding；
9. pseudo node 使用 16 维 token embedding + `[16:64)` zero padding；
10. instruction node 默认使用 63 维 opcode token dense embedding + 1 维 normalized PC scalar 生成 64 维 feature；
11. `mem_ref` pseudo node 只通过 data-flow 接入，不进入 control-flow 主链；
12. 能通过 minimal RGCN contrastive training 生成 kernel embedding；
13. embedding table 包含 12 rows，并满足 M0 输入契约；
14. 能调用 M0 selector 输出 cluster / representative anchor / structural evaluation artifacts；
15. 不声称 learned embedding quality；
16. 不声称 simulator accuracy；
17. 不引入 instruction stream compression 作为前置条件。
