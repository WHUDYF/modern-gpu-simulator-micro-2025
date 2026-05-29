# A 线 GCL-M1 Trace Graph Construction Design Spec

日期：2026-05-29

## 1. 定位

GCL-M1 是复现 GCL-Sampler 的第二个阶段。它接在 GCL-M0 之后，但不复用 M0 的 fixture embedding 作为输入。

M1 只做一件事：

```text
从 trace-like inputs 构建 deterministic heterogeneous trace graph artifacts。
```

因此，GCL-M1 是：

```text
trace-to-graph contract validation stage
```

不是：

```text
RGCN training stage
embedding generation stage
selector stage
simulator accuracy stage
```

M0 已经验证了：

```text
embedding table -> selector -> representative artifacts
```

M1 要验证的是更靠前的一段：

```text
trace records -> canonical graph bundle
```

M1 的关键价值是把 GCL-Sampler 中最容易含糊的 trace graph 表示固定下来。只有 graph artifact 稳定，后续 M2 的 RGCN encoder 才有可靠输入。

---

## 2. 目标

GCL-M1 需要回答五个问题：

1. trace records 能不能被稳定解析成 per-kernel、per-warp sequences？
2. graph builder 能不能生成 instruction、pseudo 和 variable 三类 nodes？
3. graph builder 能不能生成 control-flow 和 data-flow typed edges？
4. graph artifact 是否 deterministic、可 replay、可审计？
5. graph artifact 是否足够支持后续 tensorization / RGCN training，而不提前绑定具体模型实现？

一句话：

```text
先证明 GCL 的 graph representation contract 成立。
```

M1 不以模型效果为目标，也不声称 graph representation 已经学到了 kernel similarity。它只确认 trace 可以被转换为结构化、稳定、可验证的 heterogeneous graph。

---

## 3. M1 的最小闭环

GCL-M1 的最小闭环是：

```text
trace fixture / trace subset
  -> trace manifest validation
  -> trace entry normalization
  -> per-kernel partitioning
  -> per-warp temporal ordering
  -> node construction
  -> edge construction
  -> graph validation
  -> graph construction audit
  -> replayable graph artifacts
```

这里“闭环”的意思是：

- 不能只读取 trace manifest；
- 不能只解析 instruction rows；
- 不能只生成 nodes；
- 不能只生成 edges；
- 必须最终写出 graph bundle 和 graph construction audit。

一个 M1 run 必须能回答：

```text
哪些 trace records 被消费？
哪些 kernel invocations 生成了 graph？
每个 graph 有多少 instruction / pseudo / variable nodes？
每个 graph 有多少 control-flow / data-flow edges？
哪些 trace entries 被丢弃，原因是什么？
graph ids 和 node ids 是否 deterministic？
同一份 trace 输入能否 replay 出相同 graph hash？
```

只有这些问题都能从 artifacts 中回答，M1 才形成最小闭环。

---

## 4. 输入契约

GCL-M1 的输入是 trace manifest 加 trace records。第一版允许两种来源：

```text
fixture_trace
real_trace_subset
```

第一版推荐先实现 `fixture_trace`，再接真实 NVBit-style trace subset。原因是 M1 的第一目标是固定 graph contract，而不是立即解决 trace acquisition 环境问题。

### 4.1 Trace Manifest

每条 trace manifest row 至少包含：

```text
record_id
kernel_invocation_id
workload_id
trace_path
trace_format_version
trace_source_mode
collection_scope
selected_sm
warp_count
instruction_count
status
gap_reason
trace_hash
```

字段语义：

`record_id`：

- M1 内部稳定排序、graph id 派生和 replay 的主 id。

`kernel_invocation_id`：

- 对应 workload 中某一次真实或 fixture kernel launch。

`workload_id`：

- trace 所属 workload。

`trace_path`：

- trace record 文件路径，路径本身不参与 graph topology 决策，但用于 artifact 追踪。

`trace_format_version`：

- trace record schema 的版本。

`trace_source_mode`：

- trace 来源。第一版允许：

```text
fixture_trace
real_trace_subset
```

`collection_scope`：

- trace 覆盖范围，例如：

```text
single_sm_all_ctas
single_warp_fixture
selected_warps_fixture
full_kernel_subset
```

`selected_sm`：

- 若 trace 只来自某个 SM，则记录该 SM id；fixture 可为 `null`，但必须显式记录。

`warp_count`：

- manifest 声明的 warp 数量。

`instruction_count`：

- manifest 声明的动态 instruction entry 数量。

`status`：

- trace row 状态。允许值：

```text
collected
fixture
missing
invalid
unsupported
```

`gap_reason`：

- 当 `status != collected` 且 `status != fixture` 时必须非空。

`trace_hash`：

- trace record payload 的稳定 hash。

### 4.2 Trace Entry

每条 normalized trace entry 至少包含：

```text
kernel_invocation_id
warp_id
trace_index
pc
opcode
active_mask
destination_registers
source_registers
predicate_registers
memory_width
memory_addresses
dynamic_values
entry_hash
```

第一版可以允许字段缺失，但缺失必须进入 gap / audit，不得静默伪造。

必要字段：

```text
kernel_invocation_id
warp_id
trace_index
pc
opcode
active_mask
entry_hash
```

可选字段：

```text
destination_registers
source_registers
predicate_registers
memory_width
memory_addresses
dynamic_values
```

如果缺少 source / destination register 信息，M1 可以退化生成 control-flow graph，但必须在 audit 中把 data-flow coverage 标记为 incomplete。

---

## 5. 输出 artifacts

GCL-M1 第一版输出三类 formal artifacts：

```text
gcl_trace_manifest_l1.json
gcl_trace_graphs_l1.jsonl
gcl_graph_construction_audit_l1.json
```

如果 graph 数量或体积较大，可以把 `gcl_trace_graphs_l1.jsonl` 替换为 sharded graph bundle：

```text
gcl_trace_graphs_l1/
  manifest.json
  graphs-00000.jsonl
  graphs-00001.jsonl
```

但第一版 spec 先以单个 JSONL 为 canonical target。

### 5.1 `gcl_trace_manifest_l1.json`

记录所有输入 trace rows、状态、gap 和 trace hash。

必须包含：

```text
artifact_type = "gcl_trace_manifest_l1"
schema_version
trace_source_modes
trace_rows
manifest_hash
deterministic_replay_hash
```

### 5.2 `gcl_trace_graphs_l1.jsonl`

每行表示一个 kernel invocation graph。

每个 graph row 至少包含：

```text
graph_id
record_id
kernel_invocation_id
workload_id
trace_hash
graph_schema_version
collection_scope
warp_partitions
nodes
edges
graph_summary
graph_hash
```

`graph_id` 必须从 stable fields 派生，推荐：

```text
sha256(record_id, kernel_invocation_id, trace_hash, graph_schema_version)
```

`graph_hash` 必须由 normalized graph payload 计算，不得依赖文件路径、写入时间或 Python dict insertion order。

### 5.3 `gcl_graph_construction_audit_l1.json`

记录 graph builder 的运行过程、质量检查和 gap。

必须包含：

```text
artifact_type = "gcl_graph_construction_audit_l1"
schema_version
builder_version
input_manifest_hash
graph_schema_version
graph_count
trace_entry_count
consumed_entry_count
dropped_entry_count
drop_reasons
node_type_counts
edge_type_counts
dataflow_coverage
determinism_checks
forbidden_field_audit
audit_hash
deterministic_replay_hash
```

---

## 6. Graph Schema

M1 的 graph 是 heterogeneous directed graph。

Graph construction 以 kernel invocation 为单位：

```text
kernel invocation
  -> warp sequences
  -> warp graphs
  -> kernel graph union
```

每个 graph 必须保留 `warp_partitions`，使后续 M2 可以先做 warp-level readout，再做 kernel-level readout。

### 6.1 Node Types

M1 第一版定义三类 node：

```text
instruction
pseudo
variable
```

#### Instruction Node

表示一条动态执行的 SASS instruction。

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

`sequence_index` 是同一 warp 内按 temporal order 排序后的连续编号。

#### Pseudo Node

表示不是单条 SASS instruction、但对 graph learning 有意义的内部操作概念。

第一版允许的 pseudo classes：

```text
mem_ref
address_calc
predicate
```

第一版可以只实际生成 `mem_ref`，但 schema 必须保留 `pseudo_class` 字段。

必要字段：

```text
node_id
node_type = "pseudo"
pseudo_class
warp_id
source_trace_index
source_entry_hash
```

#### Variable Node

表示 register、memory address/value 或 predicate 的 dynamic value/version。

第一版允许的 variable classes：

```text
register_version
memory_address
memory_value
predicate_value
input_variable
unknown_variable
```

Variable nodes 必须按写入 versioning。每次写入 destination register 时创建新的 `register_version` node。之后的 reads 连接到同一 warp 内最近可见的 producer。

如果 read 找不到 producer，创建或引用 `input_variable` node。

必要字段：

```text
node_id
node_type = "variable"
variable_class
warp_id
variable_name
version_index
source_trace_index
dynamic_value_summary
source_entry_hash
```

如果 dynamic value 不可用，`dynamic_value_summary` 必须显式标记：

```text
{"status": "missing"}
```

不得用随机值或估计值填充。

### 6.2 Edge Types

M1 第一版定义 typed directed edges：

```text
control_flow
data_source
data_left_source
data_right_source
data_destination
memory_address_source
memory_value_source
predicate_source
```

必要 edge row 字段：

```text
edge_id
src_node_id
dst_node_id
edge_type
warp_id
source_trace_index
source_entry_hash
operand_position
```

`control_flow`：

- 连接同一 warp 内相邻 instruction nodes。

`data_source`：

- 当 operand ordering 不可用时，从 source variable 指向 instruction 或 pseudo node。

`data_left_source` / `data_right_source`：

- 当 operand ordering 可用时，区分不同 source operand position。

`data_destination`：

- 从 instruction 或 pseudo node 指向 destination variable node。

`memory_address_source`：

- 从地址相关 variable / pseudo node 指向 memory pseudo node。

`memory_value_source`：

- 从 memory pseudo node 指向 value variable node，或反向表达 load/store 语义时必须在 schema 中明确。

`predicate_source`：

- 从 predicate variable 指向受 predicate 控制的 instruction node。

如果第一版无法可靠区分 `data_left_source` 和 `data_right_source`，必须使用 `data_source`，并在 graph audit 里记录：

```text
operand_position_known = false
```

---

## 7. Determinism 要求

M1 必须 deterministic。

同一份 trace manifest、trace records 和 graph config 应得到相同：

```text
graph_id
node_id
edge_id
graph_hash
audit_hash
deterministic_replay_hash
```

稳定排序规则：

1. trace manifest rows 按 `record_id` 排序；
2. 每个 kernel 内按 `warp_id` 分组；
3. 每个 warp 内按 `trace_index` 排序；
4. node output 按 `(node_type, warp_id, sequence/source index, node_id)` 排序；
5. edge output 按 `(edge_type, warp_id, source_trace_index, src_node_id, dst_node_id)` 排序。

Node id 必须从 stable tuple 派生。推荐格式：

```text
instruction:{warp_id}:{trace_index}
pseudo:{pseudo_class}:{warp_id}:{source_trace_index}:{local_index}
variable:{variable_class}:{warp_id}:{variable_name}:{version_index}
```

Edge id 必须从 stable edge tuple 派生。推荐格式：

```text
edge:{edge_type}:{src_node_id}:{dst_node_id}:{source_trace_index}:{operand_position}
```

如果存在 duplicate edges，第一版允许去重，但必须在 audit 中记录：

```text
duplicate_edge_count
duplicate_edge_policy = "deduplicate_stable_tuple"
```

---

## 8. Validation 和 Gap Handling

M1 必须严格区分：

```text
fatal validation error
recoverable graph gap
```

### 8.1 Fatal Validation Error

以下情况必须停止对应 trace row 的 graph construction：

- 缺少 `record_id`；
- 缺少 `kernel_invocation_id`；
- `trace_path` 不存在；
- trace payload hash 与 `trace_hash` 不一致；
- trace entry 缺少 `warp_id`；
- trace entry 缺少 `trace_index`；
- trace entry 缺少 `pc`；
- trace entry 缺少 `opcode`；
- trace entry 包含 non-deterministic ordering，且无法稳定排序；
- graph 输出中出现重复 `node_id` 且不能确定是同一 node；
- graph 输出中出现指向不存在 node 的 edge。

### 8.2 Recoverable Graph Gap

以下情况可以继续构图，但必须进入 audit：

- source registers 缺失；
- destination registers 缺失；
- operand ordering 缺失；
- dynamic values 缺失；
- memory address 缺失；
- predicate 信息缺失；
- 部分 trace entries unsupported；
- 某个 warp 只有一条 instruction，无法生成 control-flow edge。

Recoverable gap 不应伪造 topology。比如缺少 register 信息时，可以少生成 data-flow edges，但不能创建假的 producer-consumer relation。

---

## 9. 禁止字段

M1 graph construction 不得使用以下字段决定 topology：

```text
kernel_name
source_path
family
regime
shape_hint
simulator outcome fields
full-workload cycle totals
PKA measured feature values
B-line semantic metadata
M0 cluster labels
M0 representative labels
```

这些字段可以出现在单独 explanation 或 audit artifact 中，但不能参与 node / edge construction。

原因：

```text
M1 要验证 trace evidence 能否生成 graph representation，
不能用外部语义标签或 simulator 结果辅助构图。
```

---

## 10. 与 M0 的关系

M0 的输入是：

```text
embedding table
```

M1 的输出是：

```text
canonical graph bundle
```

M1 不需要调用 M0 selector。M1 也不需要生成 `gcl_embedding_table_l1.json`。

二者的连接点在 M2：

```text
M1 graph bundle
  -> M2 RGCN encoder
  -> M2 embedding table
  -> M0-style selector
```

所以 M1 的成功标准不是“产生好的 cluster”，而是“产生后续 encoder 可以消费的可靠 graph artifact”。

M1 输出的 `graph_hash` 后续会成为 M2 embedding table 中 `source_graph_hash` 的来源。

---

## 11. 与 M2 的接口

M2 需要从 M1 读取 canonical graph，并执行 tensorization / augmentation / RGCN training。

因此 M1 artifact 必须保留 semantic graph records，而不是直接只输出 tensors。

M1 不负责：

- node feature embedding lookup；
- uniform tensor padding；
- relation-index tensor packing；
- graph augmentation；
- train / validation split；
- RGCN model config；
- contrastive loss；
- embedding export。

但 M1 必须为这些后续步骤提供足够字段：

```text
node_type
edge_type
warp_id
trace order
opcode
pc
active mask
variable version
dynamic value summary status
operand position known / unknown
graph_hash
```

---

## 12. 第一版实现建议

M1 第一版应尽量保守。

推荐实现顺序：

1. 先支持 fixture trace JSON；
2. 先支持 `instruction` nodes；
3. 先支持 `control_flow` edges；
4. 再支持 register `variable` nodes；
5. 再支持 `data_source` / `data_destination` edges；
6. 最后支持 `mem_ref` pseudo nodes；
7. 输出 graph audit 和 replay hash；
8. 再接真实 trace subset。

这个顺序的原则是：

```text
先保证 graph artifact contract 可运行，
再逐步提高 graph topology 的语义完整度。
```

第一版不应该为了追求完整 NVBit trace support 而推迟 graph schema 固化。

---

## 13. 成功标准

GCL-M1 完成标准：

1. 能读取 trace manifest；
2. 能读取至少一种 fixture trace record 格式；
3. 能校验 trace manifest 和 trace entries；
4. 能按 kernel invocation 和 warp 分组；
5. 能按 temporal order 构建 instruction nodes；
6. 能构建 control-flow edges；
7. 能构建至少一种 variable node；
8. 能构建至少一种 data-flow edge；
9. 能输出 `gcl_trace_manifest_l1.json`；
10. 能输出 `gcl_trace_graphs_l1.jsonl`；
11. 能输出 `gcl_graph_construction_audit_l1.json`；
12. 能记录 dropped / unsupported / incomplete trace entries；
13. 相同输入能 replay 出相同 graph hash；
14. 不调用 RGCN training；
15. 不调用 M0 selector；
16. 不影响现有 PKA baseline tests。

---

## 14. M1 不应声称什么

GCL-M1 不应声称：

- 已完整复现 GCL-Sampler；
- graph embedding 已经可用；
- RGCN 已经训练完成；
- graph clusters 已经合理；
- GCL 比 PKA 更准；
- sampled simulation error 已降低；
- simulator speedup 已验证；
- graph topology 一定完整表达所有程序语义。

M1 的结论只能是：

```text
trace-like inputs 已经可以被稳定转换为 canonical heterogeneous graph artifacts，
并且这些 artifacts 可以被 replay、validate、audit，作为 M2 encoder 的输入。
```

---

## 15. 与后续阶段的关系

M1 完成后：

```text
M2: graph artifact -> tensorization / augmentation -> RGCN embedding -> selector
M3: representative anchors -> sampled/full simulator metric evaluation
```

M1/M2 的目标不是重写 M0 selector，而是逐步替换 M0 的 embedding 来源：

```text
fixture/offline embedding
  -> graph-derived learned embedding
  -> selector-compatible embedding table
```

只要 M2 最终导出的 embedding table 满足 M0 输入契约，后续就可以复用同一套 selector / anchor / evaluation 语义。

---

## 16. 设计结论

GCL-M1 的核心不是“把 GCL 跑完”，而是把 GCL-Sampler 中的 trace graph representation 独立复现出来。

它应当以 deterministic graph artifact 为中心：

```text
trace records
  -> canonical heterogeneous graph
  -> graph audit
  -> graph hash
```

当 graph artifacts 可以在不训练模型、不运行 selector、不运行 simulator 的情况下独立 replay 和 validate 时，GCL-M1 完成。
