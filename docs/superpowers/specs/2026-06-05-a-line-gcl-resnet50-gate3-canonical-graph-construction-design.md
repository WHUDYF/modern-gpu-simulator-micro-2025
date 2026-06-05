# A 线 GCL ResNet-50 Gate 3 Canonical Graph Construction Design Spec

日期：2026-06-05

## 1. Gate 3 定位

Gate 3 的目标是消费 Gate 2 生成的 `representative_sm_trace_manifest.json`，把 selected representative SM 上的全部 CTA trace 转换为 GCL canonical graph artifact，并输出 graph size audit。

Gate 3 只负责：

```text
representative_sm_trace_manifest.json
  -> validate Phase B manifest
  -> selected-SM per-warp trace records
  -> per-warp graph construction
  -> kernel canonical graph bundle
  -> graph size audit
```

Gate 3 不做 tensorization，不生成 64 维 node feature tensor，不训练 RGCN，不做 augmentation，不导出 embedding，也不做 kernel classification。

## 2. 输入

Gate 3 的正式输入是 Gate 2 输出的：

```text
representative_sm_trace_manifest.json
```

Gate 3 可以读取以下 Gate 2 artifact 做 replay / audit cross-check：

```text
selected_sm_policy_report.json
scope_preview_report.json
```

但 graph construction 的正式数据来源必须是 `representative_sm_trace_manifest.json`。Gate 3 不得回读 Gate 1 bundle 或原始 ResNet trace 文件。

输入 manifest 必须满足：

```text
artifact_type = gcl_phase_b_trace_manifest
manifest_version = gcl_phase_b_trace_manifest_v1
collection_scope = single_representative_sm_all_ctas
trace_manifest_hash 可复现
```

每个 invocation 必须已经包含：

```text
selected_sm
included_cta_ids
scheduler_metadata_by_sm
cta_to_sm
all_trace_entries
selected_sm_policy_report
selected_sm_policy_report_hash
trace_hash
```

## 3. 输出

Gate 3 至少输出：

```text
phase_b_trace_records.json
canonical_graph_bundle.json
graph_size_audit.json
graph_construction_report.json
```

其中 `canonical_graph_bundle.json` 是 Gate 4 tensorization 的正式输入。

## 4. Manifest 到 Trace Records

Gate 3 必须先把 `representative_sm_trace_manifest.json` 转换为 selected-SM trace records。

转换逻辑：

```text
included_cta_ids = manifest invocation included_cta_ids
scoped_entries = all_trace_entries where cta_id in included_cta_ids
group scoped_entries by (cta_id, warp_id)
sort each warp entries by trace_index
sort CTAs by selected SM cta_start_order
assign warp_partition_id = "{cta_ordinal}:{warp_id}"
```

输出 `phase_b_trace_records.json`：

```json
{
  "artifact_type": "gcl_resnet50_phase_b_trace_records",
  "artifact_version": "gate3_trace_records_v1",
  "source_trace_manifest_hash": "...",
  "records": [
    {
      "kernel_invocation_id": "resnet50_k00017",
      "trace_family": "resnet50_real_trace",
      "collection_scope": "single_representative_sm_all_ctas",
      "selected_sm": 4,
      "included_cta_ids": ["12,0,0"],
      "selected_sm_policy_report_hash": "...",
      "warps": []
    }
  ],
  "trace_records_hash": "..."
}
```

每个 warp record 至少包含：

```text
cta_id
warp_id
warp_partition_id
entries
```

## 5. Per-Warp Graph Construction

Gate 3 必须按 warp 独立建小图，然后合并成 kernel canonical graph。

每个 warp graph 中包含三类 node：

```text
instruction node
variable node
pseudo node
```

每个 warp graph 中包含三类 edge relation：

```text
control_flow
data_source
data_destination
```

Gate 3 必须保证：

- control-flow edge 只连接同一 warp partition 内相邻 instruction node；
- 不同 warp 之间不得被串成一条 control-flow 主链；
- data edge 必须留在同一 warp partition 内；
- 每个 node 属于且只属于一个 warp partition；
- 每个 edge 属于且只属于一个 warp partition。

## 6. Node 与 Edge 的含义

Gate 3 中的 graph 不是抽象的数学图占位符，而是把 selected-SM trace 中的执行关系显式化。

Node 表示 trace 中需要被 GNN 学习的实体：

```text
instruction node
  表示一条动态 SASS instruction。

variable node
  表示寄存器版本、输入变量或未知 operand。

pseudo node
  表示不是单条 instruction、但对图学习有意义的中间语义。
  Gate 3 第一版只允许 mem_ref。
```

Edge 表示这些实体之间的关系：

```text
control_flow edge
  表示同一 warp 内动态 instruction 的前后顺序。

data_source edge
  表示某个变量或 memory reference 被 instruction 消费。

data_destination edge
  表示 instruction 产生了某个变量的新版本。
```

这些 edge relation 会进入后续 RGCN 的 `edge_type`，因此 relation type 不能随意扩展。Gate 3 strict path 只允许：

```json
{
  "control_flow": 0,
  "data_source": 1,
  "data_destination": 2
}
```

## 7. Trace 到 Graph 的示意例子

假设 selected SM 中某个 warp partition 的 trace entries 是：

```text
t0: MOV          dst=[R4]      src=[input:base]
t1: LDG.E.64.SYS dst=[R8]      src=[R4]
t2: FADD         dst=[R9]      src=[R8, input:bias]
t3: STG.E.64.SYS dst=[]        src=[R4, R9]
```

Gate 3 首先生成 instruction nodes：

```text
i:t0(MOV)
i:t1(LDG.E.64.SYS)
i:t2(FADD)
i:t3(STG.E.64.SYS)
```

然后在同一 warp 内生成 control-flow chain：

```text
i:t0 --control_flow--> i:t1
i:t1 --control_flow--> i:t2
i:t2 --control_flow--> i:t3
```

同时根据 operands 生成 variable nodes 和 data-flow edges：

```text
input:base.wp1_0 --data_source--> i:t0
i:t0 --data_destination--> R4.v1.w0
```

`LDG.E.64.SYS` 是 memory opcode，`R4.v1.w0` 被识别为地址 source，因此插入 `mem_ref` pseudo node：

```text
R4.v1.w0 --data_source--> mem_ref:t1
mem_ref:t1 --data_source--> i:t1
i:t1 --data_destination--> R8.v1.w0
```

`FADD` 消费 load 产生的 `R8.v1.w0` 和 warp-scoped input：

```text
R8.v1.w0 --data_source--> i:t2
input:bias.wp1_0 --data_source--> i:t2
i:t2 --data_destination--> R9.v1.w0
```

`STG.E.64.SYS` 是 store memory opcode，它消费 address register 和 value register：

```text
R4.v1.w0 --data_source--> mem_ref:t3
mem_ref:t3 --data_source--> i:t3
R9.v1.w0 --data_source--> i:t3
```

这个例子说明：

- instruction node 保留动态指令序列；
- variable node 保留数据流对象和寄存器版本；
- pseudo `mem_ref` node 把 memory reference 显式化；
- control-flow 与 data-flow 是不同 relation type，后续 RGCN 会用不同参数处理它们。

## 8. Instruction Nodes

每条 selected-SM trace entry 生成一个 instruction node：

```json
{
  "node_id": "i:wp1:0:t982733",
  "node_type": "instruction",
  "opcode": "LDG.E.64.SYS",
  "pc": 4096,
  "cta_id": "12,0,0",
  "warp_id": 0,
  "warp_partition_id": "1:0",
  "trace_index": 982733,
  "active_mask": 4294967295,
  "source_entry_hash": "..."
}
```

Instruction nodes 表示 warp 内动态指令流。Gate 3 不在此阶段生成 dense embedding；embedding 由 Gate 4 / Gate 5 后续阶段处理。

## 9. Variable Nodes

Gate 3 必须为 source / destination operands 生成 variable nodes。

变量 node type 包括：

```text
register_version
input_variable
unknown_variable
```

寄存器版本必须在每个 warp partition 内稳定生成：

```text
R4 first destination -> R4.v1.w0
R4 next destination  -> R4.v2.w0
source R4            -> 当前最新 version
```

该规则用于保留 warp 内数据流顺序，不能依赖 trace 输入已经预先版本化。

Input / unknown 变量必须带 warp partition scope，避免不同 warp 中同名 input 被错误合并：

```text
input:base.wp1_0
unknown:x.wp1_0
```

## 10. Pseudo Nodes

Gate 3 第一版只生成 GCL-compatible 的 `mem_ref` pseudo node。

对 memory opcode：

```text
LDG*
STG*
```

Gate 3 必须识别 source operands 中的地址寄存器，并生成：

```text
address variable node
  -> mem_ref pseudo node
  -> memory instruction node
```

Pseudo node 的作用是把 memory reference 作为图中的显式中间语义，而不是把地址 register 直接连到 memory instruction。

Gate 3 不新增额外 pseudo node 类型。任何新增 pseudo node 必须作为后续 spec，不得混入 Gate 3 strict path。

## 11. Kernel Canonical Graph Bundle

`canonical_graph_bundle.json` 至少包含：

```json
{
  "artifact_type": "gcl_resnet50_canonical_graph_bundle",
  "artifact_version": "gate3_canonical_graph_bundle_v1",
  "source_trace_manifest_hash": "...",
  "graphs": [],
  "canonical_graph_bundle_hash": "..."
}
```

每个 graph 必须满足现有 Phase B graph artifact schema：

```text
artifact_type = phase_b_canonical_graph
graph_id
kernel_invocation_id
trace_family
collection_scope
selected_sm
included_cta_ids
selected_sm_policy_report_hash
source_trace_hash
nodes
edges
edge_relation_schema
warp_partitions
graph_summary
graph_hash
```

`edge_relation_schema` 必须为：

```json
{
  "control_flow": 0,
  "data_source": 1,
  "data_destination": 2
}
```

## 12. Warp Partitions

每个 graph 必须包含 `warp_partitions`。

每个 partition 至少记录：

```text
partition_id
cta_id
warp_id
node_ids
edge_ids
instruction_node_ids
instruction_count
node_count
edge_count
first_trace_index
last_trace_index
```

`warp_partitions` 是后续 node -> warp -> kernel readout 的必要结构。Gate 3 不得使用 all-node global graph 替代 warp partitions。

## 13. Graph Summary

每个 graph 必须输出：

```text
node_count
edge_count
instruction_node_count
variable_node_count
pseudo_node_count
warp_count
node_type_counts
edge_type_counts
```

这些 summary count 必须与 `nodes`、`edges` 和 `warp_partitions` 实际内容一致。

## 14. Graph Size Audit

Gate 3 必须输出 audit-only `graph_size_audit.json`：

```json
{
  "artifact_type": "gcl_resnet50_graph_size_audit_bundle",
  "artifact_version": "gate3_graph_size_audit_bundle_v1",
  "source_canonical_graph_bundle_hash": "...",
  "audits": []
}
```

每个 audit 必须记录：

```text
graph_id
kernel_invocation_id
graph_hash
instruction_count
warp_count
node_count
edge_count
node_type_counts
edge_type_counts
max_warp_instruction_count
max_warp_node_count
max_warp_edge_count
graph_size_class
size_policy_version = phase_b_audit_guardrail_v1
training_resource_status = not_checked
trace_scope_modified_after_audit = false
phase_b_completion_status = phase_b_complete
graph_size_audit_hash
```

`graph_size_class` 只用于 audit，不得自动截断 trace scope，不得阻止 Gate 4。只有实际 tensorization / batching / training 资源失败时，后续 gate 才能输出 resource-blocked artifact。

## 15. Graph Construction Report

Gate 3 必须输出 `graph_construction_report.json`：

```json
{
  "artifact_type": "gcl_resnet50_graph_construction_report",
  "artifact_version": "gate3_graph_construction_report_v1",
  "source_trace_manifest_hash": "...",
  "graph_count": 0,
  "passed_graph_count": 0,
  "failed_invocations": [],
  "warnings": [],
  "graph_construction_report_hash": "..."
}
```

如果某个 invocation graph construction 失败，必须记录：

```text
kernel_invocation_id
failure_stage
failure_reason
source_trace_hash
```

失败 graph 不进入 formal `canonical_graph_bundle.json`。

## 16. Gate 3 通过标准

Gate 3 通过时必须满足：

1. `representative_sm_trace_manifest.json` 通过现有 Phase B manifest validator。
2. 每个 selected SM invocation 都能生成 selected-SM trace records。
3. 每个 warp partition 非空，并至少包含一个 instruction node。
4. control-flow edge 不跨 warp partition。
5. data edge 不跨 warp partition。
6. 每个 graph node 属于且只属于一个 warp partition。
7. 每个 graph edge 属于且只属于一个 warp partition。
8. register version 由 graph builder 稳定生成。
9. memory opcode 的 mem_ref pseudo node 与 address source edge 一致。
10. `graph_hash`、`canonical_graph_bundle_hash` 和 `graph_size_audit_hash` 可复现。
11. graph size audit 不修改 canonical graph。
12. Gate 4 可以只读取 `canonical_graph_bundle.json`，不需要读取 Gate 2 manifest 或原始 trace。

## 17. Failure Handling

Gate 3 必须拒绝：

```text
manifest collection_scope 不是 single_representative_sm_all_ctas
included_cta_ids 不属于 selected SM
included_cta_ids 未覆盖 selected SM 全部 CTA
selected SM scope 内 trace entries 为空
某个 warp partition 为空
跨 warp control_flow edge
edge 引用不存在 node
partition 引用不存在 node 或 edge
graph hash 不可复现
audit count 与 graph 实际 count 不一致
```

如果所有 invocation 都失败，不得生成可供 Gate 4 formal path 消费的 `canonical_graph_bundle.json`。

## 18. 非目标

Gate 3 不做：

- raw ResNet trace parsing；
- representative SM selection；
- tensorization；
- node feature schema construction；
- training augmentation；
- RGCN training；
- projection head；
- embedding export；
- GCL selector clustering；
- kernel family classification；
- graph compression；
- resource-blocked decision。

## 19. 结论

Gate 3 是真实 ResNet selected-SM trace 到 GCL graph representation 的转换层。它把 Gate 2 的 representative-SM manifest 转换为可 replay、可 audit、带 warp partitions 的 canonical graph bundle，为 Gate 4 tensorization 和后续 RGCN embedding 提供正式输入。
