# GCL-M1 Trace Graph Construction

GCL-M1 是 trace-to-graph contract validation stage。

它验证的路径是：

```text
trace records
  -> canonical heterogeneous graph bundle
```

M1 不调用 [[gcl-m0-offline-embedding-selector]]，不生成 embedding，不训练 RGCN，也不运行 simulator。

## 最小闭环

M1 的最小闭环是：

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

## 输入

M1 输入 trace manifest 和 trace records。

第一版允许：

```text
fixture_trace
real_trace_subset
```

推荐先实现 `fixture_trace`，因为 M1 第一目标是固定 graph contract，而不是立即解决真实 NVBit acquisition。

## Graph Schema

M1 生成 heterogeneous directed graph。

Node types：

```text
instruction
pseudo
variable
```

Edge types：

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

如果 operand ordering 不可靠，第一版可以退化为 `data_source`，但必须在 audit 中记录 `operand_position_known = false`。

## 输出

M1 输出：

```text
gcl_trace_manifest_l1.json
gcl_trace_graphs_l1.jsonl
gcl_graph_construction_audit_l1.json
```

其中 `gcl_trace_graphs_l1.jsonl` 是 M2 的主要输入。M1 输出的 `graph_hash` 会成为 [[gcl-m2-rgcn-embedding-and-selector]] 中 `source_graph_hash` 的来源。

## 不应声称

M1 不声称 graph embedding 可用，不声称 RGCN 已训练，也不声称 graph cluster 合理。

M1 的结论只能是：

```text
trace-like inputs 已经可以被稳定转换为 canonical heterogeneous graph artifacts。
```

相关边界见 [[stage-boundaries]]。

