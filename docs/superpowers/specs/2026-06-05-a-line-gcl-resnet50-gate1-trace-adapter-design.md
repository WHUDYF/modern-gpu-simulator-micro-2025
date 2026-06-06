# A 线 GCL ResNet-50 Gate 1 Trace Adapter Design Spec

日期：2026-06-05

## 1. Gate 1 定位

Gate 1 的目标是把 ResNet-50 的真实 NVBit trace artifacts 解析成 GCL 后续阶段可消费的 adapter bundle。

### 1.1 Formal Input 硬约束

Gate 1 formal path 的输入必须是真实 ResNet-50 NVBit trace。这个条件是 GCL ResNet-50 复现的必要条件，不能用其他数据替代。

正式通过的 Gate 1 bundle 必须同时满足：

```text
workload_id = resnet50
model = torchvision.models.resnet50
execution_mode = real_trace
trace_source = nvbit
scheduler_metadata_source = real_nvbit_smid
input_scope = full_resnet50_inference_trace
```

Gate 1 formal path 必须拒绝：

```text
synthetic trace
ResNet-like fixture
hand-written opcode sequence
mini-transformer trace
simulator replay trace
file_order_fallback scheduler metadata
partial manually selected kernel-only trace
```

这些输入可以用于单元测试、smoke test 或 debug bundle，但不得输出 `adapter_validation_report.status = passed`，不得被 Gate 2 formal path 消费。

Gate 1 不选择 representative SM，不生成 Phase B formal manifest，也不训练 GNN。它只回答一个问题：

```text
真实 ResNet trace 是否已经被稳定、完整、可审计地转换成 GCL 可消费的 per-kernel / per-CTA / per-warp 输入结构？
```

Gate 1 的输入来自 Gate 0：

```text
dynamic_trace.pb
threadblocks/
extra_info/enhanced_execution_info.json
extra_info/scheduler_metadata.json
stats.csv
```

Gate 1 的输出是：

```text
resnet50_trace_adapter_bundle.json
```

该 bundle 供 Gate 2 执行 `scheduler_signature_medoid_sm` 和 representative-SM manifest construction。

## 2. 与其他 Gate 的边界

Gate 0 负责采集真实调度证据：

```text
cta_id
sm_id
first_seen_order
last_seen_order
warp_ids
trace_entry_count
```

Gate 1 负责解析、校验、归一化和打包这些证据。

Gate 2 才负责：

```text
scheduler_metadata_by_sm
  -> scheduler_signature_medoid_sm
  -> selected_sm_policy_report
  -> representative_sm_trace_manifest.json
```

因此 Gate 1 禁止输出以下正式字段：

```text
selected_sm
selected_sm_policy_report
included_cta_ids
collection_scope = single_representative_sm_all_ctas
phase_b_complete
```

如果 Gate 1 发现输入缺少真实 scheduler metadata，它可以输出 validation failure，但不能使用 file order fallback 伪装成正式 adapter bundle。

如果 Gate 1 发现输入来自 fixture / synthetic / debug replay，即使字段 shape 与真实 trace 相同，也必须输出 debug 或 failure report，不能伪装为 formal passed bundle。

## 3. 输入 Artifact

Gate 1 必须读取以下 artifact。

### 3.1 `dynamic_trace.pb`

用于提取：

```text
kernel_id
kernel_name
function_unique_id
stream_id
device_id
grid_dim
block_dim
shared_memory_size
register_count
kernel launch order
```

### 3.2 `threadblocks/`

用于提取：

```text
kernel_id
cta_id
warp_id
dynamic PC sequence
active mask
predicate mask
memory address metadata
trace entry count
```

Gate 1 必须支持现有 trace parser 已知的 threadblock protobuf 格式：

```text
uncompressed_threadblock
compressed_threadblock
compressed_threadblock_v6
compressed_threadblock_v7
compressed_kernel_v8
```

第一版允许只展开到 GCL graph builder 需要的字段，不要求恢复 simulator 内部完整执行状态。

### 3.3 `extra_info/enhanced_execution_info.json`

用于提取：

```text
function_unique_id
kernel_name
static instruction PC
opcode
operands
predicate metadata
control bits
```

如果动态 PC 找不到静态 instruction metadata，Gate 1 必须记录 missing static metadata count，并将 opcode 标记为 `unknown_opcode`。

### 3.4 `extra_info/scheduler_metadata.json`

用于提取：

```text
kernel_invocation_id
kernel_id
cta_id
sm_id
first_seen_order
last_seen_order
warp_ids
trace_entry_count
```

该文件必须来自 Gate 0 的真实 NVBit `%smid` 采集路径：

```text
scheduler_metadata_source = real_nvbit_smid
```

其他来源只能形成 debug bundle，不得进入 Gate 2 formal path。

### 3.5 `stats.csv`

用于保留 workload provenance 和辅助校验：

```text
kernel launch count
kernel name list
runtime summary if available
```

Gate 1 不依赖 `stats.csv` 做 selected SM 决策。

## 4. 输出 Bundle Schema

`resnet50_trace_adapter_bundle.json` 至少包含：

```json
{
  "artifact_type": "gcl_resnet50_trace_adapter_bundle",
  "artifact_version": "gate1_trace_adapter_v1",
  "workload_id": "resnet50",
  "execution_mode": "real_trace",
  "scheduler_metadata_source": "real_nvbit_smid",
  "source_artifact_hashes": {},
  "kernel_invocation_table": [],
  "static_instruction_table": [],
  "cta_scheduler_records": [],
  "per_warp_trace_records": [],
  "adapter_validation_report": {},
  "adapter_bundle_hash": "..."
}
```

`adapter_bundle_hash` 必须由 canonical JSON 计算，并排除自身字段。

## 5. `kernel_invocation_table`

每个 kernel invocation 记录：

```json
{
  "kernel_invocation_id": "resnet50_k00017",
  "kernel_id": 17,
  "kernel_name": "...",
  "function_unique_id": 42,
  "device_id": 0,
  "stream_id": 0,
  "launch_order": 17,
  "grid_dim": [64, 1, 1],
  "block_dim": [128, 1, 1],
  "shared_memory_size": 0,
  "register_count": 64
}
```

`kernel_invocation_id` 必须稳定、可复现。推荐格式：

```text
resnet50_k{launch_order:05d}
```

同一个 trace 中不得出现重复 `kernel_invocation_id`。

## 6. `static_instruction_table`

每条静态 instruction 记录：

```json
{
  "function_unique_id": 42,
  "pc": 4096,
  "opcode": "LDG.E.64.SYS",
  "operands": ["R4", "R2"],
  "predicate": "P0",
  "control_bits": {
    "stall_count": 4,
    "is_yield": true,
    "wait_barrier_bits": 0
  }
}
```

Gate 1 不对 opcode 做 learned embedding，也不修改 Phase A / Phase B node feature schema。它只提供 graph builder 所需的 canonical static metadata。

## 7. `cta_scheduler_records`

每个 CTA 调度记录：

```json
{
  "kernel_invocation_id": "resnet50_k00017",
  "cta_id": "12,0,0",
  "sm_id": 4,
  "first_seen_order": 203,
  "last_seen_order": 418,
  "warp_ids": [0, 1, 2, 3],
  "trace_entry_count": 9821
}
```

验证规则：

- 同一个 `(kernel_invocation_id, cta_id)` 只能出现一次。
- 同一个 CTA 只能有一个 `sm_id`。
- `first_seen_order <= last_seen_order`。
- `warp_ids` 不能为空。
- `trace_entry_count > 0`。
- `trace_entry_count` 必须与 `per_warp_trace_records` 聚合后的实际 entry count 一致。

## 8. `per_warp_trace_records`

每个 warp trace record 记录：

```json
{
  "kernel_invocation_id": "resnet50_k00017",
  "cta_id": "12,0,0",
  "warp_id": 2,
  "trace_entries": [
    {
      "trace_index": 982733,
      "pc": 4096,
      "opcode": "LDG.E.64.SYS",
      "operands": ["R4", "R2"],
      "active_mask": 4294967295,
      "predicate_mask": 4294967295,
      "memory_address_metadata": [],
      "static_metadata_status": "resolved"
    }
  ]
}
```

`trace_index` 必须在同一 kernel invocation 内稳定排序。推荐排序：

```text
first_seen_order of CTA
  -> cta_id
  -> warp_id
  -> instruction order inside warp trace
```

Gate 1 不跨 warp 串接 control-flow 主链；跨 warp graph 语义由后续 graph construction gate 决定。

## 9. Validation Report

`adapter_validation_report` 至少包含：

```json
{
  "status": "passed",
  "kernel_invocation_count": 0,
  "cta_count": 0,
  "warp_count": 0,
  "trace_entry_count": 0,
  "missing_static_metadata_count": 0,
  "unknown_opcode_count": 0,
  "scheduler_metadata_complete": true,
  "errors": [],
  "warnings": []
}
```

正式 Gate 1 通过条件：

```text
status = passed
workload_id = resnet50
execution_mode = real_trace
trace_source = nvbit
input_scope = full_resnet50_inference_trace
scheduler_metadata_source = real_nvbit_smid
scheduler_metadata_complete = true
errors = []
kernel_invocation_count > 0
cta_count > 0
warp_count > 0
trace_entry_count > 0
```

如果失败，Gate 1 必须输出 failure report，但不得输出可被 Gate 2 formal path 消费的 passed bundle。

## 10. Debug Bundle

允许生成 debug bundle：

```text
scheduler_metadata_source = simulator_replay
adapter_status = debug_not_gate1_complete
```

或：

```text
scheduler_metadata_source = file_order_fallback
adapter_status = debug_not_gate1_complete
```

debug bundle 只能用于验证 parser、graph builder 或 RGCN pipeline 的工程连通性。Gate 2 formal representative-SM manifest construction 必须拒绝 debug bundle。

## 11. Gate 1 验收标准

Gate 1 完成时必须证明：

1. 可以读取 ResNet-50 real trace artifacts。
2. 可以建立稳定的 `kernel_invocation_id`。
3. 可以解析 static instruction metadata。
4. 可以展开 threadblock / warp dynamic trace entries。
5. 可以把 Gate 0 的 scheduler metadata 对齐到 kernel / CTA / warp。
6. 可以输出 canonical `resnet50_trace_adapter_bundle.json`。
7. bundle hash 可复现。
8. 缺少真实 scheduler metadata 时不会伪装成正式通过。
9. Gate 2 可以只依赖该 bundle 的 schema，而不直接读取原始 trace 文件。
10. synthetic / ResNet-like / hand-written fixture 不能进入 formal passed bundle。
11. formal passed bundle 必须记录并验证 `input_scope = full_resnet50_inference_trace`。

## 12. 非目标

Gate 1 不做：

- representative SM selection；
- Phase B manifest construction；
- graph construction；
- tensorization；
- RGCN training；
- GCL clustering；
- kernel family classification；
- simulator 参数调优；
- graph compression。

## 13. 结论

Gate 1 是从真实 ResNet trace 到 GCL pipeline 的第一个结构化桥接层。它把原始 trace 文件转换为可审计的 adapter bundle，使 Gate 2 可以专注于 representative-SM selection，而不再混入 trace parsing 细节。
