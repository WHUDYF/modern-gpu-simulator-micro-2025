# A 线 GCL ResNet-50 Gate 2 Representative-SM Manifest Design Spec

日期：2026-06-05

## 1. Gate 2 定位

Gate 2 的目标是消费 Gate 1 输出的 `resnet50_trace_adapter_bundle.json`，为每个 eligible ResNet-50 kernel invocation 选择 representative SM，并生成 Phase B pipeline 可直接消费的 `representative_sm_trace_manifest.json`。

Gate 2 只负责：

```text
Gate 1 adapter bundle
  -> scheduler_metadata_by_sm
  -> scheduler_signature_medoid_sm
  -> selected_sm_policy_report
  -> representative_sm_trace_manifest.json
```

Gate 2 不构建 graph，不做 tensorization，不训练 RGCN，也不进行 kernel family classification。

## 2. 输入

Gate 2 的唯一正式输入是 Gate 1 passed bundle：

```text
resnet50_trace_adapter_bundle.json
```

该 bundle 必须满足：

```text
artifact_type = gcl_resnet50_trace_adapter_bundle
artifact_version = gate1_trace_adapter_v1
workload_id = resnet50
model = torchvision.models.resnet50
execution_mode = real_trace
trace_source = nvbit
input_scope = full_resnet50_inference_trace
scheduler_metadata_source = real_nvbit_smid
adapter_validation_report.status = passed
adapter_validation_report.scheduler_metadata_complete = true
adapter_validation_report.errors = []
```

Gate 2 必须拒绝：

```text
debug_not_gate1_complete
synthetic trace bundle
ResNet-like fixture bundle
hand-written opcode bundle
mini-transformer trace bundle
simulator_replay
file_order_fallback
partial manually selected kernel-only trace
missing adapter_bundle_hash
non-reproducible adapter_bundle_hash
```

Gate 2 不允许把测试 fixture 的 adapter bundle 升级为 formal `representative_sm_trace_manifest.json`。如果输入缺少真实 ResNet-50 NVBit provenance，Gate 2 只能输出 failure / debug artifact，不能生成可被 Gate 3 formal path 消费的 manifest。

## 3. 输出

Gate 2 至少输出三个 artifact：

```text
representative_sm_trace_manifest.json
selected_sm_policy_report.json
scope_preview_report.json
```

其中 `representative_sm_trace_manifest.json` 是后续 Phase B graph stage 的正式入口。

## 4. Bundle 到 Invocation Input 的转换

Gate 2 必须按 `kernel_invocation_id` 聚合 Gate 1 bundle 中的：

```text
kernel_invocation_table
cta_scheduler_records
per_warp_trace_records
```

对每个 kernel invocation 构造 selection input：

```json
{
  "kernel_invocation_id": "resnet50_k00017",
  "trace_family": "resnet50_real_trace",
  "selected_sm_policy": "scheduler_signature_medoid_sm",
  "scheduler_metadata_by_sm": {},
  "cta_to_sm": {},
  "all_trace_entries": [],
  "instruction_count_before_scope": 0,
  "warp_count_before_scope": 0
}
```

该结构不是最终 manifest；它是运行 `scheduler_signature_medoid_sm` 前的 canonical selection input。

## 5. `scheduler_metadata_by_sm`

Gate 2 必须从 `cta_scheduler_records` 构造：

```text
scheduler_metadata_by_sm[sm_id].sm_id
scheduler_metadata_by_sm[sm_id].cta_ids
scheduler_metadata_by_sm[sm_id].warp_ids_by_cta
scheduler_metadata_by_sm[sm_id].trace_entry_count_by_cta
scheduler_metadata_by_sm[sm_id].cta_start_order
scheduler_metadata_by_sm[sm_id].cta_end_order
```

映射规则：

```text
sm_id = cta_scheduler_record.sm_id
cta_ids += cta_id
warp_ids_by_cta[cta_id] = warp_ids
trace_entry_count_by_cta[cta_id] = trace_entry_count
cta_start_order[cta_id] = first_seen_order
cta_end_order[cta_id] = last_seen_order
```

`cta_ids` 必须按 `(first_seen_order, cta_id)` 稳定排序。

如果某个 kernel invocation 没有 candidate SM，Gate 2 必须输出 failure reason：

```text
no_candidate_sm_for_kernel_invocation
```

并且该 invocation 不能进入 formal manifest。

## 6. `cta_to_sm`

Gate 2 必须为每个 kernel invocation 构造：

```text
cta_to_sm[cta_id] = sm_id
```

验证规则：

- 同一个 `cta_id` 不能映射到多个 `sm_id`。
- `cta_to_sm` 的 key 集合必须等于该 invocation 的 `cta_scheduler_records.cta_id` 集合。
- 后续 `included_cta_ids` 必须只来自 `cta_to_sm[cta_id] == selected_sm` 的 CTA。

## 7. `all_trace_entries`

Gate 2 必须把 Gate 1 的 `per_warp_trace_records` 展平为 Phase B graph builder 需要的 `all_trace_entries`。

每条 entry 至少包含：

```text
kernel_invocation_id
trace_family
collection_scope
cta_id
warp_id
trace_index
pc
opcode
active_mask
predicate_mask
destination_operands
source_operands
memory_address_metadata
source_entry_hash
```

`collection_scope` 在 `all_trace_entries` 中可以标记为：

```text
single_representative_sm_all_ctas
```

但这并不表示 entry 已被裁剪；正式裁剪由 `included_cta_ids` 决定。Gate 2 必须保留完整 `all_trace_entries`，以便 Phase B validator 计算 before-scope count。

## 8. Representative SM Selection

Gate 2 默认 policy：

```text
selected_sm_policy = scheduler_signature_medoid_sm
selected_sm_policy_version = v1
```

算法沿用 Phase B 现有 spec：

```text
1. 为每个 candidate SM 构造 raw signature:
   cta_count
   warp_count
   instruction_count_proxy
   first_cta_start_order
   last_cta_end_order
   cta_wave_coverage
   tail_cta_ratio

2. 对同一 kernel invocation 内的 candidate SM 做 min-max normalization。

3. 计算 global normalized scheduler signature。

4. 使用 equal-weight L2 distance。

5. 选择 distance 最小的 SM。

6. 若 distance 完全相同，选择 lowest_sm_id。
```

Gate 2 不允许使用：

```text
random SM
first_observed_sm
max_instruction_only_sm
LLM-generated selected SM
```

`explicit_sm_id` 只允许用于 controlled replay debug artifact，不得进入 ResNet formal path。

## 9. `selected_sm_policy_report.json`

Gate 2 必须输出 report bundle：

```json
{
  "artifact_type": "gcl_resnet50_selected_sm_policy_report_bundle",
  "artifact_version": "gate2_selected_sm_policy_report_bundle_v1",
  "source_adapter_bundle_hash": "...",
  "reports": []
}
```

每个 report 必须包含 Phase B selection report 的完整字段：

```text
artifact_name
artifact_version
kernel_invocation_id
selected_sm_policy
selected_sm_policy_version
selected_sm
selected_sm_reason
candidate_sm_count
candidate_sm_ids
signature_fields
signature_field_weights
normalization
raw_signature_by_sm
normalized_signature_by_sm
global_sm_signature
distance_metric
distance_to_global_signature_by_sm
tie_break_rule
instruction_count_proxy_source
selection_hash
```

`selection_hash` 必须可复现。`representative_sm_trace_manifest.json` 中必须 inline 同一个 `selected_sm_policy_report`，并记录：

```text
selected_sm_policy_report_hash = selection_hash
```

## 10. `representative_sm_trace_manifest.json`

输出 manifest 必须满足现有 Phase B validator：

```json
{
  "artifact_type": "gcl_phase_b_trace_manifest",
  "manifest_version": "gcl_phase_b_trace_manifest_v1",
  "collection_scope": "single_representative_sm_all_ctas",
  "kernel_invocations": [],
  "trace_manifest_hash": "..."
}
```

每个 invocation 至少包含：

```text
kernel_invocation_id
trace_family
collection_scope
selected_sm_policy
scheduler_metadata_by_sm
cta_to_sm
all_trace_entries
instruction_count_before_scope
warp_count_before_scope
selected_sm_policy_report
selected_sm_policy_report_hash
selected_sm
selected_sm_reason
candidate_sm_count
included_cta_ids
instruction_count
warp_count
trace_hash
```

`included_cta_ids` 必须等于 selected SM 上的全部 CTA：

```text
included_cta_ids = scheduler_metadata_by_sm[str(selected_sm)].cta_ids
```

`instruction_count` 必须等于 selected SM scope 内 trace entry 数量。

`warp_count` 必须等于 selected SM scope 内唯一 `(cta_id, warp_id)` 数量。

`trace_hash` 和 `trace_manifest_hash` 必须由 canonical JSON 计算并可复现。

## 11. `scope_preview_report.json`

Gate 2 必须输出 scope preview，帮助用户在进入 graph construction 前审计裁剪规模：

```json
{
  "artifact_type": "gcl_resnet50_scope_preview_report",
  "artifact_version": "gate2_scope_preview_v1",
  "source_adapter_bundle_hash": "...",
  "kernel_reports": [
    {
      "kernel_invocation_id": "resnet50_k00017",
      "selected_sm": 4,
      "candidate_sm_count": 12,
      "included_cta_count": 8,
      "instruction_count_before_scope": 100000,
      "instruction_count_after_scope": 7200,
      "warp_count_before_scope": 512,
      "warp_count_after_scope": 64
    }
  ],
  "scope_preview_hash": "..."
}
```

该 report 不改变 formal manifest，只用于审计。

## 12. Failure Handling

Gate 2 对每个 kernel invocation 分别判断 eligibility。

如果某个 invocation 失败，必须记录：

```text
kernel_invocation_id
failure_stage
failure_reason
source_adapter_bundle_hash
```

失败 invocation 不进入 formal manifest。若所有 invocation 都失败，Gate 2 必须输出 failure artifact，并且不得生成可被 Phase B graph stage 消费的 formal manifest。

允许的 failure reason 包括：

```text
missing_cta_scheduler_records
missing_per_warp_trace_records
no_candidate_sm_for_kernel_invocation
cta_to_sm_conflict
trace_entry_count_mismatch
selected_sm_policy_report_hash_mismatch
phase_b_manifest_validation_failed
```

## 13. Gate 2 通过标准

Gate 2 通过时必须满足：

1. 输入 bundle 为 Gate 1 formal passed bundle。
2. 每个 formal invocation 都有 `scheduler_metadata_by_sm`。
3. 每个 formal invocation 都有 deterministic `selected_sm_policy_report`。
4. `included_cta_ids` 只来自 selected SM，且覆盖 selected SM 上全部 CTA。
5. `instruction_count_before_scope`、`warp_count_before_scope`、`instruction_count`、`warp_count` 与 trace entries 一致。
6. `selected_sm_policy_report_hash`、`trace_hash`、`trace_manifest_hash` 可复现。
7. `representative_sm_trace_manifest.json` 被现有 Phase B validator 接受。
8. Gate 3 可以只读取该 manifest，不需要读取 Gate 1 bundle 或原始 ResNet trace。
9. manifest 必须继承并记录真实输入 provenance：`workload_id = resnet50`、`execution_mode = real_trace`、`trace_source = nvbit`、`input_scope = full_resnet50_inference_trace`。
10. fixture / synthetic / debug replay 输入不得生成 formal manifest。

## 14. 非目标

Gate 2 不做：

- 修改 NVBit tracer；
- 解析 protobuf 原始 trace；
- graph construction；
- graph size audit；
- tensorization；
- RGCN training；
- GCL clustering；
- kernel family classification；
- graph compression；
- simulator 参数调优。

## 15. 结论

Gate 2 是 ResNet-50 真实 trace 进入 GCL Phase B 的正式 manifest 构造层。它把 Gate 1 的 adapter bundle 转换为现有 Phase B pipeline 可接受的 representative-SM manifest，并为每个 selected SM 选择输出完整、可复现、可审计的 evidence。
