# A 线 GCL ResNet-50 Real Trace Acquisition Design Spec

日期：2026-06-05

## 1. 目标

本 spec 定义如何使用 ResNet-50 作为 GCL 复现的真实输入，并把真实 NVBit trace 转换为 Phase B 所需的 representative-SM trace manifest。

目标闭环是：

```text
ResNet-50 inference
  -> NVBit trace acquisition
  -> scheduler metadata artifact
  -> Phase B representative-SM trace manifest
  -> per-warp graph construction
  -> RGCN kernel embedding
  -> GCL selector clustering
  -> kernel family classification
```

本 spec 的重点不是训练 GNN，也不是证明 kernel family 分类准确率；重点是定义真实 ResNet trace 如何提供可审计的 `cta_to_sm`、`cta_start_order` 和 `cta_end_order`，使后续 GCL Phase B 不依赖 synthetic fixture。

## 2. 背景边界

当前 Phase B 已经能消费 `representative_sm_trace_manifest.json` 并完成：

```text
selected SM all CTAs
  -> per-warp graph
  -> tensorization
  -> augmentation manifests
  -> RGCN/readout
  -> 256-dimensional kernel embedding table
  -> M0 silhouette K-Means selector
```

但当前 Phase B smoke fixture 是 synthetic trace，不是从真实 ResNet 或 mini-transformer trace 中抽取的。ResNet-50 复现必须新增真实 trace acquisition / adapter 层。

## 3. 输入 Workload

正式 workload 使用 torchvision ResNet-50 inference：

```text
workload_id = resnet50
model = torchvision.models.resnet50
execution_mode = real_trace
precision = fp16_autocast
batch_size = 1
input_shape = [1, 3, 224, 224]
weights = torchvision.models.ResNet50_Weights.DEFAULT
```

第一版固定 batch size 为 1。后续 batch size 变化必须作为独立 workload variant 记录，不与 batch size 1 的 artifacts 混合。

## 4. Trace Acquisition 输出

NVBit tracer 必须为 ResNet-50 输出以下文件：

```text
dynamic_trace.pb
threadblocks/
extra_info/enhanced_execution_info.json
extra_info/scheduler_metadata.json
stats.csv
```

前三类是现有 trace-driven simulator 所需 artifact。`scheduler_metadata.json` 是本 spec 要求新增或补齐的 artifact，用于支撑 representative-SM selection。

## 5. Scheduler Metadata

`scheduler_metadata.json` 必须按 kernel invocation 记录 CTA 调度证据：

```json
{
  "artifact_type": "gcl_real_trace_scheduler_metadata",
  "artifact_version": "resnet50_scheduler_metadata_v1",
  "workload_id": "resnet50",
  "source": "nvbit_tracer",
  "kernel_invocations": [
    {
      "kernel_invocation_id": "resnet50_k00017",
      "kernel_id": 17,
      "kernel_name": "...",
      "function_unique_id": 42,
      "cta_records": [
        {
          "cta_id": "12,0,0",
          "sm_id": 4,
          "first_seen_order": 203,
          "last_seen_order": 418,
          "warp_ids": [0, 1, 2, 3],
          "trace_entry_count": 9821
        }
      ]
    }
  ]
}
```

字段含义：

- `cta_id`：threadblock 坐标的稳定字符串，例如 `"x,y,z"`。
- `sm_id`：该 CTA 在真实 GPU 上执行时读取到的 SM ID。
- `first_seen_order`：tracer 第一次观察到该 CTA 动态 trace entry 的全 kernel 顺序号。
- `last_seen_order`：tracer 最后一次观察到该 CTA 动态 trace entry 的全 kernel 顺序号。
- `warp_ids`：该 CTA 中被 trace 观察到的 warp ID。
- `trace_entry_count`：该 CTA 内观察到的动态 instruction / trace entry 数量。

`first_seen_order` 和 `last_seen_order` 不声明为硬件真实 launch / retire cycle。它们是 trace-observed order，用于描述 CTA 在采集流中的相对覆盖范围。

## 6. 如何获得 SM ID

正式路径必须从真实硬件执行中获取 `cta_id -> sm_id`。

推荐实现方式：

```text
tracer 在每个 CTA 的动态执行上下文中读取 %smid
  -> 将 blockIdx 映射为 cta_id
  -> 将 %smid 记录为该 cta_id 的 sm_id
```

同一个 CTA 的 `sm_id` 必须稳定一致。若同一 `cta_id` 出现多个不同 `sm_id`，adapter 必须拒绝该 kernel invocation，并输出 acquisition validation error。

## 7. 从 Scheduler Metadata 到 Phase B Manifest

adapter 必须把 `scheduler_metadata.json` 转换为 Phase B 已定义的字段：

```text
cta_to_sm[cta_id] = sm_id

scheduler_metadata_by_sm[sm_id].cta_ids
  = all CTA IDs whose cta_to_sm is sm_id

scheduler_metadata_by_sm[sm_id].warp_ids_by_cta[cta_id]
  = warp_ids from scheduler metadata

scheduler_metadata_by_sm[sm_id].trace_entry_count_by_cta[cta_id]
  = trace_entry_count

scheduler_metadata_by_sm[sm_id].cta_start_order[cta_id]
  = first_seen_order

scheduler_metadata_by_sm[sm_id].cta_end_order[cta_id]
  = last_seen_order
```

然后 Phase B 使用现有 `scheduler_signature_medoid_sm`：

```text
per-SM scheduler signature
  -> per-kernel min-max normalization
  -> equal-weight L2 distance to global signature
  -> deterministic selected SM
```

selected SM 确定后，Phase B manifest 只包含 selected SM 上全部 CTA 的 formal scope：

```text
collection_scope = single_representative_sm_all_ctas
included_cta_ids = all CTA IDs assigned to selected_sm
```

## 8. Trace Entries

adapter 必须从 `threadblocks/` 和 `enhanced_execution_info.json` 构造 Phase B graph builder 可消费的 trace entries。

每条 trace entry 至少包含：

```text
trace_index
kernel_invocation_id
cta_id
warp_id
pc
opcode
operands
memory_address_metadata
predicate_metadata
```

其中：

- `threadblocks/` 提供 CTA、warp、PC、active mask、predicate mask、memory address 等动态信息。
- `enhanced_execution_info.json` 提供 PC / function / static instruction 到 opcode、operand、control bits 的静态映射。
- 如果某条动态 PC 找不到静态 opcode 映射，必须使用 `unknown_opcode` 并在 manifest 中记录 missing static metadata count。

## 9. Formal 与 Debug 边界

正式 ResNet GCL 复现只接受：

```text
scheduler_metadata_source = real_nvbit_smid
phase_b_artifact_status = phase_b_complete
```

如果缺少真实 `cta_to_sm`、`first_seen_order` 或 `last_seen_order`，不得生成 `phase_b_complete`。

允许存在调试路径：

```text
scheduler_metadata_source = simulator_replay
phase_b_artifact_status = debug_not_phase_b_complete
```

或：

```text
scheduler_metadata_source = file_order_fallback
phase_b_artifact_status = debug_not_phase_b_complete
```

调试路径只能用于验证 graph / tensorization / RGCN pipeline 是否能运行，不得进入正式 ResNet GCL 复现实验、kernel family classification 结论或论文结果表。

## 10. Kernel Family Classification 的输入边界

kernel family classification 的输入必须来自 canonical non-augmented graph 的 256 维 kernel embedding：

```text
kernel_embedding_dim = 256
embedding_source = canonical_non_augmented_graph
```

不得使用 projection head 的 64 维 output 作为 classification / selector embedding。

classification 第一版可以使用 cluster-to-family mapping：

```text
GCL kernel embedding
  -> silhouette K-Means cluster
  -> cluster majority / anchor evidence
  -> family label
```

如果后续新增 supervised classification head，必须作为独立 spec，不在本 acquisition spec 中混入。

## 11. 验证要求

ResNet real trace adapter 至少需要验证：

1. `scheduler_metadata.json` 中每个 `cta_id` 只有一个 `sm_id`。
2. 每个 kernel invocation 至少有一个 candidate SM。
3. `first_seen_order <= last_seen_order`。
4. `trace_entry_count` 与实际 trace entries count 一致。
5. `included_cta_ids` 只来自 selected SM，且覆盖 selected SM 上全部 CTA。
6. `selected_sm_policy_report.selection_hash` 可复现。
7. 缺少真实 SM metadata 时输出 `debug_not_phase_b_complete`，不得伪装为正式 artifact。
8. Phase B pipeline 能消费生成的 ResNet manifest，并至少完成 graph construction 和 graph audit。

## 12. 非目标

本 spec 不实现：

- GNN 结构修改；
- supervised classification head；
- simulator 参数调优比例预测；
- graph compression；
- full ResNet 多 batch size 对比；
- 使用 file order fallback 生成正式复现结果。

## 13. 结论

ResNet-50 GCL 复现的第一步不是直接训练 GNN，而是建立真实 trace acquisition 边界：从 NVBit tracer 获取 `cta_id -> sm_id` 和 trace-observed CTA order，并把它们转换为 Phase B representative-SM manifest。

只有这一步完成后，后续 per-warp graph、RGCN embedding、GCL clustering 和 kernel family classification 才有可信的真实输入来源。
