# A 线 GCL Phase B Representative SM Selection Design Spec

日期：2026-06-04

## 1. 定位

这份 spec 定义 Phase B 中 `scheduler_signature_medoid_sm` 如何选择 representative SM。

GCL-Sampler 论文要求每个 kernel invocation 只采集一个 representative SM，并使用该 SM 上全部 CTA 的 trace 构建 kernel graph。论文没有充分展开 selected SM 的具体工程算法。因此，本 spec 定义一个可复现、可审计、HyFiSS-inspired 的工程策略：

```text
selected_sm_policy = scheduler_signature_medoid_sm
```

它的目标不是证明该 SM 一定最优，而是避免随机 SM、first observed SM、max instruction only SM 这类不可解释选择，并为每个选择输出完整 evidence。

## 2. 输入

每个 kernel invocation 的 SM selection 输入是 per-SM scheduler metadata。第一版要求每个 candidate SM 至少包含：

```text
kernel_invocation_id
sm_id
cta_ids
warp_ids_by_cta
instruction_count_by_cta
cta_start_order
cta_end_order
```

字段含义：

- `kernel_invocation_id`：当前 kernel invocation 的稳定 ID。
- `sm_id`：候选 SM 的 ID。
- `cta_ids`：该 SM 上执行过的 CTA ID 列表。
- `warp_ids_by_cta`：每个 CTA 下可见的 warp ID 列表。
- `instruction_count_by_cta`：每个 CTA 的 trace entry count 或 acquisition layer 提供的 instruction count proxy。
- `cta_start_order`：每个 CTA 在该 kernel invocation 内的调度开始序号。
- `cta_end_order`：每个 CTA 在该 kernel invocation 内的调度结束序号。

如果 acquisition layer 无法提供 `instruction_count_by_cta`，可以使用 trace entry count 作为 `instruction_count_proxy`，但 artifact 必须记录：

```text
instruction_count_proxy_source = trace_entry_count
```

不允许使用 cycle timestamp、raw memory address 或未排序的 capture order 作为默认 signature 字段。

## 3. Candidate SM

candidate SM 集合定义为：

```text
candidate_sms = SMs with cta_count > 0 for this kernel invocation
```

边界处理：

- 如果 `candidate_sms` 为空，selection 必须失败，错误为 `no_candidate_sm_for_kernel_invocation`。
- 如果只有一个 candidate SM，直接选择该 SM，但仍必须输出完整 signature、normalization、distance 和 reason artifact。
- `candidate_sm_count` 必须等于 `candidate_sms` 的数量。

## 4. Scheduler Signature

对每个 candidate SM 构造以下 raw signature：

```text
cta_count
warp_count
instruction_count_proxy
first_cta_start_order
last_cta_end_order
cta_wave_coverage
tail_cta_ratio
```

### 4.1 `cta_count`

```text
cta_count = count(unique cta_ids on this SM)
```

### 4.2 `warp_count`

```text
warp_count = count(unique (cta_id, warp_id) pairs on this SM)
```

`warp_id` 只在 CTA 内唯一时，必须使用 `(cta_id, warp_id)` 组合，不能把不同 CTA 的同名 warp 合并。

### 4.3 `instruction_count_proxy`

```text
instruction_count_proxy = sum(instruction_count_by_cta[cta_id] for cta_id in cta_ids)
```

如果使用 trace entry count 作为 proxy，则该字段表示 selected trace scope 的 dynamic entry count proxy，不声明为 exact executed instruction count。

### 4.4 `first_cta_start_order`

```text
first_cta_start_order = min(cta_start_order[cta_id] for cta_id in cta_ids)
```

该字段表示该 SM 最早开始处理 CTA 的调度位置。

### 4.5 `last_cta_end_order`

```text
last_cta_end_order = max(cta_end_order[cta_id] for cta_id in cta_ids)
```

该字段表示该 SM 最晚结束处理 CTA 的调度位置。

### 4.6 `cta_wave_coverage`

第一版用 CTA 调度顺序范围近似 wave coverage：

```text
global_first_start = min(all cta_start_order)
global_last_end = max(all cta_end_order)
global_schedule_span = max(1, global_last_end - global_first_start + 1)

cta_wave_coverage =
  (last_cta_end_order - first_cta_start_order + 1) / global_schedule_span
```

该值越高，说明该 SM 覆盖了越多 kernel invocation 的调度生命周期。

### 4.7 `tail_cta_ratio`

第一版把调度后 20% 的 CTA 视为 tail region：

```text
tail_threshold =
  global_first_start + ceil(0.8 * global_schedule_span)

tail_cta_ratio =
  count(cta_id where cta_start_order[cta_id] >= tail_threshold) / cta_count
```

该字段用于避免选择只代表 kernel tail behavior 的 SM。`cta_count = 0` 时不计算该字段，因为该 SM 不属于 candidate SM。

## 5. Normalization

所有 raw signature 字段使用 per-kernel min-max normalization：

```text
normalized_value =
  (raw_value - min_value_across_candidate_sms)
  / (max_value_across_candidate_sms - min_value_across_candidate_sms)
```

zero-variance 字段处理：

```text
if max_value == min_value:
  normalized_value = 0.0 for all candidate SMs
  normalization_note = zero_variance_feature
```

缺失字段处理：

- `cta_count`、`warp_count`、`cta_start_order`、`cta_end_order` 缺失时，selection 必须失败。
- `instruction_count_by_cta` 缺失时，可以降级到 trace entry count proxy，但必须记录 `instruction_count_proxy_source`。
- 不允许用 `0` 填充缺失值后继续选择。

## 6. Distance Function

第一版使用等权 L2 distance：

```text
global_sm_signature = mean(normalized_signature_by_sm)

distance_to_global_signature[sm_id] =
  sqrt(sum((normalized_signature[sm_id][field] - global_sm_signature[field])^2
           for field in signature_fields))
```

所有 signature fields 默认等权：

```text
signature_field_weights = {
  cta_count: 1.0,
  warp_count: 1.0,
  instruction_count_proxy: 1.0,
  first_cta_start_order: 1.0,
  last_cta_end_order: 1.0,
  cta_wave_coverage: 1.0,
  tail_cta_ratio: 1.0
}
```

Phase B 第一版不使用 learned weights，也不使用 LLM 生成权重。任何权重变更必须更新 `selected_sm_policy_version`，并作为 ablation 与等权版本对照。

## 7. Selection Rule

选择规则：

```text
selected_sm =
  argmin(
    distance_to_global_signature[sm_id],
    sm_id
  )
```

也就是：

1. 选择 distance 最小的 SM；
2. 如果 distance 完全相同，选择 `sm_id` 最小者。

`selected_sm_reason` 必须是可审计文本，例如：

```text
SM 3 selected by scheduler_signature_medoid_sm because it has the smallest
L2 distance to the global normalized scheduler signature; tie_break_rule=lowest_sm_id.
```

不允许在 tie 时随机选择。

## 8. 输出 Artifact

selection 必须输出 `selected_sm_policy_report`：

```json
{
  "artifact_name": "selected_sm_policy_report",
  "artifact_version": "gcl_phase_b_selected_sm_policy_v1",
  "kernel_invocation_id": "kernel_001",
  "selected_sm_policy": "scheduler_signature_medoid_sm",
  "selected_sm_policy_version": "v1",
  "selected_sm": 3,
  "selected_sm_reason": "SM 3 selected because it has the smallest equal-weight L2 distance to the global normalized scheduler signature.",
  "candidate_sm_count": 4,
  "candidate_sm_ids": [0, 1, 2, 3],
  "signature_fields": [
    "cta_count",
    "warp_count",
    "instruction_count_proxy",
    "first_cta_start_order",
    "last_cta_end_order",
    "cta_wave_coverage",
    "tail_cta_ratio"
  ],
  "signature_field_weights": {
    "cta_count": 1.0,
    "warp_count": 1.0,
    "instruction_count_proxy": 1.0,
    "first_cta_start_order": 1.0,
    "last_cta_end_order": 1.0,
    "cta_wave_coverage": 1.0,
    "tail_cta_ratio": 1.0
  },
  "normalization": {
    "method": "per_kernel_min_max",
    "zero_variance_policy": "set_normalized_value_to_zero_and_record_note"
  },
  "raw_signature_by_sm": {
    "3": {
      "cta_count": 8,
      "warp_count": 256,
      "instruction_count_proxy": 18420,
      "first_cta_start_order": 2,
      "last_cta_end_order": 30,
      "cta_wave_coverage": 0.91,
      "tail_cta_ratio": 0.25
    }
  },
  "normalized_signature_by_sm": {
    "3": {
      "cta_count": 0.67,
      "warp_count": 0.67,
      "instruction_count_proxy": 0.58,
      "first_cta_start_order": 0.25,
      "last_cta_end_order": 0.82,
      "cta_wave_coverage": 0.75,
      "tail_cta_ratio": 0.33
    }
  },
  "global_sm_signature": {
    "cta_count": 0.51,
    "warp_count": 0.51,
    "instruction_count_proxy": 0.49,
    "first_cta_start_order": 0.40,
    "last_cta_end_order": 0.70,
    "cta_wave_coverage": 0.68,
    "tail_cta_ratio": 0.35
  },
  "distance_metric": "equal_weight_l2",
  "distance_to_global_signature_by_sm": {
    "0": 0.43,
    "1": 0.29,
    "2": 0.31,
    "3": 0.27
  },
  "tie_break_rule": "lowest_sm_id",
  "instruction_count_proxy_source": "trace_entry_count",
  "selection_hash": "sha256:example-selected-sm-policy-report-hash"
}
```

`selection_hash` 必须由 canonical JSON 计算，覆盖 selection policy、candidate IDs、raw signatures、normalization config、distance metric、tie-break rule 和 selected SM。

## 9. 与 Trace Manifest 的关系

trace manifest 必须引用 selection report：

```text
selected_sm_policy_report_hash
selected_sm
selected_sm_policy
included_cta_ids
collection_scope = single_representative_sm_all_ctas
```

`included_cta_ids` 必须等于 `selected_sm` 对应的全部 CTA。不能在 selected SM 内再做 selected CTA / selected warp fallback。

## 10. 禁止路径

以下 policy 不能进入 Phase B complete path：

```text
first_observed_sm
random_sm
max_instruction_only_sm
max_cta_count_sm
manual_debug_sm
```

`max_cta_count_sm` 可以作为 debug / ablation report，但 artifact 必须标记：

```text
debug_not_phase_b_complete
```

## 11. 成功标准

该算法完成标准：

1. 能从 per-SM scheduler metadata 构造 raw signature；
2. 能对每个字段执行 per-kernel min-max normalization；
3. 能记录 zero-variance 字段；
4. 能计算 equal-weight L2 distance；
5. 能 deterministic 地选择 medoid SM；
6. 能输出 `selected_sm_policy_report`；
7. trace manifest 能引用 `selected_sm_policy_report_hash`；
8. 缺失必要 scheduler metadata 时必须失败；
9. 不允许随机、first observed 或 max-only 策略进入 Phase B complete path。
