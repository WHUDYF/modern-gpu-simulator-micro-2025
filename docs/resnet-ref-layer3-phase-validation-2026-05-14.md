# ResNet ref_layer3 Phase Validation

日期：2026-05-14

## 1. 验证对象

来源 worktree：

```text
/home/dyf/worktrees/trace-compression-industrial-phase
```

使用 artifacts：

```text
artifacts/resnet_forward_squash_real/reference_phase_timeline.json
artifacts/resnet_forward_squash_real/kernel_invocation_timeline.json
artifacts/resnet_forward_squash_real/squash_phase_timeline.json
artifacts/resnet_forward_squash_real/phase_alignment_report.json
```

首个验证 phase：

| 字段 | 值 |
|---|---|
| `reference_phase_id` | `ref_layer3` |
| `module_path` | `model.layer3` |
| `phase_order` | 3 |
| `input_shape` | `[1, 128, 28, 28]` |
| `output_shape` | `[1, 256, 14, 14]` |
| `kernel_range` | `[59, 87]` |
| `kernel_count` | 29 |
| `cuda_event_time_ms` | `0.2824` |
| `kernel_timeline_duration_us` | `194.976` |

Alignment status:

```text
ref_layer3 -> squash_035 ... squash_051
case = reference_split_by_squash
order_overlap_ratio = 1.0
duration_overlap_ratio = 1.0
alignment_confidence = high
```

因此 `ref_layer3` 可以作为第一轮 phase-first validation 对象。

## 2. 这一步验证什么

本次不是验证完整 B/C 线，也不是声称已经有 claim-bearing PKA 12D representation。

本次只验证：

```text
一个 ResNet reference phase
  -> 是否能稳定切出 kernel membership
  -> 是否能对齐 squash 子阶段
  -> 是否能初步形成 phase-local family / regime 候选
  -> 是否暴露 shape/resource evidence 缺口
```

当前证据状态：

| 证据 | 状态 |
|---|---|
| reference phase | measured, high confidence |
| kernel timeline | derived from PyTorch profiler trace with NVTX attribution |
| squash alignment | complete, high confidence |
| PKA 12D measured feature rows | missing in current artifact |
| resource signature | proxy only, from kernel category / grid / block / duration / name pattern |

因此本次输出应标记为：

```text
validation_status = provisional_trace_derived
evidence_status = non_claim_bearing_without_pka_12d
```

## 3. Phase 内 kernel 分组

`ref_layer3` 的 29 个 kernel invocation 可以先按实际 kernel name pattern 归成下面几类。

| 候选组 | count | duration_us | invocation_ids | 主要 grid/block | 初步解释 |
|---|---:|---:|---|---|---|
| `layout_transform` | 12 | 28.832 | 59, 60, 62, 65, 66, 68, 75, 76, 78, 81, 82, 84 | `7x8x1/256`, `1x8x256/256`, `25x4x1/256`, `1x4x256/256` | NCHW/NHWC layout conversion |
| `conv_compute` | 5 | 147.776 | 61, 67, 71, 77, 83 | `16x1x1/128`, `7x8x1/8x8x1` | CUTLASS/cuDNN convolution compute |
| `batchnorm_norm` | 5 | 9.920 | 63, 69, 72, 79, 85 | `256x1x1/64` | batchnorm inference |
| `activation` | 4 | 4.928 | 64, 74, 80, 87 | `49x1x1/128` | ReLU / activation |
| `residual_add` | 2 | 2.496 | 73, 86 | `49x1x1/128` | residual add |
| `conv_helper_index` | 1 | 1.024 | 70 | `128x1x1/1` | cuDNN precompute helper |

结论：

```text
ref_layer3 不是一个 regime。
它是一个 network phase，内部至少包含 layout、conv compute、normalization、elementwise、helper 几类候选 regime。
```

## 4. Phase -> B 线候选对象

### 4.1 `layout_transform`

建议 B 线候选：

| 字段 | 建议值 |
|---|---|
| `family` | `layout_memory` |
| `route_primitive` | `Layout / Pack / Quantize` |
| `hardware_template` | HET-10 Layout / Pack / Quantize / Cache Update |
| `shape_size_regime` | `template_refined_layout_transform_layer3` |
| `resource_signature` | `memory_coalescing_sensitive`, `layout_transform_sensitive` |
| `boundary_status` | `provisional` |

说明：

当前 shape evidence 来自 grid/block 和 kernel name pattern，不是 PKA 12D measured summary。第一版可以把它作为 layout regime 候选，但不能 claim-bearing。

### 4.2 `conv_compute`

建议 B 线候选：

| 字段 | 建议值 |
|---|---|
| `family` | `conv_tiled_compute` |
| `route_primitive` | `Convolution / Stencil Transform` |
| `hardware_template` | HET-2 Convolution / Stencil Tiled Compute |
| `shape_size_regime` | `template_refined_conv_layer3_spatial_channel_transform` |
| `resource_signature` | `tensor_core_or_conv_pipeline_sensitive`, `tile_utilization_sensitive` |
| `boundary_status` | `provisional` |

说明：

该组占 `ref_layer3` kernel timeline duration 的主要部分：`147.776us / 194.976us = 75.79%`。它应该是本 phase 的 first validation regime target。

注意 extractor 当前把多个 CUTLASS/cuDNN conv kernels 归为 `unknown`，因为 kernel name 中没有被 `classify_kernel()` 识别到 `conv` token。这是一个前端分类缺口，不应被 B 线解释为真实 unknown family。

### 4.3 `batchnorm_norm`

建议 B 线候选：

| 字段 | 建议值 |
|---|---|
| `family` | `normalization_memory` |
| `route_primitive` | `Reduction / Normalize` |
| `hardware_template` | HET-4 Reduction / Scan / Normalize, or HET-5 Elementwise Fusion for inference BN |
| `shape_size_regime` | `template_refined_batchnorm_layer3_channelwise` |
| `resource_signature` | `memory_bandwidth_sensitive`, `normalization_path_sensitive` |
| `boundary_status` | `boundary_template_ambiguity` |

说明：

BatchNorm inference 在 GPU 上可能更接近 scale/bias elementwise path，而不是训练时 reduction-heavy normalization。这里需要用 profiling / 12D evidence 决定 HET-4 还是 HET-5。

### 4.4 `activation` 和 `residual_add`

建议 B 线候选：

| 字段 | 建议值 |
|---|---|
| `family` | `elementwise_fusion` |
| `route_primitive` | `Elementwise Fusion` |
| `hardware_template` | HET-5 Elementwise / Pointwise Fusion |
| `shape_size_regime` | `template_refined_elementwise_layer3_49x128` |
| `resource_signature` | `memory_bandwidth_sensitive`, `launch_latency_sensitive` |
| `boundary_status` | `provisional` |

说明：

Activation 和 residual add 的 grid/block 一致或相近，duration 很小，但在 phase 内反复出现。第一版可作为 constraint/review object，而不是 main object。

### 4.5 `conv_helper_index`

建议 B 线候选：

| 字段 | 建议值 |
|---|---|
| `family` | `layout_or_helper_control` |
| `route_primitive` | `Layout / Pack / Quantize` or helper |
| `hardware_template` | HET-10 or boundary helper |
| `shape_size_regime` | `helper_index_precompute_layer3` |
| `resource_signature` | `latency_sensitive` |
| `boundary_status` | `boundary_helper_kernel` |

说明：

这是 cuDNN precompute helper，不能按主算法 primitive 强行合并进 conv compute。应保留为 boundary/review object，除非后续 C 线明确需要建模。

## 5. 对我们 spec 的验证结果

### 5.1 支持 phase-first

`ref_layer3` 是网络结构 phase：

```text
model.layer3
```

但它被 squash 切成：

```text
squash_035 ... squash_051
```

这说明我们之前的判断是正确的：

```text
先 phase，再在 phase 内展开 family / regime。
```

### 5.2 支持 regime 不是 phase

如果把 `ref_layer3` 直接当 regime，会把 layout transform、conv compute、batchnorm、activation、residual add 混在一起。

这会违反：

```text
same phase 不等于 same regime
```

### 5.3 支持 shape/resource 后置检查

同一 phase 内出现多种 shape signatures：

```text
conv compute: grid=16x1x1 block=128x1x1
layout transform: grid=7x8x1 block=256x1x1
batchnorm: grid=256x1x1 block=64x1x1
elementwise: grid=49x1x1 block=128x1x1
```

这说明 shape/size 和 resource signature 仍然需要在 phase 内显式检查。

### 5.4 暴露当前 artifact 缺口

当前还不能输出 stable B/C regime，原因是：

```text
missing PKA 12D measured feature rows
missing PKA cluster membership for this phase
resource signature only trace-derived proxy
some cuDNN/CUTLASS conv kernels classified as unknown
```

因此本次验证结论是：

```text
ref_layer3 phase extraction: pass
phase alignment: pass
phase-local regime candidate extraction: provisional pass
claim-bearing B/C stable regime: blocked_missing_pka_12d_representation
```

## 6. 下一步

建议下一步只做一个最小闭环：

1. 为 `ref_layer3` 的 29 个 kernels 生成 PKA-style feature summary，至少补齐 `num_instructions`、`num_thread_blocks`、memory ops、atomics、divergence 的 measured 或明确 proxy 状态。
2. 修正或补充 kernel classification，把 `cutlass__5x_cudnn::Kernel<...fprop...>` 和 `precomputed_convolve_sgemm` 标成 conv compute 候选，而不是裸 `unknown`。
3. 先把 `conv_compute` 作为 main-object regime，layout/norm/elementwise/helper 作为 review/constraint objects。
4. 用 `conv_compute` 组验证 shape consistency 和 resource signature 是否能推出一个 C-line lane。
