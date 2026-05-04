# GPU Trace Frontend RLCR 状态与设计教训

## 目的

本文记录 GPU trace frontend 必要性研究在 RLCR 运行后的真实状态。它故意独立于 plan/spec，因为上一版 plan 对“完成”的定义过于乐观。

最重要的教训是：

```text
基础设施完成，不等于证据完成。
```

这轮 RLCR 搭建了大部分测量与报告基础设施，并用 control workload 验证了 simulator 侧指标输出。但它没有完成 plan 中要求的、用于支撑主张的 AI-training 实测数据。

## 当前状态

Worktree:

```text
/home/dyf/worktrees/trace-compression-industrial
```

Branch:

```text
dyf/research/trace-compression-industrial
```

复盘时最新相关提交：

```text
66e977a fix: add provenance labels to TB/warp counts in evidence table output
```

重要说明：`.humanize/rlcr/2026-05-03_22-55-59/goal-tracker.md` 里有 RLCR 状态更新留下的本地未提交修改。不要在没有确认是否需要提交这些状态变更前覆盖或丢弃它。

当前 RLCR 状态说明：

- goal tracker 当前把基础设施部分标记为 completed。
- 证据线仍然不完整，因为用于支撑主张的 BERT/Llama 数据仍然是 placeholder 或 modeled。
- 在加入 measured AI-training workload 之前，最终研究主张仍然没有被证明。

主要 artifact 目录：

```text
artifacts/gpu_trace_frontend_difftest_necessity/
```

Simulator 插桩仓库：

```text
/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
```

## 已完成内容

### 1. 研究框架与 plan artifact

当前 plan 已经定义完整流程指标：

```text
P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done

T_kernel_to_sim_done =
  T_kernel_or_trace_export
+ T_trace_to_sim
+ T_sim_backend_execution
+ T_result_analysis
```

早期工程 gate 是：

```text
P_trace_to_sim_slice > 15%
OR
P_trace_to_sim_step > 15%
```

这个 15% 阈值只是工程 go/no-go threshold，不是论文主张阈值。

### 2. Artifact 流水线

RLCR 在以下目录下生成了预期的 Markdown 和 JSON artifact 家族：

```text
artifacts/gpu_trace_frontend_difftest_necessity/
```

关键 artifact 包括：

- `workload_evidence_table.md`
- `workload_evidence_table.json`
- `trace_to_sim_formula.md`
- `trace_to_sim_formula.json`
- `complete_flow_burden_ratio.md`
- `complete_flow_burden_ratio.json`
- `frontend_timing_breakdown.md`
- `frontend_timing_breakdown.json`
- `redundancy_profile.md`
- `redundancy_profile.json`
- `difftest_reduction_model.md`
- `difftest_reduction_model.json`
- `prototype_equivalence_report.md`
- `prototype_equivalence_report.json`
- `paper_argument_matrix.md`
- `paper_argument_matrix.json`
- `resource_bound_config.md`
- `resource_bound_config.json`
- `llama8b_full_step_attempt.md`
- `llama8b_full_step_attempt.json`

### 3. Simulator 侧 timing 插桩

simulator 侧插桩实现于：

```text
/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
```

已经实现的 timing buckets：

- `trace_read_s`
- `parse_pb_s`
- `static_bind_s`
- `tb_load_s`
- `warp_trace_build_s`
- `get_next_inst_s`
- `core_cycle_s`
- `total_sim_wall_s`
- `frontend_share`

RLCR 期间完成的重要修复：

- `static_bind_s` 与 `warp_trace_build_s` 被改成非重叠计时。
- JSON 输出加入了 `workload_id`。
- JSON 输出加入了 `kernel_count`。
- 修复了 `traces/` 布局下的 workload name 派生。
- 通过 C++17 和 move-only semantics 修复了构建问题。

### 4. Redundancy profiling 插桩

simulator 现在可以输出 redundancy counters，包括：

- `dynamic_insn_count`
- `unique_static_id_count`
- `static_info_lookup_count`
- `threadblock_count`
- `warp_trace_count`
- `metadata_obj_construction_count`
- `frontend_allocation_count`
- `unique_tb_metadata_shape_count`
- `static_reuse_ratio`
- `tb_metadata_reuse_ratio`
- `frontend_allocation_density`

### 5. 外部参考验证

已确认以下外部定义：

- XiangShan DiffTest 文档展示了本研究类比借用的 batching、delta/state-fusion、non-blocking-transfer 和 replay 思路。
- SMARTS 文档确认 simulation sampling 是另一类独立的加速方法，因此 plan 里只应把它作为背景，而不是本轮要实现的机制。

### 6. Control workload 验证

已经记录了一次 live simulator control run：

```text
rodinia-nn-control
```

Timing artifact:

```text
frontend_timing_breakdown_rodinia-nn-control.json
```

关键数值：

```text
total_sim_wall_s = 14.8957
frontend_share   = 0.264962
```

Redundancy artifact:

```text
redundancy_profile_rodinia-nn-control.json
```

关键数值：

```text
dynamic_insn_count              = 528465
unique_static_id_count          = 440642
static_info_lookup_count        = 499455
threadblock_count               = 3752
warp_trace_count                = 3752
static_reuse_ratio              = 1.19931
tb_metadata_reuse_ratio         = 3752
frontend_allocation_density     = 0.945105
```

这证明 instrumentation 和 artifact emission 路径可以在 control workload 上跑通。它不能证明 AI-training 相关主张。

## 未完成内容

### 1. 缺少用于支撑主张的 BERT/Llama 测量

plan 要求以下 workload 的 measured evidence：

- `BERT-base encoder layer slice`
- `BERT-base pretraining full step`
- `Llama 3.1 8B decoder layer slice`

当前状态：

- BERT 行仍然是 `pending_measurement`。
- Llama 3.1 8B decoder layer slice 仍然是 modeled，或没有直接测量。
- burden-ratio 报告仍然说明：有效 go/no-go 需要 simulator instrumentation data。

这意味着核心研究主张尚未被证明。

### 2. Complete-flow burden ratio 还不能用于 go/no-go

`complete_flow_burden_ratio.md` 当前写着：

```text
PENDING_MEASUREMENT — all inputs are placeholder or modeled; measured data required
```

因此，当前 `P_trace_to_sim` 数值不能用于主张 frontend path 值得优化。它们只是 planning values。

### 3. BERT-base batch scaling 没有跑通

`resource_bound_config.json` 记录了一次失败尝试：

```text
attempt: 1
batch_size: 1
status: failed
stopped_by: trace_generation_unavailable
```

失败原因是当前 session 没有 BERT-base pretraining 所需的 NVBit-instrumented PyTorch training harness。由于 trace generation 失败，所有 resource 和 timing telemetry 字段仍然是 null。

### 4. Llama 3.1 8B full-step validation 没有运行

这符合 plan，因为它是非阻塞 tail attempt。它只能在必需证据完成后运行。

当前状态：

```text
Not run — prerequisites incomplete
```

缺少的前置条件：

- 带 measured data 的 central evidence table；
- 带真实 telemetry 的 BERT batch-scaling records。

## 为什么没有拿到数据

阻塞点不是 simulator instrumentation，而是 AI-training workload 的 trace acquisition。

RLCR 已经有足够基础设施去运行 simulator control workload，并生成 frontend timing / redundancy artifacts。但它没有一条可工作的流程来：

1. 在 NVBit instrumentation 下启动 BERT-base pretraining；
2. 导出 instruction-level trace artifacts；
3. 把这些 trace 输入到已插桩 simulator；
4. 记录 export time、simulator frontend time、backend time 和 result analysis time；
5. 在 batch scaling 下重复这个过程。

没有第 1 步和第 2 步，后续用于支撑主张的证据就无法产生。

## 上一版 plan 的设计问题

上一版 plan 混淆了三种不同的“完成”：

1. **基础设施完成**
   - schema 存在；
   - calculators 存在；
   - simulator instrumentation 存在；
   - control workload 能输出 artifacts。

2. **证据完成**
   - claim-bearing BERT/Llama workload traces 存在；
   - 这些 workload 的 measured timing artifacts 存在；
   - 这些 workload 的 measured redundancy artifacts 存在；
   - burden ratio 基于 measured data 计算。

3. **研究结论完成**
   - 在 slice 或 step measured data 上观察到 `P_trace_to_sim > 15%`；
   - 或者实测结果为负，从而拒绝这条优化线。

RLCR 实际上完成了第 1 类，但 plan 没有在称为 complete 之前设置硬性停止点。未来 plan 必须明确拆分这些类别。

## 下一次写 plan 的必要规则

### Rule 1: 使用 milestone gates，而不是只写 task list

未来 plan 应定义类似这样的 gate：

```text
Gate A: Instrumentation Ready
Gate B: Trace Acquisition Ready
Gate C: Claim-Bearing Measurement Complete
Gate D: Go/No-Go Decision Complete
```

每个 gate 必须列出所需 artifact，并且在要求 measured data 时拒绝 placeholder/model-only artifact。

### Rule 2: 把 measured-data requirement 写成硬性 AC

对这项研究来说，下一版 plan 应明确写：

```text
The study is not complete unless at least one claim-bearing AI-training workload has measured:
- trace export time
- frontend timing breakdown
- redundancy profile
- simulator backend time
- result analysis time
- complete-flow burden ratio
```

modeled values 可以保留在报告里，但不能满足 measured-evidence AC。

### Rule 3: 增加 trace-acquisition prerequisite section

在任何 burden-ratio plan 之前，必须要求：

```text
NVBit-instrumented training harness exists
trace export smoke test passes
trace directory layout is accepted by the simulator
small batch trace can be replayed
```

如果这些条件缺失，plan 应该变成 trace acquisition plan，而不是 burden-ratio evidence plan。

### Rule 4: 明确 placeholder 使用，并让它在 final gate 中失败

artifact generator 可以在开发阶段写 placeholder rows，但 final validation 必须在 claim-bearing rows 包含以下内容时失败：

- `placeholder`
- `pending_measurement`
- `modeled`，当该处要求 measured 时
- `null` telemetry values

### Rule 5: 区分 control workload 和 claim-bearing workload

Rodinia `nn` 可以用于验证 instrumentation。它不能算作满足 AI-training evidence。

下一版 plan 应使用类似这样的标签：

```text
control_validation_passed: true
claim_bearing_measurement_passed: false
```

### Rule 6: 把可选 Llama full-step 当成真正可选项

Llama 3.1 8B full-step 应保持非阻塞，但只能在 BERT/Llama slice evidence 完成后执行。

必需路径应是：

```text
BERT-base slice or BERT-base step measured
AND/OR Llama 3.1 8B decoder layer slice measured
```

可选路径应是：

```text
Llama 3.1 8B full-step tail attempt
```

### Rule 7: 如果 claim-critical active tasks 仍存在，不要称 RLCR complete

如果 central evidence generation、BERT batch scaling 或 claim-bearing timing 这类任务仍然 pending，最终状态必须是：

```text
PARTIAL: infrastructure complete, evidence incomplete
```

而不是：

```text
COMPLETE
```

### Rule 8: 把 control validation 和 claim-bearing proof 分开

Rodinia `nn` 和类似 control workload 可以验证 instrumentation 和 schema 完整性。它们不能满足 AI-training evidence requirement。

## 推荐的下一版 plan

下一版 plan 不应该从继续写 calculators 开始，而应该从 trace acquisition 开始。

推荐标题：

```text
BERT/Llama Trace Acquisition And Claim-Bearing Measurement Plan
```

最小硬性交付物：

1. `bert-base-encoder-layer-slice` trace acquisition smoke test。
2. 一个 measured `frontend_timing_breakdown_bert-base-encoder-layer-slice.json`。
3. 一个 measured `redundancy_profile_bert-base-encoder-layer-slice.json`。
4. 包含 export、frontend、backend 和 analysis time 的 complete-flow record。
5. 重新生成 `complete_flow_burden_ratio.json/.md`。
6. 重新生成 `workload_evidence_table.json/.md`。
7. 只基于 measured claim-bearing data 给出明确 go/no-go verdict。

最小路径之后的 nice-to-have：

1. BERT-base pretraining full-step batch scaling。
2. Llama 3.1 8B decoder layer slice。
3. Llama 3.1 8B full-step tail attempt。

下一版 plan 推荐写法：

- 以自包含 brief 开始。
- 明确说明要测什么。
- 明确说明什么不在范围内。
- 明确说明 go/no-go 所需的精确证据。
- 分开 control validation 和 claim-bearing proof。
- 让 optional tail attempt 保持清晰的非阻塞性质。

## 当前底线

当前 RLCR 结果应总结为：

```text
Infrastructure and control validation are complete.
Claim-bearing AI-training evidence is incomplete.
The optimization necessity claim remains unproven until BERT/Llama measured data is acquired.
```

