# BERT/Llama Trace Acquisition And Claim-Bearing Measurement Plan

## 执行摘要

本轮 plan 的原始目标是拿到至少一个 claim-bearing AI-training workload 的 complete-flow measured data，并据此重新生成完整流程证据。

当前状态已经前进了一步：

- Gate A 已经完成；
- `l1_bw_32f` control trace 已经成功产出；
- `BERT-base encoder layer slice` training slice 已经成功产出 claim-bearing trace；
- 旧的“NVBit callback 不触发”诊断已经被新证据推翻。

因此，本 plan 现在的任务重心不再是 trace acquisition 本身，而是继续完成 Gate B-D 的 measured evidence 链路。

## 当前完成状态

### 已完成

- Gate A: Trace Acquisition Ready。
- `l1_bw_32f` control trace acquisition。
  - trace path: `/tmp/tracer_tool_repro/traces`
  - trace size: 3,105,111 bytes
  - trace files: 4
  - kernel rows: 1
- `bert-base-encoder-layer-slice` claim-bearing trace acquisition。
  - trace path: `/tmp/bert_trace_repro/traces`
  - trace size: 4,680,065,040 bytes
  - trace files: 19,942
  - threadblock PB files: 19,939
  - kernel rows: 77
- BERT training harness。
  - path: `/home/dyf/worktrees/trace-compression-industrial/docs/bert-inference.py`
  - workload: `BertLayer` forward + `loss.backward()`
- tracer local fix。
  - path: `/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu`
  - path: `/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/Makefile`

### 仍未完成

- Gate B: BERT trace small-window simulator replay。
- Gate C: claim-bearing complete-flow measured record。
- Gate D: measured-data-based go/no-go verdict。
- BERT full step、Llama slice、Llama full-step optional tail attempt。

上一轮 RLCR 已经完成：

- artifact schema；
- trace-to-sim formula calculator；
- complete-flow burden ratio calculator；
- simulator frontend timing instrumentation；
- simulator redundancy profiling instrumentation；
- Rodinia `nn` control workload 验证。

原始 plan 生成时仍未完成：

- `BERT-base encoder layer slice` measured trace acquisition；
- `BERT-base pretraining full step` measured trace；
- `Llama 3.1 8B decoder layer slice` measured trace；
- 基于 measured claim-bearing data 的 go/no-go verdict。

上述条目中，`BERT-base encoder layer slice` 的 trace acquisition 现在已经完成；未完成的是它的 simulator replay、timing、redundancy、burden-ratio 与 verdict 链路。

因此本轮不优先继续扩展 calculator、summary table 或 paper argument matrix。本轮优先解决 simulator replay 和 complete-flow measured evidence。

核心原则：

```text
基础设施完成，不等于证据完成。
```

## Plan-gap 处理规则

如果执行中遇到本 plan 没有显式定义的问题，并且该问题阻塞 Gate B-D、hard AC 或 claim-critical artifact，不能立即停止，也不能直接声明 `BLOCKED`。

必须针对同一个未定义问题进入 bounded troubleshooting loop：

```text
max_attempts_per_plan_gap_issue = 6
```

这里的 6 次循环指的是同一个未定义问题内部的 troubleshooting attempt，不是整个 RLCR 最多 6 个 round。

每一次 attempt 必须在 summary 或专门 artifact 中记录：

- 当前阻塞的问题；
- 本轮假设；
- 本轮执行的命令、代码修改或验证；
- 本轮结果；
- 下一步是继续、换方向，还是触发停止条件。

只有满足以下条件之一，才允许停止：

- hard AC 已经完成；
- 同一个 plan-gap issue 已经尝试 6 次仍无法解决；
- 已经用日志或命令输出证明是外部环境、权限、资源上限或工具缺失导致；
- 用户明确要求停止。

如果 6 次后仍未解决，最终状态不能写 `COMPLETE`，只能写：

```text
PARTIAL: unresolved plan-gap after bounded troubleshooting
BLOCKED: external prerequisite unavailable
```

本轮如果没有拿到至少一个 claim-bearing AI-training workload 的 complete-flow measured data，最终状态必须是：

```text
PARTIAL: infrastructure complete, evidence incomplete
```

或：

```text
BLOCKED: required prerequisite unavailable
```

不能写成：

```text
COMPLETE
```

## 研究问题

本轮要回答的问题是：

```text
对于至少一个 AI-training workload，从 trace export 到 simulator 执行和结果分析完成的完整流程中，
trace-to-simulator frontend preparation 占据多少比例？
```

主指标：

```text
P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done

T_kernel_to_sim_done =
  T_kernel_or_trace_export
+ T_trace_to_sim
+ T_sim_backend_execution
+ T_result_analysis
```

早期工程 gate：

```text
P_trace_to_sim_slice > 15%
OR
P_trace_to_sim_step > 15%
```

这个 15% 是工程 go/no-go threshold，不是论文最终主张阈值。

## 范围

### Claim-Bearing Workloads

本轮最小必需路径只要求至少一个 claim-bearing workload 拿到 complete-flow measured data。

优先级如下：

1. `BERT-base encoder layer slice`
   - 第一优先级。
   - 目标是最小可行 AI-training slice。
   - 先跑 batch size 最小配置，确保 trace acquisition 和 simulator replay 成功。

2. `BERT-base pretraining full step`
   - 第二优先级。
   - 只有在 BERT slice 跑通后再做。
   - batch size 从小开始，按资源上限逐步放大。

3. `Llama 3.1 8B decoder layer slice`
   - 第三优先级。
   - 用于代表现代 decoder-only LLM。
   - 如果 BERT path 已经产生 complete-flow measured data，可以作为扩展证据。

### Control Workloads

Rodinia `nn`、microbenchmark 和 HPC workload 只能作为 control validation。

它们可以证明：

- simulator instrumentation 能跑；
- JSON/Markdown artifact 能生成；
- schema 和 loader 没坏。

它们不能证明：

- AI-training frontend bottleneck 存在；
- BERT/Llama 的 `P_trace_to_sim > 15%`；
- DiffTest-style frontend restructuring 已经必要。

本轮 artifact 必须分开记录：

```text
control_validation_passed: true/false
claim_bearing_measurement_passed: true/false
```

### Out Of Scope

本轮不做：

- 新 trace format 设计；
- simulator backend timing semantic 修改；
- dynamic GPU instruction squash；
- RISC-V DiffTest checker 移植；
- Llama 3.1 8B full-step 强制验证；
- paper-level final claim threshold 设定。

Llama 3.1 8B full-step 仍然是 optional tail attempt，只能在必需证据线完成后尝试。

## Gate 定义

### Gate A: Trace Acquisition Ready

必须满足：

- NVBit-instrumented training harness exists；
- trace export smoke test passes；
- trace directory layout is accepted by simulator；
- small batch trace can be replayed or at least loaded far enough to produce a deterministic parser error；
- export command、environment、input config 都被记录。

必需 artifact：

- `trace_acquisition_smoke_test.md`
- `trace_acquisition_smoke_test.json`

通过条件：

- 至少一个 claim-bearing workload 完成 trace export smoke test；
- trace artifact path 存在；
- trace size 被记录；
- simulator 能识别 trace 目录结构。

当前已验证状态：

- `l1_bw_32f` control 通过；
- `bert-base-encoder-layer-slice` claim-bearing trace export 通过；
- `trace_acquisition_smoke_test.{json,md}` 已更新为 PASS。

失败条件：

- training harness 不存在；
- NVBit 无法 attach；
- trace directory 缺失；
- simulator 完全不能识别 trace layout；
- smoke test 只写 placeholder。

失败状态：

```text
BLOCKED: trace acquisition unavailable
```

### Gate B: Small Trace Replay Ready

必须满足：

- 至少一个 claim-bearing trace 能进入 instrumented simulator；
- simulator 能输出 timing JSON；
- simulator 能输出 redundancy JSON；
- 即使 simulation 运行窗口很小，也必须产生可解析 artifact。

必需 artifact：

- `frontend_timing_breakdown_<workload_id>.json`
- `redundancy_profile_<workload_id>.json`
- `small_trace_replay_report.md`
- `small_trace_replay_report.json`

通过条件：

- artifact 中 `data_label = measured`；
- `workload_id` 非空；
- `trace_read_s`、`parse_pb_s`、`static_bind_s`、`tb_load_s`、`warp_trace_build_s`、`get_next_inst_s` 至少存在；
- redundancy counters 至少包含 `dynamic_insn_count`、`unique_static_id_count`、`threadblock_count`、`warp_trace_count`。

失败条件：

- artifact 只来自 Rodinia/microbenchmark control；
- artifact 中 claim-bearing workload 仍为 `placeholder`、`pending_measurement`、`modeled`；
- timing 或 redundancy 关键字段为 `null`；
- simulator 无法生成 per-run JSON。

失败状态：

```text
PARTIAL: trace acquired, simulator replay incomplete
```

### Gate C: Claim-Bearing Measurement Complete

必须满足至少一个 claim-bearing workload 有完整 measured record：

- trace export time；
- frontend timing breakdown；
- redundancy profile；
- simulator backend time；
- result analysis time；
- complete-flow burden ratio。

必需 artifact：

- `claim_bearing_measurement_record.md`
- `claim_bearing_measurement_record.json`
- regenerated `complete_flow_burden_ratio.md`
- regenerated `complete_flow_burden_ratio.json`
- regenerated `workload_evidence_table.md`
- regenerated `workload_evidence_table.json`

通过条件：

- 至少一个 claim-bearing workload 的 `data_label = measured`；
- `T_kernel_or_trace_export` measured；
- `T_trace_to_sim` measured；
- `T_sim_backend_execution` measured；
- `T_result_analysis` measured；
- `P_trace_to_sim` 基于 measured values 计算；
- control workload 没有被当成 claim-bearing workload。

失败条件：

- 所有 claim-bearing rows 仍为 placeholder/modeled；
- 缺少 export time 或 result analysis time；
- burden ratio 使用 placeholder；
- evidence table 隐藏 modeled 数据来源。

失败状态：

```text
PARTIAL: infrastructure complete, evidence incomplete
```

### Gate D: Go/No-Go Decision Complete

必须满足：

- Gate C 通过；
- go/no-go verdict 只基于 measured claim-bearing data；
- 报告明确说明 positive、negative 或 inconclusive。

必需 artifact：

- `go_no_go_verdict.md`
- `go_no_go_verdict.json`

通过条件：

如果：

```text
P_trace_to_sim_slice > 15%
OR
P_trace_to_sim_step > 15%
```

则 verdict 为：

```text
GO: frontend prototype investigation justified
```

如果 measured 数据低于阈值，则 verdict 为：

```text
NO-GO: frontend prototype not justified by current evidence
```

如果 trace acquisition 成功但 simulation 或 analysis 不完整，则 verdict 为：

```text
INCONCLUSIVE: measured evidence incomplete
```

失败条件：

- 用 modeled/placeholder 值做 GO；
- 用 control workload 做 GO；
- 没有 verdict artifact；
- verdict 没有列出数据来源。

## 硬性 Acceptance Criteria

- AC-1: Trace acquisition smoke test 必须对至少一个 claim-bearing workload 运行。
  - Positive:
    - `trace_acquisition_smoke_test.json` 包含 workload id、export command、trace path、trace size、status。
  - Negative:
    - 只记录 Rodinia control 或 placeholder 的 smoke test 不通过。

- AC-2: 至少一个 claim-bearing trace 必须进入 instrumented simulator 并产生 measured timing artifact。
  - Positive:
    - 生成 `frontend_timing_breakdown_<workload_id>.json`，且 `data_label = measured`。
  - Negative:
    - timing artifact 来自 control workload 或包含 `null` frontend fields 不通过。

- AC-3: 至少一个 claim-bearing trace 必须产生 measured redundancy artifact。
  - Positive:
    - 生成 `redundancy_profile_<workload_id>.json`，包含 dynamic/static/TB/warp counters。
  - Negative:
    - redundancy profile 只包含 spec 或 modeled expectation 不通过。

- AC-4: Complete-flow burden ratio 必须基于 measured claim-bearing data 重新生成。
  - Positive:
    - `complete_flow_burden_ratio.json` 中至少一行 claim-bearing workload 的 `data_label = measured`。
  - Negative:
    - `placeholder`、`pending_measurement`、`modeled` 或 `null` telemetry 出现在最终 claim-bearing row 中不通过。

- AC-5: Evidence table 必须显式区分 measured、modeled、placeholder、control。
  - Positive:
    - 每一行包含 `data_label`、`claim_bearing`、`measurement_unit`、`source_artifact`、`provenance`。
  - Negative:
    - 隐藏 modeled 来源或把 control 当作 claim-bearing 不通过。

- AC-6: Final status 必须诚实。
  - Positive:
    - 如果 Gate C 未通过，最终状态只能是 `PARTIAL` 或 `BLOCKED`。
  - Negative:
- 没有 complete-flow measured claim-bearing data 却写 `COMPLETE` 不通过。

- AC-7: Go/no-go verdict 只能基于 measured claim-bearing data。
  - Positive:
    - `go_no_go_verdict.json` 引用 measured source artifacts。
  - Negative:
    - 用 planning estimate、formula estimate 或 control workload 生成 GO 不通过。

## Artifact Schema 要求

所有新生成的 JSON artifact 至少包含：

```json
{
  "workload_id": "...",
  "data_label": "measured | modeled | placeholder | control",
  "claim_bearing": true,
  "measurement_unit": "slice | step | control",
  "source_artifact": "...",
  "provenance": "...",
  "generated_at": "..."
}
```

claim-bearing final gate 只接受：

```text
data_label = measured
claim_bearing = true
```

## 执行顺序

0. 固定已完成的 Gate A 事实。
   - `trace_acquisition_smoke_test.{json,md}` 是当前 Gate A source of truth。
   - `bert-base-encoder-layer-slice` trace acquisition 已通过。
   - 禁止继续沿用“NVBit callback 不触发”的旧诊断。
   - 如果执行环境重置，必须先恢复 tracer fix，再继续后续 gate。

1. 检查已有 simulator instrumentation artifact。
   - 确认 Rodinia control 仍可作为 instrumentation sanity check。
   - 不把 Rodinia 计入 claim-bearing proof。

2. 使用已获取的 BERT-base trace 进入 simulator replay。
   - 优先使用 `trace_acquisition_smoke_test.json` 记录的 `bert-base-encoder-layer-slice` trace path。
   - 如果该临时 trace path 不存在，重新运行 artifact 中记录的 exact export command。
   - Python/PyTorch traced run 必须带 `ALLOW_CUOBJDUMP_NO_DEVICE_CODE=1`。
   - 重新 export 只用于恢复 Gate A artifact，不是本轮主任务。

3. 将 BERT trace 输入 instrumented simulator。
   - 目标是先拿到 measured timing JSON 和 redundancy JSON。
   - simulation 窗口可以小，但 artifact 必须真实。

4. 生成 complete-flow measurement record。
   - 记录 export、frontend、backend、analysis。
   - 不允许 `null` telemetry 进入 final gate。

5. 重新生成 burden ratio 和 evidence table。
   - measured row 替换 placeholder row。
   - modeled row 保留但不能满足 gate。

6. 生成 go/no-go verdict。
   - 只使用 measured claim-bearing row。

7. 如果 BERT slice 成功，再考虑扩展：
   - BERT-base pretraining full step batch scaling；
   - Llama 3.1 8B decoder layer slice；
   - Llama 3.1 8B full-step tail attempt。

## 资源上限

沿用上一轮确认的资源上限：

```text
per-GPU memory <= 28 GiB
trace + artifact size per workload unit <= 500 GiB
single complete iteration time <= 2 hours
```

如果任何 run 需要超过这些限制，必须先停止并请求用户确认。

## 成功定义

本轮只有在以下条件满足时才能写：

```text
COMPLETE
```

条件：

- Gate A 通过；
- Gate B 通过；
- Gate C 通过；
- Gate D 通过；
- 至少一个 claim-bearing AI-training workload 有 measured complete-flow record；
- go/no-go verdict 基于 complete-flow measured data。

否则只能写：

```text
PARTIAL: infrastructure complete, evidence incomplete
```

或：

```text
BLOCKED: required prerequisite unavailable
```

## 下一步输出

本 plan 执行完成后必须输出一份中文总结，包含：

- 哪个 gate 通过；
- 哪个 gate 失败；
- 如果失败，失败原因是 trace acquisition、simulator replay、analysis、resource limit 还是 harness unavailable；
- 哪些 artifact 是 measured；
- 哪些 artifact 仍是 modeled 或 placeholder；
- 是否允许进入 frontend prototype investigation。
