# GPU Trace Frontend RLCR Status And Design Lessons

## Purpose

This document records the actual state after the RLCR run for the GPU trace frontend necessity study. It is intentionally separate from the plan/spec because the plan language was too optimistic about what "complete" meant.

The main lesson is:

```text
Infrastructure completion is not evidence completion.
```

The RLCR run built most of the measurement and reporting infrastructure, and it verified simulator-side metrics with a control workload. It did not complete the claim-bearing AI-training measurements required by the plan.

## Current State

Worktree:

```text
/home/dyf/worktrees/trace-compression-industrial
```

Branch:

```text
dyf/research/trace-compression-industrial
```

Latest relevant commit at review time:

```text
66e977a fix: add provenance labels to TB/warp counts in evidence table output
```

Important note: `.humanize/rlcr/2026-05-03_22-55-59/goal-tracker.md` has local uncommitted changes from the RLCR status update. Do not overwrite or discard it without checking whether those status changes should be committed.

Current RLCR state note:

- The goal tracker currently marks infrastructure pieces as completed.
- The evidence line is still incomplete because claim-bearing BERT/Llama data is placeholder or modeled.
- The final research claim remains unproven until measured AI-training workloads are added.

Primary artifact directory:

```text
artifacts/gpu_trace_frontend_difftest_necessity/
```

Simulator instrumentation repository:

```text
/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
```

## What Was Completed

### 1. Study framing and plan artifacts

The plan now defines the complete-flow metric:

```text
P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done

T_kernel_to_sim_done =
  T_kernel_or_trace_export
+ T_trace_to_sim
+ T_sim_backend_execution
+ T_result_analysis
```

The early-stage gate is:

```text
P_trace_to_sim_slice > 15%
OR
P_trace_to_sim_step > 15%
```

This 15% threshold is an engineering go/no-go threshold only. It is not a paper-claim threshold.

### 2. Artifact pipeline

The RLCR produced the expected Markdown and JSON artifact family under:

```text
artifacts/gpu_trace_frontend_difftest_necessity/
```

Key artifacts include:

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

### 3. Simulator-side timing instrumentation

The simulator-side instrumentation was implemented in:

```text
/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
```

The implemented timing buckets are:

- `trace_read_s`
- `parse_pb_s`
- `static_bind_s`
- `tb_load_s`
- `warp_trace_build_s`
- `get_next_inst_s`
- `core_cycle_s`
- `total_sim_wall_s`
- `frontend_share`

Important fixes made during RLCR:

- `static_bind_s` and `warp_trace_build_s` were made non-overlapping.
- `workload_id` was added to JSON output.
- `kernel_count` was added to JSON output.
- workload name derivation was fixed for `traces/` layout.
- build issues were fixed by using C++17 and move-only semantics where needed.

### 4. Redundancy profiling instrumentation

The simulator now emits redundancy counters, including:

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

### 5. External reference validation

The following external definitions were checked:

- XiangShan DiffTest docs show the batching, delta/state-fusion, non-blocking-transfer, and replay concepts that were borrowed for the analogy.
- SMARTS documentation confirms that simulation sampling is a separate acceleration family, which is why the plan should mention it only as background and not as the mechanism to implement.

### 6. Control workload validation

A live simulator control run was recorded for:

```text
rodinia-nn-control
```

Timing artifact:

```text
frontend_timing_breakdown_rodinia-nn-control.json
```

Important values:

```text
total_sim_wall_s = 14.8957
frontend_share   = 0.264962
```

Redundancy artifact:

```text
redundancy_profile_rodinia-nn-control.json
```

Important values:

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

This proves the instrumentation and artifact emission path can run on a control workload. It does not prove the AI-training claim.

## What Was Not Completed

### 1. Claim-bearing BERT/Llama measurements are missing

The plan requires measured evidence for:

- `BERT-base encoder layer slice`
- `BERT-base pretraining full step`
- `Llama 3.1 8B decoder layer slice`

Current status:

- BERT rows remain `pending_measurement`.
- Llama 3.1 8B decoder layer slice remains modeled or not directly measured.
- The burden-ratio report still states that valid go/no-go requires simulator instrumentation data.

This means the central research claim is not yet proven.

### 2. Complete-flow burden ratio is not valid for go/no-go yet

`complete_flow_burden_ratio.md` currently says:

```text
PENDING_MEASUREMENT — all inputs are placeholder or modeled; measured data required
```

Therefore, the current `P_trace_to_sim` values cannot be used to claim that the frontend path is worth optimizing. They are only planning values.

### 3. BERT-base batch scaling did not run

`resource_bound_config.json` records one failed attempt:

```text
attempt: 1
batch_size: 1
status: failed
stopped_by: trace_generation_unavailable
```

The failure reason is that the session did not have an NVBit-instrumented PyTorch training harness for BERT-base pretraining. Because trace generation failed, all resource and timing telemetry fields remain null.

### 4. Llama 3.1 8B full-step validation did not run

This was correct according to the plan because it is a non-blocking tail attempt. It should only run after required evidence is complete.

Current status:

```text
Not run — prerequisites incomplete
```

Missing prerequisites:

- central evidence table with measured data;
- BERT batch-scaling records with real telemetry.

## Why The Data Was Not Obtained

The blocking issue was not the simulator instrumentation. The blocking issue was trace acquisition for AI-training workloads.

The RLCR had enough infrastructure to run a simulator control workload and produce frontend timing/redundancy artifacts. It did not have a working pipeline that can:

1. launch BERT-base pretraining under NVBit instrumentation;
2. export instruction-level trace artifacts;
3. feed those traces into the instrumented simulator;
4. record export time, simulator frontend time, backend time, and result analysis time;
5. repeat this under batch scaling.

Without step 1 and step 2, the rest of the claim-bearing evidence cannot be produced.

## Design Problem In The Previous Plan

The previous plan mixed three different notions of completion:

1. **Infrastructure completion**
   - schema exists;
   - calculators exist;
   - simulator instrumentation exists;
   - control workload emits artifacts.

2. **Evidence completion**
   - claim-bearing BERT/Llama workload traces exist;
   - measured timing artifacts exist for those workloads;
   - measured redundancy artifacts exist for those workloads;
   - burden ratio is computed from measured data.

3. **Research conclusion completion**
   - `P_trace_to_sim > 15%` is observed on slice or step measured data;
   - or the measured result is negative and the optimization line is rejected.

The RLCR effectively completed category 1, but the plan did not force a hard stop before calling the run complete. Future plans must separate these categories explicitly.

## Required Plan-Writing Rules For Next Time

### Rule 1: Use milestone gates, not only task lists

Future plans should define gates like:

```text
Gate A: Instrumentation Ready
Gate B: Trace Acquisition Ready
Gate C: Claim-Bearing Measurement Complete
Gate D: Go/No-Go Decision Complete
```

Each gate must list required artifacts and reject placeholder/model-only artifacts when measured data is required.

### Rule 2: Mark measured-data requirements as hard AC

For this study, the next plan should explicitly say:

```text
The study is not complete unless at least one claim-bearing AI-training workload has measured:
- trace export time
- frontend timing breakdown
- redundancy profile
- simulator backend time
- result analysis time
- complete-flow burden ratio
```

Modeled values may remain in the report, but they cannot satisfy the measured-evidence AC.

### Rule 3: Add a trace-acquisition prerequisite section

Before any burden-ratio plan, require:

```text
NVBit-instrumented training harness exists
trace export smoke test passes
trace directory layout is accepted by the simulator
small batch trace can be replayed
```

If these are missing, the plan should become a trace acquisition plan, not a burden-ratio evidence plan.

### Rule 4: Make placeholder usage explicit and fatal for final gates

Artifact generators may write placeholder rows during development, but final validation must fail if claim-bearing rows contain:

- `placeholder`
- `pending_measurement`
- `modeled` where measured is required
- `null` telemetry values

### Rule 5: Separate control workloads from claim-bearing workloads

Rodinia `nn` is useful for validating instrumentation. It must not be counted as satisfying AI-training evidence.

The next plan should use labels like:

```text
control_validation_passed: true
claim_bearing_measurement_passed: false
```

### Rule 6: Treat optional Llama full-step as genuinely optional

Llama 3.1 8B full-step should remain non-blocking, but only after BERT/Llama slice evidence is complete.

The required path should be:

```text
BERT-base slice or BERT-base step measured
AND/OR Llama 3.1 8B decoder layer slice measured
```

The optional path should be:

```text
Llama 3.1 8B full-step tail attempt
```

### Rule 7: Do not call RLCR complete if active tasks remain claim-critical

If tasks like central evidence generation, BERT batch scaling, or claim-bearing timing remain pending, the final status must be:

```text
PARTIAL: infrastructure complete, evidence incomplete
```

not:

```text
COMPLETE
```

### Rule 8: Keep control validation separate from claim-bearing proof

Rodinia `nn` and similar control workloads may validate instrumentation and schema integrity. They do not satisfy the AI-training evidence requirement.

## Recommended Next Plan

The next plan should not start by building more calculators. It should start with trace acquisition.

Recommended title:

```text
BERT/Llama Trace Acquisition And Claim-Bearing Measurement Plan
```

Minimum hard deliverables:

1. `bert-base-encoder-layer-slice` trace acquisition smoke test.
2. One measured `frontend_timing_breakdown_bert-base-encoder-layer-slice.json`.
3. One measured `redundancy_profile_bert-base-encoder-layer-slice.json`.
4. Complete-flow record with export, frontend, backend, and analysis time.
5. Regenerated `complete_flow_burden_ratio.json/.md`.
6. Regenerated `workload_evidence_table.json/.md`.
7. Explicit go/no-go verdict based only on measured claim-bearing data.

Nice-to-have after the minimum path:

1. BERT-base pretraining full-step batch scaling.
2. Llama 3.1 8B decoder layer slice.
3. Llama 3.1 8B full-step tail attempt.

Recommended writing pattern for the next plan:

- Start with a self-contained brief.
- State what is being measured.
- State what is explicitly out of scope.
- State the exact evidence required for go/no-go.
- Separate control validation from claim-bearing proof.
- Keep optional tail attempts clearly non-blocking.

## Current Bottom Line

The current RLCR result should be summarized as:

```text
Infrastructure and control validation are complete.
Claim-bearing AI-training evidence is incomplete.
The optimization necessity claim remains unproven until BERT/Llama measured data is acquired.
```
