# GPU Trace Frontend Necessity Study Plan

## Executor Brief

This plan is self-contained. It assumes the executor does not already know what DiffTest is.

The project studies a trace-driven GPU simulator. The immediate question is not "can we speed up the whole simulator?" The immediate question is:

```text
After a kernel or workload trace exists, how much time is spent turning trace data
into simulator-ready frontend input before the core timing model consumes it?
```

The plan must produce evidence, not assumptions. The first deliverable is a falsifiable measurement and modeling pipeline that can show whether `T_trace_to_sim` is large enough to justify a frontend optimization prototype.

## What DiffTest-Style Means Here

DiffTest is a RISC-V co-simulation and checking framework used by XiangShan. In XiangShan's documentation, the relevant optimization idea is that high-frequency hardware/software communication can become the bottleneck, and the communication path can be improved with batching, state fusion or delta handling, non-blocking transfer, and replay.

For this GPU simulator project, "DiffTest-style" is only an analogy for frontend input restructuring:

- batch: group many small trace events into larger threadblock, CTA, or warp chunks before simulator consumption;
- delta/cache: avoid repeatedly decoding or binding the same static metadata, such as `(unique_function_id, pc)`;
- validate/filter: normalize the frontend fields that the simulator actually needs and reject malformed records early;
- replay: make parser-to-frontend chunks replayable for debugging and performance regression isolation.

Do not port the RISC-V DiffTest checker. Do not compare RISC-V architectural state. Do not squash dynamic GPU instruction events. Do not change SM backend timing semantics. The safe boundary for this plan is:

```text
trace-parser -> trace-driven frontend -> shader core input
```

The core timing model, scoreboard behavior, warp issue order, and memory pipeline timing are out of scope.

## Goal Description

Establish a reproducible evidence pipeline for deciding whether trace frontend input restructuring is necessary and feasible for the trace-driven GPU simulator.

The plan turns the design spec into implementation steps for measuring `T_trace_to_sim`, building AI-training-oriented workload evidence, estimating frontend-structuring reductions, and defining a safe prototype boundary at the `trace-parser` and `trace-driven` interface.

Quantitative thresholds from the spec, including 30-60 seconds single-run cost, 10 minutes to 1 hour sweep cost, 15% / 30% / 50% reduction scenarios, and `P_trace_to_sim` bands, are planning and modeling thresholds. They are not hard performance guarantees. Later measured data may calibrate them.

## Confirmed Study Scope

The first-round study uses two measurement units:

```text
primary unit: workload slice
secondary unit: training step
```

The main early-stage metric is:

```text
P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done

T_kernel_to_sim_done =
  T_kernel_or_trace_export
+ T_trace_to_sim
+ T_sim_backend_execution
+ T_result_analysis
```

`T_kernel_or_trace_export` includes NVBit trace generation or equivalent trace export time. `T_result_analysis` is included because the user wants the complete flow from obtaining the kernel or trace through simulator completion and result processing.

The first-stage go/no-go rule is intentionally permissive:

```text
P_trace_to_sim_slice > 15%
OR
P_trace_to_sim_step > 15%
```

If either the slice-level or step-level ratio exceeds 15%, the frontend preparation path is considered worth a prototype investigation. This 15% threshold is an early-stage engineering gate, not a final paper claim threshold.

The first-round workloads are fixed as:

- T1 baseline: `BERT-base encoder layer slice`
- T1 baseline: `BERT-base pretraining full step`
- T2 representative: `Llama 3.1 8B decoder layer slice`
- T2 nice-to-have: `Llama 3.1 8B full step`, attempted only at the RLCR tail

`GPT-2 small` is not kept as a fallback workload because it is too small for the intended evidence line.

For `BERT-base pretraining full step`, batch size starts small and scales upward until the confirmed resource ceiling is reached:

```text
per-GPU memory: <= 28 GiB
trace + artifact size per workload unit: <= 500 GiB
single complete iteration time: <= 2 hours
```

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Workload catalog covers AI training scale tiers and separates controls from claim-bearing workloads.
  - Positive Tests (expected to PASS):
    - The workload table includes the required T1 rows for `BERT-base encoder layer slice` and `BERT-base pretraining full step`.
    - The workload table includes the required T2 row for `Llama 3.1 8B decoder layer slice`.
    - The workload table records `Llama 3.1 8B full step` as a non-blocking nice-to-have tail attempt.
    - Each workload row records model or slice name, approximate scale, trace granularity, expected trace-size tier, and role in the argument.
    - Existing microbenchmark and export-dominated cases are listed as controls or appendix material, not as primary AI-training evidence.
  - Negative Tests (expected to FAIL):
    - A workload catalog that contains only microbenchmarks and no AI-training or training-adjacent traces is rejected.
    - A workload catalog that uses `GPT-2 small` as the main fallback instead of the confirmed BERT-base and Llama 3.1 8B evidence line is rejected.
    - A table that uses export-dominated workloads to claim simulator-side frontend speedup is rejected.

- AC-2: Trace-to-simulator timing decomposition is measurable or explicitly modeled per workload.
  - Positive Tests (expected to PASS):
    - Each measured run can report `trace_read_s`, `parse_pb_s`, `static_bind_s`, `tb_load_s`, `warp_trace_build_s`, `get_next_inst_s`, `core_cycle_s`, `total_sim_wall_s`, and `frontend_share`.
    - The study computes `T_trace_to_sim` as the sum of trace read, protobuf parse, static binding, threadblock or warp loading, and frontend instruction delivery preparation.
    - If a large workload is not directly measurable in the first round, the report marks the value as modeled and records the model inputs.
  - Negative Tests (expected to FAIL):
    - A report that only gives total simulator wall time without frontend decomposition is insufficient.
    - A report that folds backend core timing into `T_trace_to_sim` without labeling it is rejected.

- AC-3: Trace-size planning formula is implemented as a transparent calculator artifact.
  - Positive Tests (expected to PASS):
    - The calculator implements `T_trace_to_sim ~= C_fixed + S_trace_GiB / R_frontend_GiBps`.
    - The calculator supports fast, expected, and pessimistic scenarios with configurable `R_frontend_GiBps` and `C_fixed`.
    - The generated Markdown and JSON outputs include trace-size rows from local-scale traces through at least 1 TiB scale-anchor traces.
  - Negative Tests (expected to FAIL):
    - A calculator that hardcodes only one trace size is rejected.
    - A calculator that cannot reproduce the expected shortcut `T_trace_to_sim ~= 5 + 10 * S_trace_GiB seconds` for the expected scenario is rejected.

- AC-4: Complete-flow burden ratio is computed for both slice and training-step units.
  - Positive Tests (expected to PASS):
    - The evidence pipeline computes `P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done`.
    - `T_kernel_to_sim_done` includes `T_kernel_or_trace_export`, `T_trace_to_sim`, `T_sim_backend_execution`, and `T_result_analysis`.
    - The report computes both `P_trace_to_sim_slice` and `P_trace_to_sim_step` when the corresponding measurements are available.
    - The early-stage go/no-go rule accepts either `P_trace_to_sim_slice > 15%` or `P_trace_to_sim_step > 15%`.
    - The report preserves absolute time and sweep-level cumulative time alongside the ratio.
  - Negative Tests (expected to FAIL):
    - A study that requires frontend optimization to beat backend simulation acceleration before it is considered useful is rejected.
    - A study that excludes trace export or result analysis from the complete-flow denominator without labeling an alternate denominator is rejected.
    - A study that ignores absolute time and only reports percentage share is rejected.

- AC-5: Redundancy profiling measures whether frontend caching and chunking have a local opportunity.
  - Positive Tests (expected to PASS):
    - The profile reports dynamic instruction count, unique `(unique_function_id, pc)` count, static-info lookup count, threadblock count, warp trace count, metadata object construction count, and frontend allocation count where available.
    - The profile computes `static_reuse_ratio`, `tb_metadata_reuse_ratio`, and `frontend_allocation_density`.
    - At least one AI-training workload or model slice shows whether repeated static binding is large enough to justify a frontend cache prototype.
  - Negative Tests (expected to FAIL):
    - A profile that assumes repetition without measuring unique static identifiers is rejected.
    - A profile that treats dynamic instruction squash as an allowed first-phase optimization is rejected.

- AC-6: The DiffTest-style reduction model applies only to `T_trace_to_sim`.
  - Positive Tests (expected to PASS):
    - The model computes conservative 15%, expected 30%, and optimistic 50% reductions against `T_trace_to_sim` only.
    - The output includes reduced `T_trace_to_sim`, saved time per run, and saved time per sweep.
    - The report labels the reduction model as planning evidence until prototype measurements replace it.
  - Negative Tests (expected to FAIL):
    - A model that applies the reduction rate to total simulator wall time without justification is rejected.
    - A model that presents the 15% / 30% / 50% scenarios as measured speedups before implementation is rejected.

- AC-7: The central evidence table connects workload size, frontend cost, modeled savings, and paper argument.
  - Positive Tests (expected to PASS):
    - The evidence table includes workload, measurement unit, model slice or step type, trace size, kernel count, threadblock or warp count, `T_trace_to_sim`, `T_kernel_to_sim_done`, `P_trace_to_sim`, estimated frontend-structuring reduction, reduced `T_trace_to_sim`, and complete-flow impact.
    - Rows distinguish measured values from modeled values.
    - The table can support both positive and negative conclusions about necessity.
  - Negative Tests (expected to FAIL):
    - A table that omits trace size or `T_trace_to_sim` is rejected.
    - A table that hides modeled values as measured values is rejected.

- AC-8: Prototype boundary is safe and simulator timing semantics are preserved.
  - Positive Tests (expected to PASS):
    - The allowed prototype scope is limited to decoded static-info cache, metadata normalization cache, threadblock chunk staging, and local replay at the parser / trace-driven boundary.
    - Equivalence checks compare `sim_cycle`, `sim_insn`, IPC, cache stats, kernel order, per-kernel instruction counts, warnings, and fatal conditions.
    - Any future semantic compression or dynamic instruction squash is deferred to a separate spec.
  - Negative Tests (expected to FAIL):
    - A prototype that changes scoreboard dependencies, warp issue order, memory pipeline timing, or SM backend timing state is rejected.
    - A prototype with faster frontend time but changed simulator output metrics is rejected.

- AC-9: Artifact layout is stable and reviewable.
  - Positive Tests (expected to PASS):
    - The study writes Markdown and JSON artifacts under `artifacts/gpu_trace_frontend_difftest_necessity/`.
    - Required artifacts include workload evidence, trace-to-sim formula, complete-flow burden ratio, frontend timing breakdown, redundancy profile, DiffTest reduction model, prototype equivalence report, and paper argument matrix.
    - JSON artifacts are machine-readable and Markdown artifacts are suitable for paper or thesis discussion.
  - Negative Tests (expected to FAIL):
    - Results stored only in ad hoc console output are rejected.
    - Artifacts without enough metadata to reproduce scenario assumptions are rejected.

- AC-10: Optional Llama 3.1 8B full-step validation is attempted only after the required evidence line is complete.
  - Positive Tests (expected to PASS):
    - The main deliverables for `BERT-base pretraining full step`, `Llama 3.1 8B decoder layer slice`, formula modeling, complete-flow burden ratio, and evidence table are preserved before the optional full-step attempt begins.
    - The optional `Llama 3.1 8B full step` attempt records attempt count, failure reason, partial artifacts, and whether the result was measured or abandoned.
    - Repeated failures in the optional attempt do not invalidate or overwrite the completed required artifacts.
  - Negative Tests (expected to FAIL):
    - A run plan that blocks the required evidence table on `Llama 3.1 8B full step` success is rejected.
    - An optional full-step attempt that overwrites earlier complete results with partial or failed outputs is rejected.

- AC-11: Batch scaling respects the confirmed resource ceiling.
  - Positive Tests (expected to PASS):
    - `BERT-base pretraining full step` starts with a small batch and scales upward only while the run remains within the confirmed resource ceiling.
    - The resource ceiling is recorded as per-GPU memory `<= 28 GiB`, trace plus artifact size per workload unit `<= 500 GiB`, and single complete iteration time `<= 2 hours`.
    - If scaling stops because one limit is reached, the report records which limit stopped the run.
  - Negative Tests (expected to FAIL):
    - A batch-scaling run that exceeds the confirmed resource ceiling without explicit user approval is rejected.
    - A report that changes batch size without recording resource usage is rejected.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The implementation may include a complete measurement and reporting pipeline, workload catalog, trace-size calculator, complete-flow burden ratio calculator, redundancy profiler, DiffTest-style reduction model, and a minimal no-semantics prototype with equivalence reporting. It may add simulator-side timing counters and artifact generation utilities as long as changes remain at the trace parser and trace-driven frontend boundary.

### Lower Bound (Minimum Acceptable Scope)

The minimum acceptable implementation produces the workload catalog, trace-size formula calculator, complete-flow burden ratio report, DiffTest-style reduction table, and central evidence table with clear measured versus modeled labels. It must be sufficient to decide whether frontend restructuring is worth a prototype, even before the prototype exists.

### Allowed Choices

- Can use: existing NVBit trace artifacts, existing trace bottleneck map outputs, simulator timing counters, JSON artifacts, Markdown summaries, shell or Python reporting scripts if the repository already uses them for artifact generation, and C++ instrumentation in `trace-parser` or `trace-driven`.
- Can use: modeled values for T2 and T3 scale anchors when first-round full traces are infeasible, provided the assumptions are explicit.
- Can use: DiffTest only as a methodological analogy for structured event transfer before software consumption.
- Cannot use: direct porting of the RISC-V DiffTest checker, dynamic GPU instruction squash, SM backend timing semantic changes, scoreboard or memory pipeline changes, export-time optimization claims, or unlabeled extrapolated performance claims.

## Feasibility Hints and Suggestions

### Conceptual Approach

Build the evidence line before optimizing the simulator:

1. Normalize workload metadata into a small catalog.
2. Generate trace-size timing estimates using the formula model.
3. Add or reuse timing counters to decompose simulator-side trace frontend time.
4. Emit a central evidence table with measured and modeled values.
5. Compute complete-flow burden ratio and sweep-level cumulative cost.
6. Compute DiffTest-style savings on `T_trace_to_sim` only.
7. Use the evidence to decide whether to implement the minimal frontend prototype.

The central calculation should remain simple and auditable:

```text
T_trace_to_sim =
  T_trace_read
+ T_protobuf_parse
+ T_static_bind
+ T_threadblock_warp_load
+ T_frontend_instruction_delivery_preparation

P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done
```

### Relevant References

External references that define the analogy:

- XiangShan DiffTest documentation: <https://docs.xiangshan.cc/zh-cn/latest/tools/difftest/>. Use this only for the communication-optimization ideas: batching, delta/state fusion, non-blocking transfer, and replay.
- Accel-Sim project page: <https://accel-sim.github.io/>. Use this as evidence that trace-driven GPU simulation is a mainstream method.
- Accel-Sim framework repository: <https://github.com/accel-sim/accel-sim-framework>. Use this as a reference point for GPU trace generation and trace-driven simulation workflow.
- gem5 TraceCPU documentation: <https://www.gem5.org/documentation/general_docs/cpu_models/TraceCPU>. Use this as evidence that replayable trace representations are common in architecture simulation.
- ChampSim repository: <https://github.com/ChampSim/ChampSim>. Use this as evidence that trace input format and trace consumption are first-class simulator boundaries.
- SMARTS overview: <https://users.ece.cmu.edu/~jhoe/doku/doku.php?id=smarts_simulation_sampling>. Use this as background that simulation cost can be reduced by measuring representative subsets; do not implement SMARTS in this plan.
- SimPoint project page: <https://cseweb.ucsd.edu/~calder/simpoint/>. Use this as background that architecture simulation often reduces cost by selecting representative execution, but do not implement SimPoint in this plan.

Local references:

- `docs/superpowers/specs/2026-05-03-gpu-trace-frontend-difftest-style-optimization-necessity-design.md` - source design spec for this plan.
- `docs/superpowers/specs/2026-04-28-trace-compression-engineering-bottleneck-map-design.md` - prior bottleneck map framing.
- `artifacts/trace_bottleneck_map/benchmark_cost_map.json` - existing cost map input.
- `artifacts/trace_bottleneck_map/benchmark_cost_map.md` - existing cost map summary.
- `docs/trace-benchmark-2026-04-03.md` - prior trace benchmark notes.
- `simulator-remodeled/gpu-simulator/trace-parser` - expected parser-side instrumentation boundary.
- `simulator-remodeled/gpu-simulator/trace-driven` - expected frontend consumption and replay boundary.

## Dependencies and Sequence

### Milestones

1. Workload catalog and controls
   - Define required rows for `BERT-base encoder layer slice`, `BERT-base pretraining full step`, and `Llama 3.1 8B decoder layer slice`.
   - Record `Llama 3.1 8B full step` as a non-blocking nice-to-have tail attempt.
   - Do not use `GPT-2 small` as a fallback workload in this first-round evidence line.
   - Mark microbenchmark and export-dominated cases as controls.
   - Record trace-size tiers and measurement feasibility.

2. Formula and burden modeling
   - Implement the trace-size to `T_trace_to_sim` calculator.
   - Generate fast, expected, and pessimistic scenarios.
   - Compute `P_trace_to_sim_slice`, `P_trace_to_sim_step`, and sweep-level cumulative cost using the complete-flow denominator.

3. BERT-base batch scaling guardrail
   - Start `BERT-base pretraining full step` from a small batch.
   - Increase batch size until per-GPU memory, trace plus artifact size, or single-iteration time reaches the confirmed resource ceiling.
   - Preserve resource usage records for each batch size.

4. Timing decomposition instrumentation
   - Locate existing parser and trace-driven frontend boundaries.
   - Add low-overhead timers and counters around read, parse, bind, load, warp trace build, frontend delivery, and core cycle timing.
   - Emit per-run JSON records.

5. Redundancy profiling
   - Count unique static identifiers, dynamic instructions, threadblocks, warp traces, metadata constructions, and frontend allocations.
   - Compute reuse and allocation-density ratios.
   - Compare AI-training traces against control workloads.

6. Evidence table and paper argument matrix
   - Merge workload catalog, timing breakdown, formula estimates, burden ratios, redundancy metrics, and reduction estimates.
   - Generate Markdown and JSON reports.
   - Connect XiangShan DiffTest, Accel-Sim, gem5 TraceCPU, ChampSim, SMARTS, and SimPoint to local evidence requirements.

7. Minimal no-semantics prototype decision
   - Proceed only if evidence satisfies the necessity criteria.
   - Limit implementation to decoded static-info cache, metadata normalization cache, threadblock chunk staging, and local replay.
   - Run equivalence checks before any performance claim.

8. Nice-to-have Llama 3.1 8B full-step validation
   - Attempt a `Llama 3.1 8B full training-step` run only after the required slice and local-step evidence is complete.
   - Treat this as an RLCR tail task, not as a blocker for the main result.
   - If repeated attempts fail because of trace export, storage, simulator runtime, or infrastructure limits, keep the completed required artifacts as the final usable result and record the failure evidence separately.

## Task Breakdown

Use the tasks below as execution units. Keep the output deterministic and artifact-driven.

| Task ID | Description | Target AC | Depends On |
|---------|-------------|-----------|------------|
| task1 | Create workload catalog schema and seed rows for BERT-base slice, BERT-base pretraining full step, Llama 3.1 8B decoder layer slice, optional Llama 3.1 8B full step, and control workloads | AC-1, AC-10 | - |
| task2 | Inspect existing bottleneck map artifacts and map reusable fields into the new evidence schema | AC-1, AC-7 | task1 |
| task3 | Implement trace-size formula calculator and generate Markdown / JSON planning tables | AC-3 | task1 |
| task4 | Implement complete-flow burden ratio calculator using explicit export, frontend, backend, and analysis fields for slice and step units | AC-4 | task3 |
| task5 | Identify parser and trace-driven instrumentation points for timing decomposition | AC-2 | task1 |
| task6 | Add or wire low-overhead timing counters and per-run frontend timing JSON output | AC-2, AC-9 | task5 |
| task7 | Identify available counters or insertion points for redundancy profiling | AC-5 | task5 |
| task8 | Add redundancy profile output for static reuse, threadblock metadata reuse, and frontend allocation density | AC-5, AC-9 | task7 |
| task9 | Implement DiffTest-style reduction model that applies only to `T_trace_to_sim` | AC-6 | task3, task4 |
| task10 | Build central evidence table generator with measured versus modeled labels | AC-7, AC-9 | task2, task4, task6, task8, task9 |
| task11 | Draft paper argument matrix connecting external examples to local GPU simulator evidence | AC-7, AC-9 | task10 |
| task12 | Define minimal no-semantics prototype gate and equivalence-report checklist | AC-8 | task10, task11 |
| task13 | Add BERT-base batch-scaling records and stop-condition reporting under the confirmed resource ceiling | AC-11 | task1, task4 |
| task14 | Attempt optional Llama 3.1 8B full-step validation after required artifacts are complete, preserving fallback results if the attempt fails | AC-10 | task10, task12, task13 |

## Decision Record

- The first milestone should prove necessity and feasibility before implementing simulator optimizations.
- The DiffTest analogy should be limited to structured event transfer, caching, batching, validation, and replay.
- The local optimization boundary should stay at `trace-parser -> trace-driven -> shader core` and avoid backend timing semantics.
- Quantitative thresholds are useful as planning thresholds and should be calibrated with measurements.
- The 15% `P_trace_to_sim` threshold is an early-stage go/no-go gate, not a final paper claim threshold.
- Either slice-level or step-level `P_trace_to_sim` above 15% is sufficient to justify a prototype investigation in the first stage.
- The first-round evidence line should focus on BERT-base and Llama 3.1 8B; `GPT-2 small` is too small to serve as a meaningful fallback.
- Llama 3.1 8B full-step validation is useful as scale evidence, but it should be a tail attempt after the required evidence line is complete.
- Frontend dominance versus design-loop obstruction: the chosen resolution is to prove that `T_trace_to_sim` is large enough to obstruct end-to-end iteration, not that it dominates every simulator bottleneck.
- Measured-only evidence versus modeled scale anchors: the chosen resolution is to require measured local workloads while allowing explicitly labeled modeled values for T2 and T3 large-scale anchors.
- Workload selection: the chosen resolution is `BERT-base encoder layer slice`, `BERT-base pretraining full step`, and `Llama 3.1 8B decoder layer slice`, with no `GPT-2 small` fallback.
- Llama 3.1 8B full step versus Llama 3.1 8B layer slice: the chosen resolution is to require BERT-base full-step measurement plus Llama 3.1 8B layer-slice evidence, then attempt Llama 3.1 8B full-step validation as a non-blocking nice-to-have at the end of the RLCR loop.
- BERT-base batch sizing: the chosen resolution is to start from a small pretraining batch and scale upward until the confirmed resource ceiling is reached.

## Pending User Decisions

- No pending user decisions. The current plan treats quantitative thresholds as planning and modeling thresholds, not hard performance guarantees.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must not contain plan-specific terminology such as `AC-`, `Milestone`, `Step`, `Phase`, or similar workflow markers.
- Use descriptive, domain-appropriate names in code and artifacts.
- Keep measurement overhead low and report any measurement overhead if it becomes visible.
- Preserve baseline simulator output semantics before making any performance claim.
- Clearly label measured, modeled, extrapolated, and control values in every generated report.
- Treat `P_trace_to_sim > 15%` as a first-stage engineering gate only; defer stricter paper-claim thresholds until measured data is available.

### Artifact Convention

Primary generated artifacts should live under:

```text
artifacts/gpu_trace_frontend_difftest_necessity/
```

Expected files:

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
- `resource_bound_config.md`
- `resource_bound_config.json`
- `llama8b_full_step_attempt.md` (optional)
- `llama8b_full_step_attempt.json` (optional)
