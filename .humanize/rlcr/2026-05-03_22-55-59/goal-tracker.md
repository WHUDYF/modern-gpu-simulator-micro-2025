# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

Establish a reproducible evidence pipeline for deciding whether trace frontend input restructuring is necessary and feasible for the trace-driven GPU simulator.

The plan turns the design spec into implementation steps for measuring `T_trace_to_sim`, building AI-training-oriented workload evidence, estimating frontend-structuring reductions, and defining a safe prototype boundary at the `trace-parser` and `trace-driven` interface.

Quantitative thresholds from the spec, including 30-60 seconds single-run cost, 10 minutes to 1 hour sweep cost, 15% / 30% / 50% reduction scenarios, and `P_trace_to_sim` bands, are planning and modeling thresholds. They are not hard performance guarantees. Later measured data may calibrate them.

## Confirmed Study Scope

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->


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

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 3 (Updated: Round 2)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |
| 1 | Reopened tasks 6, 8, 10, 13, 14 and removed false simulator-source blocker | External simulator source exists at `/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled`, but the Round 1 implementation is still incomplete and does not yet produce the required study evidence | AC-2, AC-5, AC-7, AC-10, AC-11 remain active work |
| 2 | Kept tasks 6, 8, 10, 13, and 14 active; updated notes to reflect partial simulator fixes and new pipeline gaps | Round 2 wired metrics emission and rewrote the evidence-table generator, but verification shows the timing split is still invalid, the burden-ratio pipeline is still placeholder-driven, and the required batch-scaling / tail-attempt artifacts remain missing | AC-2, AC-4, AC-5, AC-7, AC-9, AC-10, AC-11 remain active work |
| 3 | Verified the simulator-side timing refactor and the presence of a tail-attempt artifact, but left the evidence line incomplete | The simulator now directly accumulates `warp_trace_build_s`, but the emitted timing JSON is still not consumable by the burden-ratio loader, redundancy ratios in code still do not match the design-spec denominators, the evidence table still lacks required counts, and the BERT batch-scaling records are still missing | AC-2, AC-4, AC-5, AC-7, AC-10, AC-11 remain active work |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion and routing tag -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task1: Create workload catalog schema and seed rows | AC-1, AC-10 | completed | coding | claude | JSON + Markdown artifacts at artifacts/gpu_trace_frontend_difftest_necessity/ |
| task2: Inspect existing bottleneck map artifacts and map reusable fields | AC-1, AC-7 | completed | analyze | codex | Mapped benchmark_cost_map.json fields during task1; controls documented in workload catalog |
| task3: Implement trace-size formula calculator | AC-3 | completed | coding | claude | trace_to_sim_formula.json/.md, verified expected shortcut |
| task4: Implement complete-flow burden ratio calculator | AC-4 | completed | coding | claude | complete_flow_burden_ratio.json/.md |
| task5: Identify parser and trace-driven instrumentation points | AC-2 | completed | analyze | codex | frontend_timing_breakdown.json/.md — instrumentation spec |
| task6: Add low-overhead timing counters and per-run frontend timing JSON | AC-2, AC-9 | pending | coding | claude | Simulator now directly accumulates `warp_trace_build_s` and uses non-overlapping frontend buckets, but the emitted flat JSON schema still does not match the burden-ratio loader's workload-keyed `results` path |
| task7: Identify available counters for redundancy profiling | AC-5 | completed | analyze | codex | redundancy_profile.json/.md — counter spec |
| task8: Add redundancy profile output | AC-5, AC-9 | pending | coding | claude | Simulator now emits redundancy counters, but the runtime ratio formulas still use `metadata_obj_construction_count` / `threadblock_count` instead of the design-spec denominators and no measured AI-training redundancy artifact is integrated into the study pipeline |
| task9: Implement DiffTest-style reduction model | AC-6 | completed | coding | claude | difftest_reduction_model.json/.md, verified T_trace_to_sim only |
| task10: Build central evidence table generator | AC-7, AC-9 | pending | coding | claude | Explicit trace-size mapping is now in place, but required `kernel_count` / `threadblock_or_warp_count` remain null and the reduction numbers are still sourced from stale inputs rather than integrated measured frontend artifacts |
| task11: Draft paper argument matrix | AC-7, AC-9 | completed | analyze | codex | Argument matrix section added to paper_argument_matrix.md |
| task12: Define minimal no-semantics prototype gate and equivalence-report checklist | AC-8 | completed | analyze | codex | prototype_equivalence_report.json/.md |
| task13: Add BERT-base batch-scaling records and stop-condition reporting | AC-11 | pending | coding | claude | `resource_bound_config.json` still contains only a null `pending_measurement` template rather than real scaling attempts |
| task14: Attempt optional Llama 3.1 8B full-step validation | AC-10 | pending | analyze | codex | A tail-attempt artifact now exists, but it was recorded before task10/task13 finished; AC-10 sequencing is still unmet until the required evidence line is complete |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC-1 | task1: Workload catalog | 0 | pending | workload_evidence_table.json, workload_evidence_table.md |
| AC-3 | task3: Trace-size formula calculator | 0 | pending | trace_to_sim_formula.json, trace_to_sim_formula.md |
| AC-4 | task4: Complete-flow burden ratio | 0 | pending | complete_flow_burden_ratio.json, complete_flow_burden_ratio.md |
| AC-6 | task9: DiffTest reduction model | 0 | pending | difftest_reduction_model.json, difftest_reduction_model.md |
| AC-7 | task10: Central evidence table | 0 | pending | paper_argument_matrix.json, paper_argument_matrix.md |
| AC-9 | tasks 1,3,4,9,10: All artifacts under artifacts/ | 0 | pending | All JSON + Markdown pairs exist |
| AC-10 | task14: Llama 3.1 8B full-step attempt | 0 | pending | Attempt artifact exists, but required evidence-line sequencing is still incomplete |
| AC-11 | task13: Resource bound config | 0 | pending | resource_bound_config.json, resource_bound_config.md |
| AC-2 | task5: Timing instrumentation spec | 0 | pending | frontend_timing_breakdown.json, frontend_timing_breakdown.md |
| AC-5 | task7: Redundancy profiling spec | 0 | pending | redundancy_profile.json, redundancy_profile.md |
| AC-8 | task12: Prototype gate + equivalence | 0 | pending | prototype_equivalence_report.json, prototype_equivalence_report.md |

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
| None | - | - | - | - |

### Open Issues
<!-- Issues discovered during implementation -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
| Frontend timing JSON schema is incompatible with the burden-ratio loader | 3 | AC-2, AC-4, AC-7, AC-9 | Make the simulator-emitted `frontend_timing_breakdown.json` and the study-side loader share one schema, or teach the loader to consume the current flat per-run format directly |
| Complete-flow burden ratio still cannot ingest measured timing end-to-end | 3 | AC-4, AC-7, AC-9 | Load actual measured frontend timing records instead of relying on placeholder/model labels, and only fall back to modeled rows when no artifact exists for a workload |
| Redundancy profile ratios in code still use the wrong denominators | 3 | AC-5 | Align `frontend_metrics.h` with the design-spec ratios (`dynamic_instruction_count`, `unique_tb_metadata_shape_count`, `frontend_allocations`) and emit a measured AI-training redundancy run |
| Central evidence table still omits required counts and reduction inputs are stale | 3 | AC-7, AC-9 | Populate `kernel_count` and `threadblock_or_warp_count` from artifacts, and source reduction values from the same measured workload inputs used by the burden-ratio pipeline |
| Batch scaling records are still missing | 1 | AC-11 | Run real BERT full-step scaling attempts under the stated ceiling and record the stopping limit in `resource_bound_config.json/.md` |
| Optional Llama 3.1 8B full-step attempt must be revalidated after the evidence line is complete | 3 | AC-10 | Re-run the tail attempt only after task10 and task13 are complete, then preserve the measured/abandoned record in dedicated attempt artifacts |
