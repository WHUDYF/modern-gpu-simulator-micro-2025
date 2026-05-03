# Round 3 Review

## Findings

1. Critical: the new burden-ratio "artifact loading" path is not wired to the simulator's actual output schema, so measured frontend timing still cannot flow end-to-end. The simulator emits a flat per-run `frontend_timing_breakdown.json` in `trace_parser` destruction (`/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-parser/trace_parser.h:168-173`) using the flat fields written in `frontend_metrics.h:93-106`, but `complete_flow_burden_ratio_calc.py` only consumes a workload-keyed `results` map at `artifacts/gpu_trace_frontend_difftest_necessity/complete_flow_burden_ratio_calc.py:44-76`. Even if a real simulator run produces timing JSON today, `wid in measured.get("results", {})` at `:66-68` will never match that flat file, so the script falls back to placeholder/model rows for every workload. The summary's claim that the study repo now "loads measured timing when available" is therefore false in the current implementation.

2. High: task 8 is still not aligned with the design spec in the actual runtime implementation. The tracked JSON spec was edited to say `tb_metadata_reuse_ratio = threadblock_count / unique_tb_metadata_shape_count` and `frontend_allocation_density = frontend_allocation_count / dynamic_instruction_count` (`artifacts/gpu_trace_frontend_difftest_necessity/redundancy_profile.json:50-65`), matching the design at `docs/superpowers/specs/2026-05-03-gpu-trace-frontend-difftest-style-optimization-necessity-design.md:472-474`. But the simulator code still computes `tb_metadata_reuse_ratio` as `threadblock_count / metadata_obj_construction_count` and `frontend_allocation_density` as `frontend_allocation_count / threadblock_count` in `/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-parser/frontend_metrics.h:84-90`, and those are the values emitted in `write_redundancy_json()` at `:110-123`. The spec text and the executable implementation now disagree, so AC-5 is still open.

3. High: task 10 remains incomplete even after the trace-size mapping fix. The generator still initializes `kernel_count` and `threadblock_or_warp_count` as pending-null fields and never populates them (`artifacts/gpu_trace_frontend_difftest_necessity/evidence_table_generator.py:88-100`, `:126-129`), while the Markdown table omits those required columns entirely (`paper_argument_matrix.md:13-18`; generator at `evidence_table_generator.py:191-208`). The row-level reduction values are also internally inconsistent with the same rows' frontend times: the burden-ratio report now uses `T_trace_to_sim = 10.0/105.0/205.0/1005.0` (`complete_flow_burden_ratio.json:13-76`), but `difftest_reduction_model_calc.py` still hardcodes the older `8.0/55.0/40.0/1200.0` inputs at `artifacts/gpu_trace_frontend_difftest_necessity/difftest_reduction_model_calc.py:17-25`. That inconsistency is visible in `paper_argument_matrix.md:15-18`, where a `10.0s` frontend row claims only `2.4s` expected savings instead of `3.0s`, and a `205.0s` row claims `12.0s` instead of `61.5s`. The explicit `TRACE_SIZE_MAP` is a real improvement, but it is not enough to mark task 10 complete.

4. Medium: task 13 is still entirely unimplemented. `resource_bound_config.json` remains a null template with one `pending_measurement` record and no batch size, memory, trace-size, artifact-size, export-time, simulation-time, analysis-time, total-iteration-time, or stop-condition evidence at `artifacts/gpu_trace_frontend_difftest_necessity/resource_bound_config.json:32-45`. AC-11 requires actual scaling records or at least a measured first-batch failure; a blank template is not progress.

5. Medium: the new task-14 attempt artifact does not justify closing AC-10 yet. The plan requires the optional Llama full-step attempt only after the required evidence line is complete (`docs/superpowers/plans/2026-05-03-gpu-trace-frontend-difftest-style-optimization-necessity-plan.md:184-188`, task dependency at `:332-333`), but task 10 and task 13 are still unfinished. The recorded attempt also has `command_attempted: null` in `artifacts/gpu_trace_frontend_difftest_necessity/llama8b_full_step_attempt.json:3-10`, so it is a documented abandonment note rather than a reproducible tail execution record. The artifact is useful, but it does not satisfy AC-10's sequencing requirement, so task 14 should remain active.

## Goal Alignment Summary

`ACs: 11/11 addressed | Forgotten items: 0 | Unjustified deferrals: 0`

All 14 plan tasks are still tracked in the goal tracker, so there is no goal drift. Progress is real on AC-2 and AC-7: the simulator-side `warp_trace_build_s` refactor is present, and the evidence table no longer mis-anchors the BERT encoder row to `10.0 GiB`. The remaining gaps are still concentrated in AC-4, AC-5, AC-7, AC-10, and AC-11: measured artifacts are not yet consumable end-to-end, the runtime redundancy ratios are still wrong, required evidence-table columns are still missing, the tail attempt was recorded before its prerequisites were finished, and the BERT batch-scaling record is still empty.

## Goal Tracker Update Assessment

I updated `.humanize/rlcr/2026-05-03_22-55-59/goal-tracker.md` to reflect the verified Round 3 state.

Approved in part:
- task 6 notes now reflect the real simulator-side improvement: `warp_trace_build_s` is directly accumulated and `frontend_total_s()` is non-overlapping.
- the tracker now records that a task-14 attempt artifact exists, rather than saying the tail task is still entirely missing.

Rejected:
- marking task 10 completed, because AC-7 still fails on missing `kernel_count` / `threadblock_or_warp_count` fields and stale reduction inputs.
- treating task 8 as design-spec-aligned, because the runtime C++ formulas still do not match the updated JSON spec.
- treating task 14 as complete, because AC-10 requires the optional attempt after the required evidence line is complete, and that precondition is still false.

## Implementation Plan

1. Unify the measurement artifact schema before doing any more study-side regeneration. Make the simulator-emitted `frontend_timing_breakdown.json` and `redundancy_profile.json` consumable by the study scripts as written, or update the study scripts to consume the current flat per-run JSON directly. Then verify with one bounded local run that `complete_flow_burden_ratio_calc.py` actually promotes at least one workload row from `placeholder`/`modeled` to `measured`.

2. Repair the runtime redundancy metrics instead of only editing the spec stub. Add the missing `unique_tb_metadata_shape_count`-style source data or explicitly rename the study metric if the simulator cannot produce it yet; then make `tb_metadata_reuse_ratio` and `frontend_allocation_density` use the same denominators in both `frontend_metrics.h` and `redundancy_profile.json`.

3. Finish task 10 from the artifact layer outward. Populate `kernel_count` and `threadblock_or_warp_count` from the measured workload manifests and redundancy/timing artifacts, include the full AC-7 columns in both JSON and Markdown outputs, and regenerate the reduction model from the same `T_trace_to_sim` values used by `complete_flow_burden_ratio.json` so the evidence table is internally consistent.

4. Complete task 13 with actual BERT batch-scaling evidence. Run the smallest allowed BERT pretraining full-step configuration first; if it fails the ceiling immediately, record that measured failure. Otherwise scale by powers of two and fill `resource_bound_config.json/.md` with the real batch size, memory, trace size, artifact size, export time, simulation time, analysis time, total iteration time, and stop condition for each attempt.

5. Re-run task 14 only after tasks 10 and 13 are complete. Record the actual command, the failure stage, and any partial artifacts in the attempt report so AC-10 is satisfied as a reproducible tail attempt rather than a forward-looking note.
