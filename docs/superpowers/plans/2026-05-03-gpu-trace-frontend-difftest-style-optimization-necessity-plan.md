# GPU Trace Frontend DiffTest-Style Optimization Necessity Plan

## Goal Description

Establish a reproducible evidence pipeline for deciding whether DiffTest-style frontend input restructuring is necessary and feasible for the trace-driven GPU simulator.

The plan turns the design spec into implementation steps for measuring `T_trace_to_sim`, building AI-training-oriented workload evidence, estimating DiffTest-style reductions, and defining a safe prototype boundary at the `trace-parser` and `trace-driven` interface. The first deliverable is not a simulator optimization. The first deliverable is a falsifiable necessity study that shows whether trace-to-simulator frontend preparation is a material obstacle in the end-to-end algorithm-to-simulator loop.

Quantitative thresholds from the spec, including 30-60 seconds single-run cost, 10 minutes to 1 hour sweep cost, 15% / 30% / 50% reduction scenarios, and `P_trace_to_sim` bands, are planning and modeling thresholds. They are not hard performance guarantees. Later measured data may calibrate them.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Workload catalog covers AI training scale tiers and separates controls from claim-bearing workloads.
  - Positive Tests (expected to PASS):
    - The workload table includes at least one T0 sanity workload, one T1 local workload, one T2 representative workload, and one T3 scale anchor.
    - Each workload row records model or slice name, approximate scale, trace granularity, expected trace-size tier, and role in the argument.
    - Existing microbenchmark and export-dominated cases are listed as controls or appendix material, not as primary AI-training evidence.
  - Negative Tests (expected to FAIL):
    - A workload catalog that contains only microbenchmarks and no AI-training or training-adjacent traces is rejected.
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

- AC-4: End-to-end burden ratio is computed separately from simulator backend dominance.
  - Positive Tests (expected to PASS):
    - The evidence pipeline computes `P_trace_to_sim = T_trace_to_sim / T_e2e_iteration`.
    - `T_e2e_iteration` includes `T_trace_export`, `T_trace_to_sim`, `T_sim_backend`, and `T_result_analysis`.
    - The report classifies burden bands as `<1%`, `1%-5%`, `5%-15%`, and `>15%`, while preserving absolute time and sweep-level cumulative time.
  - Negative Tests (expected to FAIL):
    - A study that requires frontend optimization to beat backend simulation acceleration before it is considered useful is rejected.
    - A study that ignores absolute time and only reports percentage share is rejected.

- AC-5: Redundancy profiling measures whether DiffTest-style caching and chunking have a local opportunity.
  - Positive Tests (expected to PASS):
    - The profile reports dynamic instruction count, unique `(unique_function_id, pc)` count, static-info lookup count, threadblock count, warp trace count, metadata object construction count, and frontend allocation count where available.
    - The profile computes `static_reuse_ratio`, `tb_metadata_reuse_ratio`, and `frontend_allocation_density`.
    - At least one AI-training workload or model slice shows whether repeated static binding is large enough to justify a frontend cache prototype.
  - Negative Tests (expected to FAIL):
    - A profile that assumes repetition without measuring unique static identifiers is rejected.
    - A profile that treats dynamic instruction squash as an allowed first-phase optimization is rejected.

- AC-6: DiffTest-style reduction model applies only to `T_trace_to_sim`.
  - Positive Tests (expected to PASS):
    - The model computes conservative 15%, expected 30%, and optimistic 50% reductions against `T_trace_to_sim` only.
    - The output includes reduced `T_trace_to_sim`, saved time per run, and saved time per sweep.
    - The report labels the reduction model as planning evidence until prototype measurements replace it.
  - Negative Tests (expected to FAIL):
    - A model that applies the reduction rate to total simulator wall time without justification is rejected.
    - A model that presents the 15% / 30% / 50% scenarios as measured speedups before implementation is rejected.

- AC-7: Central evidence table connects workload size, frontend cost, modeled savings, and paper argument.
  - Positive Tests (expected to PASS):
    - The evidence table includes workload, model slice, trace size, kernel count, threadblock or warp count, `T_trace_to_sim`, `T_sim_total`, frontend share, estimated DiffTest-style reduction, reduced `T_trace_to_sim`, and E2E impact.
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
    - Required artifacts include workload evidence, trace-to-sim formula, E2E burden ratio, frontend timing breakdown, redundancy profile, DiffTest reduction model, prototype equivalence report, and paper argument matrix.
    - JSON artifacts are machine-readable and Markdown artifacts are suitable for paper or thesis discussion.
  - Negative Tests (expected to FAIL):
    - Results stored only in ad hoc console output are rejected.
    - Artifacts without enough metadata to reproduce scenario assumptions are rejected.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The implementation may include a complete measurement and reporting pipeline, workload catalog, trace-size calculator, E2E burden ratio calculator, redundancy profiler, DiffTest-style reduction model, and a minimal no-semantics prototype with equivalence reporting. It may add simulator-side timing counters and artifact generation utilities as long as changes remain at the trace parser and trace-driven frontend boundary.

### Lower Bound (Minimum Acceptable Scope)

The minimum acceptable implementation produces the workload catalog, trace-size formula calculator, E2E burden ratio report, DiffTest-style reduction table, and central evidence table with clear measured versus modeled labels. It must be sufficient to decide whether frontend restructuring is worth a prototype, even before the prototype exists.

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
5. Compute E2E burden ratio and sweep-level cumulative cost.
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

P_trace_to_sim = T_trace_to_sim / T_e2e_iteration
```

### Relevant References

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
   - Define T0, T1, T2, and T3 workload rows.
   - Mark microbenchmark and export-dominated cases as controls.
   - Record trace-size tiers and measurement feasibility.

2. Formula and burden modeling
   - Implement the trace-size to `T_trace_to_sim` calculator.
   - Generate fast, expected, and pessimistic scenarios.
   - Compute `P_trace_to_sim` and sweep-level cumulative cost.

3. Timing decomposition instrumentation
   - Locate existing parser and trace-driven frontend boundaries.
   - Add low-overhead timers and counters around read, parse, bind, load, warp trace build, frontend delivery, and core cycle timing.
   - Emit per-run JSON records.

4. Redundancy profiling
   - Count unique static identifiers, dynamic instructions, threadblocks, warp traces, metadata constructions, and frontend allocations.
   - Compute reuse and allocation-density ratios.
   - Compare AI-training traces against control workloads.

5. Evidence table and paper argument matrix
   - Merge workload catalog, timing breakdown, formula estimates, burden ratios, redundancy metrics, and reduction estimates.
   - Generate Markdown and JSON reports.
   - Connect XiangShan DiffTest, Accel-Sim, gem5 TraceCPU, ChampSim, SMARTS, and SimPoint to local evidence requirements.

6. Minimal no-semantics prototype decision
   - Proceed only if evidence satisfies the necessity criteria.
   - Limit implementation to decoded static-info cache, metadata normalization cache, threadblock chunk staging, and local replay.
   - Run equivalence checks before any performance claim.

## Task Breakdown

Each task includes exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Create workload catalog schema and seed rows for T0, T1, T2, T3, plus control workloads | AC-1 | coding | - |
| task2 | Inspect existing bottleneck map artifacts and map reusable fields into the new evidence schema | AC-1, AC-7 | analyze | task1 |
| task3 | Implement trace-size formula calculator and generate Markdown / JSON planning tables | AC-3 | coding | task1 |
| task4 | Implement E2E burden ratio calculator using explicit export, frontend, backend, and analysis fields | AC-4 | coding | task3 |
| task5 | Identify parser and trace-driven instrumentation points for timing decomposition | AC-2 | analyze | task1 |
| task6 | Add or wire low-overhead timing counters and per-run frontend timing JSON output | AC-2, AC-9 | coding | task5 |
| task7 | Identify available counters or insertion points for redundancy profiling | AC-5 | analyze | task5 |
| task8 | Add redundancy profile output for static reuse, threadblock metadata reuse, and frontend allocation density | AC-5, AC-9 | coding | task7 |
| task9 | Implement DiffTest-style reduction model that applies only to `T_trace_to_sim` | AC-6 | coding | task3, task4 |
| task10 | Build central evidence table generator with measured versus modeled labels | AC-7, AC-9 | coding | task2, task4, task6, task8, task9 |
| task11 | Draft paper argument matrix connecting external examples to local GPU simulator evidence | AC-7, AC-9 | analyze | task10 |
| task12 | Define minimal no-semantics prototype gate and equivalence-report checklist | AC-8 | coding | task10, task11 |

## Claude-Codex Deliberation

### Agreements

- The first milestone should prove necessity and feasibility before implementing simulator optimizations.
- The DiffTest analogy should be limited to structured event transfer, caching, batching, validation, and replay.
- The local optimization boundary should stay at `trace-parser -> trace-driven -> shader core` and avoid backend timing semantics.
- Quantitative thresholds are useful as planning thresholds and should be calibrated with measurements.

### Resolved Disagreements

- Frontend dominance versus design-loop obstruction: the chosen resolution is to prove that `T_trace_to_sim` is large enough to obstruct end-to-end iteration, not that it dominates every simulator bottleneck.
- Measured-only evidence versus modeled scale anchors: the chosen resolution is to require measured local workloads while allowing explicitly labeled modeled values for T2 and T3 large-scale anchors.

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- No pending user decisions. The current plan treats quantitative thresholds as planning and modeling thresholds, not hard performance guarantees.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must not contain plan-specific terminology such as `AC-`, `Milestone`, `Step`, `Phase`, or similar workflow markers.
- Use descriptive, domain-appropriate names in code and artifacts.
- Keep measurement overhead low and report any measurement overhead if it becomes visible.
- Preserve baseline simulator output semantics before making any performance claim.
- Clearly label measured, modeled, extrapolated, and control values in every generated report.

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
- `e2e_burden_ratio.md`
- `e2e_burden_ratio.json`
- `frontend_timing_breakdown.md`
- `frontend_timing_breakdown.json`
- `redundancy_profile.md`
- `redundancy_profile.json`
- `difftest_reduction_model.md`
- `difftest_reduction_model.json`
- `prototype_equivalence_report.md`
- `prototype_equivalence_report.json`
- `paper_argument_matrix.md`
