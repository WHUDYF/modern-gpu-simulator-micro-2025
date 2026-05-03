# GPU Trace Frontend Necessity: Central Evidence Table

Generated: 2026-05-03

## Status

**All values with `placeholder` or `modeled` labels must be recalibrated with measured timing data from simulator instrumentation (task6).**

## Go/No-Go Summary

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| P_trace_to_sim (slice) | 27.6% (placeholder) | > 15% | GO |
| P_trace_to_sim (step) | 14.3% (placeholder) | > 15% | NOT YET |
| Overall | — | Slice OR Step | GO (pending measured data) |

## Evidence Table

| Workload | Unit | Trace Size (GiB) | T_frontend (s) | T_total (s) | P_frontend (%) | Est. Reduction (s) | Reduced T_frontend (s) | Impact |
|----------|------|-----------------|---------------|------------|---------------|-------------------|----------------------|--------|
| BERT-base encoder layer slice | slice | 0.5 (placeholder) | 8.0 (placeholder) | 29.0 (placeholder) | 27.6 (derived_placeholder) | 2.4 (modeled) | 5.6 (modeled) | Moderate: saves ~2.4s per run, ~24s per 10-run sweep at expected 30% reduction |
| BERT-base pretraining full step | step | 10.0 (placeholder) | 55.0 (placeholder) | 385.0 (placeholder) | 14.3 (derived_placeholder) | 16.5 (modeled) | 38.5 (modeled) | Significant: saves ~16.5s per run, ~82.5s per 5-run sweep at expected 30% reduction |
| Llama 3.1 8B decoder layer slice | slice | 20.0 (modeled) | 40.0 (modeled) | 135.0 (modeled) | 29.6 (derived_modeled) | 12.0 (modeled) | 28.0 (modeled) | Significant: saves ~12s per run, ~120s per 10-run sweep at expected 30% reduction |
| T2 scale anchor 100 GiB | modeled_anchor | 100.0 (modeled) | 1005.0 (modeled) | 4000.0 (modeled) | 25.1 (derived_modeled) | 301.5 (modeled) | 703.5 (modeled) | Major: saves ~301.5s per run, ~904.5s per 3-run sweep at expected 30% reduction |
| T3 scale anchor 500 GiB | modeled_anchor | 500.0 (modeled) | 5005.0 (modeled) | 20000.0 (modeled) | 25.0 (derived_modeled) | 1501.5 (modeled) | 3503.5 (modeled) | Critical: saves ~1501.5s (25 min) per run, ~4504.5s (75 min) per 3-run sweep |

## Control Workloads (from Existing Bottleneck Map)

| Suite | Representative Case | Trace Size (MiB) | Export (s) | Sim (s) | Dominant Bottleneck |
|-------|-------------------|-----------------|-----------|--------|--------------------|
| GPU_Microbenchmark | MaxFlops | 8.715 | 3.52 | 1.53 | balanced / mixed |
| GPU_Microbenchmark | atomic_add_bw | 5.423 | 2.27 | 10.13 | simulator throughput |
| GPU_Microbenchmark | atomic_add_bw_conflict | 3.112 | 2.13 | 10.09 | simulator throughput |
| GPU_Microbenchmark | atomic_add_lat | 0.288 | 2.11 | 1.26 | capture / fixed overhead |
| GPU_Microbenchmark | l1_bw_128 | 24.038 | 6.59 | 2.13 | trace export / I/O |
| GPU_Microbenchmark | l1_bw_32f | 2.962 | 2.24 | 1.46 | balanced / mixed |
| GPU_Microbenchmark | l1_bw_32f_unroll | 4.544 | 2.56 | 1.49 | balanced / mixed |
| GPU_Microbenchmark | l1_bw_32f_unroll_large | 7.511 | 2.79 | 1.69 | balanced / mixed |
| GPU_Microbenchmark | l1_bw_64f | 3.026 | 2.29 | 1.49 | balanced / mixed |
| GPU_Microbenchmark | l1_lat | 37.759 | 31.05 | 1.57 | trace export / I/O |

## Conclusion (Provisional)

Based on placeholder data:
- Slice-level P_trace_to_sim exceeds the 15% engineering gate, suggesting frontend restructuring is worth a prototype investigation.
- Step-level P_trace_to_sim is close to but below 15% with placeholder values; measured data may change this.
- Scale-anchor modeling suggests frontend cost grows linearly with trace size, making the optimization increasingly valuable at industrial scale.
- **Next step**: Replace all placeholder values with measured timing data from simulator instrumentation.

## Label Legend

| Label | Meaning |
|-------|---------|
| measured | Directly measured from simulator runs |
| modeled | Estimated from formula or planning model |
| derived_measured | Computed from measured inputs |
| derived_modeled | Computed from modeled inputs |
| placeholder | Placeholder value pending measurement |
| derived_placeholder | Computed from placeholder values |
| pending | Measurement not yet available |
| not_applicable | Not applicable for this workload |

## Paper Argument Matrix

### External References to Local Context

| External Reference | Domain | Argument Role | Local Connection |
|-------------------|--------|---------------|-----------------|
| XiangShan DiffTest | RISC-V co-simulation | Methodological precedent for structured event transfer | Batch/delta-cache/validate/replay map to trace-parser to trace-driven path |
| Accel-Sim Framework | GPU trace-driven simulation | Establishes mainstream relevance | Local simulator is enhanced Accel-Sim derivative |
| gem5 TraceCPU | CPU trace replay | Cross-architecture precedent for trace input as first-class boundary | GPU trace-driven frontend is the analogous boundary |
| ChampSim | CPU cache simulation | Industry precedent for trace input optimization | Trace reader as defined boundary between format and core |
| SMARTS | Simulation sampling | Background: cost reduction through representative subsets | Complementary to frontend acceleration; not implemented here |
| SimPoint | Simulation sampling | Background: representative execution interval selection | Complementary to frontend acceleration; not implemented here |

### Argument Structure

```
External precedent (XiangShan, Accel-Sim, gem5, ChampSim)
  → Trace input is a first-class simulation boundary
    → GPU trace frontend deserves the same attention
      → Local evidence pipeline measures T_trace_to_sim
        → If P_trace_to_sim > 15%, frontend restructuring is worth prototyping
          → Prototype scope: static-info cache, chunk staging, local replay
            → Equivalence check ensures no timing semantic change
```

### Paper Positioning

The contribution is applied methodology, not novel algorithm:

1. GPU trace-driven simulation has a measurable frontend cost that is under-studied relative to backend timing optimization.
2. A reproducible measurement pipeline quantifies this cost for AI-training workloads.
3. Structured event transfer ideas (DiffTest analogy) are adapted to the GPU trace frontend, with explicit scope boundaries preserving timing semantics.
4. Frontend optimization can reduce simulation iteration time by 15-50% of T_trace_to_sim, enabling faster design-space exploration.
