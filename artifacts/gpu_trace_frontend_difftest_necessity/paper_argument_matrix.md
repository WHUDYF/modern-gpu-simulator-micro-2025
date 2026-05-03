# GPU Trace Frontend Necessity: Central Evidence Table

Generated: 2026-05-03

## Go/No-Go

**Verdict**: PENDING_MEASUREMENT
- Rule: P_trace_to_sim_slice > 15% OR P_trace_to_sim_step > 15%
- Detail: All inputs are placeholder or modeled. Run simulator instrumentation to obtain measured data.

## Evidence Rows

| Workload | Unit | Trace Size (GiB) | TB Count | Warp Count | T_frontend (s) | T_total (s) | P_frontend (%) | Est. Reduction (s) | Impact |
|----------|------|-----------------|----------|------------|---------------|------------|---------------|-------------------|--------|
| BERT-base (encoder layer slice) | slice | 0.5 (modeled) | None | None | 10.0 (placeholder) | 31.0 (placeholder) | 32.26 (placeholder) | 3.0 (modeled) | Expected 30% reduction saves 3.0s per run |
| BERT-base (pretraining full step) | step | 10.0 (modeled) | None | None | 105.0 (placeholder) | 435.0 (placeholder) | 24.14 (placeholder) | 31.5 (modeled) | Expected 30% reduction saves 31.5s per run |
| Llama 3.1 8B (decoder layer slice) | slice | 20.0 (modeled) | None | None | 205.0 (modeled) | 300.0 (modeled) | 68.33 (modeled) | 61.5 (modeled) | Expected 30% reduction saves 61.5s per run |
| Llama 3.1 8B (pretraining full step) | step | 100.0 (modeled) | None | None | 1005.0 (modeled) | 10905.0 (modeled) | 9.22 (modeled) | 301.5 (modeled) | Expected 30% reduction saves 301.5s per run |

## Control Workloads (Measured, from Existing Bottleneck Map)

| Suite | Representative Case | Trace Size (MiB) | Export (s) | Sim (s) |
|-------|-------------------|-----------------|-----------|--------|
| GPU_Microbenchmark | MaxFlops | 8.715 | 3.52 | 1.53 |
| GPU_Microbenchmark | atomic_add_bw | 5.423 | 2.27 | 10.13 |
| GPU_Microbenchmark | atomic_add_bw_conflict | 3.112 | 2.13 | 10.09 |
| GPU_Microbenchmark | atomic_add_lat | 0.288 | 2.11 | 1.26 |
| GPU_Microbenchmark | l1_bw_128 | 24.038 | 6.59 | 2.13 |
| GPU_Microbenchmark | l1_bw_32f | 2.962 | 2.24 | 1.46 |
| GPU_Microbenchmark | l1_bw_32f_unroll | 4.544 | 2.56 | 1.49 |
| GPU_Microbenchmark | l1_bw_32f_unroll_large | 7.511 | 2.79 | 1.69 |
| GPU_Microbenchmark | l1_bw_64f | 3.026 | 2.29 | 1.49 |
| GPU_Microbenchmark | l1_lat | 37.759 | 31.05 | 1.57 |

## Data Provenance

All evidence rows are merged from:
- `workload_evidence_table.json` — workload definitions
- `complete_flow_burden_ratio.json` — timing and burden ratios
- `difftest_reduction_model.json` — reduction estimates
- `trace_to_sim_formula.json` — formula-based estimates for scale anchors

Labels reflect whether data is `measured` (from simulator instrumentation), `modeled` (from planning formula), or `pending` (not yet available).
