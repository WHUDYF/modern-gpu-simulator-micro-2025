# GPU Trace Frontend Necessity: Central Evidence Table

Generated: 2026-05-04

## Go/No-Go

**Verdict**: NO-GO
- Rule: P_trace_to_sim_slice > 15% OR P_trace_to_sim_step > 15%
- Eligible measured claim-bearing rows: 1
- Slice max P_trace_to_sim: 11.5%
- Step max P_trace_to_sim: 0.0%
- Detail: Fully measured claim-bearing rows exist, but none exceed the 15% threshold.

## Evidence Rows

| Workload ID | Workload | Unit | Data Label | Trace Size (GiB) | Kernels | TB Count | Warp Count | T_frontend (s) | T_total (s) | P_frontend (%) | Reduced T_frontend (s) | Impact |
|-------------|----------|------|------------|-----------------|---------|----------|------------|---------------|------------|---------------|----------------------|--------|
| bert-base-encoder-layer-slice | BERT-base (encoder layer slice) | slice | measured | 4.36 (measured) | 9 (measured) | 3217 (measured) | 25732 (measured) | 25.87 (measured) | 225.17 (measured) | 11.49 (measured) | 18.10 (modeled) | Expected 30% reduction saves 7.8s per run |
| bert-base-pretraining-full-step | BERT-base (pretraining full step) | step | placeholder | 10.00 (modeled) | 180 (modeled) | 1500 (modeled) | 48000 (modeled) | 105.00 (placeholder) | 430.00 (placeholder) | 24.42 (placeholder) | 73.50 (modeled) | Expected 30% reduction saves 31.5s per run |
| llama3.1-8b-decoder-layer-slice | Llama 3.1 8B (decoder layer slice) | slice | modeled | 20.00 (modeled) | 24 (modeled) | 3200 (modeled) | 102400 (modeled) | 205.00 (modeled) | 570.00 (modeled) | 35.96 (modeled) | 143.50 (modeled) | Expected 30% reduction saves 61.5s per run |
| llama3.1-8b-full-step | Llama 3.1 8B (pretraining full step) | step | modeled | 100.00 (modeled) | 360 (modeled) | 48000 (modeled) | 1536000 (modeled) | 1005.00 (modeled) | 10610.00 (modeled) | 9.47 (modeled) | 703.50 (modeled) | Expected 30% reduction saves 301.5s per run |

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

Only rows with `data_label = measured` and `claim_bearing = true` are eligible for the go/no-go rule.
Rows labeled `modeled`, `placeholder`, or `pending` are planning context only.
