# Complete-Flow Burden Ratio Report

Generated: 2026-05-04

**Data Status**: All inputs are placeholder or modeled. Measured data from simulator instrumentation required for valid go/no-go.

## Formula

```text
P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done

T_kernel_to_sim_done =
  T_kernel_or_trace_export
+ T_trace_to_sim
+ T_sim_backend_execution
+ T_result_analysis
```

## Go/No-Go Rule

- Slice-level gate (measured only): P_trace_to_sim_slice > 15% → NOT YET (max: 11.5%)
- Step-level gate (measured only): P_trace_to_sim_step > 15% → NOT YET (max: N/A (no measured step rows))
- **Overall go/no-go**: NOT YET — gather measured data first
- Note: A row counts as fully measured only when T_kernel_or_trace_export, T_trace_to_sim, T_sim_backend_execution, AND T_result_analysis are all measured.

## Per-Workload Results

| Workload | Unit | T_export (s) | T_frontend (s) | T_backend (s) | T_analysis (s) | T_total (s) | P_frontend (%) | Data Label |
|----------|------|-------------|---------------|--------------|---------------|-----------|---------------|------------|
| bert-base-encoder-layer-slice | slice | 180.0 | 25.9 | 19.1 | 0.2 | 225.2 | 11.5 | measured |
| bert-base-pretraining-full-step | step | 120.0 | 105.0 | 200.0 | 5.0 | 430.0 | 24.4 | placeholder |
| llama3.1-8b-decoder-layer-slice | slice | 300.0 | 205.0 | 60.0 | 5.0 | 570.0 | 36.0 | modeled |
| llama3.1-8b-full-step | step | 3600.0 | 1005.0 | 6000.0 | 5.0 | 10610.0 | 9.5 | modeled |

## Sweep-Level Cumulative Cost (Expected Scenario, Placeholder Values)

| Workload | Single-Run Total (s) | Runs per Sweep (est.) | Sweep Total (s) | Sweep Total (min) |
|----------|---------------------|----------------------|-----------------|-------------------|
| bert-base-encoder-layer-slice | 225.2 | 10 | 2251.7 | 37.5 |
| bert-base-pretraining-full-step | 430.0 | 5 | 2150.0 | 35.8 |
| llama3.1-8b-decoder-layer-slice | 570.0 | 10 | 5700.0 | 95.0 |
| llama3.1-8b-full-step | 10610.0 | 2 | 21220.0 | 353.7 |

## Notes

- All values labeled `placeholder` must be replaced with measured data from timing instrumentation.
- Values labeled `modeled` are estimates for workloads not yet directly measured.
- The go/no-go rule uses an early-stage engineering threshold of 15%, not a final paper claim threshold.
- Sweep run counts are planning estimates.
