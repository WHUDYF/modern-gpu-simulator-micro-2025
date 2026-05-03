# Complete-Flow Burden Ratio Report

Generated: 2026-05-03

**Data Status**: Placeholder values — awaiting measured data from timing instrumentation.

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

- Slice-level gate: P_trace_to_sim_slice > 15% → PASS (max: 29.6%)
- Step-level gate: P_trace_to_sim_step > 15% → NOT YET (max: 14.3%)
- **Overall go/no-go**: GO — proceed to prototype investigation

## Per-Workload Results

| Workload | Unit | T_export (s) | T_frontend (s) | T_backend (s) | T_analysis (s) | T_total (s) | P_frontend (%) | Data Label |
|----------|------|-------------|---------------|--------------|---------------|-----------|---------------|------------|
| bert-base-encoder-layer-slice | slice | 5.0 | 8.0 | 15.0 | 1.0 | 29.0 | 27.6 | placeholder |
| bert-base-pretraining-full-step | step | 120.0 | 55.0 | 200.0 | 10.0 | 385.0 | 14.3 | placeholder |
| llama3.1-8b-decoder-layer-slice | slice | 30.0 | 40.0 | 60.0 | 5.0 | 135.0 | 29.6 | modeled |
| llama3.1-8b-full-step | step | 3600.0 | 1200.0 | 6000.0 | 300.0 | 11100.0 | 10.8 | modeled |
| bert-base-encoder-layer-slice-reduced | slice | 5.0 | 5.6 | 15.0 | 1.0 | 26.6 | 21.1 | modeled |
| bert-base-pretraining-full-step-reduced | step | 120.0 | 38.5 | 200.0 | 10.0 | 368.5 | 10.4 | modeled |

## Sweep-Level Cumulative Cost (Expected Scenario, Placeholder Values)

| Workload | Single-Run Total (s) | Runs per Sweep (est.) | Sweep Total (s) | Sweep Total (min) |
|----------|---------------------|----------------------|-----------------|-------------------|
| bert-base-encoder-layer-slice | 29.0 | 10 | 290.0 | 4.8 |
| bert-base-pretraining-full-step | 385.0 | 5 | 1925.0 | 32.1 |
| llama3.1-8b-decoder-layer-slice | 135.0 | 10 | 1350.0 | 22.5 |
| llama3.1-8b-full-step | 11100.0 | 2 | 22200.0 | 370.0 |
| bert-base-encoder-layer-slice-reduced | 26.6 | 10 | 266.0 | 4.4 |
| bert-base-pretraining-full-step-reduced | 368.5 | 5 | 1842.5 | 30.7 |

## Notes

- All values labeled `placeholder` must be replaced with measured data from timing instrumentation.
- Values labeled `modeled` are estimates for workloads not yet directly measured.
- The go/no-go rule uses an early-stage engineering threshold of 15%, not a final paper claim threshold.
- Sweep run counts are planning estimates.
