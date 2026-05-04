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

- Slice-level gate (measured only): P_trace_to_sim_slice > 15% → PASS (max: 55.2%)
- Step-level gate (measured only): P_trace_to_sim_step > 15% → NOT YET (max: N/A (no measured step rows))
- **Overall go/no-go**: GO — proceed to prototype investigation
- Note: Only measured claim-bearing rows drive the verdict. Modeled and placeholder rows are present for context but excluded from go/no-go computation.

## Per-Workload Results

| Workload | Unit | T_export (s) | T_frontend (s) | T_backend (s) | T_analysis (s) | T_total (s) | P_frontend (%) | Data Label |
|----------|------|-------------|---------------|--------------|---------------|-----------|---------------|------------|
| bert-base-encoder-layer-slice | slice | 5.0 | 25.9 | 15.0 | 1.0 | 46.9 | 55.2 | measured |
| bert-base-pretraining-full-step | step | 120.0 | 105.0 | 200.0 | 10.0 | 435.0 | 24.1 | placeholder |
| llama3.1-8b-decoder-layer-slice | slice | 30.0 | 205.0 | 60.0 | 5.0 | 300.0 | 68.3 | modeled |
| llama3.1-8b-full-step | step | 3600.0 | 1005.0 | 6000.0 | 300.0 | 10905.0 | 9.2 | modeled |

## Sweep-Level Cumulative Cost (Expected Scenario, Placeholder Values)

| Workload | Single-Run Total (s) | Runs per Sweep (est.) | Sweep Total (s) | Sweep Total (min) |
|----------|---------------------|----------------------|-----------------|-------------------|
| bert-base-encoder-layer-slice | 46.9 | 10 | 468.7 | 7.8 |
| bert-base-pretraining-full-step | 435.0 | 5 | 2175.0 | 36.2 |
| llama3.1-8b-decoder-layer-slice | 300.0 | 10 | 3000.0 | 50.0 |
| llama3.1-8b-full-step | 10905.0 | 2 | 21810.0 | 363.5 |

## Notes

- All values labeled `placeholder` must be replaced with measured data from timing instrumentation.
- Values labeled `modeled` are estimates for workloads not yet directly measured.
- The go/no-go rule uses an early-stage engineering threshold of 15%, not a final paper claim threshold.
- Sweep run counts are planning estimates.
