# Complete-Flow Burden Ratio Report

Generated: 2026-05-04

**Data Status**: 3 fully measured claim-bearing row(s) available for go/no-go.

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

- Slice-level gate (measured only): P_trace_to_sim_slice > 15% -> FAIL (max: 11.5%)
- Step-level gate (measured only): P_trace_to_sim_step > 15% -> FAIL (max: 0.0%)
- **Overall go/no-go**: NO-GO — frontend prototype not justified by current measured evidence
- Note: A row counts as fully measured only when T_kernel_or_trace_export, T_trace_to_sim, T_sim_backend_execution, AND T_result_analysis are all measured.

## Per-Workload Results

| Workload | Unit | T_export (s) | T_frontend (s) | T_backend (s) | T_analysis (s) | T_total (s) | P_frontend (%) | Row Data Label | Component Labels |
|----------|------|-------------|---------------|--------------|---------------|-----------|---------------|----------------|------------------|
| bert-base-encoder-layer-slice | slice | 180.0 | 25.9 | 19.1 | 0.2 | 225.2 | 11.5 | measured | T_kernel_or_trace_export=measured, T_trace_to_sim=measured, T_sim_backend_execution=measured, T_result_analysis=measured |
| bert-base-pretraining-full-step | step | 2805.6 | 0.1 | 7.1 | 0.2 | 2813.0 | 0.0 | measured | T_kernel_or_trace_export=measured, T_trace_to_sim=measured, T_sim_backend_execution=measured, T_result_analysis=measured |
| llama3.1-8b-decoder-layer-slice | slice | 1921.6 | 1.8 | 202.8 | 0.2 | 2126.4 | 0.1 | measured | T_kernel_or_trace_export=measured, T_trace_to_sim=measured, T_sim_backend_execution=measured, T_result_analysis=measured |
| llama3.1-8b-full-step | step | 3600.0 | 1005.0 | 6000.0 | 5.0 | 10610.0 | 9.5 | modeled | T_kernel_or_trace_export=modeled, T_trace_to_sim=modeled, T_sim_backend_execution=modeled, T_result_analysis=modeled |

## Sweep-Level Cumulative Cost (Expected Scenario, Placeholder Values)

| Workload | Single-Run Total (s) | Runs per Sweep (est.) | Sweep Total (s) | Sweep Total (min) |
|----------|---------------------|----------------------|-----------------|-------------------|
| bert-base-encoder-layer-slice | 225.2 | 10 | 2251.7 | 37.5 |
| bert-base-pretraining-full-step | 2813.0 | 5 | 14065.0 | 234.4 |
| llama3.1-8b-decoder-layer-slice | 2126.4 | 10 | 21264.1 | 354.4 |
| llama3.1-8b-full-step | 10610.0 | 2 | 21220.0 | 353.7 |

## Notes

- Rows labeled `measured` are eligible for the go/no-go rule.
- Rows labeled `placeholder` or `modeled` are planning context only and do not drive the verdict.
- The go/no-go rule uses an early-stage engineering threshold of 15%, not a final paper claim threshold.
- Sweep run counts are planning estimates.
