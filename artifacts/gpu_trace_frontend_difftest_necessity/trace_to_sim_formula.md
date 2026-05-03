# Trace-Size to T_trace_to_sim Formula Calculator

Generated: 2026-05-03

## Formula

```text
T_trace_to_sim ~= C_fixed + S_trace_GiB / R_frontend_GiBps
```

Expected shortcut: `T_trace_to_sim ~= 5 + 10 * S_trace_GiB seconds`

## Scenario Parameters

| Scenario | C_fixed (s) | R_frontend (GiB/s) | Time per GiB (s/GiB) |
|----------|------------|-------------------|---------------------|
| Fast (optimistic) | 2 | 0.50 | 2 |
| Expected (baseline) | 5 | 0.10 | 10 |
| Pessimistic (conservative) | 10 | 0.03 | 33 |

## Planning Table

| Trace Label | Size (GiB) | T (Fast, s) | T (Expected, s) | T (Pessimistic, s) | Expected Shortcut (s) |
|-------------|-----------|------------|----------------|-------------------|----------------------|
| micro (local) | 0.010 | 2.0 | 5.1 | 10.3 | 5.1 |
| small slice (local) | 0.100 | 2.2 | 6.0 | 13.3 | 6.0 |
| medium slice (local) | 0.500 | 3.0 | 10.0 | 26.7 | 10.0 |
| large slice / small step (local) | 2.000 | 6.0 | 25.0 | 76.7 | 25.0 |
| BERT-base full step (local) | 10.000 | 22.0 | 105.0 | 343.3 | 105.0 |
| Llama 3.1 8B layer slice (local) | 20.000 | 42.0 | 205.0 | 676.7 | 205.0 |
| T2 scale anchor (modeled) | 100.000 | 202.0 | 1005.0 | 3343.3 | 1005.0 |
| T3 scale anchor (modeled) | 500.000 | 1002.0 | 5005.0 | 16676.7 | 5005.0 |
| 1 TiB scale anchor (modeled) | 1024.000 | 2050.0 | 10245.0 | 34143.3 | 10245.0 |

## Notes

- All estimates are **planning and modeling thresholds**, not hard performance guarantees.
- The expected shortcut `5 + 10 * S_trace_GiB` reproduces the formula requirement from the acceptance criteria.
- Fast scenario assumes optimized frontend throughput (0.2 GiB/s, 5 s/GiB).
- Pessimistic scenario accounts for slow I/O, large metadata, or contention (0.05 GiB/s, 20 s/GiB).
- Values for T2/T3 scale anchors are **modeled**, not measured.
- Actual measured data will calibrate these parameters.

## Sweep-Level Cumulative Cost (Expected Scenario)

| Sweep Type | Runs per Sweep | Trace Size per Run (GiB) | T per Run (s) | Total Sweep T (s) | Total Sweep T (min) |
|-----------|---------------|------------------------|--------------|------------------|-------------------|
| micro slice sweep | 50 | 0.1 | 6.0 | 300.0 | 5.0 |
| medium slice sweep | 20 | 0.5 | 10.0 | 200.0 | 3.3 |
| large slice sweep | 10 | 2.0 | 25.0 | 250.0 | 4.2 |
| BERT-base full step sweep | 5 | 10.0 | 105.0 | 525.0 | 8.8 |
| scale-anchor modeled sweep | 3 | 100.0 | 1005.0 | 3015.0 | 50.2 |
