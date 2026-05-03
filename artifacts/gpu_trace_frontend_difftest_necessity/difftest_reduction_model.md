# DiffTest-Style Reduction Model (Planning Evidence)

Generated: 2026-05-03

**Status**: Planning evidence — to be replaced by prototype measurements.

## Scope

- Reductions apply **only to `T_trace_to_sim`** (trace read + protobuf parse + static bind + threadblock/warp load + frontend delivery prep).
- Backend execution time, total wall time, and SM timing semantics are **unchanged**.
- These are **planning scenarios**, not measured speedups.

## Reduction Scenarios

| Scenario | Reduction | Rationale |
|----------|----------|-----------|
| Conservative | 15% | Conservative: minor batching and caching wins |
| Expected | 30% | Expected: structured chunking and static reuse |
| Optimistic | 50% | Optimistic: aggressive batching, caching, and delta encoding |

## Per-Workload Reduction Table

| Workload | Orig T_frontend (s) | Scenario | Reduced T_frontend (s) | Saved/Run (s) | Saved/Sweep (s) |
|----------|--------------------|----------|----------------------|--------------|----------------|
| bert-base-encoder-layer-slice | 8.0 | conservative 15% | 6.8 | 1.2 | 12.0 |
| bert-base-encoder-layer-slice | 8.0 | expected 30% | 5.6 | 2.4 | 24.0 |
| bert-base-encoder-layer-slice | 8.0 | optimistic 50% | 4.0 | 4.0 | 40.0 |

| bert-base-pretraining-full-step | 55.0 | conservative 15% | 46.8 | 8.2 | 41.0 |
| bert-base-pretraining-full-step | 55.0 | expected 30% | 38.5 | 16.5 | 82.5 |
| bert-base-pretraining-full-step | 55.0 | optimistic 50% | 27.5 | 27.5 | 137.5 |

| llama3.1-8b-decoder-layer-slice | 40.0 | conservative 15% | 34.0 | 6.0 | 60.0 |
| llama3.1-8b-decoder-layer-slice | 40.0 | expected 30% | 28.0 | 12.0 | 120.0 |
| llama3.1-8b-decoder-layer-slice | 40.0 | optimistic 50% | 20.0 | 20.0 | 200.0 |

| llama3.1-8b-full-step | 1200.0 | conservative 15% | 1020.0 | 180.0 | 360.0 |
| llama3.1-8b-full-step | 1200.0 | expected 30% | 840.0 | 360.0 | 720.0 |
| llama3.1-8b-full-step | 1200.0 | optimistic 50% | 600.0 | 600.0 | 1200.0 |

| t2-scale-anchor-100GiB | 1005.0 | conservative 15% | 854.2 | 150.8 | 452.4 |
| t2-scale-anchor-100GiB | 1005.0 | expected 30% | 703.5 | 301.5 | 904.5 |
| t2-scale-anchor-100GiB | 1005.0 | optimistic 50% | 502.5 | 502.5 | 1507.5 |

| t3-scale-anchor-500GiB | 5005.0 | conservative 15% | 4254.2 | 750.8 | 2252.4 |
| t3-scale-anchor-500GiB | 5005.0 | expected 30% | 3503.5 | 1501.5 | 4504.5 |
| t3-scale-anchor-500GiB | 5005.0 | optimistic 50% | 2502.5 | 2502.5 | 7507.5 |

## Summary of Impact (Expected Scenario, 30% Reduction)

| Workload | Single-Run Savings (s) | Sweep Savings (s) | Sweep Savings (min) |
|----------|----------------------|-------------------|--------------------|
| bert-base-encoder-layer-slice | 2.4 | 24.0 | 0.4 |
| bert-base-pretraining-full-step | 16.5 | 82.5 | 1.4 |
| llama3.1-8b-decoder-layer-slice | 12.0 | 120.0 | 2.0 |
| llama3.1-8b-full-step | 360.0 | 720.0 | 12.0 |
| t2-scale-anchor-100GiB | 301.5 | 904.5 | 15.1 |
| t3-scale-anchor-500GiB | 1501.5 | 4504.5 | 75.1 |

## Notes

- All values are **planning estimates**, not measured performance data.
- The reduction applies to the frontend path only; backend simulation time is unchanged.
- If prototype measurements later contradict these estimates, the estimates must be recalibrated.
- Conservative (15%) = minimal wins from batching + caching.
- Expected (30%) = structured chunking + static-info reuse.
- Optimistic (50%) = aggressive batching + delta encoding + caching.
