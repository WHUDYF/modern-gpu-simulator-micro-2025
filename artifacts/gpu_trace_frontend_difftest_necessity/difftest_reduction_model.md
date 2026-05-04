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
| bert-base-encoder-layer-slice | 25.9 | conservative 15% | 22.0 | 3.9 | 39.0 |
| bert-base-encoder-layer-slice | 25.9 | expected 30% | 18.1 | 7.8 | 78.0 |
| bert-base-encoder-layer-slice | 25.9 | optimistic 50% | 12.9 | 12.9 | 129.0 |

| bert-base-pretraining-full-step | 105.0 | conservative 15% | 89.2 | 15.8 | 79.0 |
| bert-base-pretraining-full-step | 105.0 | expected 30% | 73.5 | 31.5 | 157.5 |
| bert-base-pretraining-full-step | 105.0 | optimistic 50% | 52.5 | 52.5 | 262.5 |

| llama3.1-8b-decoder-layer-slice | 205.0 | conservative 15% | 174.2 | 30.8 | 308.0 |
| llama3.1-8b-decoder-layer-slice | 205.0 | expected 30% | 143.5 | 61.5 | 615.0 |
| llama3.1-8b-decoder-layer-slice | 205.0 | optimistic 50% | 102.5 | 102.5 | 1025.0 |

| llama3.1-8b-full-step | 1005.0 | conservative 15% | 854.2 | 150.8 | 301.6 |
| llama3.1-8b-full-step | 1005.0 | expected 30% | 703.5 | 301.5 | 603.0 |
| llama3.1-8b-full-step | 1005.0 | optimistic 50% | 502.5 | 502.5 | 1005.0 |

| t2-scale-anchor-100GiB | 1005.0 | conservative 15% | 854.2 | 150.8 | 452.4 |
| t2-scale-anchor-100GiB | 1005.0 | expected 30% | 703.5 | 301.5 | 904.5 |
| t2-scale-anchor-100GiB | 1005.0 | optimistic 50% | 502.5 | 502.5 | 1507.5 |

| t3-scale-anchor-500GiB | 5005.0 | conservative 15% | 4254.2 | 750.8 | 2252.4 |
| t3-scale-anchor-500GiB | 5005.0 | expected 30% | 3503.5 | 1501.5 | 4504.5 |
| t3-scale-anchor-500GiB | 5005.0 | optimistic 50% | 2502.5 | 2502.5 | 7507.5 |

## Summary of Impact (Expected Scenario, 30% Reduction)

| Workload | Single-Run Savings (s) | Sweep Savings (s) | Sweep Savings (min) |
|----------|----------------------|-------------------|--------------------|
| bert-base-encoder-layer-slice | 7.8 | 78.0 | 1.3 |
| bert-base-pretraining-full-step | 31.5 | 157.5 | 2.6 |
| llama3.1-8b-decoder-layer-slice | 61.5 | 615.0 | 10.2 |
| llama3.1-8b-full-step | 301.5 | 603.0 | 10.1 |
| t2-scale-anchor-100GiB | 301.5 | 904.5 | 15.1 |
| t3-scale-anchor-500GiB | 1501.5 | 4504.5 | 75.1 |

## Notes

- All values are **planning estimates**, not measured performance data.
- The reduction applies to the frontend path only; backend simulation time is unchanged.
- If prototype measurements later contradict these estimates, the estimates must be recalibrated.
- Conservative (15%) = minimal wins from batching + caching.
- Expected (30%) = structured chunking + static-info reuse.
- Optimistic (50%) = aggressive batching + delta encoding + caching.
