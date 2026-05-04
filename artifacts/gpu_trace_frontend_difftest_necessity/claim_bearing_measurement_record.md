# Claim-Bearing Measurement Record

**Date**: 2026-05-04
**Gate**: Gate C — Claim-Bearing Measurement Complete

## Workload: BERT-base Encoder Layer Slice

### Complete-Flow Timing

| Component | Time (s) | Label |
|-----------|---------|-------|
| T_kernel_or_trace_export | 180.0 | measured |
| T_trace_to_sim | 25.87 | measured |
| T_sim_backend_execution | 19.05 | derived |
| T_result_analysis | 0.25 | measured |
| **T_kernel_to_sim_done** | **225.17** | — |

### P_trace_to_sim

- Complete flow: 25.87 / 225.17 = **11.5%** (< 15% threshold)
- Simulator-only frontend_share: 25.87 / 44.92 = **57.6%**

The complete-flow metric is below 15% because trace generation (180s) dominates. However, inside the simulator itself, the frontend consumes 57.6% of wall time.

### Frontend Timing Breakdown

| Component | Time (s) |
|-----------|---------|
| trace_read | 0.00015 |
| parse_pb | 5.79 |
| static_bind | 6.71 |
| tb_load | 0.00027 |
| warp_trace_build | 7.74 |
| get_next_inst | 5.63 |

### Interpretation

The go/no-go is INCONCLUSIVE: the simulator frontend dominates (57.6%) and justifies optimization, but the end-to-end flow is dominated by trace generation (180s). Frontend restructuring would improve simulator iteration speed but would not materially reduce the end-to-end burden unless trace generation is also optimized.
