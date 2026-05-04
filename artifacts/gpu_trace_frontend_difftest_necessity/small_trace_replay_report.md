# Small Trace Replay Report

**Date**: 2026-05-04
**Gate**: Gate B — Small Trace Replay Ready
**Status**: PASS

## Workload

BERT-base encoder layer training slice (BertLayer forward+backward).

## Simulator Config

- GPGPU-Sim config: SM120_RTX5090
- Bound: 50000 cycles
- OMP_NUM_THREADS: 4
- Simulation time: 44 sec

## Results

| Metric | Value |
|--------|-------|
| Kernels simulated | 9 of 77 |
| Dynamic instructions | 2,035,521 |
| Frontend share | 57.6% |
| T_trace_to_sim | 25.87s |
| T_total | 44.92s |

`Frontend share` is a simulator-local wall-time share for this bounded replay window. The official 15% go/no-go gate uses complete-flow `P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done`.

## Timing Breakdown

| Component | Time (s) |
|-----------|---------|
| trace_read | 0.00015 |
| parse_pb | 5.79 |
| static_bind | 6.71 |
| tb_load | 0.00027 |
| warp_trace_build | 7.74 |
| get_next_inst | 5.63 |

## Redundancy

| Counter | Value |
|---------|-------|
| Kernel count | 9 |
| Threadblocks | 3,217 |
| Warp traces | 25,732 |
| Unique static IDs | 7,960,440 |
| Static reuse ratio | 0.26 |
| TB metadata reuse ratio | 643.4 |
