# Frontend Timing Instrumentation Specification

Generated: 2026-05-04

**Status**: Implemented. C++ instrumentation committed to simulator repo at
`/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/`
(commits 69b511d, acd325c, ed89dd2, 57677ea, efbd5f7).

## Timing Decomposition

```
T_trace_to_sim =
  T_trace_read
+ T_protobuf_parse
+ T_static_bind
+ T_threadblock_warp_load
+ T_frontend_instruction_delivery_preparation
```

## Instrumentation Points

| ID | Component | Expected Code Location | Timer Type |
|----|-----------|----------------------|------------|
| trace_read | T_trace_read | trace-parser: file I/O entry point | wall_clock |
| parse_pb | T_protobuf_parse | trace-parser: protobuf message parsing loop | wall_clock |
| static_bind | T_static_bind | trace-parser: static info lookup / metadata binding | wall_clock |
| tb_load | T_threadblock_warp_load | trace-driven: threadblock construction and warp trace building | wall_clock |
| warp_trace_build | T_frontend_instruction_delivery_preparation | trace-driven: warp instruction fetch / frontend delivery loop | wall_clock |
| core_cycle | T_sim_backend_execution | trace-driven: main simulation loop | cycle_count + wall_clock |
| get_next_inst | T_frontend_instruction_delivery_preparation | trace-driven: get_next_trace_inst call | wall_clock |
| total_wall | T_kernel_to_sim_done | main(): simulation entry to exit | wall_clock |

## Per-Run JSON Output

Each simulation run produces a JSON record with these fields:

- `trace_read_s`
- `parse_pb_s`
- `static_bind_s`
- `tb_load_s`
- `warp_trace_build_s`
- `get_next_inst_s`
- `core_cycle_s`
- `total_sim_wall_s`
- `frontend_share` = (trace_read + parse_pb + static_bind + warp_trace_build + tb_load + get_next_inst) / total_wall
- `workload_id` = workload identifier (set before simulation run)

## Overhead Requirements

- Max measurement overhead: < 1% of total wall time
- Timer resolution: millisecond
- Output: single JSON file per run, written at simulation exit

## Implicit and Explicit Fields

Fields `T_kernel_or_trace_export` (NVBit trace generation) and `T_result_analysis` (post-simulation processing) are NOT measured inside the simulator. They must be recorded externally by the workflow script that invokes NVBit, the simulator, and the analysis tools.

## Frontend Share

```
frontend_share = (trace_read_s + parse_pb_s + static_bind_s + warp_trace_build_s + tb_load_s + get_next_inst_s) / total_sim_wall_s
```

All 6 buckets are non-overlapping. `static_bind_s` is per-call `parse_from_pb` time; `warp_trace_build_s` is loop time minus `static_bind_s`.
