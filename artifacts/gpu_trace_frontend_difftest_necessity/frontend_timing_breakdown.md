# Frontend Timing Instrumentation Specification

Generated: 2026-05-03

**Status**: Specification only. C++ instrumentation pending simulator source code availability. The `simulator-remodeled/gpu-simulator/` directory is not present in this repository.

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
| total_wall | T_kernel_to_sim_done | main(): simulation entry to exit | wall_clock |

## Per-Run JSON Output

Each simulation run produces a JSON record with these fields:

- `trace_read_s`
- `parse_pb_s`
- `static_bind_s`
- `tb_load_s`
- `warp_trace_build_s`
- `core_cycle_s`
- `total_sim_wall_s`
- `frontend_share` = (trace_read + parse + bind + load + delivery) / total_wall

## Overhead Requirements

- Max measurement overhead: < 1% of total wall time
- Timer resolution: millisecond
- Output: single JSON file per run, written at simulation exit

## Implicit and Explicit Fields

Fields `T_kernel_or_trace_export` (NVBit trace generation) and `T_result_analysis` (post-simulation processing) are NOT measured inside the simulator. They must be recorded externally by the workflow script that invokes NVBit, the simulator, and the analysis tools.

## Frontend Share

```
frontend_share = (trace_read_s + parse_pb_s + static_bind_s + tb_load_s + warp_trace_build_s) / total_sim_wall_s
```
