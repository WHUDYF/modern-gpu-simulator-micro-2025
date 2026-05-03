# Redundancy Profiling Counter Specification

Generated: 2026-05-03

**Status**: Specification only. Counters must be wired in the simulator C++ source when available.

## Measured Counters

| Counter | Location | Description |
|---------|----------|-------------|
| dynamic_insn_count | trace-driven: issue/delivery loop | Total dynamic instruction events |
| unique_static_id_count | trace-parser: static binding | Unique (function_id, pc) pairs |
| static_info_lookup_count | trace-parser: metadata resolution | Total static info lookups |
| threadblock_count | trace-driven: CTA launch | Total threadblocks across all kernels |
| warp_trace_count | trace-driven: warp construction | Total warp trace objects |
| metadata_obj_construction_count | trace-parser + trace-driven | Metadata object allocations |
| frontend_allocation_count | trace-parser + trace-driven | Frontend heap allocations |

## Derived Ratios

| Ratio | Formula | Interpretation |
|-------|---------|----------------|
| static_reuse_ratio | dynamic_insn_count / unique_static_id_count | Higher = more reuse opportunity for static-info cache |
| tb_metadata_reuse_ratio | threadblock_count / unique_tb_metadata_shape_count | Near 1.0 = per-CTB metadata (good reuse target); Lower = finer-grain |
| frontend_allocation_density | frontend_allocation_count / dynamic_instruction_count | Allocations per dynamic instruction; lower is better |

## AI-Training Workload Expectation

For BERT-base and Llama 3.1 8B models:
- High static_reuse_ratio expected (many dynamic instances per static instruction due to loops and repeated layer patterns)
- tb_metadata_reuse_ratio near 1.0 expected (metadata mostly per-threadblock)
- These patterns suggest a frontend static-info cache could eliminate redundant lookups

Confirmation requires measured data from the simulator.
