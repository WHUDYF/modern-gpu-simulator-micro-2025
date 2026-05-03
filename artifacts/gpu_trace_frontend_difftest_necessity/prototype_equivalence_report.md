# No-Semantics Prototype Gate and Equivalence Checklist

Generated: 2026-05-03

## Prototype Decision Gate

**Prerequisite**: Evidence pipeline must show `P_trace_to_sim_slice > 15%` OR `P_trace_to_sim_step > 15%` with **measured** (not placeholder) data.

**Current Status**: Pending — evidence currently uses placeholder data.

## Allowed Prototype Scope

| Allowed | Forbidden |
|---------|-----------|
| Decoded static-info cache: cache `(unique_function_id, pc)` lookups | Scoreboard dependency changes |
| Metadata normalization cache: cache repeated config records | Warp issue order changes |
| Threadblock chunk staging: batch CTB/warp construction | Memory pipeline timing changes |
| Local replay: replayable parser-to-frontend chunks | SM backend timing state changes |
| | Dynamic instruction squash or semantic compression |
| | Export-time optimization claims |

Boundary: `trace-parser → trace-driven frontend → shader core input`

## Equivalence Checklist

| Metric | Tolerance | Priority |
|--------|-----------|----------|
| sim_cycle | Exact match (0% deviation) | P0 |
| sim_insn | Exact match (0% deviation) | P0 |
| IPC | Exact match (derived) | P0 |
| cache_stats (L1/L2 hit/miss) | Exact match (0% deviation) | P0 |
| fatal_conditions | No new crashes | P0 |
| kernel_order | Exact match | P1 |
| per_kernel_insn_counts | Exact match | P1 |
| warnings | No new warnings | P1 |
| frontend_wall_time (T_trace_to_sim) | Expected reduction 15-50%; must not regress | P2 |

## Gate Decision Flow

1. Evidence pipeline confirms `P_trace_to_sim > 15%` with measured data
2. Prototype scope document approved
3. Baseline run: record all equivalence metrics on unmodified simulator
4. Prototype build: implement only allowed changes
5. Equivalence run: record all metrics on modified simulator
6. Compare: P0 metrics must match exactly; P1 must match; P2 must not regress
7. If any P0/P1 metric differs: prototype REJECTED
8. If all match: prototype accepted for performance evaluation
9. Performance claim: only report T_trace_to_sim reduction AFTER equivalence confirmed

## Deferred to Separate Spec

- Semantic compression (reducing dynamic instruction count)
- Dynamic instruction squash
- Any change that modifies sim_cycle or sim_insn output
