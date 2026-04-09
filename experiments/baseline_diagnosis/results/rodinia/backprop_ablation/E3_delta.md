# Diagnosis Report: backprop [E3_delta]

**Date:** 2026-04-08
**Hardware:** RTX 3080 Ti (SM_86)
**Input features:** backprop_stageB_full.json + backprop_4096_delta.json
**Mechanisms enabled:** delta
**Diagnoser:** manual (following skill protocol)

---

## Stage A: Software Utilization Check

Same as E0. Both kernels PASS.

---

## Stage B: Architecture Bottleneck Analysis

### Delta-informed insights

**Kernel-level field temperature:**

| Field | Temperature | Classification |
|-------|-------------|----------------|
| num_barriers | 0.762 | HOT |
| **uses_fp64** | **0.762** | **HOT** |
| uses_shared_memory | 0.762 | HOT |
| total_dynamic_instructions | 0.484 | HOT |
| total_static_instructions | 0.254 | WARM |
| num_tbs | 0.000 | COLD |

- **Hot fields (change between kernels):** uses_fp64, uses_shared_memory,
  num_barriers, total_dynamic_instructions
- **Cold field (stable across kernels):** num_tbs (both kernels have 256 TBs)
- Field correlations: 0 (too few data points — only 2 kernels)
- Outlier diffs: 0 (too few data points)

**TB-level field temperature (within each kernel):**
- Kernel 1 (forward): 0 hot fields, 10 cold fields
- Kernel 2 (adjust_weights): 0 hot fields, 9 cold fields

All TB-level fields are cold because backprop has perfectly uniform TBs
within each kernel (as confirmed by Batch in E2).

### Interpretation

**What Delta adds to E0:**

1. **Direct identification of behavioral axis:** Delta's kernel-level hot
   field `uses_fp64` is a **mechanically-emitted signal** that "FP64 usage
   varies between phases of this workload". This is exactly the signal
   that drove our v2 report's FP64 conclusion, but now it's **machine
   identified** rather than discovered via cross-source reasoning over raw
   opcode lists.

2. **Symmetric identification of other varying axes:** `uses_shared_memory`
   is also hot — forward uses shared memory extensively (tile-based matmul),
   adjust_weights uses none. This is the flip side of the FP64 finding and
   is equally mechanized.

3. **`num_barriers` is hot:** forward has many BAR.SYNC (8), adjust_weights
   has fewer. This is an indirect indicator of synchronization patterns
   that differ between the two phases.

4. **TB-level confirms uniformity:** both kernels have 0 hot fields at
   TB-level, mathematically confirming intra-kernel uniformity (same as
   Batch's finding but via a different mechanism).

**What Delta does NOT add:**

- No field correlations found (too few kernels — need 3+ kernels for
  meaningful correlation signals)
- No outlier diffs (same reason)
- No new prescription targets beyond FP64 and shared memory, which E0
  already discovered

### Per-kernel diagnosis (Delta-informed)

#### Kernel 1: bpnn_layerforward

- Delta signal: this kernel is the "uses_shared_memory=True, uses_fp64=False"
  side of the workload
- Bottleneck: shared memory bandwidth (same as E0)
- Delta provides no new information for this kernel's bottleneck

#### Kernel 2: bpnn_adjust_weights

- Delta signal: this kernel is the "uses_fp64=True, uses_shared_memory=False"
  side of the workload
- Bottleneck: FP64 serialization (same as E0)
- **Delta's hot `uses_fp64` field is the mechanized version of the diagnosis**

---

## Class B Prescriptions

### Prescription B.1: Increase DP pipeline initiation rate

**Target kernel:** bpnn_adjust_weights_cuda

**Modification:**
```
-trace_opcode_latency_initiation_dp 24,16 → 24,4
```

**Reason (Delta-strengthened):**
- **`uses_fp64` is a HOT field in Delta's kernel-level analysis** — a
  machine-emitted signal that FP64 usage is a key varying dimension of
  this workload
- Only kernel 2 has `uses_fp64=true`, so the prescription is automatically
  scoped to kernel 2
- IPC 0.15 + warp_cycles 286.87 + memory subsystem idle confirms the
  execution pipe (not memory) is the bottleneck
- All three pieces of evidence (Delta hot field + IPC contradiction +
  memory idleness) agree

**Expected effect / verification:** same as E0

**Confidence:** HIGH (now supported by Delta's automated hot-field signal
in addition to E0's reasoning)

**Control kernel:** bpnn_layerforward (Delta shows it has `uses_fp64=False`)

---

### Prescription B.2: Phase 1 shared memory bandwidth

**Reason (Delta-strengthened):**
- **`uses_shared_memory` is a HOT field in Delta's analysis** — only kernel 1
  uses shared memory
- `num_barriers` is also hot, and kernel 1 has 8 BAR.SYNC (tile iterations)
  vs kernel 2's minimal count → synchronization-heavy tile pattern in
  kernel 1

**Expected effect / verification:** same as E0

**Confidence:** MEDIUM (unchanged)

**Control kernel:** bpnn_adjust_weights (Delta shows uses_shared_memory=False)

---

## Summary

- Total prescriptions: 2
- High confidence: 1
- Medium confidence: 1
- Prescriptions that use mechanism features: **2** (both use Delta hot fields)
- Prescriptions that would not exist without Delta: **0**
- Prescriptions strengthened by Delta: 2

### Delta Contribution Analysis

**Value added (STRONGEST of the three mechanisms so far):**

Delta is the only mechanism that **directly surfaces the FP64 signal as a
top-level feature** without requiring opcode-list scanning. In E0, the
diagnosis must reason: "I see DMUL/DFMA/F2F.F64 in top_opcodes, therefore
FP64 is used". In E3, Delta says: "hot_field: uses_fp64" — mechanized and
unambiguous.

This matters because:
1. Opcode-list scanning is sensitive to prompt engineering (what if the
   AI misses the FP64 opcodes in a noisy list?)
2. Delta's `uses_fp64` is a pre-computed single bit that cannot be missed
3. On more complex workloads (many kernels, many opcode types), Delta's
   mechanization becomes increasingly valuable

**Value not added:** No new bottlenecks discovered beyond E0. The two
correlations and outlier-diff lists are empty because backprop has only
2 kernels (statistical insufficiency).

**Expected value on other dwarfs:**

1. Workloads with subtle precision changes (mixed FP16/FP32/FP64) →
   Delta's `uses_fp64` alone won't capture it, but similar hot-field
   analysis on FP16 usage would
2. Workloads with many kernels → correlation analysis becomes meaningful
   (e.g., "address pattern correlates with stall — memory-driven")
3. Workloads with long kernels → TB-level Delta might find hot/cold fields
   within a kernel (not trivial as in backprop)

Delta is the most **promising mechanism for mechanized bottleneck
discovery**, but its full power requires richer data (more kernels, longer
kernels, more diverse behavior).
