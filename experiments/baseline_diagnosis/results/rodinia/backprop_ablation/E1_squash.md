# Diagnosis Report: backprop [E1_squash]

**Date:** 2026-04-08
**Hardware:** RTX 3080 Ti (SM_86)
**Input features:** backprop_stageB_full.json + backprop_4096_squash.json
**Mechanisms enabled:** squash
**Diagnoser:** manual (following skill protocol)

---

## Stage A: Software Utilization Check

Same as E0. Both kernels PASS (waves_per_sm=8.53, occupancy ≥88%).

---

## Stage B: Architecture Bottleneck Analysis

### Squash-informed insights

**Kernel-level segmentation:**

| Segment | Kernels | Dominant opcodes | Behavior |
|---------|---------|------------------|----------|
| Segment 0 | kernel 0 (bpnn_layerforward) | IADD3, NOP, LDS | tile-based matmul |
| Segment 1 | kernel 1 (bpnn_adjust_weights) | LDG.E, F2F.F64.F32, NOP | FP64 weight update |

- Boundary count: **1** (between forward and adjust_weights)
- Cohesion of each segment: 1.0 (each segment has only 1 kernel)

**TB-level segmentation (within each kernel):**
- Kernel 1 (forward): 1 segment, 0 boundaries — all 256 TBs behave identically
- Kernel 2 (adjust_weights): 1 segment, 0 boundaries — all 256 TBs behave identically

### Interpretation

**What Squash adds to E0:**

1. **Explicit phase identification:** Squash formally declares "this workload
   has 2 phases with different behavioral signatures." The boundary is
   machine-identified, not a manual observation.

2. **Phase-specific dominant opcode signatures:** each segment comes with
   its own dominant opcode list, which the diagnosis can use to differentiate
   per-phase reasoning.

3. **Intra-kernel uniformity confirmed:** each kernel's 256 TBs form a single
   segment → there are no internal sub-phases to worry about. This rules
   out "a specific sub-range of TBs within forward is the bottleneck" as a
   hypothesis.

**What Squash does NOT add for backprop:**

- The same phase distinction is already visible in E0's static opcode list
  per kernel, so the information is redundant (for this workload).
- TB-level squash on backprop is trivial (1 segment per kernel) because
  backprop is highly regular. TB-level would matter for workloads with
  internal relaxation phases (e.g., iterative convergence), which
  backprop does not have.

### Per-kernel diagnosis (using Squash phases)

#### Phase 1: bpnn_layerforward_CUDA (Squash segment 0)

- Dominant bottleneck: L1/shared memory bandwidth (80.02%)
- Reasoning: LDS-heavy tile-based computation (from segment 0 dominant
  opcodes: IADD3, LDS)
- Same as E0

#### Phase 2: bpnn_adjust_weights_cuda (Squash segment 1)

- Dominant bottleneck: FP64 pipeline serialization
- Reasoning: segment 1 dominant opcodes `[LDG.E, F2F.F64.F32, NOP]` → Squash
  directly surfaces the FP32↔FP64 conversion pattern, which is a
  cheap-to-detect signal that this phase uses double precision
- Same as E0, but the FP64 signature is now phase-scoped rather than
  needing to scan the full opcode list

---

## Class B Prescriptions

### Prescription B.1: Increase DP pipeline initiation rate (Phase 2 only)

**Target kernel:** bpnn_adjust_weights_cuda (identified as Squash Phase 2)

**Modification:**
```
-trace_opcode_latency_initiation_dp 24,16 → 24,4
```

**Reason:**
- Squash Phase 2 dominant opcodes `[LDG.E, F2F.F64.F32, ...]` directly
  surface FP64 usage in this phase
- IPC 0.15 + warp_cycles_per_issued 286.87 + memory subsystem idle →
  FP64 pipeline serialization
- The phase-scoped diagnosis clarifies that this prescription applies
  only to Phase 2, not to Phase 1

**Expected effect:** (same as E0)
- IPC: 0.15 → 0.5-0.8
- Cycle count: -40% to -60%

**Verification:** (same as E0)

**Confidence:** HIGH (Squash segment signature + IPC/throughput contradiction)

**Control kernel:** bpnn_layerforward_CUDA (Squash Phase 1 — no FP64, should be unchanged)

---

### Prescription B.2: Phase 1 shared memory bandwidth (unchanged from E0)

Same as E0 B.2. Squash confirms this is a Phase 1-only issue via segment 0
dominant opcodes containing LDS.

**Confidence:** MEDIUM (same as E0)

---

## Summary

- Total prescriptions: 2 (same as E0)
- High confidence: 1
- Medium confidence: 1
- Prescriptions that use mechanism features: **2** (both use Squash phase
  scoping to justify "Phase 1 only" vs "Phase 2 only")
- Prescriptions that would not exist without Squash: **0**
- Prescriptions strengthened by Squash: 2 (phase scoping is more explicit)

### Squash Contribution Analysis

**Value added:** Squash provides **explicit phase boundaries and per-phase
opcode signatures**, which makes the control-kernel selection mechanical
(Phase 1's kernel is the obvious control for Phase 2 prescriptions, and
vice versa).

**Value not added:** No new bottlenecks found that E0 missed. On backprop
specifically, Squash is mostly a confirmation mechanism rather than a
discovery mechanism. This is because backprop's two phases are so distinct
that any per-kernel analysis (which E0 does natively) picks them up
automatically.

**Expected value on other dwarfs:** Squash's value will be higher on
workloads with:
1. Many kernels where phase boundaries are non-obvious
2. Kernels with internal TB-level phases (iterative algorithms)
3. Mixed workloads (e.g., GPT-2 with prefill/decode phases)

Backprop has 2 kernels and uniform TB behavior, which is Squash's
worst-case scenario for added value.
