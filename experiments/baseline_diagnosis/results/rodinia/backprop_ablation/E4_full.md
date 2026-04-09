# Diagnosis Report: backprop [E4_full]

**Date:** 2026-04-08
**Hardware:** RTX 3080 Ti (SM_86)
**Input features:** backprop_stageB_full.json + squash + batch + delta
**Mechanisms enabled:** squash, batch, delta
**Diagnoser:** manual (in conversation, following skill protocol)

---

## Stage A: Software Utilization Check

Same as E0. Both kernels PASS (waves_per_sm=8.53, occupancy ≥88%).

---

## Stage B: Architecture Bottleneck Analysis

### Combined Mechanism Insights

This experiment has all three mechanisms enabled. The three mechanisms
provide **orthogonal views** of the same workload:

| Mechanism | View | Key output for backprop |
|-----------|------|-------------------------|
| Squash | Time-ordered segmentation | 2 kernel-level segments; 1 boundary |
| Batch | Space-independent clustering | 0 clusters, 2 kernel outliers; TB-level uniform |
| Delta | Field-level variation | hot: uses_fp64, uses_shared_memory, num_barriers |

### Cross-mechanism cross-validation

All three mechanisms independently and consistently say:
- "The two kernels in backprop are behaviorally distinct"
- "Within each kernel, all TBs are uniform"
- "The key axes of variation between the two kernels are FP64, shared
  memory usage, and barrier count"

This **triple-convergence** makes the phase distinction extremely robust —
even a single-mechanism mis-estimate would be corrected by the other two.

### Per-kernel diagnosis (Phase 1 — bpnn_layerforward)

- **Squash:** segment 0, dominant opcodes IADD3/LDS → tile-based matmul
- **Batch:** TB-level 1 cluster, 256 TBs, homogeneity 1.0 → no outliers
- **Delta:** this is the "uses_shared_memory=True, uses_fp64=False, high
  num_barriers" side of the workload

**Stage B metrics:**
- L1/TEX throughput: 80.02% (dominant)
- Compute throughput: 72.41% (secondary)
- L1 hit rate: 57.68%
- LDS-dominant pattern confirms shared memory bandwidth bound

### Per-kernel diagnosis (Phase 2 — bpnn_adjust_weights)

- **Squash:** segment 1, dominant opcodes LDG.E/F2F.F64.F32 → FP64 conversion
- **Batch:** TB-level 1 cluster, 256 TBs, homogeneity 1.0 → no outliers
- **Delta:** this is the "uses_fp64=True" side (the critical hot field)

**Stage B metrics:**
- Compute throughput: 84.98% (appears dominant)
- IPC active: 0.15 (contradicts apparent compute saturation)
- warp_cycles_per_issued_inst: 286.87 (extreme)
- L1/L2/DRAM all < 22% (memory idle)

**Reasoning (all evidence consolidated):**
1. Delta's `uses_fp64=HOT` + Squash segment 1 dominant `F2F.F64.F32` →
   Phase 2 uses FP64 arithmetic
2. Compute throughput 85% with IPC 0.15 → serialization on scarce unit
3. warp_cycles_per_issued 287 → queue-time signature (many warps behind
   one unit)
4. Memory subsystem virtually idle → execution pipeline is the
   bottleneck, not memory
5. Consumer Ampere FP64:FP32 = 1:64 → the scarce unit is the DP pipeline

**Bottleneck:** FP64 pipeline serialization (in Phase 2 only).

---

## Class B Prescriptions

### Prescription B.1: Increase DP pipeline initiation rate (Phase 2)

**Target kernel:** bpnn_adjust_weights_cuda

**Modification:**
```
-trace_opcode_latency_initiation_dp 24,16 → 24,4
```

**Reason (triple-convergent evidence):**
- **Delta:** `uses_fp64` is a hot field at kernel-level
- **Squash:** Phase 2 dominant opcodes include `F2F.F64.F32`
- **Batch:** all 256 TBs in Phase 2 are uniform, so the fix applies
  workload-wide
- Stage B: IPC 0.15 + compute 85% contradiction + memory idleness

**Expected effect:**
- IPC active: 0.15 → 0.5-0.8 (3-5x improvement)
- warp_cycles_per_issued_inst: 287 → ~70
- Phase 2 sim_cycle: -40% to -60%
- **Phase 1 (control): unchanged**

**Expected cost:** +3-5% SM area.

**Verification:**
- Modify: `trace.config` `-trace_opcode_latency_initiation_dp 24,4`
- Rerun: `accel-sim.out -trace <backprop_trace> -config <modified>`
- Compare: per-kernel `gpu_ipc` and `gpu_sim_cycle`
- **Success criterion:**
  - Phase 2 IPC improves ≥2x
  - **Phase 1 IPC and cycle unchanged** (null control)
  - Memory metrics unchanged

**Confidence:** HIGH (triple-convergent: Delta + Squash + Stage B contradiction)

**Control kernel:** bpnn_layerforward (Phase 1 — all three mechanisms
confirm it has uses_fp64=False and no DP ops)

---

### Prescription B.2: Phase 1 shared memory bandwidth

**Target kernel:** bpnn_layerforward_CUDA

**Modification:**
```
-gpgpu_shmem_num_banks 32 → 64
```

**Reason (triple-convergent evidence):**
- **Delta:** `uses_shared_memory` is a hot field; Phase 1 is the kernel
  where it's True
- **Squash:** Phase 1 dominant opcodes include LDS (shared memory load)
- **Batch:** all 256 TBs of Phase 1 are uniform, so the fix applies uniformly
- Stage B: L1/TEX throughput 80% driven by LDS (not L1 cache hits, since
  hit rate only 58%)

**Expected effect:**
- L1/TEX throughput: 80% → 50-60%
- IPC: 1.95 → 2.5-3.0
- Phase 1 sim_cycle: -15% to -25%
- **Phase 2 (control): unchanged (uses_shared_memory=False)**

**Expected cost:** ~30% larger shared memory area.

**Verification:** same method, test null-control on adjust_weights

**Confidence:** MEDIUM (GPGPU-Sim may not finely model shared memory bank
conflicts in trace mode — our v2 closed-loop validation showed this
modification was a no-op in simulator)

**Control kernel:** bpnn_adjust_weights (Phase 2 — uses_shared_memory=False)

---

## Summary

- Total prescriptions: 2
- High confidence: 1 (B.1: DP initiation)
- Medium confidence: 1 (B.2: shmem banks)
- Prescriptions that use mechanism features: **2** (both use triple-mechanism
  cross-validation)
- Prescriptions that would not exist without mechanisms: **0**
- Prescriptions strengthened by mechanisms: 2 (cross-validation)

### E4 Contribution Analysis vs E0-E3

**Triple-convergence is the main value of E4.** In E0-E3, each mechanism
provides a single piece of evidence for the FP64 finding. In E4, all three
mechanisms independently arrive at the same conclusion, which makes the
diagnosis:

1. **More robust** (no single mechanism failure would miss the FP64
   bottleneck)
2. **More confident** (three independent data streams agree)
3. **More explainable** (multiple ways to present the same finding)

However, E4 on backprop does **NOT find any new bottleneck that E0 missed**.
The prescriptions and their confidences are identical to E0. What changes
is the quality of the **evidence trail**.

### Backprop-specific limitations

Backprop is a weak stress test for mechanism value because:
1. Only 2 kernels → Squash/Delta have minimal data
2. Uniform TBs → Batch and TB-level mechanisms are trivial
3. Clear phase distinction → E0 already identifies phases from raw opcodes

**To get strong mechanism value, Phase 3 (nn) must show a case where:**
- E0 misses a bottleneck that mechanisms surface, OR
- Mechanisms produce a more specific / higher-confidence prescription
  that translates to better closed-loop simulator results
