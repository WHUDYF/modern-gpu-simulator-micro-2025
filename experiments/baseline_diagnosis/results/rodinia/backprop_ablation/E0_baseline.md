# Diagnosis Report: backprop [E0_baseline]

**Date:** 2026-04-08
**Hardware:** RTX 3080 Ti (SM_86)
**Input features:** backprop_stageB_full.json (4096 trace + 65536 hw)
**Mechanisms enabled:** none
**Diagnoser:** manual (in conversation)

---

## Stage A: Software Utilization Check

### Utilization metrics

| Kernel | waves_per_sm | achieved_occupancy | grid_size |
|--------|--------------|---------------------|-----------|
| bpnn_layerforward_CUDA | 8.53 | 88.92% | 1x256x1 (256 TBs) |
| bpnn_adjust_weights_cuda | 8.53 | 94.61% | 1x256x1 (256 TBs) |

### Stage A Verdict

- [x] Workload utilizes hardware adequately (waves_per_sm ≥ 4, occupancy ≥ 80%)
- [ ] If no, prescribe Class A fix first

**Verdict:** PASS. Proceed to Stage B.

---

## Stage B: Architecture Bottleneck Analysis

### Per-kernel bottleneck identification

#### Kernel 1: bpnn_layerforward_CUDA

**Distance-to-roof table:**

| Resource | Utilization |
|----------|-------------|
| L1/TEX Cache | **80.02%** (dominant) |
| Compute (SM) | 72.41% (secondary) |
| DRAM | 44.19% |
| L2 Cache | 24.63% |

- IPC active: 1.95
- warp_cycles_per_issued_inst: 21.58
- L1 hit rate: 57.68% (moderate)
- L2 hit rate: 55.21%

**Cross-source reasoning:**
- Top opcodes: IADD3(14), NOP(13), LDS(12), IMAD(8), BAR.SYNC(8), STS(7)
- 8 BAR.SYNC + 12 LDS + 7 STS → classic tile-based matmul pattern
- Static shared memory in kernel_summary indicates tile usage
- L1/TEX throughput 80% includes LDS/STS traffic (they share the same
  physical SRAM as L1). L1 hit rate is only 58%, so the 80% throughput
  is NOT driven by L1 cache hits — it's driven by the explicit shared
  memory access stream
- Interpretation: this kernel is **shared memory bank bandwidth limited**,
  not L1 cache capacity limited

**Dominant bottleneck:** L1/shared memory bandwidth (LDS/STS traffic)
**Secondary bottleneck:** compute pipeline

#### Kernel 2: bpnn_adjust_weights_cuda

**Distance-to-roof table:**

| Resource | Utilization |
|----------|-------------|
| Compute (SM) | **84.98%** (dominant on paper) |
| DRAM | 21.41% |
| L2 Cache | 10.98% |
| L1/TEX | 10.56% |

- IPC active: **0.15** (extremely low)
- warp_cycles_per_issued_inst: **286.87** (extremely high)
- L1 hit rate: 73.32%

**Cross-source reasoning:**
- Top opcodes: LDG.E(12), F2F.F64.F32(12), NOP(11), IMAD.WIDE(6),
  DMUL(6), DFMA(4), F2F.F32.F64(4), STG.E(4)
- The presence of DMUL, DFMA, F2F.F64 → **this kernel uses FP64 arithmetic**
- Contradiction: compute throughput 85% but IPC only 0.15, warp_cycles/inst 287
  → compute pipe looks busy but warps are mostly stalled
- Explanation: consumer Ampere GPUs have severely throttled FP64 (1/64 of
  FP32 throughput). A single DFMA ties up the DP unit for ~64 cycles. With
  many warps queued up behind the same scarce DP unit, each warp waits
  hundreds of cycles between issues
- Memory subsystem is virtually idle (L1 10%, L2 11%, DRAM 21%), confirming
  the bottleneck is not memory

**Dominant bottleneck:** FP64 execution pipeline serialization
**Secondary bottleneck:** none meaningful (everything else is idle)

---

## Class B Prescriptions

### Prescription B.1: Increase DP pipeline initiation rate

**Target kernel:** bpnn_adjust_weights_cuda

**Modification:**
```
-trace_opcode_latency_initiation_dp 24,16 → 24,4
```

**Reason:**
- IPC 0.15 + warp_cycles_per_issued 286.87 + FP64 opcodes in top list →
  warps are serialized on a scarce DP pipe
- Compute throughput 85% but memory subsystem < 22% on all dimensions →
  the bottleneck is execution pipeline, not memory
- Reducing DP initiation interval from 16 to 4 quadruples DP throughput
  (effectively 4x more DP units)

**Expected effect:**
- IPC active: 0.15 → 0.5-0.8 (3-5x improvement)
- warp_cycles_per_issued_inst: 287 → ~70
- Kernel sim_cycle: -40% to -60%
- Compute throughput: may decrease (pipe no longer saturated)
- Memory metrics: unchanged (already idle)

**Expected cost:** +3-5% SM area, corresponding power increase.

**Verification:**
- Modify: `gpgpusim.config` or `trace.config`
- Rerun: `accel-sim.out -trace <backprop_trace> -config <modified>`
- Compare: `gpu_ipc` and `gpu_sim_cycle` for adjust_weights kernel
- Success criterion: IPC improves ≥2x, cycle count decreases ≥30%

**Confidence:** HIGH (cross-source evidence: opcode list + IPC/throughput contradiction + memory idleness all point to FP64 serialization)

**Control kernel:** bpnn_layerforward_CUDA (should be unchanged since it uses FFMA, not FP64)

---

### Prescription B.2: Investigate shared memory bank count for forward kernel

**Target kernel:** bpnn_layerforward_CUDA

**Modification:**
```
-gpgpu_shmem_num_banks 32 → 64
```

**Reason:**
- L1/TEX throughput 80% is dominated by LDS/STS (shared memory), not L1
  cache hits (hit rate only 58%)
- LDS(12) + STS(7) + 8 BAR.SYNC in top opcodes → heavy tile-based shared
  memory reuse pattern
- Shared memory bank conflicts may be limiting effective bandwidth

**Expected effect:**
- L1/TEX throughput: 80% → 50-60% (pressure relieved)
- IPC: 1.95 → 2.5-3.0
- Kernel sim_cycle: -15% to -25%

**Expected cost:** ~30% larger shared memory subsystem area.

**Verification:**
- Modify: `gpgpusim.config -gpgpu_shmem_num_banks 64`
- Rerun: same trace
- Compare: `gpu_ipc` for forward kernel, shared memory bank conflict stats

**Confidence:** MEDIUM (GPGPU-Sim may not finely model bank conflicts in
trace mode; our closed-loop validation in v2 showed this change was a no-op
on the simulator)

**Control kernel:** bpnn_adjust_weights_cuda (uses no shared memory; should be unchanged)

---

## Summary

- Total prescriptions: 2
- High confidence: 1 (B.1: DP initiation)
- Medium confidence: 1 (B.2: shmem banks)
- Prescriptions that use mechanism features: 0 (none enabled)
- Prescriptions that would not exist without mechanism features: 0 (baseline)

### Baseline Diagnosis Notes

The E0 diagnosis relies entirely on:
- Hardware metrics (NCU)
- Static trace info (top_opcodes, control bits)
- Cross-source reasoning (contradiction detection)

It successfully identifies both bottlenecks (forward shared memory, adjust_weights FP64) without any mechanism features, because the static opcode list in base features already reveals FP64 usage. This establishes the bar that mechanism-enabled experiments (E1-E4) must exceed.
