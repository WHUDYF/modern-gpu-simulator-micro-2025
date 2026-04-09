# Diagnosis Report: nn [E0_baseline]

**Date:** 2026-04-08
**Hardware:** RTX 3080 Ti (SM_86)
**Input features:** nn_full.json
**Mechanisms enabled:** none
**Diagnoser:** manual

---

## Stage A: Software Utilization Check

### Utilization metrics

| Kernel | waves_per_sm | achieved_occupancy | theoretical_occupancy | grid_size | block_dim |
|--------|--------------|---------------------|----------------------|-----------|-----------|
| euclid (×4 launches) | **0.73** | 18.1% | **33.33%** | 938x1x1 | **16x1x1** |

### Stage A Verdict

- [ ] Workload utilizes hardware adequately
- [x] Workload does NOT utilize hardware adequately

**Verdict: FAIL**

- `waves_per_sm = 0.73 < 4` → grid is too small to fill the GPU even once
- `theoretical_occupancy = 33.33%` (not the usual 100%) → **even theoretical maximum is only 1/3** of peak
- `achieved_occupancy = 18.1%` → further limited by runtime factors
- `avg_active_threads_per_warp = 11.62/32` → 36% warp utilization; 64% of each warp is wasted

### Root Cause Analysis

The `theoretical_occupancy = 33.33%` is the smoking gun. With 48 max warps/SM,
33.33% means only 16 warps/SM can ever be active. With `block_dim = 16`
(half a warp), each block still consumes a full warp slot but only uses
half of its threads. This limits both **per-SM warp count** and
**per-warp active threads**.

This is fundamentally a **kernel launch configuration** problem, not a
data-size problem. Increasing input data will add more blocks but each
block will still have only 16 threads, still waste half the warp, still
cap per-SM warps at 16.

### Class A Prescription

**Prescription A.1: Change block_dim from 16 to 64+**

- **Modification:** In `nn_cuda.cu`, increase block size from 16 to 64 (or 128)
- **Reason:** block_dim = 16 is a half-warp, permanently wasting 50% of
  thread slots per warp AND limiting theoretical warps/SM to 16 (33%)
- **Expected:**
  - avg_active_threads_per_warp: 11.62 → 32 (full warp utilization)
  - theoretical_occupancy: 33% → 100% (no longer block-size-limited)
  - achieved_occupancy: 18% → 60%+
- **Verification:** rebuild nn with larger block_dim, rerun NCU, compare metrics
- **Cost:** requires kernel source modification; may change correctness if
  block-level synchronization depends on current size

**Stage B is not performed** per protocol (Stage A failed).

---

## Speculative Stage B Notes (for comparison with mechanism-enabled experiments)

Even though Stage B is formally suspended, here is what the raw metrics
suggest (to be ignored unless Class A is fixed):

- L1 hit rate 83% (good), L2 hit rate 16% (bad, most L1 misses go to DRAM)
- compute throughput 7% (nothing is compute-bound)
- DRAM 17%, L1/L2 both ~9% (nothing is memory-bound in absolute terms)
- IPC 0.5, warp_cycles_per_issued 16.66 (moderate stalls)
- NCU explicitly flags: **92% of global accesses are uncoalesced**

If Class A fix is applied, the remaining Class B bottleneck is likely
**uncoalesced global access pattern** (a kernel-level optimization, not a
simulator config change). This is distinct from both backprop's FP64
serialization and shared memory bandwidth issues.

---

## Summary

- Total prescriptions: 1 (Class A only — Stage A failed)
- Class A prescriptions: 1 (change block_dim)
- Class B prescriptions: 0 (Stage B suspended)
- **Key observation**: nn's Class A failure is **structurally different**
  from backprop's. backprop was "grid scales with input, too small input".
  nn is "grid is hardcoded, block_dim is intrinsically half-warp".
  This is a qualitatively different Class A failure mode.
