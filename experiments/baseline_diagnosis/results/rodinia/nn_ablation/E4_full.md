# Diagnosis Report: nn [E4_full]

**Date:** 2026-04-08
**Hardware:** RTX 3080 Ti (SM_86)
**Input features:** nn_full.json + squash + batch + delta
**Mechanisms enabled:** all three

---

## Triple-mechanism cross-validation

All three mechanisms independently certify the same structural property
of nn: **all 4 kernel launches are behaviorally identical**.

| Mechanism | Signal | Interpretation |
|-----------|--------|----------------|
| Squash | 1 segment, 0 boundaries | temporal uniformity |
| Batch | 1 cluster of 4, 0 outliers, homogeneity=1.0 | spatial uniformity |
| Delta | 0 hot fields, 6 cold, 0 correlations at kernel-level | field-level uniformity |

**Triple convergence** on "nn is behaviorally uniform across launches"
gives maximum confidence in this structural observation.

---

## Stage A: FAIL

Same as E0-E3: `waves_per_sm = 0.73`, `theoretical_occupancy = 33.33%`,
`achieved_occupancy = 18.1%`, `avg_active_threads_per_warp = 11.62/32`.

---

## Class A Prescription

**Prescription A.1: Change block_dim from 16 to 64 (kernel source change)**

**Target:** `nn_cuda.cu` kernel launch configuration

**Modification:**
```c
// before:
euclid<<<num_blocks, 16>>>(...);
// after:
euclid<<<num_blocks/4, 64>>>(...);  // and adjust thread indexing
```

(Exact modification depends on kernel internals; the point is to use
full warps.)

**Reason (triple-convergent):**
1. **Squash**: 1 segment, 0 boundaries → launch count is not the lever
2. **Batch**: 1 homogeneous cluster + no outliers → no special-case TBs
3. **Delta**: 0 hot fields at kernel-level → 4 launches carry zero
   information diversity; "run more" cannot help
4. **Base metrics**: theoretical_occupancy=33.33% (half-warp block size
   cap), avg_active_threads_per_warp=11.62 (half the warp is always
   inactive)

Together: the only actionable lever is **per-kernel thread structure**,
specifically block_dim.

**Expected effect:**
- avg_active_threads_per_warp: 11.62 → 32 (2.75x)
- theoretical_occupancy: 33.33% → 100%
- achieved_occupancy: 18.1% → 60%+ (still may be limited by register or
  shared memory, but the fundamental block-size cap is removed)
- Runtime: **significant improvement is likely**, but exact speedup depends
  on whether memory coalescing also improves

**Expected cost:** Requires kernel source modification. May affect
correctness if the original algorithm assumed block_dim=16 (e.g.,
intra-warp reductions).

**Verification:**
- Modify `nn_cuda.cu` kernel launch line and kernel body (indexing)
- Rebuild
- Rerun NCU, compare: theoretical_occupancy, avg_active_threads_per_warp,
  overall runtime
- **Success criterion**: theoretical_occupancy reaches ≥80%, runtime
  improves ≥1.5x

**Confidence:** HIGH (triple-convergent mechanism evidence + clear
structural signal in base metrics)

---

## Speculative Stage B (after Class A is applied)

Once block_dim is increased, we expect the following additional bottlenecks
to become visible (currently masked by the occupancy limit):

1. **Uncoalesced global access** (NCU flagged 92% uncoalesced) — likely
   still the dominant issue after Class A fix. This is a kernel-level
   optimization (change data layout or access pattern), not a simulator
   config change.

2. **L2 miss rate 84%** — with more active warps, the L2 pressure will
   increase. Possible future Class B prescription: `-gpgpu_cache:dl2`
   size increase.

These are **speculative** until Class A is validated.

---

## Summary

- Total prescriptions: 1 (Class A)
- High confidence: 1
- Prescriptions that use mechanism features: 1 (all three mechanisms
  support the Class A reasoning)
- New bottlenecks found: 0 (same Class A as E0)
- **Mechanism value**: triple-convergent certification that "launch-pattern
  changes cannot help — must change kernel structure"

### Key Comparison: E4 on nn vs E4 on backprop

| | backprop E4 | nn E4 |
|---|------------|-------|
| Stage A status | PASS (after input=65536 merge) | **FAIL** (cannot be merged away) |
| Prescriptions found | 2 (Class B) | 1 (Class A only) |
| Mechanisms added new info? | No (all signals derivable from E0) | **Yes (Delta's "all cold" is non-trivial)** |
| Mechanism value type | confirming / cross-validating | **discovering / mechanizing** |
| Triple convergence | confirms FP64 bottleneck | confirms structural uniformity |

**On nn, the mechanisms finally show a clear advantage over E0**: the
"all cold" Delta signal and "1 segment" Squash signal mechanize a
conclusion (launches are identical) that E0 can only derive via explicit
cross-kernel comparison.
