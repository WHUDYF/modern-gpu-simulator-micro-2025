# Diagnosis Report: backprop [E2_batch]

**Date:** 2026-04-08
**Hardware:** RTX 3080 Ti (SM_86)
**Input features:** backprop_stageB_full.json + backprop_4096_batch.json
**Mechanisms enabled:** batch
**Diagnoser:** manual (following skill protocol)

---

## Stage A: Software Utilization Check

Same as E0. Both kernels PASS.

---

## Stage B: Architecture Bottleneck Analysis

### Batch-informed insights

**Kernel-level clustering:**
- Clusters formed: **0**
- Outlier kernels: [1, 2] (both kernels flagged as outliers)
- Homogeneity: 0.0

Interpretation: DBSCAN with `min_samples=2` cannot form a cluster because
the two kernels are so different from each other that they don't meet the
density criterion. Both are flagged as outliers. **This is actually a
strong signal that forward and adjust_weights have fundamentally different
behavior profiles.**

**TB-level clustering (within each kernel):**

| Kernel | Clusters | Outliers | Homogeneity |
|--------|----------|----------|-------------|
| bpnn_layerforward | 1 cluster (256 TBs) | 0 | 1.000 |
| bpnn_adjust_weights | 1 cluster (256 TBs) | 0 | 1.000 |

Interpretation: within each kernel, all 256 TBs form a single perfectly
homogeneous cluster. This confirms backprop has **no outlier TBs**
(no boundary-condition TBs, no edge-case handling).

### Interpretation

**What Batch adds to E0:**

1. **Machine-certified intra-kernel uniformity:** Batch mathematically
   confirms that every TB in each kernel is behaviorally identical (homogeneity
   = 1.0). This rules out any "outlier TB" hypothesis.

2. **Machine-certified inter-kernel divergence:** Both kernels being
   flagged as outliers at kernel-level is a machine-produced signal that
   the workload has **high behavioral diversity between kernels** — which
   is the flip side of uniformity within each kernel.

3. **Ruling out edge-case prescriptions:** Because there are no outlier
   TBs, prescriptions like "fix the boundary TB" or "add scheduler slots
   for outlier warps" are definitively ruled out.

**What Batch does NOT add for backprop:**

- No new bottleneck found. The diagnosis still reduces to the same two:
  forward = shared memory bandwidth, adjust_weights = FP64 serialization.
- No actionable prescription unique to Batch. Any prescription that would
  benefit from "target outlier TBs" is inapplicable because there are no
  outliers.

### Per-kernel diagnosis

Same as E0 — Batch doesn't surface new bottlenecks, only confirms the
absence of outlier-driven bottlenecks.

---

## Class B Prescriptions

### Prescription B.1: Increase DP pipeline initiation rate

Same as E0 B.1.

**Batch contribution:** Batch confirms that all 256 TBs of adjust_weights
are identical, which means the FP64 bottleneck is workload-wide (not
confined to specific TBs). This supports treating the whole kernel as
a single target for the prescription.

**Confidence:** HIGH (unchanged from E0)

---

### Prescription B.2: Phase 1 shared memory bandwidth

Same as E0 B.2.

**Batch contribution:** Confirms forward kernel's 256 TBs are uniform,
so a single shared memory bank count change applies uniformly.

**Confidence:** MEDIUM (unchanged from E0)

---

## Summary

- Total prescriptions: 2 (same as E0)
- High confidence: 1
- Medium confidence: 1
- Prescriptions that use mechanism features: **2** (Batch confirms uniformity)
- Prescriptions that would not exist without Batch: **0**
- Prescriptions strengthened by Batch: 2 (uniformity guarantees)

### Batch Contribution Analysis

**Value added:** Batch acts as a **negative check** — it confirms what
we assumed (intra-kernel uniformity, no outlier TBs). This is a useful
safety net: if Batch had found outlier TBs in backprop, the diagnosis
would have to be revised.

**Value not added:** No new bottlenecks discovered. The kernel-level
"cluster count = 0" outcome is technically informative (both kernels are
outliers relative to each other) but doesn't lead to new prescriptions.

**Expected value on other dwarfs:** Batch's value will be high on:
1. Workloads with boundary-condition TBs (e.g., sequence length not
   divisible by tile size)
2. Workloads with data-dependent divergence (e.g., BFS, irregular graphs)
3. Workloads with mixed behavior (e.g., MoE routing where some TBs go
   through different code paths)

Backprop is uniform by design, which is Batch's worst-case scenario for
added value.

### Key Observation: Batch's "Outlier-Only" Kernel-Level Result

The fact that Batch found 0 clusters and 2 outliers at kernel-level is
actually a **meaningful structural observation**: it means the two
kernels in backprop are behaviorally distant enough that DBSCAN's density
requirement cannot bridge them. This is a machine-produced signal of
"phase distance" — complementary to Squash's boundary identification.

However, converting this into a prescription would require a mechanism
beyond what Batch currently exposes (e.g., "use the larger outlier distance
to prioritize which kernel to optimize first").
