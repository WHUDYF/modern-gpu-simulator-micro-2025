# Diagnosis Report: nn [E2_batch]

**Date:** 2026-04-08
**Hardware:** RTX 3080 Ti (SM_86)
**Input features:** nn_full.json + nn_batch.json
**Mechanisms enabled:** batch

---

## Batch-informed Observation

**Kernel-level clustering:**
- 1 cluster containing all 4 kernels
- 0 outlier kernels
- Homogeneity: **1.0** (perfect)
- Centroid summary: single shared behavior across all 4 euclid launches

**TB-level clustering (within each kernel):**
- Kernel 1: 1 cluster, 0 outliers, homogeneity=1.000 (938 TBs)
- Kernel 2: 1 cluster, 0 outliers, homogeneity=1.000 (938 TBs)
- Kernel 3: 1 cluster, 0 outliers, homogeneity=1.000 (938 TBs)
- Kernel 4: 1 cluster, 0 outliers, homogeneity=1.000 (938 TBs)

### Contrast with backprop

On backprop, Batch produced **0 clusters and 2 outliers at kernel-level**
because the 2 kernels were too different for DBSCAN to form a cluster.

On nn, Batch produces **1 cluster of 4 kernels with perfect homogeneity**,
the opposite extreme.

### What Batch adds over E0

Batch provides **machine-certified dual uniformity**:
1. **Inter-kernel uniformity**: all 4 launches are in one cluster with
   homogeneity 1.0 — no diversity between launches
2. **Intra-kernel uniformity**: each kernel's 938 TBs form a single perfect
   cluster — no outliers, no boundary-condition TBs

**Implication for prescriptions:**
- E0 can hypothesize "maybe some specific boundary TBs are the bottleneck"
- E2 **rules this out mechanically**: there are no outliers at any level
- The diagnosis can focus on the **kernel-wide issue** (block_dim) without
  worrying about special-case TBs

---

## Stage A (same as E0): FAIL

---

## Class A Prescription

**Prescription A.1: Change block_dim from 16 to 64+** (same as E0 and E1)

**Batch-strengthened reasoning:**
- The 1-cluster result confirms this is a **uniform whole-workload issue**,
  not a pocket of bad TBs
- The fix applies identically to all 3752 TBs (938 × 4 launches)
- No edge-case handling needed

**Confidence:** HIGH (Batch's uniformity certification supports applying
a single blanket fix)

---

## Summary

- Total prescriptions: 1 (Class A only)
- Batch contribution: rules out outlier-driven and cluster-specific fixes
- New bottlenecks found: 0
- Prescription count delta vs E0: 0

### Batch value on nn vs backprop

| Dimension | backprop | nn |
|-----------|---------|-----|
| Kernel-level result | 0 clusters, 2 outliers | 1 cluster of 4, 0 outliers |
| Interpretation | kernels are **diverse** | kernels are **identical** |
| Actionable signal | "each kernel needs its own prescription" | "one prescription applies to all" |

In both cases, Batch's contribution is a **structural certification**
rather than a bottleneck discovery. Its value is in **ruling out
hypotheses**, not generating new ones.

The **opposite-extreme results** (0 clusters vs 1 perfect cluster)
demonstrate that Batch correctly distinguishes workload structure.
