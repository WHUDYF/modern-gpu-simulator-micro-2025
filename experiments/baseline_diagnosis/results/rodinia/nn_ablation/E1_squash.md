# Diagnosis Report: nn [E1_squash]

**Date:** 2026-04-08
**Hardware:** RTX 3080 Ti (SM_86)
**Input features:** nn_full.json + nn_squash.json
**Mechanisms enabled:** squash

---

## Squash-informed Observation

Kernel-level segmentation:
- **1 segment** covering all 4 kernel launches
- 0 boundaries
- Dominant opcodes: IMAD.MOV.U32, FFMA, IADD, LDG
- Interpretation: all 4 launches are the **same kernel with identical behavior**

TB-level segmentation (per kernel):
- 1 segment per kernel, 0 boundaries
- All 938 TBs within each kernel are uniform

### What Squash adds over E0

Squash provides a **new piece of structural information**: the 4 kernel
launches are behaviorally identical (forming a single segment). E0 could
infer this from the stats_csv showing 4 entries with the same kernel name
and similar instruction counts, but **Squash formalizes it**.

**Why this matters for nn:**

E0 might speculate "perhaps launching nn with more kernel invocations
could help amortize overhead". Squash's "1 segment, 0 boundaries" result
directly rules this out: **each launch is identical, so launching more
of them just produces more identical copies, not new behavior**. The
bottleneck cannot be "insufficient kernel diversity".

This narrows the Class A fix space: Squash confirms we need **per-kernel
restructuring** (change block_dim, not launch pattern).

---

## Stage A (same as E0)

**Verdict: FAIL** (`waves_per_sm = 0.73`, `theoretical_occupancy = 33.33%`)

---

## Class A Prescription

**Prescription A.1: Change block_dim from 16 to 64+** (same as E0)

**Squash-strengthened reasoning:**
- E0's reasoning: "block_dim=16 limits warp utilization"
- E1 adds: "Squash confirms all 4 launches are identical; changing launch
  count or pattern cannot help. The fix must be **within** the kernel
  structure, specifically block_dim."

This narrows the fix space more precisely than E0.

**Confidence: HIGH** (Squash's "identical launches" finding is a strong
signal that launch-level changes are futile)

---

## Summary

- Total prescriptions: 1 (Class A)
- Squash contribution: narrows the Class A fix space (rules out
  launch-pattern changes)
- New bottlenecks found: 0
- Prescription count delta vs E0: 0
- Confidence improvement: E0's Class A was HIGH already; Squash makes the
  reasoning **more explicit about why launch-pattern changes won't help**

### Squash value on nn vs backprop

- **On backprop**: Squash confirmed phase distinction (2 phases, 1 boundary).
  Information was redundant because E0 could see it from per-kernel opcodes.
- **On nn**: Squash confirms **non-distinction** (1 segment, 0 boundaries).
  This is a **different kind of signal**: it doesn't just confirm what E0
  sees, it **rules out a whole class of hypotheses** (launch-pattern changes).

**Conclusion**: Squash on nn is a **discriminating negative signal**,
which is more valuable than its confirming-positive signal on backprop.
