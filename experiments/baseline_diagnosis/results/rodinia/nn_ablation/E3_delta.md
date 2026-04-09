# Diagnosis Report: nn [E3_delta]

**Date:** 2026-04-08
**Hardware:** RTX 3080 Ti (SM_86)
**Input features:** nn_full.json + nn_delta.json
**Mechanisms enabled:** delta

---

## Delta-informed Observation

**Kernel-level field temperature:**

| Field | Temperature | Classification |
|-------|-------------|----------------|
| num_barriers | 0.0 | COLD |
| num_tbs | 0.0 | COLD |
| total_dynamic_instructions | 0.0 | COLD |
| total_static_instructions | 0.0 | COLD |
| uses_fp64 | 0.0 | COLD |
| uses_shared_memory | 0.0 | COLD |

- **All 6 fields are COLD**
- **0 hot fields**
- 0 field correlations (4 kernel launches, but all identical → no variance)
- 0 outlier diffs

**This is the strongest possible Delta signal for uniformity**: every
measured field is perfectly stable across all 4 kernel launches. The 4
euclid invocations are behavioral carbon copies.

### Critical insight Delta provides over E0

**E0 alone** sees 4 kernel launches in `stats_csv` and `enhanced_execution_info`.
It can observe "4 kernels, same name". But it might still hypothesize:
"maybe the 4 launches do slightly different things (different data
partitions) and one of them is the bottleneck".

**E3 (Delta) mechanically rules this out**: "all 6 behavioral fields are
COLD = 4 launches are indistinguishable = there is no 'bad kernel' among
them".

**This is the first instance in Phase 2-3 where a mechanism provides a
signal E0 cannot cheaply produce.** On backprop, Delta's hot field
(`uses_fp64`) was information E0 could also derive by scanning opcodes.
On nn, Delta's "all cold" finding is a **global statistical statement**
that E0 would have to compute explicitly.

### Implication for Class A prescription

**E0's reasoning:** "block_dim=16 is too small" (derived from static info).

**E3's reasoning:** "Class A fix is unambiguously **kernel restructuring**
because:
1. **All 4 launches are identical** (Delta: 0 hot fields across kernels)
2. Therefore, changing the **number** of launches cannot help (nothing
   would differ between more launches)
3. Therefore, the fix must change **what each launch does internally**
4. The only internal structural issue visible is block_dim=16"

E3's reasoning is **mechanized and tighter** than E0's.

---

## TB-level Delta

**Kernel 1 (analogous for all 4 kernels):**
- 0 hot fields
- 9 cold fields
- 15 "correlations" (likely spurious — see note below)
- 0 outlier diffs

**Note on spurious correlations:** Delta TB-level analysis on nn reports
15 field correlations, but inspection of the underlying feature values
shows all TB features are bit-identical within each kernel (zero variance,
with floating-point noise on the order of 1e-16). The correlation
coefficients of ±1.0 are artifacts of dividing near-zero noise vectors.
**This is a Delta implementation bug**: it should filter out fields with
std below a numerical epsilon before computing correlations. See
"Known Bugs" at the end of this report.

Once the bug is fixed, we expect TB-level Delta on nn to report:
0 hot, 9 cold, 0 correlations, 0 outliers — consistent with the uniform
nature of nn.

---

## Stage A (same as E0): FAIL

---

## Class A Prescription

**Prescription A.1: Change block_dim from 16 to 64+** (same as E0/E1/E2)

**Delta-strengthened reasoning:**
- All kernel-level fields are COLD → 4 launches are identical → "more
  launches" is not a fix
- The only degree of freedom is within-kernel structure
- block_dim=16 is the most proximate structural issue

**Confidence:** HIGH (this is the strongest evidence of the three mechanism
experiments — Delta's "all cold" signal is the most **direct mechanization**
of the reasoning chain)

---

## Summary

- Total prescriptions: 1 (Class A only)
- New bottlenecks found: 0
- **Mechanism value**: Delta provides a **genuinely new signal** on nn
  ("all kernel-level fields are cold") that E0 cannot cheaply produce
- **First instance in the ablation where a mechanism provides
  non-redundant information**

### Delta value on nn vs backprop

| | backprop | nn |
|---|---|---|
| Kernel-level hot fields | 4 (uses_fp64, ...) | 0 |
| Information content | can be derived from opcode scanning | **cannot** be derived without computing across-kernel diffs |
| Value to diagnosis | confirms FP64 bottleneck | **mechanizes "launches are identical" finding** |
| Contribution type | confirming | **discovering** |

On nn, Delta is **discovering-type**: it mechanizes a statement that E0
could only produce via explicit comparison of kernel signatures. On
backprop, Delta was confirming-type.

---

## Known Bugs (to fix in Phase 4 or later)

**Bug 1: Spurious correlations on zero-variance fields**
- Description: Delta TB-level analysis reports correlations of ±1.0
  between fields that have zero real variance (floating-point noise only)
- Cause: correlation computation does not check if std is numerically
  meaningful
- Fix: add minimum std threshold (e.g., 1e-10) before including a field
  in correlation analysis
- Impact: currently contaminates TB-level correlation output on nn; no
  impact on kernel-level output for either backprop or nn
- Priority: MEDIUM (does not affect the main finding, but produces
  misleading JSON output)
