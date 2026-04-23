# Squash + Batch Family Criteria Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first `mini-transformer` prototype of the squash+batch family-criteria workflow, producing kernel analysis cards, family explanation cards, and a qualitative validation-track summary from existing diagnosis artifacts.

**Architecture:** Reuse existing `mini_transformer_v4` diagnosis artifacts as the source of truth, normalize them into a small structured kernel-card dataset, and derive family cards from that dataset using the approved three-layer criteria framework. Keep the first version documentation-first and lightweight, with only enough scripting to make the evidence extraction repeatable.

**Tech Stack:** Markdown, Python 3, pytest, existing `mini_transformer_v4` E0-E5 reports and APE JSON outputs

---

### Task 1: Set Up the Family-Criteria Workspace

**Files:**
- Create: `docs/family_criteria/README.md`
- Create: `docs/family_criteria/mini_transformer_v4/`
- Create: `docs/family_criteria/mini_transformer_v4/analysis_cards/`
- Create: `docs/family_criteria/mini_transformer_v4/family_cards/`
- Create: `docs/family_criteria/mini_transformer_v4/kernel_card_schema.md`

**Step 1: Create the directory layout**

Run:

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
mkdir -p docs/family_criteria/mini_transformer_v4/analysis_cards
mkdir -p docs/family_criteria/mini_transformer_v4/family_cards
```

Expected: both directories exist under `docs/family_criteria/mini_transformer_v4/`

**Step 2: Write the workspace README**

Create `docs/family_criteria/README.md` with:

```md
# Family Criteria Workspace

This directory stores the first prototype artifacts for the squash+batch
family-criteria workflow.

## Scope

- Input workload: `mini_transformer_v4`
- Source evidence: existing E0-E5 diagnosis reports and APE JSON outputs
- Output artifacts:
  - kernel analysis cards
  - family explanation cards
  - a qualitative validation-track summary

## Non-Goals

- no delta integration
- no simulator prescription execution
- no cross-workload generalization in this version
```

**Step 3: Write the kernel-card schema note**

Create `docs/family_criteria/mini_transformer_v4/kernel_card_schema.md` with:

```md
# mini_transformer_v4 Kernel Analysis Card Schema

Each kernel card must contain:

1. Basic info
   - kernel name
   - operator semantics
   - workload role
   - representative note
2. Execution mode
   - compute-heavy / memory-heavy / mixed / uncertain
3. Key observed metrics
   - occupancy
   - compute throughput
   - dram throughput
   - l1/l2 hit behavior
   - warp cycles
   - shmem usage
   - waves / launch shape / block limit when available
4. Dominant resource candidates
   - one primary candidate
   - optional secondary candidate
5. Family decision
   - tentative family
   - boundary notes
   - outlier / ambiguous marker if needed
```

**Step 4: Verify the workspace files exist**

Run:

```bash
test -f docs/family_criteria/README.md
test -f docs/family_criteria/mini_transformer_v4/kernel_card_schema.md
```

Expected: both commands exit with code `0`

**Step 5: Commit**

```bash
git add docs/family_criteria/README.md \
        docs/family_criteria/mini_transformer_v4/kernel_card_schema.md
git commit -m "docs: add family criteria workspace scaffold"
```

---

### Task 2: Add a Repeatable Kernel Evidence Extractor

**Files:**
- Create: `experiments/baseline_diagnosis/build_kernel_cards.py`
- Create: `tests/test_build_kernel_cards.py`

**Step 1: Write the failing test**

Create `tests/test_build_kernel_cards.py` with:

```python
from pathlib import Path

from experiments.baseline_diagnosis.build_kernel_cards import load_sources


def test_load_sources_contains_expected_reports():
    base = Path("/home/dyf/modern-gpu-simulator-micro-2025")
    sources = load_sources(base)

    assert "E0_baseline" in sources
    assert "E1_squash" in sources
    assert "E2_batch" in sources
    assert "E4_full" in sources
    assert "E5_stageC_validation" in sources
```

**Step 2: Run the test to verify it fails**

Run:

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
pytest tests/test_build_kernel_cards.py::test_load_sources_contains_expected_reports -v
```

Expected: FAIL with `ModuleNotFoundError` or import failure for `build_kernel_cards`

**Step 3: Write the minimal implementation**

Create `experiments/baseline_diagnosis/build_kernel_cards.py` with:

```python
from pathlib import Path


def load_sources(repo_root: Path) -> dict[str, Path]:
    result_dir = repo_root / "experiments" / "baseline_diagnosis" / "results" / "mini_transformer_v4"
    return {
        "E0_baseline": result_dir / "E0_baseline.md",
        "E1_squash": result_dir / "E1_squash.md",
        "E2_batch": result_dir / "E2_batch.md",
        "E4_full": result_dir / "E4_full.md",
        "E5_stageC_validation": result_dir / "E5_stageC_validation.md",
        "baseline_ape": result_dir / "baseline_ape.json",
    }
```

**Step 4: Run the test to verify it passes**

Run:

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
pytest tests/test_build_kernel_cards.py::test_load_sources_contains_expected_reports -v
```

Expected: PASS

**Step 5: Extend the test for output shape**

Append to `tests/test_build_kernel_cards.py`:

```python
from experiments.baseline_diagnosis.build_kernel_cards import default_kernel_names


def test_default_kernel_names_are_the_expected_representatives():
    assert default_kernel_names() == [
        "gemm_tiled",
        "attention_score",
        "residual_add",
        "softmax_kernel",
        "context_mul",
        "layernorm_kernel",
    ]
```

**Step 6: Run the new test to verify it fails**

Run:

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
pytest tests/test_build_kernel_cards.py::test_default_kernel_names_are_the_expected_representatives -v
```

Expected: FAIL with missing `default_kernel_names`

**Step 7: Implement the helper**

Update `experiments/baseline_diagnosis/build_kernel_cards.py` to:

```python
from pathlib import Path


def load_sources(repo_root: Path) -> dict[str, Path]:
    result_dir = repo_root / "experiments" / "baseline_diagnosis" / "results" / "mini_transformer_v4"
    return {
        "E0_baseline": result_dir / "E0_baseline.md",
        "E1_squash": result_dir / "E1_squash.md",
        "E2_batch": result_dir / "E2_batch.md",
        "E4_full": result_dir / "E4_full.md",
        "E5_stageC_validation": result_dir / "E5_stageC_validation.md",
        "baseline_ape": result_dir / "baseline_ape.json",
    }


def default_kernel_names() -> list[str]:
    return [
        "gemm_tiled",
        "attention_score",
        "residual_add",
        "softmax_kernel",
        "context_mul",
        "layernorm_kernel",
    ]
```

**Step 8: Run the focused tests**

Run:

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
pytest tests/test_build_kernel_cards.py -v
```

Expected: PASS

**Step 9: Commit**

```bash
git add experiments/baseline_diagnosis/build_kernel_cards.py \
        tests/test_build_kernel_cards.py
git commit -m "feat: add kernel card source loader"
```

---

### Task 3: Generate the Six Kernel Analysis Cards

**Files:**
- Create: `docs/family_criteria/mini_transformer_v4/analysis_cards/gemm_tiled.md`
- Create: `docs/family_criteria/mini_transformer_v4/analysis_cards/attention_score.md`
- Create: `docs/family_criteria/mini_transformer_v4/analysis_cards/residual_add.md`
- Create: `docs/family_criteria/mini_transformer_v4/analysis_cards/softmax_kernel.md`
- Create: `docs/family_criteria/mini_transformer_v4/analysis_cards/context_mul.md`
- Create: `docs/family_criteria/mini_transformer_v4/analysis_cards/layernorm_kernel.md`

**Step 1: Draft the gemm_tiled card**

Create `docs/family_criteria/mini_transformer_v4/analysis_cards/gemm_tiled.md` with:

```md
# Kernel Analysis Card: gemm_tiled

## Basic Info
- Operator semantics: GEMM / matrix multiply
- Workload role: compute backbone in mini-transformer v4
- Representative note: appears repeatedly and anchors the main compute path

## Execution Mode
- Tentative mode: compute-heavy

## Key Observed Metrics
- High compute throughput in baseline reports
- Low L1 hit, strong L2-backed behavior
- register/block-limit signal appears near the compute-dominant regime

## Dominant Resource Candidates
- Primary: register / occupancy
- Secondary: compute pipeline saturation

## Family Decision
- Tentative family: compute-heavy -> register-limited
- Boundary note: separate from attention_score if shared-memory evidence dominates the explanation
```

**Step 2: Repeat the card pattern for the remaining five representative kernels**

Use the same section structure for:

- `attention_score.md`
- `residual_add.md`
- `softmax_kernel.md`
- `context_mul.md`
- `layernorm_kernel.md`

Each card must include:

- one tentative execution mode
- one primary resource candidate
- an explicit boundary or uncertainty note

**Step 3: Check that all six cards exist**

Run:

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
find docs/family_criteria/mini_transformer_v4/analysis_cards -maxdepth 1 -type f | wc -l
```

Expected: `6`

**Step 4: Review card completeness**

Run:

```bash
rg -n "^## (Basic Info|Execution Mode|Key Observed Metrics|Dominant Resource Candidates|Family Decision)" \
  docs/family_criteria/mini_transformer_v4/analysis_cards/*.md
```

Expected: each file reports all five section headers

**Step 5: Commit**

```bash
git add docs/family_criteria/mini_transformer_v4/analysis_cards/*.md
git commit -m "docs: add mini-transformer kernel analysis cards"
```

---

### Task 4: Derive the First Family Explanation Cards

**Files:**
- Create: `docs/family_criteria/mini_transformer_v4/family_cards/compute-heavy-register-limited.md`
- Create: `docs/family_criteria/mini_transformer_v4/family_cards/memory-heavy-dram-dominated.md`
- Create: `docs/family_criteria/mini_transformer_v4/family_cards/mixed-cache-or-shmem-sensitive.md`
- Create: `docs/family_criteria/mini_transformer_v4/family_cards/outliers.md`

**Step 1: Write the first family card**

Create `docs/family_criteria/mini_transformer_v4/family_cards/compute-heavy-register-limited.md` with:

```md
# Family Card: compute-heavy -> register-limited

## Core Explanation
- This family groups kernels whose dominant behavior is compute-heavy and whose strongest shared explanatory signal points to register / occupancy pressure.

## Representative Kernels
- gemm_tiled
- attention_score (tentative)

## Boundary Conditions
- Exclude kernels whose shared-memory signature becomes the primary explanation.
- Exclude kernels whose memory throughput dominates the explanation.

## Uncertainty
- attention_score may remain borderline if shared-memory coupling is stronger than register pressure.

## Validation Meaning
- This family is a candidate for shared simulator-side validation around occupancy-sensitive explanations.
```

**Step 2: Add the DRAM-dominated family card**

Create `docs/family_criteria/mini_transformer_v4/family_cards/memory-heavy-dram-dominated.md` with:

```md
# Family Card: memory-heavy -> dram-dominated

## Core Explanation
- This family groups kernels whose dominant behavior is memory-heavy and whose strongest explanatory signal points to DRAM bandwidth pressure.

## Representative Kernels
- residual_add

## Boundary Conditions
- Exclude kernels that look memory-heavy only because of cache-locality artifacts.

## Uncertainty
- Single-member family in the first version is acceptable.

## Validation Meaning
- This family is a candidate for a dedicated memory-system validation track.
```

**Step 3: Add the mixed family card and outlier card**

Create:

- `mixed-cache-or-shmem-sensitive.md`
- `outliers.md`

The mixed card must explain why kernels are not cleanly compute-heavy or memory-heavy.
The outlier card must list kernels that cannot be stably absorbed into the current families.

**Step 4: Validate that every analysis card is referenced**

Run:

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
for kernel in gemm_tiled attention_score residual_add softmax_kernel context_mul layernorm_kernel; do
  rg -n "$kernel" docs/family_criteria/mini_transformer_v4/family_cards
done
```

Expected: every kernel name appears in at least one family or outlier card

**Step 5: Commit**

```bash
git add docs/family_criteria/mini_transformer_v4/family_cards/*.md
git commit -m "docs: derive first mini-transformer family cards"
```

---

### Task 5: Write the Qualitative Validation-Track Synthesis

**Files:**
- Create: `docs/family_criteria/mini_transformer_v4/mini_transformer_family_synthesis.md`

**Step 1: Draft the synthesis document**

Create `docs/family_criteria/mini_transformer_v4/mini_transformer_family_synthesis.md` with:

```md
# mini_transformer_v4 Family Criteria Synthesis

## Goal
- Show that squash+batch can organize representative kernels into a small number of interpretable families.
- Show that these families qualitatively reduce the validation problem from per-kernel guessing to a few validation tracks plus outliers.

## Families
- compute-heavy -> register-limited
- memory-heavy -> dram-dominated
- mixed -> cache-or-shmem-sensitive

## Outliers
- Record kernels that remain structurally unresolved in version 1.

## Why This Matters
- The workflow no longer starts by guessing a simulator prescription for each kernel independently.
- Instead, it first groups kernels by shared architectural explanation, then uses those families to organize validation thinking.

## Version-1 Limits
- no delta integration
- no automatic threshold derivation
- no quantitative cost-reduction claim
```

**Step 2: Verify that the synthesis mentions all family cards**

Run:

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
rg -n "compute-heavy|memory-heavy|mixed|outlier" \
  docs/family_criteria/mini_transformer_v4/mini_transformer_family_synthesis.md
```

Expected: all four terms appear in the synthesis

**Step 3: Run the test suite again**

Run:

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
pytest tests/test_build_kernel_cards.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add docs/family_criteria/mini_transformer_v4/mini_transformer_family_synthesis.md \
        tests/test_build_kernel_cards.py \
        experiments/baseline_diagnosis/build_kernel_cards.py
git commit -m "docs: add mini-transformer family criteria synthesis"
```

---

### Task 6: Close the Loop with Spec Alignment

**Files:**
- Modify: `draft_squash_batch.md`
- Modify: `docs/superpowers/specs/2026-04-15-squash-batch-family-criteria-design.md`

**Step 1: Update the draft to reference the implemented family workspace**

Add one short section to `draft_squash_batch.md` describing:

- where analysis cards live
- where family cards live
- how the `mini-transformer` prototype is used to refine family rules

**Step 2: Update the spec with implementation status notes**

Append a brief section to `docs/superpowers/specs/2026-04-15-squash-batch-family-criteria-design.md`:

```md
## Prototype Status

- Kernel analysis cards exist for the six representative kernels.
- First family cards exist for the initial mini-transformer prototype.
- The first version is qualitative and documentation-first.
```

**Step 3: Verify both docs contain the new status references**

Run:

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
rg -n "Prototype Status|analysis cards|family cards" \
  draft_squash_batch.md \
  docs/superpowers/specs/2026-04-15-squash-batch-family-criteria-design.md
```

Expected: matching lines in both files

**Step 4: Commit**

```bash
git add draft_squash_batch.md \
        docs/superpowers/specs/2026-04-15-squash-batch-family-criteria-design.md
git commit -m "docs: align family criteria draft and spec"
```

