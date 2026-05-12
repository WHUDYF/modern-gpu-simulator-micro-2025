# PKA-M1 Workload Sweep Evaluation Design

Date: 2026-05-12

## 1. Background

We have already collected multiple workloads into the PKA-M1 path and produced at least some complete 12D measured feature records. The next question is not whether the acquisition pipeline exists, but whether PKA is actually useful across the workload set we already have.

This design defines a workload sweep evaluation that answers two questions:

1. Can each collected workload reliably produce the full 12D PKA measured feature vector?
2. For workloads with complete 12D rows, can PKA compress them into a small number of representative anchors while preserving coverage of the important behavior?

This design deliberately stops before simulator-side validation. It only evaluates the PKA front-end quality on the collected workload corpus.

## 2. Goals

The evaluation must produce both a corpus-wide view and a workload-by-workload view.

The core goals are:

- Measure 12D feature completeness for every collected workload.
- Separate acquisition failures from selector-quality failures.
- Evaluate PKA compression ratio on workloads that are fully measured.
- Evaluate representative coverage, not only anchor count.
- Keep enough detail to explain why a workload performs well or poorly.

## 3. Non-goals

This design does not:

- Introduce new NCU acquisition logic.
- Require privileged NCU execution.
- Re-measure workloads that already have complete 12D rows just for this evaluation.
- Evaluate simulator speedup, simulator accuracy, or end-to-end architecture improvement.
- Collapse the results into a single scalar score.
- Replace the existing PKA-M1 pipeline.

## 4. Corpus Scope

The sweep consumes all previously collected workloads that already have PKA feature outputs, including data under:

- `artifacts/a_line/l1/pka_feature_table_l1.json`
- `artifacts/a_line/l1/*/pka_feature_table_l1.json`
- `artifacts/a_line/l1/m1_smoke_inputs/*/pka_feature_table_l1.json`

The source of truth for the sweep is the measured feature table, not the raw source code tree.

### 4.1 Workload Categories

Each workload must be assigned to one of these categories:

- `microbench`
- `rodinia`
- `ai_workload`
- `smoke`

If a workload does not fit cleanly, the category must be recorded explicitly rather than inferred later from the summary.

## 5. Evaluation Layers

The evaluation has two layers.

### 5.1 Layer 1: 12D Completeness

This layer answers whether a workload can consistently produce the full 12D PKA feature vector.

For each workload, compute:

- `total_records`
- `complete_12d_records`
- `complete_rate`
- `missing_feature_counts`
- `timing_unit`
- `usable_for_selector`

Rules:

- `complete_rate == 1.0` means the workload is fully usable for selector evaluation.
- `0 < complete_rate < 1.0` means the workload is partially usable and must be flagged.
- `complete_rate == 0` means the workload is not usable for selector evaluation and should be treated as an acquisition or alignment blocker, not as a selector failure.

The 12D feature order is the fixed PKA-M1 order already used by the shared selector core.

### 5.2 Layer 2: Compression and Representativeness

This layer runs PKA selector evaluation only on workloads that are usable for selector evaluation.

For each eligible workload, compute:

- `input_records`
- `anchor_count`
- `compression_ratio`
- `cluster_count`
- `coverage_count`
- `coverage_weight`
- `top_1_coverage`
- `top_2_coverage`
- `top_3_coverage`
- `anchor_balance`
- `representative_workloads`
- `representative_kernel_invocations`

This layer must keep compression and coverage separate. A workload is not good just because its anchor count is small; it must also preserve the dominant coverage mass.

`anchor_balance` is defined as the largest single-anchor coverage weight within the workload. Higher values mean the workload's coverage is concentrated into fewer anchors.

## 6. Required Outputs

The evaluation must emit both machine-readable and human-readable outputs.

### 6.1 Corpus-wide Summary

The corpus summary must include:

- total workload count
- count per category
- complete 12D rate per workload
- usable-for-selector rate per category
- average compression ratio per category
- average top-1 coverage per category
- average top-2 coverage per category
- average top-3 coverage per category

### 6.2 Workload Detail Table

The workload detail table must include one row per workload with:

- workload id
- category
- source feature table path
- total records
- complete 12D records
- complete rate
- missing feature counts
- timing unit
- usable_for_selector
- anchor count
- compression ratio
- top-1 / top-2 / top-3 coverage
- representative workload ids

## 7. Artifact Layout

The sweep should write outputs under:

```text
artifacts/a_line/l1/pka_m1_workload_sweep/
```

Expected artifacts:

- `workload_feature_completeness.json`
- `workload_feature_completeness.md`
- `pka_selector_sweep.json`
- `pka_selector_sweep.md`
- `pka_workload_sweep_summary.json`
- `pka_workload_sweep_summary.md`

The JSON files are the canonical outputs. The Markdown files are the human-readable summary.

## 8. Evaluation Semantics

### 8.1 12D Completeness Semantics

The completeness layer must not silently fill missing values with zeros.

Missing features must remain missing and be counted explicitly.

The evaluation must preserve the reason a workload is incomplete, where possible:

- missing metric availability
- parser failure
- timing unit conflict
- incomplete selector input
- incomplete measured row

If multiple reasons exist, the evaluation should keep the earliest blocking reason and any secondary reasons separately in the JSON summary.

### 8.2 Selector Semantics

Selector evaluation must use the shared PKA selector core.

The sweep must not duplicate PCA or k-means logic in a second implementation.

The selector summary must preserve:

- anchor count
- compression ratio
- cluster assignments
- representative selection
- top-k coverage

### 8.3 Comparison Semantics

Workload results must be comparable across categories, but not flattened into one score.

The primary comparison axes are:

- completeness
- compression ratio
- representative coverage

This keeps the evaluation useful even when different workload classes have very different launch shapes or timing scales.

## 9. Acceptance Criteria

This design is satisfied when the evaluation can produce:

- a corpus-wide summary over all collected workloads
- a per-workload completeness table
- a selector sweep over all workloads with complete 12D rows
- a category summary for microbench, rodinia, ai_workload, and smoke
- a stable list of representative anchors for each eligible workload

The evaluation must clearly distinguish:

- workloads that are incomplete because the data are missing
- workloads that are complete but compress poorly
- workloads that are complete and compress well

## 10. Relationship to Existing PKA-M1 Artifacts

This evaluation reuses the existing PKA-M1 feature order and selector core. It does not redefine the 12D feature set.

It should consume existing measured feature tables such as:

- `pka_feature_table_l1.json`
- the smoke feature tables already produced for `mini_transformer_v2`
- the smoke feature tables already produced for `vector_add_permission_smoke`

The design is intentionally corpus-oriented: it is about how the full collected set behaves, not about a single representative smoke example.

## 11. Expected Outcome Shape

At the end of this sweep, we should be able to say:

- which workloads are fully ready for PKA
- which workloads are only partially usable
- which workload categories compress well
- which categories preserve coverage poorly
- whether the collected corpus supports PKA as a stable front-end abstraction

That is the main evidence needed before deciding whether to keep expanding the same PKA path or to change the experimental direction.
