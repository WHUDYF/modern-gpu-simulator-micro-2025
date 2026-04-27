# A-Line L1 RLCR Implementation Plan

## Goal Description

Establish a correctness-gated, feature-sanity-gated, and downstream-interface-gated L1 pipeline for the PKA baseline input, 12-dimensional feature extraction, representative anchor output, and B-line consumption interface on a small set of interpretable kernels.

**L1 has one primary success mode and one valid early-stop mode:**

1. **Full closure success**: All 10 P0 objects produce 12 measured PKA features, the selector runs successfully on the PKA-only feature space, the anchor table is emitted, and B-line consumption validates the anchor table schema successfully. This proves the full loop closes on L1 inputs.

2. **Acquisition-gate success**: One or more P0 objects cannot produce all 12 measured PKA features. The pipeline stops at Stage 2 and emits the manifest, feature audit, and acquisition gap report. This is **not a failure** — it is the expected and correct behavior when acquisition is incomplete. The audit and gap report are the primary round-1 deliverables in this case and provide actionable input for the next acquisition iteration.

Both outcomes are valid L1 completions. The key invariant is: **the pipeline never proceeds past an unresolved acquisition gap**. A round that produces only audit/gap artifacts is a successful correctness-gate round. A round that reaches anchor/B-line emission is a successful loop-closure round.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

### Core Pipeline Criteria

**Object-to-Invocation Cardinality**: One manifest object (e.g., `L1_MB_01`) may map to one or more kernel invocations, depending on the source file. Microbench JSON files typically contain a single invocation. Rodinia trace JSON and mini-transformer full JSON contain multiple invocations, and not all invocations in a source file necessarily correspond to the manifest object's `kernel_or_case`. The feature extractor must filter invocations by matching `kernel_name` (or equivalent identity field) to the manifest object's `kernel_or_case`, then produce one `PkaFeatureRecord` or gap row per matched invocation. `kernel_invocation_id` is derived during invocation expansion (not during manifest parsing) following the rule: `{kernel_or_case}#{occurrence_index}` where `occurrence_index` is 1-based and ordered by `trace_order` (or file order when `trace_order` is absent).

This means:
- One manifest object -> N invocations -> N `PkaFeatureRecord` rows (if all measured) or N gap rows (if any metric missing)
- AC-6 applies at the invocation level (per-invocation outcome), not at the manifest-object level
- The stage-gate checks all invocations of all P0 objects; a gap in any P0 invocation blocks Stage 3

- AC-1: L1 manifest is machine-readable and schema-validated
  - Positive Tests (expected to PASS):
    - All 10 P0 manifest entries are present in the JSON output with stable `id` (matching the entry's `L1_*` identifier), `source_type`, `benchmark_name`, `kernel_or_case`, `priority`, and `local_input_path`
    - The manifest JSON passes validation against `kernel_validation_manifest_schema.json` (the existing schema uses `id` and `local_input_path`; the manifest builder maps the draft's `validation_id` and `source_path` concepts to these schema fields)
    - P1 entries are also present when the manifest builder is configured to include them
    - Each entry has `expected_behavior_axis` populated as a human-readable label (not used for grouping)
  - Negative Tests (expected to FAIL):
    - Manifest with missing `id` is rejected by schema validation
    - Manifest with duplicate `id` values is rejected
    - Manifest entry whose `local_input_path` does not resolve to an existing file fails a path-existence check, with special handling for entries whose source is a multi-kernel file (the file must exist; the specific kernel/case is validated during feature extraction)
    - Manifest entry with an invalid `source_type` value (not in enum) is rejected
  - AC-1.1: Manifest path-existence and source-type validation
    - Positive: All P0 `local_input_path` values resolve to existing files; source-type-specific checks confirm the file has the expected structure (e.g., JSON parse succeeds, expected top-level keys present)
    - Negative: Manifest with a broken `local_input_path` halts pipeline with a clear error message naming the entry and path; a manifest entry with `source_type: local_microbench` pointing to a file that is not valid JSON is rejected

- AC-2: PKA feature table is generated with 12-dimensional measured features for every P0 object
  - Positive Tests (expected to PASS):
    - Every P0 invocation in the feature table has exactly 12 entries under `features`, one per PKA field
    - Every feature value has `status: "measured"` with a non-null numeric `value` and a non-empty `source` string traceable to an Nsight metric name or launch metadata field
    - `num_thread_blocks` status is `measured` with source `launch_grid_size` (from profiler or launch record)
    - `feature_mode` is set to `pka_l1_measured_only` when all 12 fields are measured
    - Running the feature extractor twice on identical inputs produces identical output (deterministic)
  - Negative Tests (expected to FAIL):
    - An invocation missing any of the 12 PKA fields is absent from `PkaFeatureTable` and appears only in `PkaAcquisitionGap`
    - An invocation with a non-measured feature value (e.g., imputed, default-zero, or semantically substituted) is rejected from the feature table
    - An invocation whose `num_thread_blocks` is derived from anything other than profiler/launch metadata is rejected from the feature table
    - A P0 object whose acquisition gap remains unresolved blocks selector execution (AC-3/AC-4 are not reachable)

- AC-3: PKA baseline selector groups solely on the 12-dimensional feature space, with no forbidden fields in the grouping key
  - Positive Tests (expected to PASS):
    - Selector output includes `feature_mode`, the list of fields actually used, and each field's status
    - Selector produces identical cluster assignments and anchor choices on identical input (deterministic)
    - Every cluster has an explicit `cluster_id`, member list, representative, and membership count
    - The output anchor table schema is machine-checkable (required fields present, forbidden fields absent)
  - Negative Tests (expected to FAIL):
    - Selector that receives `kernel_name` in its input grouping key raises a forbidden-field error
    - Selector that receives `grid_dim` or `block_dim` strings in its grouping key raises a forbidden-field error
    - Selector that receives `cross_tb_offset_coverage`, `squash_boundary_crossing_flag`, or any compression-side field in its grouping key raises a forbidden-field error
    - Selector that receives `family_id`, `regime_id`, `route_primitive`, `execution_template`, or `simulator_lane_id` in its grouping key raises a forbidden-field error
    - Selector output row containing any of the forbidden fields listed in AC-3 is rejected by the anchor table validator
  - AC-3.1: Precondition gate
    - Positive: Selector only runs when `PkaAcquisitionGap` contains zero blocking P0 objects
    - Negative: Selector refuses to run when any P0 object still has an unresolved acquisition gap

- AC-4: Representative anchor table is parseable by B-line with schema validation
  - Positive Tests (expected to PASS):
    - B-line consumer can parse the anchor table JSON without schema errors
    - B-line consumer verifies every anchor row has all required fields (`rep_kernel_id`, `kernel_name`, `cluster_id`, `member_invocations`, `coverage_count`, `coverage_weight`, `time_weight`)
    - B-line consumer verifies no forbidden downstream keys (`family_id`, `regime_id`, `route_primitive`, `execution_template`, `simulator_lane_id`) are present in any anchor row
    - The consumption report records: anchor count consumed, schema check result (pass/fail per row), and a list of any missing or forbidden fields found
  - Negative Tests (expected to FAIL):
    - Anchor table missing `rep_kernel_id` is rejected by B-line consumer with a clear error naming the missing field and row
    - Anchor table missing `member_invocations` is rejected by B-line consumer
    - Anchor table containing forbidden downstream keys (e.g., `family_id` in anchor rows) is rejected by B-line consumer
  - AC-4.1: Precondition gate
    - Positive: B-line consumption only runs when `RepresentativeAnchorTable` exists and passes its own schema validation
    - Negative: B-line consumer refuses to run on an empty or schema-invalid anchor table

- AC-5: Regression tests are automated and re-runnable
  - Positive Tests (expected to PASS):
    - Manifest schema validation test passes on a valid manifest and fails on a deliberately broken one
    - Feature table completeness test verifies all 12 fields exist with `measured` status for every P0 record in a valid table
    - Selector forbidden-field test injects `kernel_name` into the grouping key and asserts rejection
    - Anchor output schema test verifies required fields present and forbidden fields absent
    - B-line smoke consumption test verifies that a valid anchor table produces a consumption report with pass/fail schema check results for each row
  - Negative Tests (expected to FAIL):
    - Feature completeness test fails on a table where one record has a `null` value for `coalesced_global_loads`
    - Forbidden-field test passes (does not reject) when only allowed PKA fields are in the grouping key
    - B-line smoke test fails when given an anchor table with a missing `member_invocations` field
    - All tests produce clear assertion messages naming the specific violation

### Supplementary Criteria (from Codex review)

- AC-6: Every P0 invocation yields exactly one of two deterministic outcomes
  - Positive: Each invocation of a P0 object produces either one valid `PkaFeatureRecord` with 12 measured features, or one acquisition-gap row enumerating the missing metrics, invocation identity, and source path. Objects with multiple invocations produce one outcome per invocation; a single object may mix measured rows and gap rows (all gaps block the stage-gate).
  - Negative: An invocation producing both a feature record and a gap row is rejected as ambiguous; an invocation producing neither halts the pipeline with an error naming the invocation and source file

- AC-7: Acquisition gap blocks downstream artifact emission
  - Positive: When any P0 object appears in the acquisition gap report, `RepresentativeAnchorTable` and `BLineConsumptionReport` are not emitted
  - Negative: Emission of anchor table while a P0 gap exists is rejected by the stage-gate validator

- AC-8: Selector runtime emits its actual feature allowlist, exactly matching the approved 12 PKA fields
  - Positive: The emitted allowlist contains exactly 12 field names matching the PKA spec
  - Negative: An allowlist with 11 fields or 13 fields is rejected; an allowlist containing a non-PKA field name is rejected

- AC-9: `kernel_invocation_id` is unique and stable across all P0 rows
  - Positive: All P0 invocations have distinct `kernel_invocation_id` values; re-running the manifest builder with identical inputs produces identical IDs
  - Negative: Duplicate `kernel_invocation_id` values cause the manifest builder to fail fast with a clear error

- AC-10: Mixed timing units are rejected by default
  - Positive: When all invocations use the same timing unit (all `duration_ns` or all `elapsed_cycles`), weight computation proceeds normally
  - Negative: When invocations mix `duration_ns` and `elapsed_cycles`, any weight computation aborts with an error naming the conflicting sources; this check runs before selector execution

- AC-11: Source adapters are test-covered independently
  - Positive: A dedicated test exists for the microbench JSON adapter, the Rodinia artifact adapter, and the mini-transformer JSON adapter, each verifying correct extraction of the 12 PKA fields
  - Negative: An adapter fed a malformed input (missing required metric field) fails with a diagnostic naming the field and the source file

- AC-12: Audit output records per-feature provenance
  - Positive: `PkaFeatureAudit` includes for every feature of every P0 invocation: metric name, source artifact path, measured status, and missing-metric reason (when applicable)
  - Negative: An audit record with a `measured` status but an empty `source` field is rejected by the audit validator

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes all of the following:
- A manifest builder that reads the L1 manifest document and produces a schema-validated `kernel_validation_manifest_l1.json` with all P0 and P1 entries, including path-existence pre-checks
- A PKA feature extractor with dedicated source adapters for microbench JSON, Rodinia NCU/trace artifacts, and mini-transformer full/dual-source JSON, each producing the 12 PKA feature values with per-feature `status` and `source` provenance
- A PKA baseline selector that groups purely on the 12-dimensional feature space using a deterministic algorithm (bucketed or distance-threshold), with no dependency on `kernel_name`, `grid_dim`, or `block_dim` in its grouping key
- A representative selection rule (`first_chronological`) with deterministic tie-breaking, documented and tested
- Full audit artifacts: `pka_feature_audit_l1.md`, `pka_feature_audit_l1.json`, and `pka_acquisition_gap_l1.json` with per-feature provenance and missing-metric reasons
- A B-line stub consumer that parses the anchor table, generates at least family/regime/writeback lineage, and produces a consumption report recording every interface check result
- Regression tests covering all 12 ACs with both positive and negative test cases
- Explicit deprecation or retention markers on the existing non-PKA selector modes (`name-only`, `pka-like-coarse`, `hybrid`) with tests reflecting the boundary

### Lower Bound (Minimum Acceptable Scope)

The implementation includes at minimum:
- A manifest builder that produces `kernel_validation_manifest_l1.json` with at least all 10 P0 objects, passing schema validation against the existing `kernel_validation_manifest_schema.json`
- A PKA feature extractor with source adapters for all three P0-bearing source types (microbench JSON, Rodinia artifacts, mini-transformer JSON). If any P0 invocation cannot produce all 12 measured PKA features, the feature extractor routes it to the acquisition gap report and the pipeline stops at Stage 2. This is the expected lower-bound outcome when acquisition data is insufficient.
- A PKA feature audit and acquisition gap report that clearly enumerates which P0 invocations are missing which metrics
- A B-line consumer that parses the anchor table and validates required/forbidden fields (parse-only; this only executes if the pipeline reaches Stage 4, which requires all P0 invocations to pass Stage 2)
- Regression tests covering AC-1 through AC-5 (manifest, feature table, forbidden fields, anchor schema, B-line parse smoke), with tests designed to pass even when the pipeline stops at Stage 2 (i.e., tests for Stage 2 outputs work independently of Stage 3/4)
- The existing selector modes are left untouched (neither deprecated nor refactored) but the new PKA selector is clearly separated in its own module

> **Lower Bound Clarification**: The lower bound is **not** "microbench-only adapter with full loop closure." It is "all adapters implemented, with the realistic expectation that the pipeline may stop at Stage 2 (audit/gap) if existing artifacts lack the canonical 12 PKA Nsight metrics." Both outcomes (full closure via Stages 3-4, or audit-only stop at Stage 2) satisfy the lower bound. The lower bound defines the minimum code that must exist; the stage-gate determines which outputs are produced at runtime.

### Allowed Choices

- Can use: The 12 PKA feature fields and only those fields for grouping; `first_chronological` or `max_exec_time` for representative selection (must be explicitly chosen and documented); exact-vector grouping or bucketed grouping or distance-threshold clustering for the grouping algorithm; Python standard library, existing project JSON utilities, and `pytest` for tests; the existing `kernel_validation_manifest_schema.json` as a base with optional extensions for invocation-level records
- Cannot use: `kernel_name`, `grid_dim` string, `block_dim` string, `shape_hint`, `trace_order`, `cross_tb_offset_coverage`, `squash_boundary_crossing_flag`, compression-side features, `family_id`, `regime_id`, `route_primitive`, `execution_template`, or `simulator_lane_id` in the PKA baseline grouping key; imputed, default-zero, or semantically-substituted values masquerading as `measured`; the existing curated middle-layer bundle path as the B-line consumption target (L1 B-line consumer must operate on the anchor table directly)

> **Note on Deterministic Designs**: The 12 PKA feature set is fixed per `docs/a-line-pka-feature-general-spec-2026-04-27.md`. The two-layer object design (`KernelValidationRecord` -> `PkaFeatureRecord`) is fixed per the draft. The stage-gate execution order is fixed. The allowed choices above reflect these deterministic constraints—upper and lower bounds differ mainly in adapter breadth and B-line consumption depth, not in core contract.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The implementation follows a strict stage-gate pipeline with five stages. Each stage must pass before the next can execute.

```
+-------------------+     +-------------------+     +-------------------+
| Stage 1: Manifest | --> | Stage 2: Feature  | --> | Stage 3: PKA      |
| Builder           |     | Extractor + Audit |     | Selector          |
+-------------------+     +-------------------+     +-------------------+
        |                         |                         |
        v                         v                         v
  manifest_l1.json       feature_table_l1.json      anchor_table_l1.json
                         feature_audit_l1.json
                         acquisition_gap_l1.json
                                                              |
                                                              v
+-------------------+                               +-------------------+
| Stage 5: Tests    | <---------------------------- | Stage 4: B-line   |
| (runs alongside)  |                               | Consumption       |
+-------------------+                               +-------------------+
                                                            |
                                                            v
                                                  consumption_report_l1.md
```

**Stage 1 — Manifest Builder**: Read the L1 manifest document and produce `kernel_validation_manifest_l1.json`. Each entry maps to a `KernelValidationRecord`-like structure with `validation_id`, `source_type`, `benchmark_name`, `kernel_or_case`, `priority`, `source_path`, `expected_behavior_axis`. Validate against the existing manifest schema. Run a path-existence pre-check: every `source_path` must resolve to an existing file.

**Stage 2 — Feature Extractor + Audit**: For each manifest entry, dispatch to the appropriate source adapter based on `source_type`. Each adapter reads raw data and attempts to extract the 12 PKA features. For microbench JSON, check whether the canonical 12 Nsight metric names are present in the source data. If present, extract the value directly with `status: measured`. If absent, the metric is missing and the invocation is routed to the acquisition gap report. No name-mapping, no semantic substitution, no approximate-field fallback is permitted — this would violate the measured-only rule. For Rodinia artifacts, parse NCU CSV or trace JSON. For mini-transformer, extract from full/dual-source JSON.

Every feature carries `{value, status, source}`. If all 12 are `measured`, produce a `PkaFeatureRecord` row. If any is missing, produce an acquisition gap row. The stage-gate checks: are all P0 objects fully measured? If yes, proceed to Stage 3. If no, emit audit/gap artifacts and stop.

**Stage 3 — PKA Selector**: Input is the `PkaFeatureTable` (only measured-invocation rows). The grouping algorithm follows the PKA paper's approach: first apply dimensionality reduction (PCA or similar) to the 12-dimensional feature space, then apply k-means clustering on the reduced space. The number of clusters (k) and PCA components are recorded in the selector output as part of the `feature_mode` metadata. Select representative via `first_chronological` (earliest `trace_order` in cluster). Output `representative_anchor_table_l1.json` with schema validated against the anchor table contract (required fields present, forbidden fields absent).

**Stage 4 — B-line Consumption**: A lightweight consumer reads the anchor table and validates the interface contract. The consumer does NOT generate family/regime/writeback lineage — this is a parse-and-validate check only. The derivation contract is:

1. **Parse**: Read `representative_anchor_table_l1.json`, parse all rows.
2. **Validate required fields**: For every row, verify presence of `rep_kernel_id`, `kernel_name`, `cluster_id`, `member_invocations`, `coverage_count`, `coverage_weight`, `time_weight`. Record any missing fields with row index and field name.
3. **Validate forbidden fields**: For every row, verify absence of `family_id`, `regime_id`, `route_primitive`, `execution_template`, `simulator_lane_id`, and any compression-side fields. Record any leaked fields with row index and field name.
4. **Report**: Emit `b_line_consumption_report_l1.md` recording: anchor count consumed, required-field check result (pass/fail per row, with missing-field details), forbidden-field check result (pass/fail per row, with leaked-field details), and overall interface status (all rows pass, or N rows with issues).

This consumer does NOT depend on `artifacts/middle_layer/mini_transformer_v4/bundle.json`, does NOT depend on `experiments/backend_pipeline/backend_builder.py`, and does NOT generate family/regime/writeback artifacts. It operates directly on the anchor table and proves only that the A-line output schema is compatible with B-line consumption expectations.

**Stage 5 — Regression Tests**: Implemented alongside each stage. Tests cover: manifest schema validation (valid + invalid inputs), feature table completeness (12 measured fields + gap routing), selector forbidden-field rejection, anchor table schema validation, B-line parse smoke.

### Key Design Decisions from the Draft

1. **Two-layer object model**: `KernelValidationRecord` (validation/audit/traceability) -> `PkaFeatureRecord` (selector input). This separation is mandatory per the draft and prevents engineering metadata from polluting the PKA grouping space.

2. **`measured`-only policy**: Only Nsight-measured or profiler-recorded values may enter the selector. No imputation, no semantic substitution, no default-zero backfill.

3. **Forbidden-field isolation**: The grouping key must be mechanically prevented from accessing `kernel_name`, `grid_dim`, `block_dim`, and compression-side fields. Testing must inject these fields and assert rejection.

4. **Stage-gate rigidness**: If any P0 object cannot produce 12 measured features, the pipeline stops at Stage 2. This is intentional—the draft explicitly says "不能把后续步骤当作'尽力而为'的 smoke test."

5. **`first_chronological` representative selection**: Per the PKA feature spec, the representative is the earliest invocation by `trace_order` in the cluster. This differs from the current codebase's `max_exec_time` rule.

### Relevant References

- `experiments/baseline_diagnosis/frontend_anchor/selector.py` — Current selector (uses forbidden fields; serves as reference for what NOT to use in grouping key but provides useful output structure patterns)
- `experiments/baseline_diagnosis/frontend_anchor/exporter.py` — Anchor table exporter (the `_validate_anchor_table` forbidden-field check is reusable as a pattern; the output schema is a useful starting point)
- `experiments/baseline_diagnosis/frontend_anchor/invocation_table.py` — Invocation table builder (the dual-source alignment logic is reusable for cross-source identity resolution)
- `experiments/baseline_diagnosis/schemas/kernel_validation_manifest_schema.json` — Existing manifest schema (reuse for Stage 1 manifest validation)
- `experiments/backend_pipeline/backend_builder.py` — B-line builder (the family/regime/lane structure is useful for understanding the B-line interface contract; L1 consumer should produce structurally compatible output)
- `docs/a-line-pka-feature-general-spec-2026-04-27.md` — 12 PKA feature definitions, field status rules, and forbidden-field list (normative reference for all stages)
- `docs/a-line-l1-validation-manifest-2026-04-26.md` — L1 validation manifest with 10 P0 + 8 P1 objects (primary input for Stage 1)
- `experiments/baseline_diagnosis/results/microbench/` — Local microbench results for P0 objects (l1_bw_32f, l2_bw_32f, mem_bw, mem_lat, shared_bw, MaxFlops)
- `experiments/baseline_diagnosis/results/rodinia/` — Rodinia results (nn)
- `experiments/mini_transformer/mini_transformer_v4_full.json` — Mini-transformer full feature JSON

## Dependencies and Sequence

### Milestones

1. Milestone 1: Manifest Builder — Machine-readable L1 input
   - Phase A: Parse the L1 manifest document into structured entries with all required fields
   - Phase B: Write `kernel_validation_manifest_l1.json` and validate against the JSON schema
   - Phase C: Run path-existence pre-check on all P0 `source_path` values

2. Milestone 2: Feature Extractor and Audit — 12-dimensional PKA feature table
   - Phase A: Implement the PKA feature extraction logic with per-feature `{value, status, source}` structure
   - Phase B: Implement source adapters for microbench JSON, Rodinia artifacts, and mini-transformer JSON
   - Phase C: Generate `PkaFeatureTable` (measured invocations) and `PkaAcquisitionGap` (incomplete invocations)
   - Phase D: Generate `PkaFeatureAudit` with per-feature provenance
   - Gate: Verify all P0 objects have 12 measured features; if not, emit gap report and stop

3. Milestone 3: PKA Baseline Selector — Forbidden-field-free grouping
   - Phase A: Implement selector that groups purely on the 12-dimensional feature space
   - Phase B: Implement forbidden-field guard (reject any grouping key containing banned fields)
   - Phase C: Implement representative selection with `first_chronological` rule and deterministic tie-breaking
   - Phase D: Output `RepresentativeAnchorTable` with schema validation (required fields present, forbidden fields absent)

4. Milestone 4: B-line Consumption Check — Interface closure proof
   - Phase A: Implement B-line consumer that parses anchor table and validates schema
   - Phase B: Validate required fields and forbidden fields for every anchor row
   - Phase C: Produce `BLineConsumptionReport` recording pass/fail results per row

5. Milestone 5: Regression Tests — Automated constraint verification
   - Phase A: Manifest schema validation tests
   - Phase B: Feature table completeness and gap routing tests
   - Phase C: Selector forbidden-field rejection tests
   - Phase D: Anchor table schema tests
   - Phase E: B-line consumption smoke tests

### Dependency Graph

```
Milestone 1 ──> Milestone 2 ──> Milestone 3 ──> Milestone 4
     │               │               │               │
     └───────────────┴───────┬───────┴───────────────┘
                             │
                        Milestone 5
                      (runs alongside)
```

- Milestone 2 depends on Milestone 1 (needs manifest to locate input files)
- Milestone 3 depends on Milestone 2 passing its gate (all P0 objects measured)
- Milestone 4 depends on Milestone 3 (needs anchor table)
- Milestone 5 tests can be written alongside each milestone but integration tests depend on all milestones completing

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| T1 | Manifest builder: parse L1 manifest document into machine-readable JSON with schema validation and path-existence pre-check | AC-1, AC-1.1 | coding | - |
| T2 | PKA feature extractor: implement 12-feature extraction with per-field `{value, status, source}`, source adapters for microbench/Rodinia/mini-transformer, invocation expansion, `kernel_invocation_id` derivation, and acquisition gap routing | AC-2, AC-6, AC-9, AC-10, AC-12 | coding | T1 |
| T3 | PKA feature audit generator: produce `pka_feature_audit_l1.md`, `pka_feature_audit_l1.json`, and `pka_acquisition_gap_l1.json` | AC-6, AC-7, AC-12 | coding | T2 |
| T4 | Stage-gate validator: enforce that selector (T5) and B-line (T6) stages cannot execute when any P0 invocation has an unresolved acquisition gap; enforce anchor table existence before B-line consumption | AC-3.1, AC-4.1, AC-7 | coding | T2, T3 |
| T5 | PKA baseline selector: implement 12-D feature-space-only grouping with forbidden-field guard, representative selection, and anchor table export; selector refuses to run if the stage-gate (T4) reports blocking gaps | AC-3, AC-3.1, AC-8 | coding | T2, T4 |
| T6 | B-line consumer: parse anchor table, validate required fields and forbidden fields for every row, and produce consumption report with pass/fail results | AC-4, AC-4.1 | coding | T4, T5 |
| T7 | Regression test suite: manifest schema tests, feature completeness tests, stage-gate tests, forbidden-field rejection tests, anchor schema tests, B-line smoke tests; each test module runs independently of pipeline stage completion | AC-1 through AC-12 | coding | T1, T2, T4, T5, T6 |
| T8 | Codex review of PKA feature extraction completeness: verify all 12 features are correctly mapped from Nsight metrics and that source adapters handle edge cases (missing metrics, multi-invocation sources, cross-source alignment) | AC-2, AC-6, AC-9 | analyze | T2 |
| T9 | Codex review of selector forbidden-field isolation: verify no forbidden field leaks into the grouping key and that the 12-D algorithm is coherent | AC-3, AC-8 | analyze | T5 |
| T10 | Codex review of B-line interface contract: verify anchor table schema is sufficient for downstream consumption and that the parse-and-validate consumer correctly checks all required/forbidden fields | AC-4 | analyze | T6 |

## Claude-Codex Deliberation

### Phase 3 Codex Analysis v1 Summary

Codex identified six categories of concerns:

**CORE_RISKS** (7 items): The highest-risk items are (a) the current codebase does not support the 12 PKA feature set—`invocation_table.py` extracts a different 13-field vector; (b) the current selector is built around `kernel_name`/`grid_dim`/`block_dim`, all forbidden by the draft; (c) B-line backend depends on a curated middle-layer bundle, not generic anchor consumption; (d) the heterogeneous input sources lack a unified invocation model and timing unit; (e) the stage-gate is correct but brittle—high probability RLCR stops at acquisition gap in round 1; (f) representative selection differs between current code (`max_exec_time`) and PKA spec (`first_chronological`); (g) existing manifest schema is for high-level entries, not invocation-level records with feature payloads.

**MISSING_REQUIREMENTS** (11 items): Schemas for all five output artifact classes; source-adapter contracts per `source_type`; deterministic `kernel_invocation_id` rule; definition of `measured` for launch metadata; formal grouping algorithm; formal representative selection rule; mixed timing unit policy; B-line "consumption" definition; singleton/one-invocation kernel handling; artifact versioning and path conventions; P1 object policy during round 1.

**TECHNICAL_GAPS** (8 items): Selector is not a PKA-12 selector; invocation_table.py produces non-PKA features; build_frontend_anchor_outputs.py hardcodes current modes; no invocation-level record schema; no anchor-table schema validator; backend consumption not wired to frontend output; planner/writeback downstream of missing family/regime step; existing tests validate current flow, not draft contract.

**ALTERNATIVE_DIRECTIONS** (6 items): Strict PKA-first L1; split L1 into L1A/L1B; add `pka_baseline` mode beside existing modes; narrow B-line to stub consumer; run round 1 on single source family; relax `num_thread_blocks` to `recorded_launch`.

### Agreements

- **Stage-gate is architecturally correct**: Both Claude and Codex agree the rigid stage-gate (stop at acquisition gap if any P0 incomplete) is the right architectural decision for L1's purpose as a correctness gate, even though it creates a high probability of stopping before selector/B-line in round 1.
- **Two-layer object model is necessary**: Both agree `KernelValidationRecord` -> `PkaFeatureRecord` is essential to prevent engineering metadata from contaminating PKA grouping.
- **Current codebase is a different feature space**: Both agree the existing selector and invocation table operate on a different feature set and must be either replaced or supplemented for L1—this is not a tuning issue but a definitional one.
- **B-line "consumption" needs explicit scoping**: Both agree the draft's B-line consumption check is underspecified and needs a concrete definition before implementation.

### Resolved Disagreements

- **Selector implementation strategy**: Codex proposed keeping current selector modes as evidence-only and adding `pka_baseline` as a parallel mode. Claude initially considered replacing the selector entirely. **Resolution**: The plan keeps current selector modes untouched (no deprecation, no refactoring) and implements the new PKA selector in a clearly separated module. This preserves the existing evidence pipeline while adding the draft-contract selector. Rationale: lower risk, preserves debug capability, and the draft does not require removing existing modes.

- **B-line consumption depth**: Codex questioned whether "parse-only" counts as consumption. Claude proposed full family/regime/writeback generation. **Resolution**: User chose parse-only. The B-line consumer validates required/forbidden fields in the anchor table and produces a consumption report with pass/fail per row. This proves schema compatibility without generating family/regime/writeback lineage. Rationale: L1 is a correctness gate; proving the anchor table schema is compatible with B-line expectations is sufficient for L1.

- **Representative selection rule**: Codex noted the current code uses `max_exec_time` while the PKA spec says `first_chronological`. **Resolution**: The plan adopts `first_chronological` as the L1 rule, matching the PKA feature spec. Rationale: the PKA spec explicitly recommends `first_chronological` for the first version, and L1 is about PKA baseline fidelity.

- **`num_thread_blocks` provenance strictness**: Codex suggested relaxing the requirement to accept existing `#blocks/total_threadblocks` from local artifacts. Claude insisted on profiler/launch-metadata origin. **Resolution**: The plan keeps the strict requirement (profiler or launch metadata only) but documents in the Acquisition Risk note that this is likely the hardest feature to source and may need explicit acquisition work before L1 can pass Stage 2. Rationale: the draft explicitly says `num_thread_blocks` must come from "profiler / launch metadata 记录" and the PKA spec confirms `launch_grid_size` as the canonical source.

### Convergence Round 1 (Claude v1 -> Codex v2 -> Claude v2)

**Round 1 Convergence Matrix:**

| Topic | Claude Position (v1) | Codex Position (v2) | Resolution |
|-------|---------------------|---------------------|------------|
| Object-to-invocation cardinality | Not defined; AC-6 assumed one-object-one-outcome | REQUIRED_CHANGES: multi-invocation sources break AC-6 | resolved — added cardinality definition; AC-6 now per-invocation; `kernel_invocation_id` derives during invocation expansion |
| Lower bound vs stage gate | Lower bound allowed microbench-only adapter | REQUIRED_CHANGES: microbench-only contradicts stage gate since Rodinia/AI P0s remain | resolved — lower bound redefined: all adapters required; stage-gate determines runtime output; both audit-stop and loop-closure are valid lower-bound outcomes |
| AC-4/AC-11/T7 consistency | AC-4 required family/regime/writeback; lower bound allowed parse-only | REQUIRED_CHANGES: three sections define different minimums | resolved — lower bound unified: all adapters + full audit; B-line may be parse-only at lower bound; AC-4 now reflects parse-or-generate |
| Manifest schema contract | Used `validation_id` and `source_path` | REQUIRED_CHANGES: existing schema uses `id` and `local_input_path` | resolved — AC-1 now maps draft concepts to existing schema fields |
| Stage 2 semantic substitution | "Map existing metric names to PKA metric names where possible" | REQUIRED_CHANGES: this is semantic substitution, violating measured-only rule | resolved — removed mapping language; Stage 2 checks for exact Nsight metric names only; absent metrics route to gap report |
| B-line derivation contract | "Generate family/regime/writeback lineage" without specification | REQUIRED_CHANGES: not implementable without a concrete contract | resolved — added deterministic 5-step stub derivation contract |
| Task dependency graph | T5 depended on T4 while gating T4 (circular); AC-9 on T1 | REQUIRED_CHANGES: T4-T5 circular; AC-9 belongs to T2 | resolved — T4 (stage-gate) now precedes T5 (selector); T5 depends on T2+T4; AC-9 moved to T2 |
| Risk note accuracy | "Likely missing some metrics" | DISAGREE: canonical 12 PKA metrics are likely entirely absent, not just "some" | resolved — risk notes upgraded to CRITICAL; explicitly state metrics are likely absent; 10x12 matrix recommended as first Stage 2 output |
| Exact-vector grouping sufficiency | "Likely sufficient" for L1 | DISAGREE: near-singletons prove determinism but not behavioral sanity | resolved — noted as a known limitation; upper bound allows bucketed grouping |

**Unresolved (carried to Pending User Decisions):**
- What counts as L1 success: The plan now defines two valid success modes. The user must confirm this framing.
- B-line success threshold: The plan defines both parse-only (lower bound) and stub-generation (upper bound). The user must choose.
- Mixed timing-unit policy: Retained as DEC-9.

### Convergence Round 2 (Claude v2 -> Codex v3) — Not Executed

Round 1 resolved all REQUIRED_CHANGES. No DISAGREE items remain unresolved. The two UNRESOLVED items from Codex v2 are carried to Pending User Decisions. Per loop termination rules: no REQUIRED_CHANGES remain and no high-impact DISAGREE remains. Convergence achieved after 1 round.

### Convergence Status

- **Final Status**: `converged`
- **Rounds Executed**: 1 (of max 3)
- **Agreements**: Two-layer object model, keeping old selector modes untouched, measured-only policy, forbidden-field isolation — both Claude and Codex agree these are correct
- **Resolved Disagreements**: 7 REQUIRED_CHANGES and 2 DISAGREE items addressed in Claude v2
- **Carried to User**: 2 UNRESOLVED items (L1 success definition, B-line threshold) merged into Pending User Decisions

## Pending User Decisions

The following items were identified during Phase 3 (Codex analysis v1), Phase 4 (Claude plan synthesis), Phase 5 (Convergence Loop), and Phase 6 (User Resolution).

### Resolved

- DEC-1: New `pka_baseline` selector implementation strategy → **Resolved**: Implement new selector in separate module; leave current modes untouched. Both Claude and Codex agree.
- DEC-2: B-line success definition → **Resolved by user**: Parse-only. B-line consumer validates required/forbidden fields in the anchor table; does not generate family/regime/writeback lineage.
- DEC-3: P0 acquisition gap behavior → **Resolved**: Emit audit/gap artifacts and mark round as "blocked on acquisition" — not a failure. Both valid outcomes captured in the Goal.
- DEC-4: Manifest schema strategy → **Resolved**: Reuse existing `kernel_validation_manifest_schema.json` with its field names (`id`, `local_input_path`). New schemas for `KernelValidationRecord` and `PkaFeatureRecord` as needed.
- DEC-5: Grouping algorithm → **Resolved by user**: PKA paper-faithful — dimensionality reduction (PCA-like) followed by k-means clustering on the reduced feature space.
- DEC-6: Representative selection rule → **Resolved**: `first_chronological` per PKA feature spec.
- DEC-7: `num_thread_blocks` provenance → **Resolved by user**: Strict — must come from profiler or launch metadata. No relaxation.
- DEC-8: P1 objects in manifest → **Resolved**: Include P1 entries in manifest with non-blocking status. P1 gaps do not trigger stage-gate stop.
- DEC-9: Mixed timing units policy → **Resolved**: Global rejection by default. Weight computation aborts on mixed `duration_ns` / `elapsed_cycles`. Selector does not consume timing, so mixed timing only matters when computing coverage weights.
- DEC-10: Old selector modes disposition → **Resolved**: Retain untouched (neither deprecated nor refactored). New PKA selector is in a separate module.

### Pending

No pending user decisions remain. All 10 DEC items are resolved.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### L1-Specific Constraints
- All 12 PKA features must be in `measured` status before entering the selector; no imputation, no default values, no semantic substitution
- `kernel_name` may appear in metadata/audit fields but must not enter the grouping key
- `expected_behavior_axis` is for human sanity-check only and must not be used for grouping or labeling
- The stage-gate is rigid: if any P0 object has an unresolved acquisition gap, Stages 3 and 4 must not execute
- Output artifacts follow the paths defined in the draft: `artifacts/a_line/l1/` for all L1 outputs
- The `RepresentativeAnchorTable` must not contain `family_id`, `regime_id`, `route_primitive`, `execution_template`, or `simulator_lane_id` fields
- Mixed timing units across invocations cause weight computation to abort by default

### Risk Notes
- **Acquisition Risk (CRITICAL)**: The canonical 12 PKA Nsight metric names (e.g., `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, `smsp__sass_inst_executed_op_global_atom.sum`) are likely entirely absent from the existing in-repo artifacts. The current codebase uses a different 13-field feature vector (compute_throughput_pct, dram_throughput_pct, ipc_active, etc.) that is not a subset of the PKA 12. This means round 1 is primarily an acquisition-readiness exercise: it will identify exactly which metrics are missing from which P0 objects, producing a detailed gap report as its primary deliverable. Reaching Stage 3 (selector) or Stage 4 (B-line) in round 1 is unlikely without new NCU data collection. This is not a plan defect — it is the expected behavior of a correctness gate applied to existing data collected under a different feature regime.
- **NCU Data Availability**: The 12 PKA metrics are: 3 `l1tex__t_sectors_pipe_lsu_mem_global_op_*`, 6 `smsp__inst_executed_op_*`, 1 `smsp__sass_inst_executed_op_global_atom`, 1 `smsp__inst_executed`, and 1 `smsp__thread_inst_executed_per_inst_executed`. Each of these must be present in the source artifact with the exact Nsight metric name to qualify as `measured`. Existing artifacts in `experiments/baseline_diagnosis/results/microbench/` use a different metric vocabulary. A 10-by-12 metric-availability matrix should be the first output of Stage 2 to make the acquisition status immediately visible.
- **Backend Coupling Risk (HIGH)**: The current B-line (`backend_builder.py`) depends on a curated middle-layer bundle at `artifacts/middle_layer/mini_transformer_v4/bundle.json` with hardcoded anchor/family/regime IDs. The L1 B-line consumer must NOT depend on this curated bundle; it must implement the stub derivation contract described in Stage 4. This is a new code path that shares no code with the existing `backend_builder.py`.

### Output File Convention

The translated language variant is not generated (`ALT_PLAN_LANGUAGE` is empty).

--- Original Design Draft Start ---

# A 线 L1 RLCR Spec

日期：2026-04-27

## 1. 目标

这份 spec 定义 A 线第一轮 `L1 RLCR` 的任务边界、输入、输出和验收标准。

当前 L1 的核心目标不是证明 A 线 compression 效果最好，
而是先证明：

**PKA baseline 的输入、特征、anchor 输出，以及 B 线消费接口能够在一组小而可解释的 kernel 上稳定闭环。**

因此 L1 是：

- correctness gate
- feature sanity gate
- downstream interface gate

而不是：

- compression quality gate
- large-scale benchmark evaluation
- extension superiority proof

---

## 2. 为什么 L1 要独立跑 RLCR

L1 和 L2 关注的问题不同。

L1 关注：

- 输入字段是否可信；
- PKA 12 维特征能否被稳定抽取；
- baseline selector 是否错误依赖 `kernel_name` / `grid_dim` / `block_dim`；
- `Representative Anchor Table` 是否能被 B 线消费；
- B 线 family / regime / writeback 是否能在小集合上跑通。

L2 关注：

- 压缩率；
- top-k coverage；
- cluster 内方差；
- 大样本稳定性；
- baseline 与 extension 的统计差异。

如果 L1 和 L2 混在一轮 RLCR 中，
很容易出现两个问题：

1. L1 的接口问题被 L2 的数据规模问题掩盖；
2. L2 的 compression 结果不稳定时，很难判断根因来自 A 线特征、selector、数据采集，还是 B 线消费逻辑。

因此，当前推荐顺序是：

`L1 RLCR -> L2 RLCR`

其中 L2 必须继承 L1 的 schema、feature extractor 和 anchor 输出契约。

---

## 3. L1 的输入范围

L1 使用已有的 `L1 基础验证集 Manifest` 作为输入清单：

- `L1_MB_*`: canonical microbench
- `L1_RD_*`: 少量 Rodinia / benchmark kernel
- `L1_AI_*`: mini-transformer target kernels

参考文件：

- `docs/a-line-l1-validation-manifest-2026-04-26.md`
- `experiments/baseline_diagnosis/schemas/kernel_validation_manifest_schema.json`

### 3.1 第一批必须接入对象

第一批只要求接入 P0 对象：

- `l1_bw_32f`
- `l2_bw_32f`
- `mem_bw`
- `mem_lat`
- `shared_bw`
- `MaxFlops`
- Rodinia `nn`
- mini-transformer `gemm_tiled`
- mini-transformer `attention_score`
- mini-transformer `softmax_kernel`

### 3.2 第二批可选对象

在第一批跑通后，再接入：

- `shared_lat`
- `atomic_add_bw`
- `atomic_add_lat`
- Rodinia `backprop`
- mini-transformer `context_mul`
- mini-transformer `layernorm_kernel`
- mini-transformer `residual_add`

### 3.3 输入约束

L1 允许使用已有仓库结果，
但不能让不同来源的原始格式直接进入 PKA selector。

原因是 L1 的输入来自：

- microbench JSON
- Rodinia 本地结果
- mini-transformer full JSON / feature sources
- 后续可能补充的 NCU CSV

这些来源的字段命名、粒度和可信度并不一致。
因此 L1 必须先经过一层工程适配。

这里明确区分两类对象：

1. `KernelValidationRecord`
2. `PkaFeatureRecord`

其中：

- `KernelValidationRecord` 是验证集对象，用于溯源、审计和回归；
- `PkaFeatureRecord` 是 PKA baseline selector 的真正输入。

`KernelValidationRecord` 不是 PKA 方法本身的一部分，
也不代表 PKA 论文内部存在同名处理步骤。
它只是我们为了在混合来源数据上复现 PKA baseline 而引入的工程输入适配层。

#### 3.3.1 `KernelValidationRecord`

`KernelValidationRecord` 至少包含：


- `validation_id`
- `dataset_level`
- `source_type`
- `benchmark_name`
- `kernel_or_case`
- `kernel_invocation_id`
- `kernel_name`
- `exec_time_or_cycle_observed`
- `expected_behavior_axis`
- `pka_feature_vector`
- `feature_status`
- `source_path`

各字段作用如下：

| 字段 | 作用 | 是否允许进入 PKA selector |
|---|---|---|
| `validation_id` | L1 验证对象的稳定 id，用于测试、报告和错误定位 | `No` |
| `dataset_level` | 标明对象属于 `L1` 还是后续 `L2`，防止混用验收标准 | `No` |
| `source_type` | 标明来源类型，例如 microbench、Rodinia、AI workload | `No` |
| `benchmark_name` | 记录 benchmark 名称，用于人工审查和报告 | `No` |
| `kernel_or_case` | 记录该对象对应的 kernel 或 benchmark case 名称 | `No` |
| `kernel_invocation_id` | invocation 级唯一标识，用于 membership 和 writeback | `Identity only` |
| `kernel_name` | 原始 kernel 名称，用于溯源和报告 | `No` |
| `exec_time_or_cycle_observed` | 记录实测时间或周期，用于 weight / audit，不作为 PKA 主特征 | `No` |
| `expected_behavior_axis` | 人工预期行为轴，用于 sanity check，不作为聚类标签 | `No` |
| `pka_feature_vector` | PKA 12 维特征容器，后续转换成 `PkaFeatureRecord.features` | `Only contained 12 features` |
| `feature_status` | 记录 12 维特征是否均已实测；未采齐时标记 acquisition incomplete | `Audit only` |
| `source_path` | 原始数据路径，用于复现和追踪问题 | `No` |

关键约束：

- `expected_behavior_axis` 只能用于 sanity check，不能用于 grouping；
- `kernel_name` 只能用于溯源，不能用于 PKA baseline 主 grouping；
- `source_type` / `benchmark_name` / `dataset_level` 不能进入 selector；
- `pka_feature_vector` 中只有通用 PKA feature spec 定义的 12 个字段可以进入 selector。

#### 3.3.2 `PkaFeatureRecord`

`PkaFeatureRecord` 是 selector 的真正输入对象。

它从 `KernelValidationRecord` 派生得到，
只保留：

- 最小 identity
- PKA 12 维 feature values
- 每个 feature 的状态
- 必要的 audit metadata

建议结构如下：

```json
{
  "record_id": "L1_MB_01",
  "kernel_invocation_id": "l1_bw_32f#1",
  "feature_mode": "pka_l1_measured_only",
  "features": {
    "coalesced_global_loads": {"value": 0.0, "status": "measured", "source": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"},
    "coalesced_global_stores": {"value": 0.0, "status": "measured", "source": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"},
    "coalesced_local_loads": {"value": 0.0, "status": "measured", "source": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"},
    "thread_global_loads": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed_op_global_ld.sum"},
    "thread_global_stores": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed_op_global_st.sum"},
    "thread_local_loads": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed_op_local_ld.sum"},
    "thread_shared_loads": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed_op_shared_ld.sum"},
    "thread_shared_stores": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed_op_shared_st.sum"},
    "thread_global_atomics": {"value": 0.0, "status": "measured", "source": "smsp__sass_inst_executed_op_global_atom.sum"},
    "num_instructions": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed.sum"},
    "divergence_efficiency": {"value": 0.0, "status": "measured", "source": "smsp__thread_inst_executed_per_inst_executed.ratio"},
    "num_thread_blocks": {"value": 0.0, "status": "measured", "source": "launch_grid_size"}
  },
  "metadata": {
    "kernel_name": "l1_bw_32f",
    "source_path": "experiments/baseline_diagnosis/results/microbench/l1_bw_32f.json",
    "expected_behavior_axis": "L1 bandwidth / coalesced load-heavy"
  }
}
```

Selector 只能读取：

- `record_id`
- `kernel_invocation_id`
- `features`
- `feature_mode`

Selector 不得使用：

- `metadata.kernel_name`
- `metadata.source_path`
- `metadata.expected_behavior_axis`

#### 3.3.3 为什么需要两层对象

两层对象的目的不是增加方法复杂度，
而是防止 PKA baseline 被工程 metadata 污染。

`KernelValidationRecord` 解决：

- 这个验证对象来自哪里；
- 预期行为是什么；
- 原始数据是否可追踪；
- 字段状态是否可信。

`PkaFeatureRecord` 解决：

- 哪些数值真正进入 PKA behavior feature space；
- selector 实际使用了哪些字段；
- 当前结果是 `pka_complete` 还是 `pka_l1_measured_only`。

因此，L1 的数据流应固定为：

`raw local result -> KernelValidationRecord -> PkaFeatureRecord -> pka_baseline selector`

---

## 4. PKA 12 维特征要求

L1 必须以 PKA 12 维信号作为 feature extraction 的目标字段。

通用字段契约见：

- `docs/a-line-pka-feature-general-spec-2026-04-27.md`

目标字段如下：

| 字段 | 含义 | PKA / Nsight metric name | L1 处理要求 |
|---|---|---|
| `coalesced_global_loads` | 合并全局加载 | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `coalesced_global_stores` | 合并全局存储 | `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `coalesced_local_loads` | 合并局部加载 | `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_global_loads` | 线程级全局加载 | `smsp__inst_executed_op_global_ld.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_global_stores` | 线程级全局存储 | `smsp__inst_executed_op_global_st.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_local_loads` | 线程级局部加载 | `smsp__inst_executed_op_local_ld.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_shared_loads` | 线程级共享内存加载 | `smsp__inst_executed_op_shared_ld.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_shared_stores` | 线程级共享内存存储 | `smsp__inst_executed_op_shared_st.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_global_atomics` | 全局原子操作 | `smsp__sass_inst_executed_op_global_atom.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `num_instructions` | 总指令数 | `smsp__inst_executed.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `divergence_efficiency` | 分支发散效率 | `smsp__thread_inst_executed_per_inst_executed.ratio` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `num_thread_blocks` | 线程块数量 | `launch_grid_size` | 必须从 profiler / launch metadata 记录；状态标记为 `measured` |

### 4.1 字段状态

进入 `PkaFeatureRecord.features` 的每个字段必须带状态：

- `measured`

如果任意 PKA 字段没有 measured value，
该 invocation 不得生成可供 selector 消费的 `PkaFeatureRecord`，
只能进入 acquisition gap report。

L1 spec 只定义 `measured` feature status，
避免实现层把替代字段误接入正式 baseline。

### 4.2 Selector 字段策略

L1 阶段正式 selector 只允许使用本 12 维字段中状态为：

- `measured`

的字段。

但每次 selector 运行必须输出：

- 实际使用字段列表；
- 每个使用字段的状态；
- `feature_mode`: `pka_l1_measured_only` 或 `pka_complete`。

未采集到的字段必须保留在 acquisition gap report 中，
用于说明当前 NCU acquisition 还缺什么。
gap report 不得进入正式 `pka_baseline` 主 grouping。

禁止进入 `pka_baseline` 主 grouping 的字段包括：

- `kernel_name`
- `grid_dim` string
- `block_dim` string
- `shape_hint`
- `trace_order`
- family / regime / lane 字段
- squash / batch / delta 机制字段

### 4.3 L1 的最低通过条件

第一版 L1 的最低通过条件与正式 selector 条件一致：

- PKA 12 维字段必须全部以 `measured` 状态稳定生成；
- `num_thread_blocks` 必须来自 profiler / launch metadata 记录；
- 任一字段未采齐时，该 invocation 只能进入 acquisition gap report。

如果 12 维 measured feature table 无法稳定生成，
则 L1 不应继续进入 selector / B 线消费阶段。

### 4.4 Stage-gate 执行约束

L1 必须按 stage-gate 执行，
不能把后续步骤当作“尽力而为”的 smoke test。

执行顺序固定为：

1. 先完成 trace / NCU acquisition；
2. 再生成 12 维 measured `PkaFeatureTable`；
3. 只有 feature table 通过完整性检查后，才能运行 `pka_baseline` selector；
4. 只有 selector 产出 `RepresentativeAnchorTable` 后，才能进入 B 线消费检查。

如果任一 P0 invocation 无法读取或生成完整 12 维 measured feature，
当前 stage 必须停止，
并回到 trace / NCU acquisition 部分继续迭代。

此时允许输出：

- `pka_feature_audit_l1.md`
- `pka_feature_audit_l1.json`
- `pka_acquisition_gap_l1.json`

但禁止输出或消费：

- `representative_anchor_table_l1.json`
- `b_line_consumption_report_l1.md`

原因是 selector 和 B 线消费都依赖已经完成的 PKA baseline 输入。
如果输入阶段没有通过，
后续 anchor / family / regime 结果都没有可信基础。

---

## 5. L1 输出产物

L1 RLCR 完成后至少应产出下面 5 类文件。

### 5.1 `KernelValidationManifest`

用途：

- 固定 L1 输入对象；
- 记录每个对象的来源、状态、预期行为轴和优先级。

建议路径：

- `artifacts/a_line/l1/kernel_validation_manifest_l1.json`

### 5.2 `PkaFeatureTable`

用途：

- 将 L1 对象统一转换成 PKA 12 维 feature table。

建议路径：

- `artifacts/a_line/l1/pka_feature_table_l1.json`

### 5.3 `PkaFeatureAudit`

用途：

- 记录每个 PKA 字段是否已经获得 measured value；
- 明确哪些字段仍需要后续 NCU acquisition 补齐；
- 对未采齐对象输出 acquisition gap report，而不是补齐 feature table。

建议路径：

- `artifacts/a_line/l1/pka_feature_audit_l1.md`
- `artifacts/a_line/l1/pka_feature_audit_l1.json`
- `artifacts/a_line/l1/pka_acquisition_gap_l1.json`

### 5.4 `RepresentativeAnchorTable`

用途：

- 在 L1 feature table 上运行 `pka_baseline` selector；
- 输出 representative anchors、membership 和 weight。

建议路径：

- `artifacts/a_line/l1/representative_anchor_table_l1.json`

### 5.5 `BLineConsumptionReport`

用途：

- 验证 B 线能消费 L1 anchor；
- 输出 family / regime / writeback 最小闭环状态。

建议路径：

- `artifacts/a_line/l1/b_line_consumption_report_l1.md`

---

## 6. L1 RLCR 工作包

### Task 1：Manifest builder

目标：

- 将 `docs/a-line-l1-validation-manifest-2026-04-26.md` 中的对象转成机器可读 manifest。

输入：

- L1 manifest 文档
- 本地已有结果路径

输出：

- `kernel_validation_manifest_l1.json`

验收：

- 每个 P0 对象都有稳定 id、source path、expected behavior axis；
- manifest 能通过 `kernel_validation_manifest_schema.json`。

### Task 2：PKA feature extractor

目标：

- 从 L1 对象中抽取 PKA 12 维 feature table。
- 如果无法抽取完整 12 维 measured feature，
  则迭代 trace / NCU acquisition，
  不进入 selector。

输入：

- PKA NCU CSV / profile report
- profiler / launch metadata
- L1 manifest 中记录的 source path

输出：

- `pka_feature_table_l1.json`
- `pka_feature_audit_l1.json`
- `pka_feature_audit_l1.md`
- `pka_acquisition_gap_l1.json`

验收：

- 只有 12 维均为 measured 的对象才能进入 feature table；
- 每个进入 feature table 的 PKA 字段都有 value、status=`measured` 和 source；
- `num_instructions`、`divergence_efficiency`、`num_thread_blocks` 必须为 measured；
- 未采齐对象必须进入 acquisition gap report。
- 如果任何 P0 对象未采齐 12 维 measured feature，
  本 task 判定为 blocked on acquisition，
  后续 Task 3 / Task 4 不得执行。

### Task 3：PKA baseline selector

目标：

- 在 L1 PKA feature table 上实现或调用 `pka_baseline` selector。

前置条件：

- Task 2 必须通过；
- `pka_acquisition_gap_l1.json` 中不得存在阻塞 P0 对象；
- 输入 feature table 中每个字段 status 都必须为 `measured`。

要求：

- 主 grouping 不依赖 `kernel_name`；
- 主 grouping 不依赖 `grid_dim` / `block_dim`；
- 主 grouping 不使用 compression-side / downstream-side 字段。

输出：

- `representative_anchor_table_l1.json`

验收：

- 每个 anchor 有 explicit membership；
- 每个 anchor 有 coverage count / weight；
- 每个 anchor 有代表对象；
- 输出不包含 family / regime / lane 字段。

### Task 4：B 线消费检查

目标：

- 用 L1 anchor table 验证 B 线是否能稳定消费 A 线输出。

前置条件：

- Task 3 必须通过；
- `representative_anchor_table_l1.json` 必须存在且 schema 完整；
- 不允许直接消费 acquisition gap report 或不完整 feature table。

输出：

- `b_line_consumption_report_l1.md`

验收：

- B 线能读取 anchor table；
- 能生成或更新 family / regime 的最小对象；
- 不要求 family 结论最终正确；
- 只要求接口完整、字段齐、writeback 关系不断裂。

### Task 5：L1 regression tests

目标：

- 把 L1 的核心约束转成测试。

建议测试：

- manifest schema validation；
- PKA feature table required fields；
- selector forbidden field check；
- representative anchor output schema；
- B line consumption smoke test。

---

## 7. 明确不做的事情

L1 RLCR 不做：

- 大规模 benchmark acquisition；
- Rodinia / Altis 全量跑通；
- CUTLASS sweep；
- HeCBench 泛化；
- baseline vs extension 的统计显著性分析；
- compression ratio 最大化；
- family / regime 最终正确性证明。

这些属于 L2 或后续 B/C 线验证。

---

## 8. 验收标准

L1 RLCR 完成时，必须满足下面条件。

### AC-1：L1 manifest 可机器读取

至少所有 P0 对象被写入 JSON manifest，
并能通过 schema 检查。

### AC-2：PKA feature table 可生成

进入 feature table 的所有对象都有 PKA 12 维 measured 字段。
采不齐的 P0 对象必须进入 acquisition gap report，
不得用替代字段补齐。
如果 P0 对象仍存在 acquisition gap，
L1 必须回到 trace / NCU acquisition 继续迭代，
不得进入 selector。

### AC-3：baseline selector 不依赖禁止字段

前置条件：

- AC-2 必须通过；
- 否则 AC-3 不执行。

`pka_baseline` 主 grouping 不使用：

- `kernel_name`
- `grid_dim`
- `block_dim`
- `cross_tb_offset_coverage`
- squash boundary fields
- family / regime / lane 字段

### AC-4：anchor 输出可被 B 线消费

前置条件：

- AC-3 必须通过；
- 否则 AC-4 不执行。

B 线能读取 L1 anchor table，
并完成最小 family / regime / writeback 接口检查。

### AC-5：测试可回归

至少存在一组自动化测试或检查脚本，
能覆盖：

- manifest schema
- feature table completeness
- selector forbidden fields
- anchor output schema
- B line smoke consumption

---

## 9. 与 L2 的接口

L1 完成后，必须向 L2 输出稳定接口：

- `KernelValidationManifest` schema
- `PkaFeatureTable` schema / implicit contract
- `pka_baseline` selector contract
- `RepresentativeAnchorTable` schema / implicit contract
- L1 regression tests

L2 只允许扩大输入规模和增加 compression quality metrics，
不应重新定义 L1 已经固定的核心 schema。

---

## 10. 建议执行顺序

建议下一轮 RLCR 按下面顺序执行：

1. `Manifest builder`
2. `Trace / NCU acquisition`
3. `PKA feature extractor`
4. `PKA feature audit`
5. `Gate: 12 measured PKA features complete`
6. `pka_baseline selector`
7. `RepresentativeAnchorTable` export
8. `B line consumption smoke`
9. `L1 regression tests`

如果第 5 步失败，
必须回到第 2 步继续迭代采集，
不能继续第 6 步。

---

## 11. 简短结论

L1 RLCR 的目标可以压成一句话：

**先用一组小而可解释的 kernel，把 PKA baseline 输入、特征、anchor 输出和 B 线消费接口打稳。**

只有 L1 通过后，L2 才应该开始做大规模 acquisition 和 compression quality evaluation。

--- Original Design Draft End ---
