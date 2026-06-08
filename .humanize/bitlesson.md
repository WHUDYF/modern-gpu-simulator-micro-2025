# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

<!-- Add lessons below using the strict template. -->

## Lesson: formal-export-path
Lesson ID: BL-20260606-formal-export-path
Scope: experiments/gcl_phase_b/embedding_export.py, experiments/gcl_phase_b/pipeline.py, experiments/gcl_phase_b/readout.py
Problem Description: A helper-level readout implementation can satisfy unit tests while the formal artifact export path still uses an older pooling method.
Root Cause: Gate 5 export reused Phase A `encoder.encode_kernel()` instead of deriving `kernel_embedding_table.json` rows from the Phase B readout manifest.
Solution: Add a Phase B embedding export function that computes node embeddings, calls `build_readout_manifest()`, and writes embedding rows with `readout_manifest_hash` and the CTA-aware hierarchy in `weight_input`.
Constraints: Keep Phase A export unchanged; Phase B selector must validate Phase B embedding tables directly.
Validation Evidence: `pytest -q tests/gcl_phase_b/test_embedding_export.py`; `pytest -q tests/gcl_phase_a tests/gcl_phase_b` passed in Round 1.
Source Rounds: 1

## Lesson: source-invocation-identity
Lesson ID: BL-20260606-source-invocation-identity
Scope: experiments/gcl_phase_b/resnet50_adapter.py, tests/gcl_phase_b/test_resnet50_adapter.py
Problem Description: Raw trace adapter records can collapse repeated launches when fallback provenance maps only by `kernel_id`.
Root Cause: `kernel_id` is a static/function identity and is not unique per launch; using it as a key overwrites earlier invocations when repeated launches share the same kernel ID.
Solution: Preserve explicit `kernel_invocation_id` when present, otherwise align raw records by `launch_order`; reject repeated `kernel_id` records without `kernel_invocation_id` or `launch_order`.
Constraints: Ambiguous raw records must fail fast instead of silently merging launches; launch-order fallback is only valid when raw records include launch order.
Validation Evidence: `pytest -q tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py`; `pytest -q tests/gcl_phase_a tests/gcl_phase_b` passed in Round 2.
Source Rounds: 2

## Lesson: scheduler-trace-reconciliation
Lesson ID: BL-20260606-scheduler-trace-reconciliation
Scope: experiments/gcl_phase_b/resnet50_adapter.py
Problem Description: A trace adapter can emit a passed bundle even when CTA scheduler metadata and per-warp trace records disagree.
Root Cause: Validation only checked that each invocation had some scheduler and trace records, not that each `(kernel_invocation_id, cta_id)` had matching CTA presence, warp IDs, trace entry counts, and scheduler ordering.
Solution: Aggregate per-warp trace records by `(kernel_invocation_id, cta_id)` and validate CTA set equality, warp ID equality, exact trace-entry count equality, positive counts, and `first_seen_order <= last_seen_order`.
Constraints: Fixture scheduler metadata must include only CTAs with corresponding trace records; debug/proxy scheduler records cannot enter formal Gate 1 bundles.
Validation Evidence: `pytest -q tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py`; `pytest -q tests/gcl_phase_a tests/gcl_phase_b` passed in Round 3.
Source Rounds: 3

## Lesson: persisted-formal-boundaries
Lesson ID: BL-20260606-persisted-formal-boundaries
Scope: experiments/gcl_phase_b/resnet50_gate0.py, experiments/gcl_phase_b/selector.py, experiments/gcl_phase_b/resnet50_gate_pipeline.py
Problem Description: Formal GCL gates can be bypassed when validation only checks self-consistency among caller-controlled or root-local JSON dictionaries.
Root Cause: Gate0 accepted a handwritten attestation over the same untrusted root, and Gate6 accepted in-memory Gate5 manifest dicts supplied by the caller instead of loading persisted artifacts.
Solution: Bind Gate0 formal acceptance to acquisition-produced session/attestation artifacts, and make Gate6 formal validation load Gate5 lineage and manifest files from a persisted artifact root before recomputing hashes.
Constraints: This does not prove real ResNet-50 trace availability; fixture-backed producer-path tests remain contract coverage only.
Validation Evidence: `pytest -q tests/gcl_resnet50 tests/gcl_phase_b` passed with 179 tests; `pytest -q tests/gcl_phase_a tests/gcl_phase_b tests/gcl_resnet50` passed with 242 tests; `git diff --check` passed.
Source Rounds: 7

## Lesson: real-root-schema-contract
Lesson ID: BL-20260607-real-root-schema-contract
Scope: experiments/gcl_phase_b/resnet50_adapter.py, experiments/gcl_phase_b/graph_builder.py, tests/gcl_resnet50/real_chain.py
Problem Description: Fixture and artifact-shape tests can pass while the formal Gate1 loader fails on the real Gate0 root because the real NVBit artifact layout and SASS operand schema differ from synthetic fixtures.
Root Cause: Gate1 read root-level `enhanced_execution_info.json` and required fixture-only `launch_order` / `threadblock_pb` fields, while the real root stores static metadata at `extra_info/enhanced_execution_info.json`, represents instructions under `kernels[].instructions`, identifies invocations as `d_<device>_s_<stream>_k_<kernel>`, and uses compound memory operands such as `desc[UR10][R4.64]`.
Solution: Add real-root schema normalization, derive threadblock paths from scheduler invocation metadata, align invocation IDs with real scheduler metadata, materialize representative-SM CTA trace records from real threadblock protobufs, and parse compound memory operands into register-version address sources.
Constraints: Regression tests may use a deterministic real-root invocation slice for runtime, but the slice must preserve real Gate0 provenance and must not be replaced by synthetic fixture data.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate1_adapter.py tests/gcl_resnet50/test_gate2_representative_sm.py tests/gcl_resnet50/test_gate3_canonical_graph.py tests/gcl_resnet50/test_gate4_tensorization.py tests/gcl_resnet50/test_gate5_rgcn_training.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py` passed with 29 tests; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 80 tests; `git diff --check` passed.
Source Rounds: 1

## Lesson: gate1-slice-before-materialization
Lesson ID: BL-20260607-gate1-slice-before-materialization
Scope: experiments/gcl_phase_b/resnet50_adapter.py, experiments/gcl_phase_b/resnet50_gate_pipeline.py, tests/gcl_resnet50/real_chain.py
Problem Description: A real-root regression may appear tractable because downstream artifacts are sliced, while Gate1 still materializes every kernel invocation and scheduler-referenced threadblock protobuf before the slice is applied.
Root Cause: `run_resnet50_gate1_to_gate7(..., invocation_limit=1)` called the full formal adapter first, then trimmed the adapter bundle after all 265 real invocations and selected-SM protobuf records had already been loaded.
Solution: Pass `invocation_limit` into Gate1 source loading, limit the dynamic trace invocation list, filter scheduler metadata to matching real `kernel_invocation_id` values, and only then read representative-SM threadblock protobufs.
Constraints: Full formal runs must keep the default unbounded behavior; limited regression runs must preserve real Gate0 provenance and record `formal_replay_invocation_limit` in the adapter validation report.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_invocation_limit_bounds_real_root_materialization_before_threadblock_reads` passed in 1.24s; `pytest -q tests/gcl_resnet50/test_gate1_adapter.py tests/gcl_resnet50/test_gate6_selector.py tests/gcl_resnet50/test_gate7_correctness.py tests/gcl_resnet50/test_gate8_tuning.py tests/gcl_resnet50/test_gate9_simulator_evaluation.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py` passed with 40 tests; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 87 tests; `git diff --check` passed.
Source Rounds: 2

## Lesson: gate6-nondegenerate-family-evidence
Lesson ID: BL-20260607-gate6-nondegenerate-family-evidence
Scope: experiments/gcl_phase_b/selector.py, experiments/gcl_phase_b/embedding_export.py, experiments/gcl_phase_b/tensorizer.py, tests/gcl_resnet50/test_gate6_selector.py, tests/gcl_resnet50/test_gate7_correctness.py
Problem Description: Real-root Gate6/Gate7 acceptance tests can pass schema checks while only exercising the single-embedding selector fallback path and leaving family evidence empty.
Root Cause: The direct real-root tests used `invocation_limit=1`, which produced one Gate5 embedding, triggered `fallback_reason = single_embedding_batch`, and emitted `cluster_family_evidence_report.clusters = []`, so Gate7 family metrics stayed `None`.
Solution: Use a bounded but non-degenerate real-root slice with multiple embeddings, assert no selector fallback is used, carry `trace_family` as post-clustering metadata, and build non-empty family evidence from completed cluster assignments.
Constraints: Family evidence must remain post-clustering only; `family_labels_used_for_clustering` stays false and family metadata must not enter normalization, silhouette-K, K-Means, or centroid calculations.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_invocation_limit_bounds_real_root_materialization_before_threadblock_reads tests/gcl_resnet50/test_gate6_selector.py::test_gate6_accepts_real_resnet50_gate5_embedding_table tests/gcl_resnet50/test_gate6_selector.py::test_gate6_runs_silhouette_k_and_deterministic_kmeans_on_real_root tests/gcl_resnet50/test_gate6_selector.py::test_gate6_real_root_family_evidence_is_post_clustering_only tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_records_embedding_geometry_metrics_from_real_root_gate6 tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_records_family_representative_metric_and_stability_from_real_root tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate_pipeline_real_root_reaches_gate9_with_baseline_artifacts` passed with 7 tests; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 87 tests; `git diff --check` passed.
Source Rounds: 3

## Lesson: gate6-real-slice-must-prove-k2
Lesson ID: BL-20260607-gate6-real-slice-must-prove-k2
Scope: experiments/gcl_phase_b/resnet50_adapter.py, experiments/gcl_phase_b/selector.py, tests/gcl_resnet50/real_chain.py, tests/gcl_resnet50/test_gate6_selector.py, tests/gcl_resnet50/test_gate7_correctness.py
Problem Description: A multi-row real-root slice can still be degenerate when every exported Gate5 embedding is identical, causing silhouette-K to return only `k = 1`.
Root Cause: The direct tests used a prefix `invocation_limit=2` slice whose two real Gate5 embeddings were identical, so `choose_silhouette_k` short-circuited on unique-row count and never evaluated `k >= 2` candidates.
Solution: Add Gate1 filtering by explicit real `kernel_invocation_id`, choose a bounded non-prefix slice of lightweight but structurally different invocations, and assert `selected_k >= 2`, candidate `k >= 2`, unique embeddings, and multiple assigned cluster ids.
Constraints: The slice must still come from the real Gate0 root and persisted Gate5 artifacts; do not use synthetic fixtures or family labels for clustering proof.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate6_selector.py::test_gate6_runs_silhouette_k_and_deterministic_kmeans_on_real_root tests/gcl_resnet50/test_gate6_selector.py::test_gate6_real_root_family_evidence_is_post_clustering_only` passed with 2 tests; `pytest -q tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_weighted_purity_uses_cluster_weights_not_cluster_count tests/gcl_resnet50/test_gate7_correctness.py::test_gate7_records_family_representative_metric_and_stability_from_real_root` passed with 2 tests; `pytest -q tests/gcl_resnet50/test_gate6_selector.py tests/gcl_resnet50/test_gate7_correctness.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_resnet50/test_gate1_adapter.py` passed with 37 tests; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 88 tests; `git diff --check` passed.
Source Rounds: 4

## Lesson: gate7-family-metrics-need-member-evidence
Lesson ID: BL-20260607-gate7-family-metrics-need-member-evidence
Scope: experiments/gcl_phase_b/selector.py, experiments/gcl_phase_b/correctness.py, tests/gcl_resnet50/test_gate7_correctness.py
Problem Description: Gate7 family-alignment reports can keep `ari` and `nmi` as placeholders even when cluster-level purity is available.
Root Cause: Cluster-level majority-family and purity aggregates do not preserve per-record family labels, so ARI/NMI/homogeneity/completeness cannot be recomputed from the Gate6 evidence.
Solution: Persist post-clustering member-level family evidence in Gate6 selector artifacts and compute Gate7 alignment metrics from paired `(family_label, cluster_id)` samples.
Constraints: Member family labels remain report-only and must not enter normalization, silhouette-K, K-Means, centroid selection, or embedding computation.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate7_correctness.py` passed with 13 tests; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 90 tests; `git diff --check` passed.
Source Rounds: 5

## Lesson: gate7-missing-inputs-are-report-states
Lesson ID: BL-20260607-gate7-missing-inputs-are-report-states
Scope: experiments/gcl_phase_b/correctness.py, experiments/gcl_phase_b/selector.py, tests/gcl_resnet50/test_gate7_correctness.py
Problem Description: Gate7 can overclaim evidence or crash when optional report inputs are missing or partially populated.
Root Cause: Missing family labels were treated as available family evidence, and metric rows were indexed as if `measured` and `predicted` were always present.
Solution: Preserve missing family labels as missing, emit `family_alignment_claim_status = no_family_claim`, and treat incomplete metric rows as `metric_missing` / `partial_metric_missing` report states while continuing complete-row evaluation.
Constraints: Missing-input states must not block embedding geometry or representative-quality reporting; they only limit the affected claim layer.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate7_correctness.py` passed with 16 tests; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 93 tests; `git diff --check` passed.
Source Rounds: 6

## Lesson: gate7-artifact-contract-is-acceptance
Lesson ID: BL-20260607-gate7-artifact-contract-is-acceptance
Scope: experiments/gcl_phase_b/correctness.py, experiments/gcl_phase_b/resnet50_gate_pipeline.py, experiments/gcl_phase_b/tuning.py, tests/gcl_resnet50/test_gate7_correctness.py
Problem Description: Gate7 behavior tests can pass while the persisted manifest still violates the tracked plan's artifact name, artifact type, claim status, and source/report hash contract.
Root Cause: Earlier rounds focused on report calculations and kept the legacy `gate7_correctness_manifest` surface, so downstream pipeline and Gate8 consumers were not forced onto the plan-defined `gate7_cluster_correctness_manifest` contract.
Solution: Treat manifest file names, artifact types, claim statuses, source hashes, and report hashes as acceptance criteria, and update both producers and consumers in the same change.
Constraints: Compatibility aliases may exist for transitional hash fields, but the canonical contract must be the plan-defined Gate7 cluster correctness manifest.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate7_correctness.py tests/gcl_resnet50/test_gate8_tuning.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py` passed with 26 tests; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 94 tests; `git diff --check` passed.
Source Rounds: 7

## Lesson: planned-artifact-bundles-must-persist
Lesson ID: BL-20260607-planned-artifact-bundles-must-persist
Scope: experiments/gcl_phase_b/resnet50_gate_pipeline.py, experiments/gcl_phase_b/tuning.py, experiments/gcl_phase_b/simulator_eval.py, tests/gcl_phase_b/test_resnet50_gate_pipeline.py
Problem Description: A combined in-memory artifact can pass behavior tests while the original plan's standalone report files and manifest-bound hashes are absent from disk.
Root Cause: Pipeline tests checked the combined Gate7/Gate8/Gate9 artifacts but did not assert that each planned report/bundle file was persisted and hash-bound.
Solution: Add red tests for every planned artifact filename, emit standalone report files in the pipeline, and bind manifests to the persisted report hashes.
Constraints: Combined compatibility artifacts may remain, but they do not replace the plan-defined bundle files.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate7_correctness.py tests/gcl_resnet50/test_gate8_tuning.py tests/gcl_resnet50/test_gate9_simulator_evaluation.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py` passed with 28 tests; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 94 tests; `git diff --check` passed.
Source Rounds: 8

## Lesson: extension-artifacts-need-row-provenance
Lesson ID: BL-20260607-extension-artifacts-need-row-provenance
Scope: experiments/gcl_phase_b/tuning.py, experiments/gcl_phase_b/simulator_eval.py, experiments/gcl_phase_b/resnet50_gate_pipeline.py
Problem Description: Gate8/Gate9 extension files can exist while individual tuning vectors and simulator evaluation reports are not tied back to representative anchors, Gate7 evidence, or Gate8 tuning provenance.
Root Cause: Tests checked top-level artifact presence and coarse hashes but did not require row-level evidence hashes or Gate9 provenance inputs.
Solution: Bind every Gate8 tuning vector to representative-anchor, family-alignment, metric-error, and Gate7 manifest hashes; require Gate9 to consume Gate8 tuning manifest plus representative-anchor provenance and report p95/high-weight bad-case/tuning-effect fields.
Constraints: Gate8/Gate9 remain our extension, not original GCL-Sampler reproduction; provenance binding does not imply simulator accuracy claims.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate8_tuning.py tests/gcl_resnet50/test_gate9_simulator_evaluation.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py` passed with 13 tests; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 96 tests; `git diff --check` passed.
Source Rounds: 9

## Lesson: incremental-gate5-export
Lesson ID: BL-20260609-incremental-gate5-export
Scope: experiments/gcl_phase_b/embedding_export.py, experiments/gcl_phase_b/resnet50_gate_pipeline.py, scripts/run_resnet50_full_trace_gcl.py
Problem Description: Full ResNet50 trace Gate5 export can exceed a deadline and lose progress when the embedding table is written only after every kernel graph has been exported.
Root Cause: Gate5 had no per-row export progress artifact and the pipeline retrained or restarted export instead of reusing a compatible checkpoint, progress prefix, or completed embedding table.
Solution: Persist Gate5 export progress keyed by ordered tensor hashes and encoder manifest hash, skip completed tensor rows on rerun, reuse compatible RGCN checkpoints and completed embedding tables, and keep the stable runtime artifact root ignored but available for resume.
Constraints: Final `kernel_embedding_table.json` schema remains unchanged; progress files are internal runtime state and are removed after the complete table validates.
Validation Evidence: Formal full-trace command completed with `formal_full_trace_run=true`, `input_kernel_invocation_count=265`, `input_cta_record_count=124876`, `final_gate=gate9_report_only`, `embedding_rows=265`, and `selected_k=2`; `pytest -q tests/gcl_phase_b/test_embedding_export.py::test_phase_b_embedding_export_resumes_partial_progress tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate5_reuses_existing_checkpoint_for_export_resume tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate8_report_only_handles_weak_representatives_without_blocking tests/gcl_resnet50/test_full_trace_reproduction_runner.py` passed with 14 tests; `pytest -q tests/gcl_phase_a/test_rgcn_training.py tests/gcl_phase_b/test_readout.py tests/gcl_phase_b/test_embedding_export.py tests/gcl_resnet50/test_gate5_rgcn_training.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py` passed with 46 tests; `git diff --check && git diff --cached --check` passed.
Source Rounds: 3
