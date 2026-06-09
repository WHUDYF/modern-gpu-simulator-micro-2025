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
Problem Description: Raw trace adapter records can collapse repeated launches when fallback provenance maps only by `kernel_id`, and bounded formal replay can accidentally include multiple repeated launches when a legacy scheduler alias is used as the filter input.
Root Cause: `kernel_id` is a static/function identity and is not unique per launch; using it as a key overwrites earlier invocations when repeated launches share the same kernel ID, formal `dynamic_trace.pb` can repeat the same `kernel.id` across distinct launches, legacy alias filtering compared every invocation directly against `d_<device>_s_<stream>_k_<kernel_id>` before resolving the requested alias to one canonical launch-order ID, and `invocation_limit` reused that legacy alias set to filter scheduler metadata, widening repeated-kernel bounded slices when scheduler rows lacked `launch_order`.
Solution: Preserve explicit `kernel_invocation_id` when present for raw debug records, otherwise align raw records by `launch_order`; formal protobuf records must generate `source_kernel_invocation_id` from launch order, preserve original launch order after filtering, resolve explicit legacy scheduler IDs to one canonical launch-order ID before filtering dynamic trace and scheduler metadata, and for `invocation_limit` without explicit IDs limit scheduler metadata by the same prefix count instead of adding non-unique legacy aliases.
Constraints: Ambiguous raw records must fail fast instead of silently merging launches; launch-order fallback is only valid when raw records include launch order; legacy `d_<device>_s_<stream>_k_<kernel_id>` IDs are accepted only for explicit compatibility and must not become final formal adapter IDs or select every repeated launch; bounded `invocation_limit` assumes scheduler metadata preserves the same launch prefix order as `dynamic_trace.pb` when no scheduler launch IDs are present.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_formal_pb_uses_launch_order_invocation_ids_for_reused_kernel_id`; `pytest -q tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_legacy_invocation_alias_selects_single_repeated_launch`; `pytest -q tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_invocation_limit_does_not_widen_repeated_kernel_fallback_scheduler`; `pytest -q tests/gcl_resnet50/test_gate1_adapter.py tests/gcl_phase_b/test_resnet50_adapter.py`; `pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_replay.py`; `python3 -m py_compile experiments/gcl_phase_b/resnet50_adapter.py tests/gcl_resnet50/test_gate1_adapter.py`.
Source Rounds: 2, 40, 45, 49

## Lesson: scheduler-trace-reconciliation
Lesson ID: BL-20260606-scheduler-trace-reconciliation
Scope: experiments/gcl_phase_b/resnet50_adapter.py
Problem Description: A trace adapter can emit a passed bundle even when CTA scheduler metadata and per-warp trace records disagree.
Root Cause: Validation only checked that each invocation had some scheduler and trace records, not that each `(kernel_invocation_id, cta_id)` had matching CTA presence, warp IDs, trace entry counts, and scheduler ordering.
Solution: Aggregate per-warp trace records by `(kernel_invocation_id, cta_id)` and validate CTA set equality, warp ID equality, exact trace-entry count equality, positive counts, and `first_seen_order <= last_seen_order`.
Constraints: Fixture scheduler metadata must include only CTAs with corresponding trace records; debug/proxy scheduler records cannot enter formal Gate 1 bundles.
Validation Evidence: `pytest -q tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py`; `pytest -q tests/gcl_phase_a tests/gcl_phase_b` passed in Round 3.
Source Rounds: 3

## Lesson: final-state-artifacts-mutually-exclusive
Lesson ID: BL-20260609-final-state-artifacts-mutually-exclusive
Scope: scripts/run_resnet50_full_trace_gcl.py, tests/gcl_resnet50/test_full_trace_reproduction_runner.py, artifacts/gcl_resnet50_full_trace_reproduction/
Problem Description: A stable output root can contain both a previous blocker report and a later success manifest after a rerun completes, leaving contradictory final-state evidence for consumers.
Root Cause: The failure path removed the success manifest before writing a blocker report, but the success path did not remove `resnet50_full_trace_reproduction_blocker_report.json`.
Solution: Delete the stale blocker report immediately before writing a successful full-trace reproduction manifest, and add a regression test that seeds a blocker report before a successful monkeypatched run.
Constraints: Failure-path behavior stays unchanged; blocker reports remain valid final evidence only when no success manifest is emitted.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_full_trace_reproduction_runner.py` passed with 12 tests; the real stable-root command completed with `resource_status=completed`, `final_gate=gate9_report_only`, `embedding_rows=265`, `selected_k=2`, and no blocker report present; `git diff --check && git diff --cached --check` passed.
Source Rounds: 4

## Lesson: persisted-formal-boundaries
Lesson ID: BL-20260606-persisted-formal-boundaries
Scope: experiments/gcl_phase_b/resnet50_gate0.py, experiments/gcl_phase_b/selector.py, experiments/gcl_phase_b/resnet50_gate_pipeline.py
Problem Description: Formal GCL gates can be bypassed when validation only checks self-consistency among caller-controlled or root-local JSON dictionaries, but an over-broad revalidation guard can also reject legitimate first-time formalization from the live collector path.
Root Cause: Gate0 accepted a handwritten attestation over the same untrusted root, restart/revalidation skipped live collector session binding when `active_collector_session_id` was absent, Gate0 later treated every no-active-session recording as revalidation requiring an existing manifest, a later attempted relaxation let root-local evidence fields or an active session ID alone mint a first formal manifest without a process-owned collector session record, acquisition minted attestation for any runner that wrote self-consistent local evidence under the output root, and an attempted runtime nonce fix exposed that nonce to the workload command while also blocking valid external first formalization.
Solution: Bind Gate0 live acquisition acceptance to a trusted collector runner plus a process-owned active collector session record containing the session hash and output root, keep runtime proof material out of workload-command environment, derive acquisition runtime proof from the collector session hash after the trusted runner returns, allow no-active-session first formalization when persisted collector evidence/session/attestation include a valid runtime proof, and require existing manifest hash revalidation only for restart/revalidation without active acquisition.
Constraints: This separates trusted collector orchestration from arbitrary workload commands but still does not cryptographically prove an honest external collector if it can forge a valid persisted session/attestation/proof set; fixture-backed producer-path tests remain contract coverage only; no-active-session restart/revalidation without runtime proof is allowed only for an already formalized root; process-local active session IDs without the recorded session hash/root/runtime proof are not trusted.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py`; `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py tests/gcl_resnet50/test_gate0_formal_trace_runner.py tests/gcl_resnet50/test_gate1_adapter.py`; `pytest -q tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_replay.py`; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b` passed with 179 tests; `pytest -q tests/gcl_phase_a tests/gcl_phase_b tests/gcl_resnet50` passed with 242 tests; `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py tests/gcl_resnet50/test_gate0_formal_trace_runner.py tests/gcl_resnet50/test_full_trace_reproduction_runner.py` passed with 46 tests; `pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 34 tests; `python3 -m py_compile experiments/gcl_phase_b/resnet50_gate0.py tests/gcl_resnet50/test_gate0_trace_acquisition.py`; `git diff --check && git diff --cached --check` passed.
Source Rounds: 7, 32, 35, 39, 43, 45, 46

## Lesson: real-root-schema-contract
Lesson ID: BL-20260607-real-root-schema-contract
Scope: experiments/gcl_phase_b/resnet50_adapter.py, experiments/gcl_phase_b/graph_builder.py, tests/gcl_resnet50/real_chain.py
Problem Description: Fixture and artifact-shape tests can pass while the formal Gate1 loader fails on the real Gate0 root because the real NVBit artifact layout and SASS operand schema differ from synthetic fixtures; committed real-root regression tests can also fail in clean checkouts when large ignored trace artifacts are absent.
Root Cause: Gate1 read root-level `enhanced_execution_info.json` and required fixture-only `launch_order` / `threadblock_pb` fields, while the real root stores static metadata at `extra_info/enhanced_execution_info.json`, represents instructions under `kernels[].instructions`, identifies invocations as `d_<device>_s_<stream>_k_<kernel>`, uses compound memory operands such as `desc[UR10][R4.64]`, and later tests hard-coded a local `artifacts/gcl_resnet50_gate0_formal_trace/traces` root without guarding missing ignored runtime files.
Solution: Add real-root schema normalization, derive threadblock paths from scheduler invocation metadata, align invocation IDs with real scheduler metadata, materialize representative-SM CTA trace records from real threadblock protobufs, parse compound memory operands into register-version address sources, and gate real-root regression helpers with a required-artifact check that skips when local trace artifacts are unavailable.
Constraints: Regression tests may use a deterministic real-root invocation slice for runtime, but the slice must preserve real Gate0 provenance and must not be replaced by synthetic fixture data; committed tests that require ignored local trace artifacts must skip cleanly or use self-contained fixtures.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate1_adapter.py tests/gcl_resnet50/test_gate2_representative_sm.py tests/gcl_resnet50/test_gate3_canonical_graph.py tests/gcl_resnet50/test_gate4_tensorization.py tests/gcl_resnet50/test_gate5_rgcn_training.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py` passed with 29 tests; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 80 tests; `pytest -q tests/gcl_resnet50/test_gate1_adapter.py::test_real_chain_skips_when_formal_root_artifacts_are_missing tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_builds_formal_adapter_from_real_resnet50_trace_root tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_invocation_limit_bounds_real_root_materialization_before_threadblock_reads tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_invocation_ids_mark_real_root_adapter_as_bounded_slice`; `pytest -q tests/gcl_resnet50/test_gate2_representative_sm.py tests/gcl_resnet50/test_gate3_canonical_graph.py tests/gcl_resnet50/test_gate4_tensorization.py tests/gcl_resnet50/test_gate5_rgcn_training.py tests/gcl_resnet50/test_gate6_selector.py tests/gcl_resnet50/test_gate7_correctness.py`; `pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_replay.py`; `python3 -m py_compile tests/gcl_resnet50/real_chain.py`; `git diff --check && git diff --cached --check` passed.
Source Rounds: 1, 36

## Lesson: gate1-slice-before-materialization
Lesson ID: BL-20260607-gate1-slice-before-materialization
Scope: experiments/gcl_phase_b/resnet50_adapter.py, experiments/gcl_phase_b/resnet50_gate_pipeline.py, tests/gcl_resnet50/real_chain.py
Problem Description: A real-root regression may appear tractable because downstream artifacts are sliced, while Gate1 still materializes every kernel invocation and scheduler-referenced threadblock protobuf before the slice is applied.
Root Cause: `run_resnet50_gate1_to_gate7(..., invocation_limit=1)` called the full formal adapter first, then trimmed the adapter bundle after all 265 real invocations and selected-SM protobuf records had already been loaded; later bounded adapters also retained `input_scope = full_resnet50_inference_trace`, making slices look like full-trace provenance.
Solution: Pass `invocation_limit` into Gate1 source loading, limit the dynamic trace invocation list, filter scheduler metadata to matching real `kernel_invocation_id` values, only then read representative-SM threadblock protobufs, and mark bounded adapters with `input_scope = bounded_resnet50_invocation_slice`.
Constraints: Full formal runs must keep the default unbounded behavior and `input_scope = full_resnet50_inference_trace`; limited regression runs must preserve real Gate0 provenance and record `formal_replay_invocation_limit` or `formal_replay_invocation_ids` in the adapter validation report.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate1_adapter.py::test_gate1_invocation_limit_bounds_real_root_materialization_before_threadblock_reads` passed in 1.24s; `pytest -q tests/gcl_resnet50/test_gate1_adapter.py tests/gcl_resnet50/test_gate6_selector.py tests/gcl_resnet50/test_gate7_correctness.py tests/gcl_resnet50/test_gate8_tuning.py tests/gcl_resnet50/test_gate9_simulator_evaluation.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py` passed with 40 tests; `pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_replay.py` passed with 87 tests; `pytest -q tests/gcl_resnet50/test_gate1_adapter.py tests/gcl_resnet50/test_gate2_representative_sm.py tests/gcl_phase_b/test_resnet50_manifest.py` passed with 17 tests; `pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_resnet50/test_full_trace_reproduction_runner.py` passed with 46 tests; `git diff --check` passed.
Source Rounds: 2, 31

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
Root Cause: Gate5 had no per-row export progress artifact and the pipeline retrained or restarted export instead of reusing a compatible checkpoint, progress prefix, or completed embedding table; later stale progress files with mismatched tensor hashes or encoder manifest hashes were treated as fatal instead of restartable.
Solution: Persist Gate5 export progress keyed by ordered tensor hashes and encoder manifest hash, skip completed tensor rows on rerun, reuse compatible RGCN checkpoints and completed embedding tables, delete stale progress when tensor or encoder lineage no longer matches, and keep the stable runtime artifact root ignored but available for resume.
Constraints: Final `kernel_embedding_table.json` schema remains unchanged; progress files are internal runtime state and are removed after the complete table validates; stale progress is discarded only for lineage mismatch, while corrupt same-lineage progress still fails validation.
Validation Evidence: Formal full-trace command completed with `formal_full_trace_run=true`, `input_kernel_invocation_count=265`, `input_cta_record_count=124876`, `final_gate=gate9_report_only`, `embedding_rows=265`, and `selected_k=2`; `pytest -q tests/gcl_phase_b/test_embedding_export.py::test_phase_b_embedding_export_resumes_partial_progress tests/gcl_phase_b/test_embedding_export.py::test_phase_b_embedding_export_discards_stale_progress_on_tensor_mismatch`; `pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate5_reuses_existing_checkpoint_for_export_resume tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate8_report_only_handles_weak_representatives_without_blocking tests/gcl_resnet50/test_full_trace_reproduction_runner.py` passed with 14 tests; `pytest -q tests/gcl_phase_a/test_rgcn_training.py tests/gcl_phase_b/test_readout.py tests/gcl_phase_b/test_embedding_export.py tests/gcl_resnet50/test_gate5_rgcn_training.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py` passed with 46 tests; `git diff --check && git diff --cached --check` passed.
Source Rounds: 3, 49

## Lesson: blocked-artifact-preflight-order
Lesson ID: BL-20260609-blocked-artifact-preflight-order
Scope: scripts/run_resnet50_full_trace_gcl.py, experiments/gcl_phase_b/resnet50_gate_pipeline.py, tests/gcl_resnet50/test_full_trace_reproduction_runner.py
Problem Description: A wrapper runner can make a formal blocked state unreachable when it eagerly requires the success manifest before checking for the paired blocker report.
Root Cause: The full-trace runner loaded `gate0_trace_acquisition_manifest.json` before detecting `gate0_trace_acquisition_blocker_report.json`, so Gate0 acquisition failures were converted into generic manifest-missing blockers and lost the original blocker evidence.
Solution: Check for the blocker-report-without-manifest state before loading the success manifest, delegate to the gate pipeline's blocked path, and preserve the original blocker hash in the final wrapper manifest.
Constraints: Stale blocker reports must still be ignored when a formal success manifest exists; success-path validation remains manifest-backed.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_full_trace_reproduction_runner.py tests/gcl_resnet50/test_gate0_formal_trace_runner.py` passed with 22 tests; `pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_resnet50/test_full_trace_reproduction_runner.py` passed with 42 tests; `git diff --check && git diff --cached --check` passed.
Source Rounds: 26

## Lesson: resource-blocked-final-state-clears-success-lineage
Lesson ID: BL-20260609-resource-blocked-clears-success-lineage
Scope: experiments/gcl_phase_b/pipeline.py, tests/gcl_phase_b/test_pipeline.py, tests/gcl_phase_b/test_replay.py
Problem Description: A later-stage resource failure can leave a pipeline manifest marked `resource_blocked` while still retaining hashes and artifacts from earlier successful downstream stages.
Root Cause: Selector-stage resource blocking removed only selector artifacts and selector hash, but replay validation treats any resource-blocked final state as mutually exclusive with Gate5/selector success lineage.
Solution: On selector resource failure, remove all success artifacts and clear `EMBEDDING_DOWNSTREAM_HASH_NULLS` together with the new `resource_blocked_hash`; retries must rebuild Gate5 before rerunning selector.
Constraints: Graph/tensor artifacts remain valid inputs for recovery; success-path reruns can clear the resource-blocked state after rebuilding the downstream artifacts.
Validation Evidence: `pytest -q tests/gcl_phase_b/test_pipeline.py::test_from_disk_selector_stage_failure_marks_resource_blocked_and_clears_stale_selector tests/gcl_phase_b/test_pipeline.py::test_from_disk_embedding_stage_selector_failure_preserves_gate5_outputs`; `pytest -q tests/gcl_phase_b/test_pipeline.py tests/gcl_phase_b/test_replay.py` passed with 65 tests.
Source Rounds: 30

## Lesson: portable-tool-path-roots
Lesson ID: BL-20260609-portable-tool-path-roots
Scope: experiments/trace_compression_behavior/catalog.py, scripts/clone_workload_sources.sh, scripts/generate_source_registry.py, scripts/generate_workload_registry.py, tests/test_workload_registry_tools.py, registry/source_registry.json
Problem Description: Tooling can pass in the author checkout while failing for generated external catalogs, installed packages, CI, or another developer machine.
Root Cause: Catalog relative `source_path` values were resolved against the repository root, catalog duplicate IDs could silently overwrite records, workload/source-registry defaults used absolute `/home/dyf/...` workload roots, clone status rows wrote paths relative to the caller instead of relative to `clone_status.tsv`, generated source registries serialized local absolute paths without a portable root, workload registry generation resolved portable paths against either the registry directory or a hardcoded default root instead of the source registry's true root, existing checkout sparse failures shared the new-clone destructive cleanup path, and curated workload IDs could be emitted from stale `source_available` paths that no longer existed.
Solution: Resolve catalog relative paths against the catalog file directory, reject duplicate catalog entry IDs at load time, keep fixture catalogs catalog-relative, use portable relative default roots, write and serialize clone status paths as `sources/<name>` when they are inside the workload root, persist a portable `source_root` in generated source registries, preserve existing checkouts on sparse setup failure, resolve relative source registry paths through explicit `--workload-root` or `source_root`, skip missing `source_available` roots, keep full-checkout curated workload IDs table-driven only when the source root exists, and reserve per-workload path filtering for sparse sources.
Constraints: Absolute catalog/source paths outside the workload root remain valid; callers can still pass `--root` or `--workload-root` explicitly for external workload locations; clone status relative paths must remain relative to the workload root/status file directory; generated `source_root` should be relative to the source registry output location when possible; failed sparse setup after a fresh clone may still remove the newly created incomplete target; sparse source registries still filter candidates to retained paths.
Validation Evidence: `pytest -q experiments/trace_compression_behavior/tests/test_trace_compression_behavior_catalog.py`; `pytest -q tests/test_workload_registry_tools.py`; `python3 -m py_compile experiments/trace_compression_behavior/catalog.py scripts/generate_source_registry.py`; `python3 -m py_compile experiments/trace_compression_behavior/catalog.py`; `python3 -m py_compile scripts/generate_workload_registry.py`; `python3 -m py_compile scripts/generate_source_registry.py scripts/generate_workload_registry.py tests/test_workload_registry_tools.py`; `bash -n scripts/clone_workload_sources.sh`; `git diff --check && git diff --cached --check`.
Source Rounds: 33, 34, 38, 44, 48, 50

## Lesson: gate0-trace-runner-framework-compatibility
Lesson ID: BL-20260609-gate0-trace-runner-framework-compatibility
Scope: scripts/run_resnet50_gate0_formal_trace.py, tests/gcl_resnet50/test_gate0_formal_trace_runner.py
Problem Description: Gate0 formal trace collection can fail before NVBit profiling starts when the runner assumes newer PyTorch or torchvision APIs on hosts that are pinned to older framework builds.
Root Cause: The runner called `models.resnet50(weights=None)` without a fallback for older torchvision `pretrained=` APIs, and used PyTorch 2.x `torch.amp.autocast("cuda", ...)` without a fallback for older `torch.cuda.amp.autocast(...)`.
Solution: Build the offline ResNet-50 model with `weights=None` first and fall back to `pretrained=False` on `TypeError`; wrap CUDA autocast selection in a helper that prefers `torch.amp.autocast` but falls back to `torch.cuda.amp.autocast`.
Constraints: Do not introduce pretrained downloads; do not pin a new framework minimum in tests unless the collection environment is intentionally narrowed.
Validation Evidence: `pytest -q tests/gcl_resnet50/test_gate0_formal_trace_runner.py`; `python3 -m py_compile scripts/run_resnet50_gate0_formal_trace.py`; `git diff --check && git diff --cached --check`.
Source Rounds: 37

## Lesson: sass-predicate-destination-operands
Lesson ID: BL-20260609-sass-predicate-destination-operands
Scope: experiments/gcl_phase_b/resnet50_adapter.py, tests/gcl_phase_b/test_resnet50_adapter.py
Problem Description: Formal ResNet-50 trace decoding can misclassify the second predicate output of `ISETP.*` and `PSETP.*` instructions as a source operand.
Root Cause: `_split_operands()` treated only `LEA` as a two-destination opcode, so predicate setter instructions emitted one missing destination edge and one false source dependency in downstream graph construction.
Solution: Include `ISETP` and `PSETP` in the two-destination opcode prefix set and add an operand split regression test for both opcode families.
Constraints: Store and control-flow zero-destination handling remains unchanged; this fix only changes opcodes whose SASS operand schema has two predicate destinations.
Validation Evidence: `pytest -q tests/gcl_phase_b/test_resnet50_adapter.py`; `python3 -m py_compile experiments/gcl_phase_b/resnet50_adapter.py`.
Source Rounds: 41

## Lesson: augmentation-derived-metadata-and-launch-order
Lesson ID: BL-20260609-augmentation-derived-metadata-launch-order
Scope: experiments/gcl_phase_a/train.py, experiments/gcl_phase_b/resnet50_adapter.py
Problem Description: GCL augmentation and trace parsing can keep stale or guessed provenance when derived metadata is copied across a structural transform.
Root Cause: `augment_tensor()` rebuilt node features, edges, and `warp_partitions` but copied pre-augmentation `warp_partition_tensors`; `_load_dynamic_trace_pb()` assigned launch-order IDs by grouped device/stream iteration even though multi-stream protobufs do not expose a cross-stream global chronology.
Solution: Rebuild `warp_partition_tensors` from augmented node and edge arrays after node/edge dropping, and reject active multi-stream `dynamic_trace.pb` inputs unless a scheduler-aligned global launch order is available before invocation IDs are minted.
Constraints: Single-stream formal ResNet-50 traces keep the existing protobuf order; multi-stream inputs must fail fast rather than silently binding scheduler metadata to guessed launch IDs.
Validation Evidence: `pytest -q tests/gcl_phase_a/test_rgcn_training.py`; `pytest -q tests/gcl_phase_b/test_tensorizer.py tests/gcl_phase_b/test_embedding_export.py`; `pytest -q tests/gcl_resnet50/test_gate1_adapter.py tests/gcl_phase_b/test_resnet50_adapter.py`; `pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_replay.py`; `pytest -q tests/gcl_resnet50/test_gate5_rgcn_training.py tests/gcl_resnet50/test_gate6_selector.py tests/gcl_resnet50/test_gate7_correctness.py`; `python3 -m py_compile experiments/gcl_phase_a/train.py experiments/gcl_phase_b/resnet50_adapter.py tests/gcl_phase_a/test_rgcn_training.py tests/gcl_resnet50/test_gate1_adapter.py`; `git diff --check && git diff --cached --check` passed.
Source Rounds: 47

## Lesson: selector-blocked-keeps-gate5-artifacts
Lesson ID: BL-20260609-selector-blocked-keeps-gate5-artifacts
Scope: experiments/gcl_phase_b/pipeline.py, tests/gcl_phase_b/test_pipeline.py, tests/gcl_phase_b/test_replay.py
Problem Description: A transient selector-only resource failure can delete completed Gate5 training/export artifacts and force unnecessary retraining on the next retry.
Root Cause: `_mark_selector_stage_resource_blocked()` and the full `run_pipeline()` selector failure path reused the full success-artifact cleanup path intended for earlier training/export failures, and replay validation treated every resource-blocked state as incompatible with any downstream success artifacts.
Solution: Selector-stage resource blocking now removes only `selector_artifacts.json`, clears only `selector_manifest_hash`, preserves Gate5 files and hashes in both from-disk retry and full `run_pipeline()` paths, and replay validation has a selector-blocked branch that validates Gate5 artifacts while requiring selector artifacts to be absent.
Constraints: Training and embedding-stage resource failures still clear Gate5/downstream success artifacts; preserving Gate5 is only valid when the recorded `failed_stage` is `selector`.
Validation Evidence: `pytest -q tests/gcl_phase_b/test_pipeline.py::test_selector_resource_failure_writes_resource_blocked_artifact`; `pytest -q tests/gcl_phase_b/test_pipeline.py::test_from_disk_selector_stage_failure_marks_resource_blocked_and_clears_stale_selector tests/gcl_phase_b/test_pipeline.py::test_from_disk_embedding_stage_selector_failure_preserves_gate5_outputs`; `pytest -q tests/gcl_phase_b/test_pipeline.py tests/gcl_phase_b/test_replay.py`; `python3 -m py_compile experiments/gcl_phase_b/pipeline.py`.
Source Rounds: 41, 42

## Lesson: frontend-timing-fallback-workload-identity
Lesson ID: BL-20260609-frontend-timing-fallback-workload-identity
Scope: artifacts/gpu_trace_frontend_difftest_necessity/complete_flow_burden_ratio_calc.py, tests/test_complete_flow_burden_ratio_calc.py
Problem Description: A generic `frontend_timing_breakdown.json` file can be reused as measured timing for unrelated workloads when per-workload timing files are absent.
Root Cause: The fallback loader checked only timing field presence and did not verify that the single-run timing artifact was generated for the requested workload ID.
Solution: Require generic fallback timing artifacts to declare a matching `workload_id` or include the workload in `workload_ids`; otherwise skip the artifact and use the modeled/formula fallback for that workload.
Constraints: Per-workload `frontend_timing_breakdown_<workload_id>.json` files remain trusted by filename; the identity check applies to the generic single-run fallback file.
Validation Evidence: `pytest -q tests/test_complete_flow_burden_ratio_calc.py`; `python3 -m py_compile artifacts/gpu_trace_frontend_difftest_necessity/complete_flow_burden_ratio_calc.py tests/test_complete_flow_burden_ratio_calc.py`.
Source Rounds: 42
