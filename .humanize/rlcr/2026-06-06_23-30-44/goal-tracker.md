# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

实现真实 ResNet-50 输入上的 GCL-Sampler 核心复现，并把复现结果推进到可量化验证的 cluster correctness evidence。

Formal reproduction path:

```text
真实 ResNet-50 NVBit trace
  -> Gate 0 trace acquisition
  -> Gate 1 trace adapter
  -> Gate 2 representative SM manifest
  -> Gate 3 canonical graph
  -> Gate 4 graph tensorization
  -> Gate 5 RGCN contrastive embedding
  -> Gate 6 silhouette-K / deterministic K-Means selector
  -> Gate 7 cluster correctness evaluation
```

Gate 8 / Gate 9 are extension gates after the GCL-Sampler core reproduction:

```text
Gate 8: cluster / family -> simulator tuning vector proposal
Gate 9: sampled-vs-full simulator evaluation
```

Hard boundary:

```text
formal GCL reproduction cannot use artificial trace, ResNet-like fixture,
mini-transformer trace, simulator replay trace, or file-order fallback
as a substitute for real ResNet-50 NVBit trace.
```

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->

- AC-1: Gate 0 produces real ResNet-50 NVBit trace artifacts with auditable scheduler metadata.
  - Covers plan AC-1.
  - Formal artifacts must include `workload_id = resnet50`, `execution_mode = real_trace`, `trace_source = nvbit`, `scheduler_metadata_source = real_nvbit_smid`, and `input_scope = full_resnet50_inference_trace`.
  - Synthetic, simulator replay, or missing `%smid` metadata cannot produce a formal acquisition manifest.

- AC-2: Gate 1 formal adapter consumes only real ResNet-50 trace artifacts and emits stable kernel / CTA / warp / instruction records.
  - Covers plan AC-2 and AC-3.
  - Fixture, hand-written opcode, mini-transformer, simulator replay, or ambiguous launch identity must not produce `adapter_validation_report.status = passed`.

- AC-3: Gate 2 representative-SM manifest is deterministic, formal, and scoped to all CTAs on the selected SM.
  - Covers plan AC-4 and AC-5.
  - `scheduler_signature_medoid_sm` is the formal policy.
  - Partial selected-SM scope, mixed-SM CTA scope, random SM, file-order fallback, or debug adapter must fail.

- AC-4: Gate 3 and Gate 4 convert formal representative-SM trace into GCL-compatible graph and tensor artifacts.
  - Covers plan AC-6, AC-7, AC-8, and AC-9.
  - Gate 3 must preserve one typed canonical graph with instruction nodes, variable nodes, optional `mem_ref` pseudo nodes, and allowed edge relations only.
  - Gate 4 must emit 64-wide node features, edge tensors, partition tensors, representation metadata, and real ResNet-50 provenance.

- AC-5: Gate 5 trains / exports GCL-compatible RGCN embeddings without leaking augmented or projection-head outputs into selector inputs.
  - Covers plan AC-10 and AC-11.
  - Augmentation is training-only.
  - Selector embeddings must be canonical non-augmented 256D kernel embeddings from the projection-head input side.

- AC-6: Gate 6 selector consumes only Gate 5 formal embeddings and produces reproducible silhouette-K / deterministic K-Means clusters, representative anchors, and post-clustering family evidence.
  - Covers plan AC-12, AC-13, and AC-14.
  - Clustering input is only the 256D canonical embedding.
  - Family labels, kernel names, runtime, graph size, and weight fields cannot influence clustering.

- AC-7: Gate 7 correctness evaluation quantifies cluster trustworthiness, and Gate 8 / Gate 9 only make tuning or speedup claims when their prerequisites are proven.
  - Covers plan AC-15 through AC-20.
  - Gate 7 must report embedding geometry, family alignment, representative quality, metric error, and stability evidence with `threshold_policy = report_only_v1` until real baseline thresholds exist.
  - Gate 8 cannot consume high-risk clusters directly.
  - Gate 9 cannot claim speedup / accuracy without full or measured baseline comparison.

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 9 (Updated: Round 8)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initialized tracker from `2026-06-06-a-line-gcl-resnet50-full-reproduction-humanize-plan.md`; compressed 20 detailed ACs into 7 top-level tracker ACs. | Round 0 goal tracker setup requires a stable immutable anchor and an active task map. | No AC scope reduction; tracker ACs map to all 20 plan ACs. |
| 1 | Rejected Round 1 completion move requests; kept all active tasks pending and recorded new blocker issues instead. | Gate0 is still manifest-only and fixture-backed in tests, Gate1 still consumes legacy JSON fixtures rather than Gate0 formal artifacts, and Gate2-5 / Gate7-9 remain only partially verified against the original plan. | AC-1 through AC-7 remain in progress; no task reached "Completed and Verified". |
| 2 | Accepted boundary-repair progress only: Gate0 now emits a formal blocker when real NVBit collection is unavailable, fixture-backed roots are explicitly rejected, fixture/debug adapter paths are visibly non-formal, and Round 2 added provenance propagation plus new Gate2-6 test files. Kept all tasks active. | Round 2 improved honesty at the formal/debug boundary and expanded coverage, but it still did not implement the real Gate0 acquisition runner, the Gate1 formal artifact-format transition, or the formal Gate7-9 execution path. | AC-1 and AC-2 boundary reporting improved; AC-3 through AC-7 remain incomplete and unverified end-to-end. |
| 3 | Accepted Round 3 structural progress only: Gate0 now has a callable runner entrypoint, Gate1 formally consumes `dynamic_trace.pb` plus scheduler-referenced `threadblocks/`, Gate5 lineage is exported and Gate6 enforces it, and Gate7 computes geometry from Gate5/Gate6 artifacts while the pipeline writes Gate8/Gate9 extension artifacts in report-only / no-claim mode. Kept all tasks active. | Round 3 closed the Gate1 artifact-format transition and the Gate6 handcrafted-table loophole, but the positive path is still proven only by synthetic in-repo protobuf fixtures rather than a verified real ResNet-50 NVBit acquisition, and Gate9 still does not execute the baseline-backed sampled-vs-full evaluation required by the original plan. | AC-2 and AC-6 materially advanced; AC-1 and AC-3 through AC-7 remain incomplete for full-plan acceptance. |
| 4 | Accepted Round 4 progress in part: synthetic protobuf fixture coverage is now explicitly debug/artifact-shape by default, Gate1 missing-static-metadata negative coverage now reaches the protobuf parser path, and the pipeline can ingest baseline artifacts for Gate7 metric rows and Gate9 comparison. Kept all tasks active. | Round 4 improved boundary honesty and added the baseline-backed Gate7/Gate9 code path, but the formal boundary is still spoofable because a synthetic helper can set `evidence_scope = real_resnet50_nvbit_collection`, Gate6 still accepts a forged self-consistent table when paired with a forged lineage bundle, and the successful pipeline path still writes `gate1_5_pipeline_manifest.json` instead of a Gate1-7 manifest. | AC-2 and AC-7 structurally advanced; AC-1 and AC-6 remain incomplete, and AC-3 through AC-5 still lack verified real-trace evidence. |
| 5 | Accepted Round 5 progress only for the successful pipeline manifest filename contract; kept all tasks active. Rejected the claimed Gate0 and Gate6 closures. | The success path now writes `gate1_7_pipeline_manifest.json`, but the new Gate0 `collector_attestation_hash` can still be forged from the same untrusted evidence payload and source hashes, and Gate6 still accepts a forged table plus forged matching lineage bundle when the new `persisted_manifest_hashes` field is populated self-consistently. Round 5 also removed the only end-to-end baseline-backed pipeline success test, reducing AC-7 coverage. | AC-7 contract consistency improved, but AC-1 and AC-6 remain open, and AC-2 through AC-5 still lack verified real-trace evidence. |
| 6 | Rejected the claimed Round 6 Gate0 and Gate6 closures; kept all tasks active. | Round 6 added a required persisted `nvbit_collector_attestation.json` and a `gate5_manifests` selector parameter, but both checks are still forgeable: a synthetic helper root can still emit a formal Gate0 manifest once a matching handwritten attestation artifact is added, and Gate6 still accepts a forged table plus forged lineage bundle when the caller supplies forged manifest dicts that hash-match the forged lineage. Round 6 also did not implement any writer for `nvbit_collector_attestation.json` and did not restore baseline-backed end-to-end pipeline success coverage. | AC-1 and AC-6 remain open, AC-7 coverage is still weakened, and AC-2 through AC-5 still lack verified real-trace evidence. |
| 7 | Accepted Round 7 progress in part: Gate6 formal selector validation now loads persisted Gate5 artifacts from `gate5_artifact_root`, and the baseline-backed Gate7/Gate9 pipeline success coverage was restored as a debug/report-path test. Rejected the claimed Gate0 closure and kept all tasks active. | Round 7 closed the specific in-memory `gate5_manifests` loophole and restored the missing baseline-backed pipeline test, but Gate0 still accepts a synthetic helper root when a handwritten `.nvbit_collector_session.json` and matching `nvbit_collector_attestation.json` are added, and Gate6 still does not prove that the persisted Gate5 root comes from a verified real Gate0-5 artifact chain. | AC-6 boundary enforcement improved and AC-7 coverage improved, but AC-1 remains open and AC-2 through AC-7 still lack verified real-trace evidence for full-plan acceptance. |
| 8 | Rejected the claimed Round 8 Gate0 closure; kept all tasks active and retained the baseline-backed Gate7/Gate9 path as debug/report-path coverage only. | Round 8 added `_reject_synthetic_artifact_shape_root(...)`, but the check is only a marker-based denylist over `evidence_scope`, `runner_invocation`, selected `artifact_type` fields, and a protobuf trace-name substring. The same synthetic artifact-shape root is still accepted as formal Gate0 once those markers are rewritten, as shown by the new `_contract_style_root(...)` helpers used in the Gate0 and pipeline tests. The restored baseline-backed pipeline success path therefore still depends on synthetic root promotion rather than a verified real acquisition root. | AC-1 remains open, AC-7 remains debug/report-path only, and AC-2 through AC-7 still lack verified real-trace evidence for full-plan acceptance. |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion and routing tag -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| Gate0 real ResNet-50 NVBit trace acquisition and formal acquisition manifest | AC-1 | pending | coding | claude | Gate0 still lacks a trustworthy formal boundary: the Round 8 marker-based synthetic rejection is bypassed by rewriting a few fixture markers (`evidence_scope`, `runner_invocation`, selected `artifact_type` values, protobuf trace name), after which the same synthetic artifact-shape root is accepted as a formal Gate0 manifest with either acquisition-written or handwritten collector attestation artifacts. |
| Gate1 formal adapter boundary and kernel / CTA / warp / instruction record extraction | AC-2 | pending | coding | claude | Gate1 now consumes Gate0 `dynamic_trace.pb` and scheduler-referenced `threadblocks/` directly, and the missing-static-metadata negative test now reaches the intended protobuf parser path. Remaining work is to validate this path against an actual collected ResNet-50 NVBit trace rather than synthetic in-repo protobuf fixtures. |
| Gate2 deterministic representative-SM manifest with selected-SM all-CTA scope | AC-3 | pending | coding | claude | Artifact-shape protobuf coverage is now explicitly debug-only, but the positive-path evidence is still synthetic in-repo fixture data rather than a verified real Gate0/Gate1 acquisition. |
| Gate3 canonical graph and Gate4 graph tensorization | AC-4 | pending | coding | claude | Artifact-shape graph/tensor coverage is now explicitly debug-only, but the remaining evidence still comes from synthetic Gate0/Gate1 fixtures instead of a real collected trace. |
| Gate5 RGCN contrastive embedding export | AC-5 | pending | coding | claude | Gate5 now exports auditable lineage fields, but the current positive-path evidence still originates from the synthetic artifact-shape chain rather than a verified real Gate0-4 artifact chain. |
| Gate6 selector / representative anchor / post-clustering family evidence | AC-6 | pending | coding | claude | Gate6 formal validation now requires `gate5_artifact_root` and loads the persisted lineage/manifests from disk instead of trusting caller-supplied manifest dicts. Remaining work is to bind that persisted Gate5 root to a verified real Gate0-5 artifact chain rather than allowing a fully synthetic but self-consistent persisted root to satisfy the selector boundary. |
| Gate7 correctness evaluation plus Gate8 / Gate9 extension gates | AC-7 | pending | coding | claude | The baseline-backed Gate7/Gate9 pipeline success test is still debug/report-path coverage only. In Round 8 it now depends on Gate0 accepting a marker-rewritten synthetic artifact-shape root (`_contract_style_root(...)`) as formal input, which does not satisfy the plan's real-trace requirement. |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|

### Open Issues
<!-- Issues discovered during implementation -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
| Gate0 still has no verified in-workspace real ResNet-50 NVBit acquisition, scheduler metadata capture, or collected production artifact set. Round 8's new synthetic-root rejection is still forgeable because it only rejects a small set of artifact-shape markers; rewriting those markers on the same synthetic root allows both `acquire_resnet50_gate0_trace(...)` and `record_resnet50_gate0_trace_acquisition(...)` to emit a formal Gate0 manifest. | 8 | AC-1 | Execute and validate the actual Gate0 collection path against a real ResNet-50 NVBit run, and bind formal acceptance to provenance that a repo-local synthetic helper cannot fabricate by editing local files or metadata fields after the fact. |
| The Gate2-Gate5 artifact-shape tests are now honestly debug-only, but the positive-path evidence still comes from in-repo synthetic protobuf trace/threadblock artifacts instead of a verified real ResNet-50 NVBit acquisition. | 4 | AC-3, AC-4, AC-5 | Keep artifact-shape tests as parser/unit coverage only, and add the real-trace positive/negative suites required by the plan on top of an actual Gate0 acquisition root. |
| Gate6 now loads and hashes the persisted Gate5 manifests from `gate5_artifact_root`, but the selector boundary still does not prove that the persisted Gate5 root comes from a verified real Gate0-5 artifact chain rather than a fully synthetic self-consistent export written to disk. | 7 | AC-6 | Keep the persisted-root validation, and add end-to-end provenance binding from verified Gate0-4 formal artifacts through Gate5 export so a synthetic persisted root cannot satisfy formal Gate6 acceptance. |
| The baseline-backed pipeline success test is restored, but it still does not exercise a real formal path: in Round 8 the test obtains a formal Gate0 manifest from a marker-rewritten synthetic artifact-shape root via `_contract_style_root(...)` rather than from a verified real Gate0/Gate1 collection root. | 8 | AC-7 | Keep the restored baseline-backed coverage as debug/report-path only, and drive the formal success path from a verified real Gate0/Gate1 acquisition root before treating AC-7 as complete. |
