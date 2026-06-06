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

### Plan Version: 4 (Updated: Round 3)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initialized tracker from `2026-06-06-a-line-gcl-resnet50-full-reproduction-humanize-plan.md`; compressed 20 detailed ACs into 7 top-level tracker ACs. | Round 0 goal tracker setup requires a stable immutable anchor and an active task map. | No AC scope reduction; tracker ACs map to all 20 plan ACs. |
| 1 | Rejected Round 1 completion move requests; kept all active tasks pending and recorded new blocker issues instead. | Gate0 is still manifest-only and fixture-backed in tests, Gate1 still consumes legacy JSON fixtures rather than Gate0 formal artifacts, and Gate2-5 / Gate7-9 remain only partially verified against the original plan. | AC-1 through AC-7 remain in progress; no task reached "Completed and Verified". |
| 2 | Accepted boundary-repair progress only: Gate0 now emits a formal blocker when real NVBit collection is unavailable, fixture-backed roots are explicitly rejected, fixture/debug adapter paths are visibly non-formal, and Round 2 added provenance propagation plus new Gate2-6 test files. Kept all tasks active. | Round 2 improved honesty at the formal/debug boundary and expanded coverage, but it still did not implement the real Gate0 acquisition runner, the Gate1 formal artifact-format transition, or the formal Gate7-9 execution path. | AC-1 and AC-2 boundary reporting improved; AC-3 through AC-7 remain incomplete and unverified end-to-end. |
| 3 | Accepted Round 3 structural progress only: Gate0 now has a callable runner entrypoint, Gate1 formally consumes `dynamic_trace.pb` plus scheduler-referenced `threadblocks/`, Gate5 lineage is exported and Gate6 enforces it, and Gate7 computes geometry from Gate5/Gate6 artifacts while the pipeline writes Gate8/Gate9 extension artifacts in report-only / no-claim mode. Kept all tasks active. | Round 3 closed the Gate1 artifact-format transition and the Gate6 handcrafted-table loophole, but the positive path is still proven only by synthetic in-repo protobuf fixtures rather than a verified real ResNet-50 NVBit acquisition, and Gate9 still does not execute the baseline-backed sampled-vs-full evaluation required by the original plan. | AC-2 and AC-6 materially advanced; AC-1 and AC-3 through AC-7 remain incomplete for full-plan acceptance. |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion and routing tag -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| Gate0 real ResNet-50 NVBit trace acquisition and formal acquisition manifest | AC-1 | pending | coding | claude | Gate0 now has a runner entrypoint and command/env hookup, but no verified in-workspace real ResNet-50 NVBit acquisition, scheduler capture, or production artifact collection evidence exists yet. |
| Gate1 formal adapter boundary and kernel / CTA / warp / instruction record extraction | AC-2 | pending | coding | claude | Gate1 now consumes Gate0 `dynamic_trace.pb` and scheduler-referenced `threadblocks/` directly. Remaining work is to validate this path against an actual collected ResNet-50 NVBit trace rather than synthetic in-repo protobuf fixtures. |
| Gate2 deterministic representative-SM manifest with selected-SM all-CTA scope | AC-3 | pending | coding | claude | Positive-path tests now use the formal protobuf artifact shape, but the only current evidence source is the minimal synthetic fixture chain rather than a verified real Gate0/Gate1 acquisition. |
| Gate3 canonical graph and Gate4 graph tensorization | AC-4 | pending | coding | claude | Positive-path tests now run from the formal protobuf artifact shape, but the graph/tensor evidence is still derived from synthetic in-repo Gate0/Gate1 fixtures instead of a real collected trace. |
| Gate5 RGCN contrastive embedding export | AC-5 | pending | coding | claude | Gate5 now exports auditable lineage, but the positive-path evidence still originates from the synthetic fixture chain rather than a verified real Gate0-4 artifact chain. |
| Gate6 selector / representative anchor / post-clustering family evidence | AC-6 | pending | coding | claude | Gate6 now rejects missing Gate5 lineage, but it still accepts forged self-consistent lineage payloads that are not tied to real Gate5 manifests. Remaining work is manifest-backed lineage verification plus end-to-end validation from a real Gate0-5 artifact chain. |
| Gate7 correctness evaluation plus Gate8 / Gate9 extension gates | AC-7 | pending | coding | claude | Gate7 now computes geometry from Gate5/Gate6 artifacts and Gate8/Gate9 are emitted by the pipeline, but Gate7 still lacks baseline-fed metric error inputs in the pipeline and Gate9 still does not perform the plan’s baseline-backed sampled-vs-full evaluation. |

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
| Gate0 no longer treats fixtures as formal and now exposes a runner entrypoint, but no verified in-workspace real ResNet-50 NVBit acquisition, scheduler metadata capture, or collected production artifact set exists yet. | 1 | AC-1 | Execute and validate the actual Gate0 collection path against a real ResNet-50 NVBit run; until then, keep the blocker path and stop any formal execution at Gate0. |
| The new "formal" Gate2-Gate5 positive-path tests are driven by `tests/gcl_resnet50/formal_fixture.py`, which synthesizes protobuf trace/threadblock artifacts in-repo instead of consuming a verified real ResNet-50 NVBit acquisition. | 3 | AC-3, AC-4, AC-5 | Replace the synthetic fixture chain with evidence from an actual Gate0 acquisition root, or explicitly keep these tests as artifact-format unit tests while adding the real-trace positive/negative suites required by the plan. |
| Gate6 now rejects missing Gate5 lineage fields, but it still accepts handcrafted embedding tables that include a forged self-consistent `gate5_lineage` payload not tied to any actual Gate5 manifests. | 3 | AC-6 | Make Gate6 verify lineage against persisted Gate5 manifests or embedded manifest hashes, and add a negative test for forged-but-self-consistent lineage. |
| Gate7 now computes embedding geometry from Gate5/Gate6 artifacts, but the pipeline still does not feed measured/full-baseline metric rows into Gate7 and Gate9 still emits only `baseline_missing_no_speedup_or_accuracy_claim` rather than running the required sampled-vs-full evaluation when baselines are available. | 3 | AC-7 | Extend the pipeline to ingest baseline artifacts, compute Gate7 metric-error evidence from them, and invoke the real Gate9 comparison path whenever full/measured baselines exist. |
