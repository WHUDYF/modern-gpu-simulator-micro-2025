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

### Plan Version: 3 (Updated: Round 2)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initialized tracker from `2026-06-06-a-line-gcl-resnet50-full-reproduction-humanize-plan.md`; compressed 20 detailed ACs into 7 top-level tracker ACs. | Round 0 goal tracker setup requires a stable immutable anchor and an active task map. | No AC scope reduction; tracker ACs map to all 20 plan ACs. |
| 1 | Rejected Round 1 completion move requests; kept all active tasks pending and recorded new blocker issues instead. | Gate0 is still manifest-only and fixture-backed in tests, Gate1 still consumes legacy JSON fixtures rather than Gate0 formal artifacts, and Gate2-5 / Gate7-9 remain only partially verified against the original plan. | AC-1 through AC-7 remain in progress; no task reached "Completed and Verified". |
| 2 | Accepted boundary-repair progress only: Gate0 now emits a formal blocker when real NVBit collection is unavailable, fixture-backed roots are explicitly rejected, fixture/debug adapter paths are visibly non-formal, and Round 2 added provenance propagation plus new Gate2-6 test files. Kept all tasks active. | Round 2 improved honesty at the formal/debug boundary and expanded coverage, but it still did not implement the real Gate0 acquisition runner, the Gate1 formal artifact-format transition, or the formal Gate7-9 execution path. | AC-1 and AC-2 boundary reporting improved; AC-3 through AC-7 remain incomplete and unverified end-to-end. |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion and routing tag -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| Gate0 real ResNet-50 NVBit trace acquisition and formal acquisition manifest | AC-1 | pending | coding | claude | Current code now emits a formal blocker when real NVBit collection is unavailable, but the actual runner / collector / scheduler capture path is still missing. |
| Gate1 formal adapter boundary and kernel / CTA / warp / instruction record extraction | AC-2 | pending | coding | claude | Formal Gate1 now requires a Gate0 manifest and fixture paths are debug-only, but the formal adapter still parses legacy JSON inputs instead of Gate0 `dynamic_trace.pb` and `threadblocks/`. |
| Gate2 deterministic representative-SM manifest with selected-SM all-CTA scope | AC-3 | pending | coding | claude | Round 2 propagated provenance into Gate2 artifacts and added tests, but the new tests still rely on debug fixtures rather than formal Gate0/Gate1 outputs. |
| Gate3 canonical graph and Gate4 graph tensorization | AC-4 | pending | coding | claude | Round 2 propagated provenance and added tests, but the positive path is still exercised through debug-smoke artifacts rather than a formal trace-to-graph chain. |
| Gate5 RGCN contrastive embedding export | AC-5 | pending | coding | claude | Provenance is richer and tests exist, but formal completion is still not tied to a real Gate0-4 chain and Gate5 proof remains debug-smoke based. |
| Gate6 selector / representative anchor / post-clustering family evidence | AC-6 | pending | coding | claude | Formal guard tightened, but the selector still accepts handcrafted embedding tables that only mimic top-level provenance instead of requiring a genuine Gate5 lineage. |
| Gate7 correctness evaluation plus Gate8 / Gate9 extension gates | AC-7 | pending | coding | claude | Gate0 blocker now stops the formal pipeline honestly, but Gate7 metrics are still helper-driven and Gate8/Gate9 are not wired into a verified baseline-backed workflow. |

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
| Gate0 no longer treats fixtures as formal, but it still only records an existing root or writes a blocker report; no real ResNet-50 runner, NVBit collection hook, or scheduler metadata capture path exists in-workspace. | 1 | AC-1 | Implement the actual Gate0 collection path; until then, keep the blocker path and stop any formal execution at Gate0. |
| Gate1 now requires Gate0 manifest provenance, but the formal adapter still parses `dynamic_trace.json` and `threadblocks.json` instead of consuming Gate0 `dynamic_trace.pb` and `threadblocks/` outputs. | 1 | AC-2 | Rework Gate1 to read the Gate0 formal artifact set directly and keep provenance hashes aligned with those exact inputs. |
| Round 2 added Gate2-Gate5 `tests/gcl_resnet50` files, but they mostly exercise debug fixture smoke paths and do not satisfy the original plan's formal positive/negative contracts. | 2 | AC-3, AC-4, AC-5 | Replace debug-smoke stand-ins with formal Gate2-Gate5 AC suites built from the real Gate0/Gate1 artifact chain. |
| Gate6 requires top-level formal provenance fields, but it still accepts handcrafted embedding tables that are not demonstrably produced by Gate5 training/readout artifacts. | 2 | AC-6 | Require auditable Gate5 lineage on the embedding table and reject handcrafted formal-looking tables that lack source manifests and hashes from the actual Gate5 path. |
| The Gate7 pipeline invocation still emits a correctness manifest without computed geometry or measured/simulator error inputs, and Gate8/Gate9 remain standalone helpers rather than an end-to-end evaluated extension path. | 1 | AC-7 | Compute Gate7 metrics from actual selector/baseline artifacts and wire Gate8/Gate9 into a verified extension workflow. |
