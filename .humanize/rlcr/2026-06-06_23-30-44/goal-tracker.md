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

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initialized tracker from `2026-06-06-a-line-gcl-resnet50-full-reproduction-humanize-plan.md`; compressed 20 detailed ACs into 7 top-level tracker ACs. | Round 0 goal tracker setup requires a stable immutable anchor and an active task map. | No AC scope reduction; tracker ACs map to all 20 plan ACs. |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion and routing tag -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| Gate0 real ResNet-50 NVBit trace acquisition and formal acquisition manifest | AC-1 | pending | coding | claude | Requires real trace provenance and scheduler metadata. |
| Gate1 formal adapter boundary and kernel / CTA / warp / instruction record extraction | AC-2 | pending | coding | claude | Must reject fixture / synthetic / replay inputs as formal artifacts. |
| Gate2 deterministic representative-SM manifest with selected-SM all-CTA scope | AC-3 | pending | coding | claude | Uses `scheduler_signature_medoid_sm`; rejects random / fallback policies. |
| Gate3 canonical graph and Gate4 graph tensorization | AC-4 | pending | coding | claude | Produces typed canonical graph and RGCN tensor bundle. |
| Gate5 RGCN contrastive embedding export | AC-5 | pending | coding | claude | Must export canonical non-augmented 256D selector embeddings. |
| Gate6 selector / representative anchor / post-clustering family evidence | AC-6 | pending | coding | claude | Clustering uses only 256D canonical embeddings. |
| Gate7 correctness evaluation plus Gate8 / Gate9 extension gates | AC-7 | pending | coding | claude | Gate7 report-only metrics precede tuning and speedup claims. |

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
