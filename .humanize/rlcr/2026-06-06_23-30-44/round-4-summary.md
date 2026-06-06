# Round 4 Summary

## Implemented

- Stopped treating in-repo synthetic protobuf fixtures as formal ResNet-50 evidence:
  - `write_minimal_artifact_shape_resnet50_root(...)` now defaults to `evidence_scope = synthetic_artifact_shape_unit_test_only`.
  - Gate0 rejects any evidence scope other than `real_resnet50_nvbit_collection` for formal manifest creation.
  - Existing protobuf fixture chain tests were relabeled as artifact-shape / debug coverage and no longer assert formal input eligibility.

- Added a debug-only protobuf artifact-shape Gate1 adapter:
  - `build_resnet50_artifact_shape_trace_adapter_bundle(...)`
  - Reads `dynamic_trace.pb` and scheduler-referenced `threadblocks/` for parser coverage.
  - Emits `artifact_status = debug_not_formal` and `formal_input_eligible = false`.

- Repaired Gate1 negative coverage:
  - Missing static instruction metadata now enters the protobuf adapter path and fails for the intended `static instruction metadata` reason.
  - Missing scheduler-referenced threadblock protobuf still fails at the intended missing file boundary.

- Closed the Gate6 forged-lineage gap:
  - Gate5 embedding tables now carry `gate5_lineage_bundle_hash`.
  - Gate6 formal selector requires a persisted `gcl_resnet50_gate5_lineage_bundle`.
  - A handcrafted table with self-consistent fake lineage is rejected without that persisted bundle.

- Added baseline-backed Gate7/Gate9 pipeline support:
  - `run_resnet50_gate1_to_gate7(..., baseline_artifacts_path=...)`.
  - Gate7 receives `metric_rows` from baseline artifacts when present.
  - Gate9 invokes `evaluate_gate9_sampled_vs_full(...)` when sampled/full/measured baselines are present.
  - Missing baselines still emit the no-claim fallback.

## BitLesson Applied

- `BL-20260606-scheduler-trace-reconciliation`
  - Used for the Gate1 negative-path repair so scheduler/threadblock protobuf records remain the validation source.

- `BL-20260606-formal-export-path`
  - Used for Gate6 lineage hardening: formal selector now requires persisted Gate5 lineage evidence, not only a self-consistent table payload.

## BitLesson Delta

- Action: none
- Lesson ID(s): NONE
- Notes: Existing lessons were sufficient; this round did not add a new reusable failure mode to `.humanize/bitlesson.md`.

## Real Trace Status

I searched the workspace for full ResNet-50 NVBit trace artifacts. I found only ResNet-related scripts and a separate `resnet_ref_layer3` ROI trace under another worktree, not a full real ResNet-50 NVBit acquisition root satisfying the plan boundary. Therefore Round 4 does not claim AC-1 completion. The code now blocks synthetic artifact-shape roots from producing formal Gate0 manifests.

## Validation

- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py tests/gcl_resnet50/test_gate1_adapter.py tests/gcl_resnet50/test_gate2_representative_sm.py tests/gcl_resnet50/test_gate3_canonical_graph.py tests/gcl_resnet50/test_gate4_tensorization.py tests/gcl_resnet50/test_gate5_rgcn_training.py tests/gcl_resnet50/test_gate6_selector.py tests/gcl_resnet50/test_gate7_correctness.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py`
  - `38 passed in 58.49s`

- `pytest -q tests/gcl_phase_b/test_selector_integration.py::test_selector_rejects_resource_blocked_rows tests/gcl_resnet50/test_gate6_selector.py`
  - `7 passed in 40.03s`

- `pytest -q tests/gcl_resnet50 tests/gcl_phase_b`
  - `171 passed in 317.82s`

- `pytest -q tests/gcl_phase_a tests/gcl_phase_b tests/gcl_resnet50`
  - `234 passed in 644.11s`

- `git diff --check`
  - passed with no output

## Commit

- `40e02598 fix: separate synthetic coverage from formal GCL evidence`

## Goal Tracker Update Request

### Requested Changes:

- Record that synthetic protobuf fixture coverage is now explicitly debug/artifact-shape coverage and cannot produce a formal Gate0 manifest.
- Record that Gate1 missing-static-metadata negative coverage now reaches the intended protobuf parser path.
- Record that Gate6 now requires a persisted Gate5 lineage bundle for formal selector execution and rejects forged self-consistent lineage without that bundle.
- Record that the pipeline now supports baseline-backed Gate7 metric rows and Gate9 sampled-vs-full comparison when baseline artifacts are provided.
- Keep AC-1 and real-trace formal acceptance tasks active because no full real ResNet-50 NVBit acquisition root exists in this workspace.

### Justification:

Round 4 addresses the Round 3 review blockers without overstating the result: synthetic fixture tests are no longer presented as formal evidence, Gate6 lineage cannot be forged by table-only self-consistency, and Gate7/Gate9 have a real baseline-backed code path while preserving the no-claim fallback when baselines are absent.
