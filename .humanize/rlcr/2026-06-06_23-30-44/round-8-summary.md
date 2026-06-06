# Round 8 Summary

## Work Completed
- Closed the specific Gate0 bypass reported in Round 7 review.
  - Added a negative test for a synthetic artifact-shape root with a handwritten `.nvbit_collector_session.json`, matching `nvbit_collector_attestation.json`, and matching evidence fields.
  - Gate0 now rejects synthetic artifact-shape roots before collector attestation validation, even when the local JSON triplet is self-consistent.
- Tightened Gate0 synthetic-marker detection.
  - Rejects `synthetic_artifact_shape_unit_test_only` evidence scope.
  - Rejects the unit-test runner invocation marker.
  - Rejects artifact-shape scheduler metadata and enhanced execution info markers.
  - Rejects the protobuf unit trace name marker.
- Updated tests that previously expected later attestation failures on synthetic roots.
  - Those now correctly expect the earlier `synthetic artifact-shape` boundary rejection.
  - A contract-style local root is used only to keep lower-level attestation mismatch branch coverage available.
- Updated the baseline-backed pipeline test so it no longer depends on an artifact-shape root being accepted by Gate0.
  - It remains debug/report-path coverage and is not claimed as formal real-trace evidence.

## Files Changed
- `experiments/gcl_phase_b/resnet50_gate0.py`
- `tests/gcl_resnet50/test_gate0_trace_acquisition.py`
- `tests/gcl_phase_b/test_resnet50_gate_pipeline.py`

## Validation
- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py::test_gate0_rejects_handwritten_session_attestation_triplet_on_synthetic_root tests/gcl_resnet50/test_gate0_trace_acquisition.py::test_gate0_acquisition_runner_writes_attestation_but_rejects_synthetic_trace`
  - Initially failed before implementation, confirming the Round 7 Gate0 bypass.
- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py`
  - `10 passed in 0.05s`
- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py tests/gcl_resnet50/test_gate6_selector.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py`
  - `23 passed in 34.18s`
- `pytest -q tests/gcl_resnet50 tests/gcl_phase_b`
  - `180 passed in 307.92s`
- `pytest -q tests/gcl_phase_a tests/gcl_phase_b tests/gcl_resnet50`
  - `243 passed in 600.27s`
- `git diff --check`
  - passed

## Remaining Items
- No verified full real ResNet-50 NVBit trace root exists in this workspace.
- AC-1 through AC-7 must remain active because the original formal positive path still requires a real collected root and downstream validation on that real chain.
- The baseline-backed pipeline coverage remains debug/report-path coverage, not formal proof.
- Gate6 persisted-root validation still needs eventual binding to a verified real Gate0-5 artifact chain for full-plan acceptance.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: Applied existing `BL-20260606-persisted-formal-boundaries`; no new reusable lesson was added.

## Goal Tracker Update Request

### Requested Changes:
- Record that Round 8 closed the specific Gate0 synthetic artifact-shape promotion bypass involving a handwritten session + attestation + evidence triplet.
- Keep AC-1 active because no real full ResNet-50 NVBit collection root exists in this workspace.
- Keep AC-2 through AC-7 active because their formal positive evidence still depends on a verified real Gate0 acquisition chain.
- Keep the baseline-backed Gate7/Gate9 coverage classified as debug/report-path coverage.

### Justification:
Round 8 addresses the concrete code-level Gate0 loophole found by Codex without claiming completion of the original real-trace reproduction plan. The remaining work is still the real ResNet-50 NVBit acquisition and end-to-end formal validation, so the active task state should remain conservative.
