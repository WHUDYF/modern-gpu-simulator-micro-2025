# Round 9 Summary

## Work Completed
- Removed the marker-rewritten synthetic Gate0 success path.
  - Deleted `_contract_style_root(...)` helpers from Gate0 and pipeline tests.
  - Removed tests that depended on `acquire_resnet50_gate0_trace(...)` accepting a rewritten artifact-shape root.
  - Added `test_gate0_rejects_marker_rewritten_synthetic_root`.
- Hardened Gate0 evidence validation.
  - `nvbit_collection_evidence.json` must now include `collector_artifact_origin = real_nvbit_runtime`.
  - `_write_collector_attestation(...)` validates evidence before writing `nvbit_collector_attestation.json`, so synthetic outputs do not receive producer attestation.
  - `record_resnet50_gate0_trace_acquisition(...)` and `acquire_resnet50_gate0_trace(...)` now reject synthetic roots before formal manifest creation.
- Reworked baseline pipeline coverage.
  - Removed the Gate1-7 synthetic success path from the baseline-backed pipeline test.
  - Baseline artifacts are now verified to remain blocked when Gate0 is blocked and no real acquisition root exists.

## Files Changed
- `experiments/gcl_phase_b/resnet50_gate0.py`
- `tests/gcl_resnet50/test_gate0_trace_acquisition.py`
- `tests/gcl_phase_b/test_resnet50_gate_pipeline.py`

## Validation
- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py`
  - `14 passed in 1.00s`
- `pytest -q tests/gcl_resnet50/test_gate6_selector.py tests/gcl_resnet50/test_gate7_correctness.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py`
  - `21 passed in 36.94s`
- `pytest -q tests/gcl_resnet50 tests/gcl_phase_b`
  - `180 passed in 303.10s`
- `pytest -q tests/gcl_phase_a tests/gcl_phase_b tests/gcl_resnet50`
  - `243 passed in 632.45s`
- `git diff --check`
  - passed

## Remaining Items
- AC-1 through AC-7 remain active because no verified full real ResNet-50 NVBit trace root exists in this workspace.
- There is no formal positive Gate1-7 success path without a real Gate0 acquisition root.
- Gate6/Gate7 debug and artifact-shape coverage remains unit/debug coverage only, not formal reproduction evidence.
- The original plan still requires implementing and validating the real ResNet-50 NVBit acquisition chain.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: Applied existing `BL-20260606-persisted-formal-boundaries`; no new reusable lesson was added.

## Goal Tracker Update Request

### Requested Changes:
- Record that Round 9 removed the marker-rewritten synthetic Gate0 formal success path.
- Record that Gate0 now requires `collector_artifact_origin = real_nvbit_runtime` before formal attestation/manifest creation.
- Keep AC-1 through AC-7 active because no verified real ResNet-50 NVBit root exists.
- Keep baseline-backed Gate7/Gate9 coverage classified as blocked/debug coverage until a real Gate0 root exists.

### Justification:
This round fixes the concrete Round 8 review issue without claiming full formal reproduction. The system no longer treats marker-rewritten artifact-shape roots as formal Gate0 inputs. The remaining work is the real acquisition path and downstream formal validation required by the original plan.
