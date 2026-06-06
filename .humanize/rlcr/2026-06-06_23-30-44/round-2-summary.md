# Round 2 Summary

## What Was Implemented

Addressed the Round 1 review by removing fixture-backed formal claims.

- Gate0 formal manifest now requires `nvbit_collection_evidence.json` and rejects fixture-backed roots.
- Added Gate0 blocker report output for the current in-workspace state where real full ResNet-50 NVBit trace has not been collected.
- Gate1 formal adapter still requires a valid Gate0 formal manifest; legacy JSON fixture roots cannot enter the formal path.
- Added a separate debug adapter builder for existing smoke coverage so fixture tests remain useful but visibly `debug_not_formal`.
- Updated ResNet-50 pipeline behavior so a Gate0 blocker stops the formal path at `gate0_blocked` instead of continuing to fake Gate1-7 artifacts.
- Propagated provenance fields through trace records, canonical graphs, tensors, and Gate5 embedding tables.
- Tightened Gate6 selector input validation so formal clustering requires a formal, provenance-carrying Gate5 embedding table by default.
- Kept debug Phase B smoke/replay paths explicit via `allow_debug=True`.
- Added missing `tests/gcl_resnet50` Gate2-Gate5 suites covering debug/formal boundaries and core artifact contracts.

## Files Changed

- `.humanize/rlcr/2026-06-06_23-30-44/goal-tracker.md`
- `experiments/gcl_phase_b/resnet50_gate0.py`
- `experiments/gcl_phase_b/resnet50_adapter.py`
- `experiments/gcl_phase_b/resnet50_gate_pipeline.py`
- `experiments/gcl_phase_b/resnet50_manifest.py`
- `experiments/gcl_phase_b/trace_scope.py`
- `experiments/gcl_phase_b/graph_builder.py`
- `experiments/gcl_phase_b/tensorizer.py`
- `experiments/gcl_phase_b/embedding_export.py`
- `experiments/gcl_phase_b/selector.py`
- `experiments/gcl_phase_b/pipeline.py`
- `tests/gcl_resnet50/test_gate0_trace_acquisition.py`
- `tests/gcl_resnet50/test_gate1_adapter.py`
- `tests/gcl_resnet50/test_gate2_representative_sm.py`
- `tests/gcl_resnet50/test_gate3_canonical_graph.py`
- `tests/gcl_resnet50/test_gate4_tensorization.py`
- `tests/gcl_resnet50/test_gate5_rgcn_training.py`
- `tests/gcl_resnet50/test_gate6_selector.py`
- related `tests/gcl_phase_b/*` smoke/replay migrations

## Validation

- BitLesson selector for Gate0/Gate1 formal boundary:
  - Applied `BL-20260606-source-invocation-identity`
  - Applied `BL-20260606-scheduler-trace-reconciliation`
- BitLesson selector for Gate2-Gate5 provenance tests:
  - Applied `BL-20260606-formal-export-path`
  - Applied `BL-20260606-source-invocation-identity`
- BitLesson selector for Gate6 provenance tightening:
  - Applied `BL-20260606-formal-export-path`
- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py tests/gcl_resnet50/test_gate1_adapter.py tests/gcl_phase_b/test_resnet50_adapter.py`
  - `19 passed`
- `pytest -q tests/gcl_resnet50/test_gate2_representative_sm.py tests/gcl_resnet50/test_gate3_canonical_graph.py tests/gcl_resnet50/test_gate4_tensorization.py tests/gcl_resnet50/test_gate5_rgcn_training.py`
  - `9 passed`
- `pytest -q tests/gcl_resnet50 tests/gcl_phase_b`
  - `159 passed in 259.35s`
- `git diff --check`
  - passed with no output

## Remaining Items

The actual full ResNet-50 NVBit acquisition runner is still not implemented in this workspace. The code now reports that honestly as a Gate0 blocker rather than treating fixtures as formal input.

Gate7/Gate8/Gate9 remain limited by the same missing real trace and measured/simulator baselines. The formal pipeline will not reach them until Gate0 produces real formal artifacts.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Existing lessons covered the relevant export-path and adapter identity/reconciliation issues. No new recurring pattern needs to be added yet.

## Goal Tracker Update Request

### Requested Changes:

- Keep Gate0 active, but update its notes to say the implementation now emits a formal blocker when real ResNet-50 NVBit collection is unavailable, instead of accepting fixture-backed roots.
- Keep Gate1 active, but update its notes to say legacy fixture JSON paths are now debug-only and formal Gate1 requires Gate0 formal manifest provenance.
- Keep Gate2-Gate6 active, but record that provenance propagation and plan-level Gate2-Gate5 tests were added in Round 2.
- Keep Gate7/Gate8/Gate9 active because real quantified evidence and baseline-driven extension execution remain blocked by missing real Gate0 artifacts.

### Justification:

Round 2 removed the false formal path and added the missing test surface, but it did not create the actual NVBit ResNet-50 acquisition runner. The tracker should therefore reflect progress on correctness boundaries without marking any AC completed.
