# Round 1 Summary

## What Was Implemented

Implemented the first formal ResNet-50 GCL reproduction contracts requested by the Round 0 review.

- Added Gate0 formal acquisition manifest support for real ResNet-50 NVBit trace roots.
- Repaired Gate1 so formal adapter construction requires a Gate0 manifest and rejects plain fixture roots.
- Marked fixture paths as `debug_not_formal` / `formal_input_eligible = false`.
- Preserved Gate1 launch identity and scheduler/trace reconciliation behavior from prior BitLessons.
- Migrated existing ResNet-50 smoke tests so they create a formal Gate0 root before exercising formal Gate1-5 behavior.
- Upgraded selector output to Gate6 formal artifacts:
  - `embedding_normalization_report`
  - `k_selection_report`
  - `kmeans_cluster_assignment_table`
  - `representative_anchor_table`
  - `cluster_family_evidence_report`
- Added Gate7 report-only correctness evaluation artifacts.
- Added Gate8 and Gate9 extension contracts, explicitly labeled as our extension and not original GCL-Sampler reproduction.
- Extended the ResNet-50 gate pipeline from Gate1-5 to Gate1-7, while keeping the old function name as a compatibility alias.
- Updated Phase B replay validation to understand the new Gate6 selector artifact structure.

## Files Changed

- `experiments/gcl_phase_b/resnet50_gate0.py`
- `experiments/gcl_phase_b/resnet50_adapter.py`
- `experiments/gcl_phase_b/resnet50_gate_pipeline.py`
- `experiments/gcl_phase_b/selector.py`
- `experiments/gcl_phase_b/correctness.py`
- `experiments/gcl_phase_b/tuning.py`
- `experiments/gcl_phase_b/simulator_eval.py`
- `experiments/gcl_phase_b/pipeline.py`
- `tests/gcl_resnet50/*`
- `tests/gcl_phase_b/test_resnet50_adapter.py`
- `tests/gcl_phase_b/test_resnet50_manifest.py`
- `tests/gcl_phase_b/test_resnet50_gate_pipeline.py`
- `tests/gcl_phase_b/test_resnet50_gate_replay.py`
- `tests/gcl_phase_b/test_embedding_export.py`
- `tests/gcl_phase_b/test_selector_integration.py`
- `tests/gcl_phase_b/test_replay.py`

## Validation

- BitLesson selector for Gate0/Gate1 formal boundary:
  - Applied `BL-20260606-source-invocation-identity`
  - Applied `BL-20260606-scheduler-trace-reconciliation`
- BitLesson selector for Gate6 and Gate7 formal artifacts:
  - Applied `BL-20260606-formal-export-path`
- BitLesson selector for Gate8/Gate9 extensions:
  - `LESSON_IDS: NONE`
- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py tests/gcl_resnet50/test_gate1_adapter.py`
  - `6 passed`
- `pytest -q tests/gcl_resnet50/test_gate6_selector.py tests/gcl_resnet50/test_gate7_correctness.py tests/gcl_resnet50/test_gate8_tuning.py tests/gcl_resnet50/test_gate9_simulator_evaluation.py tests/gcl_phase_b/test_selector_integration.py`
  - `17 passed`
- `pytest -q tests/gcl_resnet50 tests/gcl_phase_b`
  - `151 passed in 289.15s`
- `git diff --check`
  - passed with no output

## Remaining Items

This round implements the formal contracts and testable artifact path through Gate7, plus Gate8/Gate9 extension contracts. It does not perform an actual NVBit ResNet-50 trace collection run in this workspace. The Gate0 manifest function records and validates an already collected formal trace root.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Existing lessons were applied where relevant; no new recurring failure mode was discovered that requires a new BitLesson entry.

## Goal Tracker Update Request

### Requested Changes:

- Mark `Gate0 real ResNet-50 NVBit trace acquisition and formal acquisition manifest` as completed for the artifact contract implementation, with evidence from `tests/gcl_resnet50/test_gate0_trace_acquisition.py`.
- Mark `Gate1 formal adapter boundary and kernel / CTA / warp / instruction record extraction` as completed for the formal/debug boundary and adapter validation contract, with evidence from `tests/gcl_resnet50/test_gate1_adapter.py` and migrated `tests/gcl_phase_b/test_resnet50_adapter.py`.
- Mark `Gate6 selector / representative anchor / post-clustering family evidence` as completed for the formal selector artifact contract, with evidence from `tests/gcl_resnet50/test_gate6_selector.py`.
- Mark `Gate7 correctness evaluation plus Gate8 / Gate9 extension gates` as partially completed: Gate7 report-only correctness and Gate8/Gate9 extension contracts are implemented and tested, but real measured/simulator baseline execution remains future work.
- Add an open issue: actual real ResNet-50 NVBit collection has not been executed in this workspace; Gate0 currently validates and records an existing formal trace root rather than launching NVBit collection itself.

### Justification:

The code now enforces the formal input boundary that Round 0 review identified as blocking, moves the pipeline to Gate7, and adds tested Gate8/Gate9 extension contracts. The remaining open issue is operational data acquisition, not a fixture substitution: fixtures are now explicitly rejected as formal input.
