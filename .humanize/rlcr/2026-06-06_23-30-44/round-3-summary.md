# Round 3 Summary

## Implemented

- Added a testable Gate0 ResNet-50 NVBit acquisition runner:
  - `ResNet50NvbitAcquisitionConfig`
  - `acquire_resnet50_gate0_trace(...)`
  - command/env construction with `LD_PRELOAD` and `GCL_RESNET50_TRACE_OUT`
  - formal manifest recording after runner completion

- Reworked Gate1 formal adapter to consume Gate0 formal artifacts directly:
  - reads `dynamic_trace.pb`
  - reads scheduler-referenced `threadblocks/**/*.pb`
  - keeps debug JSON fixture path separate via `build_resnet50_debug_trace_adapter_bundle`
  - preserves launch-order identity for repeated `kernel_id`
  - validates scheduler/threadblock CTA and warp reconciliation through existing adapter validation

- Added formal ResNet-50 protobuf fixture helpers for tests:
  - `tests/gcl_resnet50/formal_fixture.py`
  - `tests/gcl_resnet50/formal_chain.py`
  - formal chain helper covers Gate0 -> Gate1 -> Gate2 -> Gate3 -> Gate4

- Replaced debug-smoke-only evidence with formal Gate2-Gate5 positive path tests:
  - Gate2 representative-SM manifest from formal Gate1 bundle
  - Gate3 canonical graph from formal Gate2 manifest
  - Gate4 tensorization from formal graphs
  - Gate5 256D embedding export from formal tensors

- Added auditable Gate5 lineage and Gate6 enforcement:
  - embedding table now carries `gate5_lineage`
  - rows carry `gate5_lineage_hash`
  - Gate6 rejects handcrafted formal-looking tables without Gate5 lineage
  - selector still rejects augmented/projection/forbidden clustering inputs

- Wired Gate7-Gate9 artifact path:
  - Gate7 computes embedding geometry from Gate5 embedding table and Gate6 assignments
  - pipeline writes Gate7 correctness manifest with `source_gate5_embedding_table_hash`
  - pipeline writes Gate8 report-only tuning vector proposal
  - pipeline writes Gate9 no-claim baseline-missing report instead of claiming speedup/accuracy

## BitLesson Applied

- `BL-20260606-source-invocation-identity`
  - Gate1 protobuf parser preserves launch-order identity so repeated `kernel_id` launches do not collapse.

- `BL-20260606-scheduler-trace-reconciliation`
  - Gate1 formal threadblock loading is driven by scheduler metadata, then existing validation checks CTA/warp/entry-count consistency.

- `BL-20260606-formal-export-path`
  - Gate5 export now carries readout/training lineage, and Gate6/Gate7 consume formal embedding artifacts rather than helper-only stand-ins.

## BitLesson Delta

- Action: none
- Lesson ID(s): NONE
- Notes: Existing lessons were sufficient for this round; no new reusable failure mode was added to `.humanize/bitlesson.md`.

## Validation

- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py tests/gcl_resnet50/test_gate1_adapter.py`
  - `9 passed in 0.04s`

- `pytest -q tests/gcl_resnet50/test_gate2_representative_sm.py tests/gcl_resnet50/test_gate3_canonical_graph.py tests/gcl_resnet50/test_gate4_tensorization.py tests/gcl_resnet50/test_gate5_rgcn_training.py tests/gcl_resnet50/test_gate6_selector.py`
  - `18 passed in 52.28s`

- `pytest -q tests/gcl_resnet50/test_gate7_correctness.py tests/gcl_resnet50/test_gate8_tuning.py tests/gcl_resnet50/test_gate9_simulator_evaluation.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py`
  - `15 passed in 12.63s`

- `pytest -q tests/gcl_resnet50 tests/gcl_phase_b`
  - `169 passed in 311.20s`

- `pytest -q tests/gcl_phase_a tests/gcl_phase_b tests/gcl_resnet50`
  - `232 passed in 631.87s`

- `git diff --check`
  - passed with no output

## Commit

- `d1799311 feat: wire formal ResNet50 GCL artifact path`

## Goal Tracker Update Request

### Requested Changes:

- Mark AC-1 as partially advanced: Gate0 now has a concrete acquisition runner entrypoint and command/env hookup, while actual GPU/NVBit execution in this workspace remains environment-dependent.
- Mark AC-2 as implemented for artifact-format transition: Gate1 formal path now consumes `dynamic_trace.pb` and scheduler-referenced `threadblocks/` protobuf files instead of legacy JSON fixtures.
- Mark AC-3 through AC-5 as substantially advanced: formal positive tests now exercise Gate2-Gate5 from the formal Gate0/Gate1 artifact chain.
- Mark AC-6 as substantially advanced: Gate6 now requires auditable Gate5 lineage and rejects handcrafted formal-looking embedding tables.
- Mark AC-7 as advanced: Gate7 geometry is computed from actual Gate5/Gate6 artifacts, and Gate8/Gate9 are wired into the pipeline as report-only/no-claim extension artifacts.
- Add an open issue if needed: the in-repo formal positive tests use minimal protobuf fixtures to validate the formal artifact format; a full real ResNet-50 NVBit run is still required before claiming empirical reproduction on a production trace.

### Justification:

Round 3 closes the structural review blockers around formal artifact formats, formal positive-path tests, Gate5 lineage, and Gate7-Gate9 pipeline wiring. It does not claim that a large real production ResNet-50 trace has been collected in this environment; it makes the formal path executable and auditable once such artifacts are present.
