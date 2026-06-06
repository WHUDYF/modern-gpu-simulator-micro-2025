# Round 5 Summary

## Implemented

- Closed the Gate0 evidence-scope spoofing hole:
  - Gate0 now requires `collector_attestation_hash` in `nvbit_collection_evidence.json`.
  - The attestation hash must match the source artifact hashes and formal collection fields.
  - A synthetic helper root with `evidence_scope = real_resnet50_nvbit_collection` is rejected without collector attestation.

- Closed the stronger Gate6 forged lineage case:
  - Gate5 lineage bundles now include `persisted_manifest_hashes`.
  - Gate6 validates those persisted hashes against the embedding table lineage.
  - A forged formal table plus forged matching lineage bundle is rejected.

- Fixed the successful pipeline manifest filename contract:
  - Introduced `GATE1_7_PIPELINE_MANIFEST_FILENAME = "gate1_7_pipeline_manifest.json"`.
  - Success and blocked paths now both write `gate1_7_pipeline_manifest.json`.
  - Tests assert the old `gate1_5_pipeline_manifest.json` path is not written.

## BitLesson Applied

- `BL-20260606-formal-export-path`
  - Used while tightening Gate6 lineage validation so selector validation depends on persisted Gate5 manifest hashes rather than table/bundle self-consistency alone.

## BitLesson Delta

- Action: none
- Lesson ID(s): NONE
- Notes: Existing lessons covered the relevant Gate6 formal export boundary. No new reusable lesson was added.

## Remaining Boundary

No full real ResNet-50 NVBit acquisition root exists in this workspace. AC-1 remains active and externally blocked until such a trace is collected. Round 5 only hardens the formal boundary so synthetic helpers cannot promote themselves into formal Gate0 evidence.

## Validation

- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py tests/gcl_resnet50/test_gate6_selector.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py`
  - `15 passed in 34.62s`

- `pytest -q tests/gcl_resnet50 tests/gcl_phase_b`
  - `172 passed in 311.80s`

- `pytest -q tests/gcl_phase_a tests/gcl_phase_b tests/gcl_resnet50`
  - `235 passed in 642.50s`

- `git diff --check`
  - passed with no output

## Commit

- `97622c66 fix: harden formal GCL artifact boundaries`

## Goal Tracker Update Request

### Requested Changes:

- Record that Gate0 synthetic scope spoofing is closed by collector attestation hash validation.
- Record that Gate6 forged table plus forged lineage bundle is rejected through persisted Gate5 manifest hash validation.
- Record that the successful pipeline manifest filename contract now uses `gate1_7_pipeline_manifest.json`.
- Keep AC-1 active because no full real ResNet-50 NVBit acquisition root exists in this workspace.

### Justification:

Round 5 directly addresses the three Round 4 review findings. It removes the synthetic formal promotion loophole, hardens Gate6 against self-consistent forged lineage bundles, and fixes the pipeline manifest path inconsistency without claiming real-trace completion.
