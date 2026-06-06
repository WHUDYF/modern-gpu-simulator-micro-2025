# Round 6 Summary

## Implemented

- Gate0 now validates a separately persisted collector attestation artifact:
  - Requires `nvbit_collector_attestation.json`.
  - Recomputes `collector_attestation_hash` from the persisted attestation artifact.
  - Checks that the evidence file references the same attestation hash.
  - Checks that attested `source_artifact_hashes` match the current Gate0 artifacts.
  - Added negative coverage for self-declared evidence hash and mismatched persisted attestation.

- Gate6 now validates persisted Gate5 manifest objects:
  - Formal selector accepts `gate5_manifests`.
  - Recomputes hashes for:
    - `rgcn_training_run_manifest`
    - `rgcn_checkpoint_manifest`
    - `readout_manifest_bundle`
    - `embedding_export_report`
  - Requires recomputed hashes to match the embedding table lineage.
  - A forged table plus forged populated lineage bundle is rejected without matching persisted manifest objects.

- Pipeline formal selector call now passes actual Gate5 manifest objects from the run:
  - `training_run_manifest`
  - `checkpoint_manifest`
  - `readout_bundle`
  - `export_report`

## BitLesson Applied

- `BL-20260606-formal-export-path`
  - Applied to Gate6: selector validation now depends on actual persisted Gate5 manifests, not helper-only or table-only self-consistency.

## BitLesson Delta

- Action: none
- Lesson ID(s): NONE
- Notes: Existing BitLessons were sufficient. No new reusable lesson was added.

## Remaining Boundary

No full real ResNet-50 NVBit acquisition root exists in this workspace. Round 6 does not claim AC-1 completion or formal end-to-end GCL reproduction. It hardens the code so repo-local synthetic artifacts cannot self-promote through Gate0/Gate6.

The baseline-backed Gate7/Gate9 code path remains implemented, but end-to-end formal baseline pipeline coverage still requires a valid real Gate0 root. I did not reintroduce a synthetic formal success test.

## Validation

- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py tests/gcl_resnet50/test_gate6_selector.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py`
  - `17 passed in 34.21s`

- `pytest -q tests/gcl_resnet50 tests/gcl_phase_b`
  - `174 passed in 312.43s`

- `pytest -q tests/gcl_phase_a tests/gcl_phase_b tests/gcl_resnet50`
  - `237 passed in 609.35s`

- `git diff --check`
  - passed with no output

## Commit

- `4cc12787 fix: require persisted formal GCL attestations`

## Goal Tracker Update Request

### Requested Changes:

- Record that Gate0 now requires persisted `nvbit_collector_attestation.json` and rejects evidence-only self-declared attestation hashes.
- Record that Gate6 now requires persisted Gate5 manifest objects and rejects forged table plus forged populated lineage bundle without matching manifests.
- Keep AC-1 active because a full real ResNet-50 NVBit acquisition root is still unavailable in this workspace.
- Keep AC-7 active for formal completion because baseline-backed end-to-end pipeline coverage still needs a real Gate0 root.

### Justification:

Round 6 addresses the two self-consistency loopholes identified in Round 5 while preserving the hard boundary that synthetic fixtures cannot stand in for real ResNet-50 NVBit evidence.
