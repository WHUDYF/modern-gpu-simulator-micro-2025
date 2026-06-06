# Round 7 Summary

## Work Completed
- Gate0 formal acceptance now requires collector-produced provenance from `acquire_resnet50_gate0_trace`.
  - The acquisition path writes `.nvbit_collector_session.json`.
  - It passes `GCL_RESNET50_COLLECTOR_SESSION_ID` to the runner environment.
  - After a successful runner return, it writes `nvbit_collector_attestation.json` and updates `nvbit_collection_evidence.json` with producer/session/attestation hashes.
  - `record_resnet50_gate0_trace_acquisition` now rejects a handwritten matching attestation on a synthetic root because it is not bound to the collector producer/session.
- Gate6 formal selector validation now loads persisted Gate5 artifacts from disk through `gate5_artifact_root`.
  - Formal Gate6 no longer succeeds from caller-supplied in-memory `gate5_manifests`.
  - It loads and hashes `gate5_lineage_bundle.json`, `rgcn_training_run_manifest.json`, `rgcn_checkpoint_manifest.json`, `readout_manifest.json`, and `embedding_export_report.json`.
  - It rejects forged table + forged lineage bundle + forged manifest dicts when no matching persisted Gate5 artifact root exists.
- The ResNet50 Gate1-7 pipeline now binds the embedding table lineage to persisted Gate5 manifests before selector execution.
  - Selector invocation now passes `gate5_artifact_root=out_dir`.
  - The pipeline writes a baseline-backed debug/report-path test that propagates baseline artifacts into Gate7 metric rows and Gate9 sampled-vs-full evaluation.
- No direct edits were made to `goal-tracker.md` in this round. It remains modified only from the RLCR review-side update that existed at round start.

## Files Changed
- `experiments/gcl_phase_b/resnet50_gate0.py`
- `experiments/gcl_phase_b/selector.py`
- `experiments/gcl_phase_b/resnet50_gate_pipeline.py`
- `tests/gcl_resnet50/test_gate0_trace_acquisition.py`
- `tests/gcl_resnet50/test_gate6_selector.py`
- `tests/gcl_phase_b/test_resnet50_gate_pipeline.py`

## Validation
- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py::test_gate0_rejects_handwritten_matching_attestation_on_synthetic_root tests/gcl_resnet50/test_gate0_trace_acquisition.py::test_gate0_acquisition_runner_writes_collector_attestation`
  - Initially failed before implementation, confirming the Gate0 bypass and missing producer writer.
- `pytest -q tests/gcl_resnet50/test_gate6_selector.py::test_gate6_rejects_forged_lineage_even_with_forged_manifest_dicts`
  - Initially failed before implementation, confirming the forged in-memory manifest loophole.
- `pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate_pipeline_propagates_baseline_artifacts_in_debug_report_path`
  - Initially failed before implementation because the acquisition path did not write collector attestation.
- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py::test_gate0_rejects_handwritten_matching_attestation_on_synthetic_root tests/gcl_resnet50/test_gate0_trace_acquisition.py::test_gate0_acquisition_runner_writes_collector_attestation tests/gcl_resnet50/test_gate6_selector.py::test_gate6_rejects_forged_lineage_even_with_forged_manifest_dicts tests/gcl_resnet50/test_gate6_selector.py::test_gate6_accepts_formal_table_with_persisted_gate5_artifact_root tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate_pipeline_propagates_baseline_artifacts_in_debug_report_path`
  - `5 passed in 6.62s`
- `pytest -q tests/gcl_resnet50/test_gate0_trace_acquisition.py tests/gcl_resnet50/test_gate6_selector.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py`
  - `22 passed in 38.09s`
- `pytest -q tests/gcl_resnet50 tests/gcl_phase_b`
  - `179 passed in 297.28s`
- `pytest -q tests/gcl_phase_a tests/gcl_phase_b tests/gcl_resnet50`
  - `242 passed in 606.31s`
- `git diff --check`
  - passed

## Remaining Items
- No real full ResNet-50 NVBit trace root exists in this workspace, so AC-1 through AC-7 still cannot be marked complete as a formal reproduction.
- The Gate0 producer-path positive coverage uses a real-style runner fixture to verify the writer/validator contract; it is not evidence of a real ResNet-50 NVBit acquisition.
- Baseline-backed pipeline coverage is restored as a debug/report-path test because a verified real Gate0 root is still unavailable.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260606-persisted-formal-boundaries
- Notes: Round 7 closed two self-consistency loopholes by requiring formal boundaries to load producer-owned or persisted artifacts from disk instead of trusting caller-supplied dictionaries. Gate6 also applied `BL-20260606-formal-export-path` by binding selector validation to the Phase B persisted Gate5 export path.

## BitLesson Entry Request

```markdown
## Lesson: persisted-formal-boundaries
Lesson ID: BL-20260606-persisted-formal-boundaries
Scope: experiments/gcl_phase_b/resnet50_gate0.py, experiments/gcl_phase_b/selector.py, experiments/gcl_phase_b/resnet50_gate_pipeline.py
Problem Description: Formal GCL gates can be bypassed when validation only checks self-consistency among caller-controlled or root-local JSON dictionaries.
Root Cause: Gate0 accepted a handwritten attestation over the same untrusted root, and Gate6 accepted in-memory Gate5 manifest dicts supplied by the caller instead of loading persisted artifacts.
Solution: Bind Gate0 formal acceptance to acquisition-produced session/attestation artifacts, and make Gate6 formal validation load Gate5 lineage and manifest files from a persisted artifact root before recomputing hashes.
Constraints: This does not prove real ResNet-50 trace availability; fixture-backed producer-path tests remain contract coverage only.
Validation Evidence: `pytest -q tests/gcl_resnet50 tests/gcl_phase_b` passed with 179 tests; `pytest -q tests/gcl_phase_a tests/gcl_phase_b tests/gcl_resnet50` passed with 242 tests; `git diff --check` passed.
Source Rounds: 7
```

## Goal Tracker Update Request

### Requested Changes:
- Record Round 7 as a code-level boundary repair: Gate0 handwritten matching attestation on a synthetic root is now rejected unless bound to the acquisition producer/session.
- Record Round 7 as a Gate6 boundary repair: formal selector validation now requires a persisted Gate5 artifact root and rejects forged table + forged lineage bundle + forged manifest dicts.
- Record that baseline-backed Gate7/Gate9 pipeline coverage was restored as debug/report-path coverage, not formal real-trace evidence.
- Keep all ACs active because no verified full real ResNet-50 NVBit trace root exists in the workspace.

### Justification:
These changes address the Round 6 blocking review issues without claiming full formal reproduction. The implementation closes the code-level trust-boundary holes and restores the missing AC-7 coverage, while preserving the plan boundary that fixture/debug inputs cannot substitute for a real ResNet-50 NVBit trace.
