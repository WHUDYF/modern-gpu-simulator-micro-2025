# A-Line PKA-M1 Measured Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first PKA-M1 measured loop across all L1 P0 workloads, from workload resolution through exact NCU capture, strict 12D measured feature extraction, selector eligibility / repair reporting, and formal PCA/k-means anchor evaluation.

**Architecture:** Implement M1 as five explicit gates with machine-readable artifacts between gates. Gate1 resolves and smoke-runs workloads, Gate2 captures exact NCU CSVs, Gate3 extracts strict measured 12D records, Gate4 decides selector eligibility and repair actions, and Gate5 runs the shared selector core to produce formal M1 anchors and structural evaluation. M0 and M1 must share `pka_selector_core.py` to prevent PCA/k-means algorithm drift.

**Tech Stack:** Python 3, `numpy`, JSON artifacts, `pytest`, Nsight Compute CLI (`ncu`), existing `experiments/baseline_diagnosis` scripts and artifacts.

---

## Normative References

- `docs/superpowers/specs/2026-05-07-a-line-pka-m1-measured-loop-design.md`
- `docs/superpowers/specs/2026-05-07-a-line-pka-m1-gate1-workload-resolver-design.md`
- `docs/superpowers/specs/2026-05-07-a-line-pka-m1-gate2-ncu-capture-dispatcher-design.md`
- `docs/superpowers/specs/2026-05-08-a-line-pka-m1-gate3-measured-feature-extractor-design.md`
- `docs/superpowers/specs/2026-05-08-a-line-pka-m1-gate4-selector-eligibility-repair-design.md`
- `docs/superpowers/specs/2026-05-08-a-line-pka-m1-gate5-formal-selector-evaluation-design.md`
- `docs/superpowers/plans/2026-05-06-a-line-pka-m0-minimal-loop-ablation.plan.md`

If this plan conflicts with any listed spec, follow the spec.

## Confirmed Product Decisions

- Run the full Gate1-Gate5 M1 loop in the first RLCR.
- Cover all L1 P0 workloads, not only easy seed workloads.
- Allow two outcomes:
  - `measured_rows >= 3`: run Gate5 and produce formal M1 selector artifacts.
  - `measured_rows < 3`: do not run Gate5, but complete Gate1-Gate4 and emit a complete per-entry backward repair report.
- Allow controlled build only from `workload_registry_l1.json`; do not guess build commands.
- Extract `pka_selector_core.py`; M0 and M1 must share PCA/k-means/anchor logic.
- Do not include B-line consumption.
- M1 evaluation is structural-only, not simulator accuracy or measured speedup.
- Gate5 only reads Gate4 selector input projection.
- Timing unit policy is strict: all no timing -> member-count fallback; one timing unit -> timing weight; mixed timing -> block Gate5.
- Stop-hook must be hard: missing required artifacts, incomplete P0 coverage, missing repair report under `<3`, fake Gate5 completion, or M0 artifact overwrite must fail.

## File Structure

Create:

- `experiments/baseline_diagnosis/workload_registry_l1.json`
  - Static L1 P0 workload registry. Maps workload ids to real binaries, build commands, run args, smoke args, working directories, and capture timeout.
- `experiments/baseline_diagnosis/m1_workload_resolver.py`
  - Gate1 resolver. Reads manifest + registry, optionally runs allowlisted build commands, runs smoke tests, writes resolution / Gate1 gap artifacts.
- `experiments/baseline_diagnosis/m1_ncu_capture_dispatcher.py`
  - Gate2 dispatcher. Reads Gate1 resolution, deduplicates commands, resolves selected metrics, runs / dry-runs exact `ncu --metrics` capture, writes attempts / Gate2 gap artifacts.
- `experiments/baseline_diagnosis/m1_measured_feature_extractor.py`
  - Gate3 extractor. Reads Gate2 eligible capture artifacts, parses NCU CSVs, emits strict measured feature table / acquisition gaps / audits.
- `experiments/baseline_diagnosis/m1_selector_eligibility.py`
  - Gate4 eligibility and repair reporter. Checks measured rows, feature table quality, selector input projection, timing policy, and emits repair report.
- `experiments/baseline_diagnosis/pka_selector_core.py`
  - Shared pure selector core used by M0 and M1.
- `experiments/baseline_diagnosis/pka_m1_selector.py`
  - Gate5 M1 wrapper around shared selector core.
- `experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py`
- `experiments/baseline_diagnosis/tests/test_m1_ncu_capture_dispatcher.py`
- `experiments/baseline_diagnosis/tests/test_m1_measured_feature_extractor.py`
- `experiments/baseline_diagnosis/tests/test_m1_selector_eligibility.py`
- `experiments/baseline_diagnosis/tests/test_m1_selector.py`

Modify:

- `experiments/baseline_diagnosis/pka_m0_pipeline.py`
  - Replace internal selector algorithm functions with calls into `pka_selector_core.py` while preserving M0 artifact schema and behavior.
- `experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py`
  - Keep existing M0 tests passing after selector core extraction.
- `experiments/baseline_diagnosis/test_l1_regression.py`
  - Add smoke coverage for the M1 orchestrated gates if lightweight enough; keep existing regression tests stable.

Do not modify:

- B-line consumers.
- `pka_m0_*` artifact paths except by running existing M0 tests / pipeline.
- Formal M1 selector output paths from Gate5 except via `pka_m1_selector.py`.

## Required Final Verification

Run these commands before claiming completion:

```bash
python experiments/baseline_diagnosis/m1_workload_resolver.py
python experiments/baseline_diagnosis/m1_ncu_capture_dispatcher.py
python experiments/baseline_diagnosis/m1_measured_feature_extractor.py
python experiments/baseline_diagnosis/m1_selector_eligibility.py
python experiments/baseline_diagnosis/pka_m1_selector.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_ncu_capture_dispatcher.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_measured_feature_extractor.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_selector_eligibility.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_selector.py
pytest -q experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py
pytest -q experiments/baseline_diagnosis/test_l1_regression.py
```

If the local machine cannot execute NCU successfully, the Gate2 command may still produce `environment_blocked` / `permission_blocked` evidence. The RLCR may still complete only if Gate4 emits the required blocked-on-acquisition repair report.

## Completion Outcomes

### Outcome A: Full M1 Selector Success

Required:

- Every L1 P0 manifest entry has a Gate1 outcome.
- Every Gate1 resolved entry is covered by Gate2 attempts.
- Every Gate2 eligible capture job is attempted by Gate3.
- `pka_feature_table_l1.json` has at least 3 complete `pka_m1_measured` rows.
- Gate4 emits `selector_ready` or `selector_ready_with_remaining_gaps`.
- Gate5 emits:
  - `artifacts/a_line/l1/pka_pca_projection_l1.json`
  - `artifacts/a_line/l1/pka_kmeans_clusters_l1.json`
  - `artifacts/a_line/l1/representative_anchor_table_l1.json`
  - `artifacts/a_line/l1/pka_compression_evaluation_l1.json`
- Remaining gaps, if any, are listed in `m1_backward_repair_report_l1.json`.
- M0 tests pass and M0 artifacts are not overwritten by M1.

### Outcome B: Correct Blocked-on-Acquisition

Allowed only when `measured_rows < 3`.

Required:

- Gate1-Gate4 have run.
- Gate5 has not run and does not claim success.
- `m1_selector_eligibility_l1.json` has `gate5_allowed: false`.
- `m1_backward_repair_report_l1.json` and `.md` exist.
- Every L1 P0 entry has `earliest_failed_gate` or `measured`.
- Repair actions are concrete and sourced from registry, repo scripts, capture artifacts, metric artifacts, or manual environment instructions.

## Tasks

### Task 1: Add M1 Workload Registry

**Files:**
- Create: `experiments/baseline_diagnosis/workload_registry_l1.json`
- Test: `experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py`

- [ ] **Step 1: Write registry shape tests**

Add tests that load `workload_registry_l1.json` and assert required entries exist:

```python
def test_registry_contains_all_l1_p0_workloads():
    import json
    from pathlib import Path

    registry = json.loads(Path("experiments/baseline_diagnosis/workload_registry_l1.json").read_text())
    ids = {entry["workload_id"] for entry in registry["entries"]}
    assert {
        "l1_bw_32f", "l2_bw_32f", "mem_bw", "mem_lat", "shared_bw", "MaxFlops",
        "rodinia_nn", "mini_transformer_v4",
    }.issubset(ids)


def test_registry_build_commands_are_explicit_lists_or_null():
    import json
    from pathlib import Path

    registry = json.loads(Path("experiments/baseline_diagnosis/workload_registry_l1.json").read_text())
    for entry in registry["entries"]:
        assert "build_command" in entry
        assert entry["build_command"] is None or isinstance(entry["build_command"], list)
        assert "binary_path" in entry
        assert "run_command_template" in entry
        assert "smoke_timeout_seconds" in entry
        assert "capture_timeout_seconds" in entry
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py -k registry
```

Expected: FAIL because the registry file does not exist.

- [ ] **Step 3: Create initial registry**

Create `workload_registry_l1.json` with this structure:

```json
{
  "registry_name": "m1_workload_registry_l1",
  "dataset_level": "L1",
  "entries": [
    {
      "workload_id": "l1_bw_32f",
      "source_type": "local_microbench",
      "binary_path": "experiments/baseline_diagnosis/bin/l1_bw_32f",
      "build_command": null,
      "run_args": [],
      "run_command_template": "{binary_path}",
      "working_directory": ".",
      "expected_kernel_or_case": "l1_bw_32f",
      "capture_target_type": "single_kernel_binary",
      "smoke_args": [],
      "smoke_timeout_seconds": 30,
      "capture_timeout_seconds": 300,
      "expected_output_regex": null
    },
    {
      "workload_id": "l2_bw_32f",
      "source_type": "local_microbench",
      "binary_path": "experiments/baseline_diagnosis/bin/l2_bw_32f",
      "build_command": null,
      "run_args": [],
      "run_command_template": "{binary_path}",
      "working_directory": ".",
      "expected_kernel_or_case": "l2_bw_32f",
      "capture_target_type": "single_kernel_binary",
      "smoke_args": [],
      "smoke_timeout_seconds": 30,
      "capture_timeout_seconds": 300,
      "expected_output_regex": null
    },
    {
      "workload_id": "mem_bw",
      "source_type": "local_microbench",
      "binary_path": "experiments/baseline_diagnosis/bin/mem_bw",
      "build_command": null,
      "run_args": [],
      "run_command_template": "{binary_path}",
      "working_directory": ".",
      "expected_kernel_or_case": "mem_bw",
      "capture_target_type": "single_kernel_binary",
      "smoke_args": [],
      "smoke_timeout_seconds": 30,
      "capture_timeout_seconds": 300,
      "expected_output_regex": null
    },
    {
      "workload_id": "mem_lat",
      "source_type": "local_microbench",
      "binary_path": "experiments/baseline_diagnosis/bin/mem_lat",
      "build_command": null,
      "run_args": [],
      "run_command_template": "{binary_path}",
      "working_directory": ".",
      "expected_kernel_or_case": "mem_lat",
      "capture_target_type": "single_kernel_binary",
      "smoke_args": [],
      "smoke_timeout_seconds": 30,
      "capture_timeout_seconds": 300,
      "expected_output_regex": null
    },
    {
      "workload_id": "shared_bw",
      "source_type": "local_microbench",
      "binary_path": "experiments/baseline_diagnosis/bin/shared_bw",
      "build_command": null,
      "run_args": [],
      "run_command_template": "{binary_path}",
      "working_directory": ".",
      "expected_kernel_or_case": "shared_bw",
      "capture_target_type": "single_kernel_binary",
      "smoke_args": [],
      "smoke_timeout_seconds": 30,
      "capture_timeout_seconds": 300,
      "expected_output_regex": null
    },
    {
      "workload_id": "MaxFlops",
      "source_type": "local_microbench",
      "binary_path": "experiments/baseline_diagnosis/bin/MaxFlops",
      "build_command": null,
      "run_args": [],
      "run_command_template": "{binary_path}",
      "working_directory": ".",
      "expected_kernel_or_case": "MaxFlops",
      "capture_target_type": "single_kernel_binary",
      "smoke_args": [],
      "smoke_timeout_seconds": 30,
      "capture_timeout_seconds": 300,
      "expected_output_regex": null
    },
    {
      "workload_id": "rodinia_nn",
      "source_type": "local_benchmark_result",
      "binary_path": "experiments/baseline_diagnosis/bin/rodinia_nn",
      "build_command": null,
      "run_args": [],
      "run_command_template": "{binary_path}",
      "working_directory": ".",
      "expected_kernel_or_case": "euclid",
      "capture_target_type": "single_kernel_binary",
      "smoke_args": [],
      "smoke_timeout_seconds": 30,
      "capture_timeout_seconds": 300,
      "expected_output_regex": null
    },
    {
      "workload_id": "mini_transformer_v4",
      "source_type": "local_ai_workload",
      "binary_path": "experiments/baseline_diagnosis/bin/mini_transformer_v4",
      "build_command": null,
      "run_args": [],
      "run_command_template": "{binary_path}",
      "working_directory": ".",
      "expected_kernel_or_case": "gemm_tiled|attention_score|softmax_kernel",
      "capture_target_type": "multi_kernel_binary",
      "smoke_args": [],
      "smoke_timeout_seconds": 30,
      "capture_timeout_seconds": 600,
      "expected_output_regex": null
    }
  ]
}
```

If actual binary paths differ, update the registry to existing project binaries or add allowlisted `build_command` arrays. Do not make the resolver guess build commands.

- [ ] **Step 4: Run registry tests**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py -k registry
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline_diagnosis/workload_registry_l1.json experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py
git commit -m "feat: add M1 workload registry"
```

### Task 2: Implement Gate1 Workload Resolver

**Files:**
- Create: `experiments/baseline_diagnosis/m1_workload_resolver.py`
- Modify: `experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py`

- [ ] **Step 1: Add resolver unit tests**

Add tests for registry missing, dispatcher final provenance rejection, binary missing, smoke failure, and shared binary entries:

```python
def test_gate1_emits_gap_when_registry_entry_missing(tmp_path):
    from experiments.baseline_diagnosis import m1_workload_resolver as r

    manifest = {"entries": [{"id": "L1_X", "priority": "P0", "benchmark_name": "x", "kernel_or_case": "x", "source_type": "local_microbench", "run_args": "missing"}]}
    registry = {"entries": []}
    result = r.resolve_manifest_entries(manifest, registry, repo_root=tmp_path, run_smoke=False)
    assert result["records"] == []
    assert result["gaps"][0]["gap_reason"] == "registry_missing"
    assert result["gaps"][0]["failed_gate"] == "Gate1"


def test_gate1_rejects_dispatcher_as_final_binary(tmp_path):
    from experiments.baseline_diagnosis import m1_workload_resolver as r

    dispatcher = tmp_path / "dispatch_ncu_capture.sh"
    dispatcher.write_text("#!/usr/bin/env bash\n")
    dispatcher.chmod(0o755)
    entry = {"id": "L1_MB_01", "priority": "P0", "benchmark_name": "l1_bw_32f", "kernel_or_case": "l1_bw_32f", "source_type": "local_microbench", "run_args": "l1_bw_32f"}
    registry = {"entries": [{"workload_id": "l1_bw_32f", "binary_path": str(dispatcher), "build_command": None, "run_args": [], "run_command_template": "{binary_path}", "working_directory": ".", "smoke_args": [], "smoke_timeout_seconds": 1, "capture_timeout_seconds": 1}]}
    result = r.resolve_manifest_entries({"entries": [entry]}, registry, repo_root=tmp_path, run_smoke=False)
    assert result["gaps"][0]["gap_reason"] == "binary_unresolved"


def test_gate1_shared_binary_keeps_three_records(tmp_path):
    from experiments.baseline_diagnosis import m1_workload_resolver as r

    binary = tmp_path / "mini"
    binary.write_text("#!/usr/bin/env bash\nexit 0\n")
    binary.chmod(0o755)
    entries = [
        {"id": "L1_AI_01", "priority": "P0", "benchmark_name": "gemm_tiled", "kernel_or_case": "gemm_tiled", "source_type": "local_ai_workload", "run_args": "mini_transformer_v4"},
        {"id": "L1_AI_02", "priority": "P0", "benchmark_name": "attention_score", "kernel_or_case": "attention_score", "source_type": "local_ai_workload", "run_args": "mini_transformer_v4"},
        {"id": "L1_AI_03", "priority": "P0", "benchmark_name": "softmax_kernel", "kernel_or_case": "softmax_kernel", "source_type": "local_ai_workload", "run_args": "mini_transformer_v4"},
    ]
    registry = {"entries": [{"workload_id": "mini_transformer_v4", "binary_path": str(binary), "build_command": None, "run_args": [], "run_command_template": "{binary_path}", "working_directory": ".", "smoke_args": [], "smoke_timeout_seconds": 5, "capture_timeout_seconds": 30}]}
    result = r.resolve_manifest_entries({"entries": entries}, registry, repo_root=tmp_path, run_smoke=True)
    assert [rec["manifest_entry_id"] for rec in result["records"]] == ["L1_AI_01", "L1_AI_02", "L1_AI_03"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py
```

Expected: FAIL because `m1_workload_resolver.py` does not exist.

- [ ] **Step 3: Implement resolver API**

Create `m1_workload_resolver.py` with these public functions:

- `load_json(path: Path) -> dict[str, Any]`: read a JSON object and raise `ValueError` on invalid JSON.
- `write_json(path: Path, value: Any) -> None`: write deterministic JSON with `indent=2` and sorted keys.
- `resolve_manifest_entries(manifest: dict[str, Any], registry: dict[str, Any], repo_root: Path, run_smoke: bool = True) -> dict[str, Any]`: return a dictionary with `records`, `gaps`, and `summary` keys.
- `main(argv: list[str] | None = None) -> int`: write Gate1 artifacts and return 0 when artifacts are produced.

Implementation requirements:

- Filter `priority == "P0"`.
- Match manifest entry to registry by `entry["run_args"]` first, then `benchmark_name`.
- Resolve real binary path relative to repo root.
- Reject final binary paths ending with `dispatch_ncu_capture.sh`.
- If binary missing and `build_command` exists, execute it with `subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=entry_timeout_seconds)`.
- Recheck binary exists and executable.
- Build `resolved_run_command` from `run_command_template`.
- Run smoke command with `timeout`.
- Emit `m1_workload_resolution_l1.json` and `m1_workload_resolution_gap_l1.json`.

- [ ] **Step 4: Run Gate1 tests**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py
```

Expected: PASS.

- [ ] **Step 5: Run Gate1 script**

Run:

```bash
python experiments/baseline_diagnosis/m1_workload_resolver.py
```

Expected:

- Writes `artifacts/a_line/l1/m1_workload_resolution_l1.json`.
- Writes `artifacts/a_line/l1/m1_workload_resolution_gap_l1.json`.
- Exits 0 even if some workloads are Gate1 gaps.

- [ ] **Step 6: Commit**

```bash
git add experiments/baseline_diagnosis/m1_workload_resolver.py experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py artifacts/a_line/l1/m1_workload_resolution_l1.json artifacts/a_line/l1/m1_workload_resolution_gap_l1.json
git commit -m "feat: implement M1 gate1 workload resolver"
```

### Task 3: Implement Gate2 NCU Capture Dispatcher

**Files:**
- Create: `experiments/baseline_diagnosis/m1_ncu_capture_dispatcher.py`
- Create: `experiments/baseline_diagnosis/tests/test_m1_ncu_capture_dispatcher.py`

- [ ] **Step 1: Add dispatcher tests**

Test command deduplication, selected metrics, and status classification:

```python
def test_gate2_deduplicates_by_resolved_run_command(tmp_path):
    from experiments.baseline_diagnosis import m1_ncu_capture_dispatcher as d

    gate1 = {"records": [
        {"manifest_entry_id": "A", "workload_id": "w", "kernel_or_case": "k1", "resolved_run_command": "/bin/true", "working_directory": str(tmp_path), "capture_timeout_seconds": 10},
        {"manifest_entry_id": "B", "workload_id": "w", "kernel_or_case": "k2", "resolved_run_command": "/bin/true", "working_directory": str(tmp_path), "capture_timeout_seconds": 10},
    ]}
    jobs = d.build_capture_jobs(gate1, selected_metrics=["smsp__inst_executed"], query_ref={"query_artifact_hash": "q", "resolution_table_hash": "r"})
    assert len(jobs) == 1
    assert jobs[0]["consuming_manifest_entry_ids"] == ["A", "B"]


def test_gate2_rejects_empty_selected_metrics():
    from experiments.baseline_diagnosis import m1_ncu_capture_dispatcher as d

    status = d.classify_metric_selection([])
    assert status["capture_status"] == "metric_selection_failed"
    assert status["gate3_eligible"] is False


def test_gate2_nonzero_exit_with_valid_csv_is_gate3_eligible(tmp_path):
    from experiments.baseline_diagnosis import m1_ncu_capture_dispatcher as d

    csv_path = tmp_path / "capture.csv"
    csv_path.write_text("ID,Kernel Name,Grid Size,Metric Name,Metric Value\n0,k,(1,1,1),smsp__inst_executed,1\n")
    status = d.classify_capture_result(exit_code=7, stdout="", stderr="", csv_path=csv_path, timed_out=False)
    assert status["capture_status"] == "capture_non_zero_exit_with_partial_csv"
    assert status["gate3_eligible"] is True
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_ncu_capture_dispatcher.py
```

Expected: FAIL because dispatcher module does not exist.

- [ ] **Step 3: Implement dispatcher**

Create `m1_ncu_capture_dispatcher.py` with these public functions:

- `build_capture_jobs(gate1_resolution: dict[str, Any], selected_metrics: list[str], query_ref: dict[str, str]) -> list[dict[str, Any]]`: group Gate1 records by `resolved_run_command` and return deterministic capture jobs.
- `classify_metric_selection(selected_metrics: list[str]) -> dict[str, Any]`: return `metric_selection_failed` when metrics are empty, otherwise return a capture-ready status.
- `classify_capture_result(exit_code: int, stdout: str, stderr: str, csv_path: Path, timed_out: bool) -> dict[str, Any]`: return the Gate2 capture status and `gate3_eligible` boolean.
- `main(argv: list[str] | None = None) -> int`: write Gate2 attempts / gap artifacts and return 0 when artifacts are produced.

Implementation requirements:

- Read Gate1 resolution.
- Build selected metrics from `ncu_metric_resolution_table_l1.json`, accepting only `available` and `rollup_resolved`.
- Exclude `launch_grid_size` from selected metrics.
- Deduplicate jobs by `resolved_run_command`.
- Generate deterministic `capture_job_id`.
- Support `--dry-run` for tests / no-NCU environments.
- Write per-job output directory under `experiments/baseline_diagnosis/results/m1_ncu/<capture_job_id>/`.
- Write `capture_command.json`, `selected_metrics.json`, `query_artifact_ref.json`.
- If not dry-run, execute `ncu --csv --target-processes all --metrics <selected_metrics_csv> -- <target_run_command>` with capture timeout.
- Classify statuses exactly as Gate2 spec.

- [ ] **Step 4: Run dispatcher tests**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_ncu_capture_dispatcher.py
```

Expected: PASS.

- [ ] **Step 5: Run dispatcher dry-run**

Run:

```bash
python experiments/baseline_diagnosis/m1_ncu_capture_dispatcher.py --dry-run
```

Expected:

- Writes `m1_ncu_capture_attempts_l1.json`.
- Does not run NCU.
- Attempts have deterministic command artifacts.

- [ ] **Step 6: Commit**

```bash
git add experiments/baseline_diagnosis/m1_ncu_capture_dispatcher.py experiments/baseline_diagnosis/tests/test_m1_ncu_capture_dispatcher.py
git commit -m "feat: implement M1 gate2 ncu capture dispatcher"
```

### Task 4: Implement Gate3 Measured Feature Extractor

**Files:**
- Create: `experiments/baseline_diagnosis/m1_measured_feature_extractor.py`
- Create: `experiments/baseline_diagnosis/tests/test_m1_measured_feature_extractor.py`

- [ ] **Step 1: Add Gate3 tests**

Add tests for strict 12D measured records, missing metrics, selected allowlist rejection, shared captures, and nonzero-exit provenance:

```python
def test_gate3_complete_csv_becomes_measured_record(tmp_path):
    from experiments.baseline_diagnosis import m1_measured_feature_extractor as e
    from experiments.baseline_diagnosis.shared_acquisition import PKA_CANONICAL_METRICS

    selected = [{"pka_feature_name": k, "canonical_metric": v, "actual_source_metric": v.replace(".sum", "").replace(".ratio", ""), "resolution_status": "rollup_resolved"} for k, v in PKA_CANONICAL_METRICS.items() if k != "num_thread_blocks"]
    metric_rows = {row["actual_source_metric"]: 1.0 for row in selected}
    metric_rows["Grid Size"] = "(2, 3, 1)"
    record = e.build_feature_record(
        manifest_entry={"id": "L1_MB_01", "dataset_level": "L1", "source_type": "local_microbench", "benchmark_name": "l1_bw_32f", "kernel_or_case": "l1_bw_32f"},
        invocation={"csv_invocation_id": "0", "kernel_name": "l1_bw_32f", "metric_map": metric_rows, "grid_size_raw": "(2, 3, 1)", "grid_size_normalized": 6},
        selected_metrics=selected,
        capture={"capture_job_id": "cap", "capture_status": "captured", "capture_csv_path": "capture.csv"},
    )
    assert record["feature_status"] == "complete_measured"
    assert record["feature_mode"] == "pka_m1_measured"
    assert len(record["features"]) == 12
    assert record["features"]["num_thread_blocks"]["value"] == 6


def test_gate3_missing_metric_routes_to_gap(tmp_path):
    from experiments.baseline_diagnosis import m1_measured_feature_extractor as e

    gap = e.make_gap_row(
        manifest_entry_id="L1_MB_01",
        capture_job_id="cap",
        kernel_or_case="l1_bw_32f",
        gap_reason="missing_canonical_metric",
        missing_features=["num_instructions"],
    )
    assert gap["failed_gate"] == "Gate3"
    assert gap["gap_reason"] == "missing_canonical_metric"
    assert gap["missing_features"] == ["num_instructions"]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_measured_feature_extractor.py
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement extractor**

Create `m1_measured_feature_extractor.py` with these public functions:

- `parse_ncu_csv(path: Path) -> list[dict[str, Any]]`: parse Nsight Compute CSV rows after locating the `ID` header.
- `build_feature_record(manifest_entry: dict[str, Any], invocation: dict[str, Any], selected_metrics: list[dict[str, Any]], capture: dict[str, Any]) -> dict[str, Any]`: build one complete `pka_m1_measured` record or raise a structured extractor error.
- `make_gap_row(manifest_entry_id: str, capture_job_id: str, kernel_or_case: str, gap_reason: str, missing_features: list[str]) -> dict[str, Any]`: build one Gate3 acquisition gap row.
- `run_gate3(attempts: dict[str, Any]) -> dict[str, Any]`: write feature table, gap table, feature audit, and join audit.
- `main(argv: list[str] | None = None) -> int`: run Gate3 using default artifact paths.

Implementation requirements:

- Only consume attempts with `gate3_eligible is True`.
- Parse NCU CSV preamble by locating `ID` header.
- Build invocation maps by CSV `ID` and kernel occurrence.
- Join consuming manifest entries by `kernel_or_case` and occurrence order.
- Build feature records only when all 12 features are measured.
- Record `capture_warning: non_zero_exit` when capture status is partial.
- Route every missing or invalid item to `pka_acquisition_gap_l1.json`.
- Emit `pka_feature_table_l1.json`, `pka_acquisition_gap_l1.json`, `pka_feature_audit_l1.json`, and `pka_join_audit_l1.json`.

- [ ] **Step 4: Run Gate3 tests**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_measured_feature_extractor.py
```

Expected: PASS.

- [ ] **Step 5: Run Gate3 script**

Run:

```bash
python experiments/baseline_diagnosis/m1_measured_feature_extractor.py
```

Expected: Exits 0, writes feature / gap / audit artifacts. It may write 0 measured rows if capture artifacts are blocked.

- [ ] **Step 6: Commit**

```bash
git add experiments/baseline_diagnosis/m1_measured_feature_extractor.py experiments/baseline_diagnosis/tests/test_m1_measured_feature_extractor.py
git commit -m "feat: implement M1 gate3 measured feature extractor"
```

### Task 5: Implement Gate4 Selector Eligibility and Repair Report

**Files:**
- Create: `experiments/baseline_diagnosis/m1_selector_eligibility.py`
- Create: `experiments/baseline_diagnosis/tests/test_m1_selector_eligibility.py`

- [ ] **Step 1: Add Gate4 tests**

Add tests for state transitions, timing policy, forbidden projection, and earliest failed gate:

```python
def test_gate4_blocks_when_measured_rows_less_than_three():
    from experiments.baseline_diagnosis import m1_selector_eligibility as g

    state = g.decide_selector_state(measured_rows=2, total_p0_entries=10, gap_rows=8, preflight_ok=True, timing_status="ok")
    assert state == "selector_blocked_insufficient_measured_records"


def test_gate4_ready_with_remaining_gaps():
    from experiments.baseline_diagnosis import m1_selector_eligibility as g

    state = g.decide_selector_state(measured_rows=4, total_p0_entries=10, gap_rows=6, preflight_ok=True, timing_status="ok")
    assert state == "selector_ready_with_remaining_gaps"


def test_gate4_mixed_timing_blocks():
    from experiments.baseline_diagnosis import m1_selector_eligibility as g

    timing = g.check_timing_units([
        {"record_id": "a", "timing": {"unit": "duration_ns", "value": 1}},
        {"record_id": "b", "timing": {"unit": "elapsed_cycles", "value": 2}},
    ])
    assert timing["status"] == "mixed_timing_unit"
    assert timing["gate5_allowed"] is False


def test_gate4_earliest_failed_gate_priority():
    from experiments.baseline_diagnosis import m1_selector_eligibility as g

    entry = g.classify_entry_status(gate1_status="blocked", gate2_status="not_attempted", gate3_status="not_attempted")
    assert entry["earliest_failed_gate"] == "Gate1"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_selector_eligibility.py
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement Gate4**

Create `m1_selector_eligibility.py` with these public functions:

- `decide_selector_state(measured_rows: int, total_p0_entries: int, gap_rows: int, preflight_ok: bool, timing_status: str) -> str`: return one of the Gate4 selector states.
- `check_timing_units(records: list[dict[str, Any]]) -> dict[str, Any]`: enforce member-count fallback, single timing unit, or mixed-unit block.
- `validate_selector_projection(records: list[dict[str, Any]]) -> dict[str, Any]`: validate allowed fields and 12D measured completeness.
- `classify_entry_status(gate1_status: str, gate2_status: str, gate3_status: str) -> dict[str, Any]`: compute `earliest_failed_gate`.
- `run_gate4() -> dict[str, Any]`: write eligibility, selector input projection, and backward repair reports.
- `main(argv: list[str] | None = None) -> int`: run Gate4 using default artifact paths.

Implementation requirements:

- Build `m1_selector_input_l1.json` only from complete `pka_m1_measured` records.
- Projection fields are exactly `record_id`, `kernel_invocation_id`, `features`, `feature_mode`, `weight_input`.
- Check `measured_rows >= 3`.
- Check complete 12D rows, feature status, feature mode, and forbidden fields.
- Enforce timing policy.
- Emit `m1_selector_eligibility_l1.json`.
- Always emit `m1_backward_repair_report_l1.json` and `.md`.
- Mark `gate5_allowed` according to Gate4 spec.

- [ ] **Step 4: Run Gate4 tests**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_selector_eligibility.py
```

Expected: PASS.

- [ ] **Step 5: Run Gate4 script**

Run:

```bash
python experiments/baseline_diagnosis/m1_selector_eligibility.py
```

Expected:

- Writes `m1_selector_eligibility_l1.json`.
- Writes `m1_selector_input_l1.json` if measured rows exist.
- Writes `m1_backward_repair_report_l1.json` and `.md`.
- If measured rows < 3, exits 0 but `gate5_allowed` is false.

- [ ] **Step 6: Commit**

```bash
git add experiments/baseline_diagnosis/m1_selector_eligibility.py experiments/baseline_diagnosis/tests/test_m1_selector_eligibility.py
git commit -m "feat: implement M1 gate4 selector eligibility"
```

### Task 6: Extract Shared PKA Selector Core

**Files:**
- Create: `experiments/baseline_diagnosis/pka_selector_core.py`
- Modify: `experiments/baseline_diagnosis/pka_m0_pipeline.py`
- Modify: `experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py`

- [ ] **Step 1: Add selector core import tests**

Add tests that assert M0 can call the shared core and still emits the same M0 modes:

```python
def test_m0_pipeline_uses_shared_selector_core():
    import inspect
    import pka_m0_pipeline
    import pka_selector_core

    assert hasattr(pka_selector_core, "run_pca")
    source = inspect.getsource(pka_m0_pipeline)
    assert "pka_selector_core" in source
```

- [ ] **Step 2: Run M0 tests before refactor**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py
```

Expected: PASS before refactor.

- [ ] **Step 3: Create `pka_selector_core.py`**

Move pure algorithm functions from `pka_m0_pipeline.py` into `pka_selector_core.py`:

The shared module must expose:

- `FEATURE_ORDER`: the same 12 feature names used by M0 and M1.
- `COUNT_FEATURES` and `RATIO_FEATURES`: preprocessing field groups.
- `canonical_bytes(value: Any) -> bytes`: deterministic JSON bytes.
- `sha256_json(value: Any) -> str`: SHA-256 hash of canonical JSON.
- `with_replay_hash(value: dict[str, Any]) -> dict[str, Any]`: add deterministic replay hash.
- `validate_selector_records(records: list[dict[str, Any]], feature_mode: str, allowed_record_fields: set[str]) -> dict[str, Any]`: validate mode, fields, feature completeness, and numeric values.
- `raw_matrix(records: list[dict[str, Any]]) -> np.ndarray`: build fixed-order numeric matrix.
- `preprocess(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]`: apply log1p / clip / z-score.
- `run_pca(records: list[dict[str, Any]], normalized: np.ndarray, input_hash: str, mode: str, artifact_name: str, components_count: int = 3) -> dict[str, Any]`: run numpy SVD PCA.
- `run_kmeans(pca: dict[str, Any], mode: str, artifact_name: str, max_iters: int = 300) -> dict[str, Any]`: run deterministic farthest-first k-means.
- `label_clusters(records: list[dict[str, Any]], clusters: dict[str, Any]) -> dict[str, Any]`: generate debug labels from feature means.
- `select_anchors(clusters: dict[str, Any], labels: dict[str, Any], audit: dict[str, Any], mode: str, artifact_name: str, normalization_config: dict[str, Any], input_hash: str) -> dict[str, Any]`: choose nearest-centroid real records.
- `evaluate_structural(records: list[dict[str, Any]], normalized: np.ndarray, input_hash: str, pca: dict[str, Any], clusters: dict[str, Any], anchors: dict[str, Any], weight_mode: str, timing_unit: str | None) -> dict[str, Any]`: compute structural-only metrics.

Preserve M0 algorithm semantics exactly.

- [ ] **Step 4: Update M0 wrapper**

Modify `pka_m0_pipeline.py` so it:

- Loads fixture.
- Calls `pka_selector_core`.
- Writes the same four `pka_m0_*` artifacts.
- Preserves M0 artifact names, mode, feature mode, selector name, replay hashes, tests.

- [ ] **Step 5: Run M0 tests**

Run:

```bash
python experiments/baseline_diagnosis/pka_m0_pipeline.py
pytest -q experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/baseline_diagnosis/pka_selector_core.py experiments/baseline_diagnosis/pka_m0_pipeline.py experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py
git commit -m "refactor: share PKA selector core"
```

### Task 7: Implement Gate5 M1 Selector

**Files:**
- Create: `experiments/baseline_diagnosis/pka_m1_selector.py`
- Create: `experiments/baseline_diagnosis/tests/test_m1_selector.py`

- [ ] **Step 1: Add Gate5 tests**

Add tests for Gate4 block, formal artifact names, no M0 overwrite, structural evaluation, and weight modes:

```python
def test_gate5_aborts_when_gate4_disallows(tmp_path):
    from experiments.baseline_diagnosis import pka_m1_selector as s

    eligibility = {"gate5_allowed": False, "selector_eligibility_state": "selector_blocked_insufficient_measured_records"}
    assert s.run_selector(eligibility, {"records": []}, out_dir=tmp_path)["status"] == "aborted_gate4_blocked"


def test_gate5_writes_formal_m1_artifacts(tmp_path):
    from experiments.baseline_diagnosis import pka_m1_selector as s

    records = []
    for i in range(3):
        records.append({
            "record_id": f"r{i}",
            "kernel_invocation_id": f"k#{i}",
            "feature_mode": "pka_m1_measured",
            "features": {name: {"value": float(i + 1), "status": "measured"} for name in s.FEATURE_ORDER},
            "weight_input": {"weight_mode": "member_count_fallback", "weight": 1},
        })
    eligibility = {"gate5_allowed": True, "selector_eligibility_state": "selector_ready", "timing_check": {"weight_mode": "member_count_fallback", "timing_unit": None}}
    result = s.run_selector(eligibility, {"records": records}, out_dir=tmp_path)
    assert result["status"] == "completed"
    assert (tmp_path / "pka_pca_projection_l1.json").exists()
    assert (tmp_path / "pka_kmeans_clusters_l1.json").exists()
    assert (tmp_path / "representative_anchor_table_l1.json").exists()
    assert (tmp_path / "pka_compression_evaluation_l1.json").exists()
    assert not list(tmp_path.glob("pka_m0_*.json"))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_selector.py
```

Expected: FAIL because `pka_m1_selector.py` does not exist.

- [ ] **Step 3: Implement M1 selector wrapper**

Create `pka_m1_selector.py` with these public functions:

- `load_gate4_inputs(eligibility_path: Path, selector_input_path: Path) -> tuple[dict[str, Any], dict[str, Any]]`: load Gate4 eligibility and projection artifacts.
- `run_selector(eligibility: dict[str, Any], selector_input: dict[str, Any], out_dir: Path) -> dict[str, Any]`: run shared selector core and write formal M1 artifacts.
- `main(argv: list[str] | None = None) -> int`: run Gate5 using default artifact paths.

Implementation requirements:

- Abort unless Gate4 allows Gate5.
- Validate selector input projection contains no forbidden metadata.
- Validate at least 3 records.
- Validate `feature_mode == pka_m1_measured`.
- Call `pka_selector_core` for preprocessing, PCA, k-means, labels, anchors, evaluation.
- Write formal M1 artifacts:
  - `pka_pca_projection_l1.json`
  - `pka_kmeans_clusters_l1.json`
  - `representative_anchor_table_l1.json`
  - `pka_compression_evaluation_l1.json`
- Do not write `pka_m0_*`.
- Use Gate4 `weight_mode` for weighted coverage.

- [ ] **Step 4: Run Gate5 tests**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_selector.py
```

Expected: PASS.

- [ ] **Step 5: Run Gate5 script**

Run:

```bash
python experiments/baseline_diagnosis/pka_m1_selector.py
```

Expected:

- If Gate4 blocks, exits nonzero or exits 0 with explicit `aborted_gate4_blocked` artifact according to implementation decision, but must not write formal selector outputs.
- If Gate4 allows, writes all four formal M1 artifacts.

- [ ] **Step 6: Commit**

```bash
git add experiments/baseline_diagnosis/pka_m1_selector.py experiments/baseline_diagnosis/tests/test_m1_selector.py
git commit -m "feat: implement M1 gate5 formal selector"
```

### Task 8: Add M1 End-to-End Orchestration and Stop Gates

**Files:**
- Create: `experiments/baseline_diagnosis/run_m1_measured_loop.py`
- Modify: `experiments/baseline_diagnosis/test_l1_regression.py`

- [ ] **Step 1: Add orchestration regression tests**

Add tests that validate the two allowed final outcomes:

```python
def test_m1_completion_allows_blocked_with_repair_report():
    from experiments.baseline_diagnosis import run_m1_measured_loop as loop

    status = loop.classify_completion({
        "gate5_allowed": False,
        "measured_rows": 2,
        "backward_repair_report_exists": True,
        "per_entry_earliest_gate_complete": True,
    })
    assert status == "blocked_on_acquisition_with_repair_report"


def test_m1_completion_rejects_less_than_three_without_report():
    from experiments.baseline_diagnosis import run_m1_measured_loop as loop

    status = loop.classify_completion({
        "gate5_allowed": False,
        "measured_rows": 2,
        "backward_repair_report_exists": False,
        "per_entry_earliest_gate_complete": False,
    })
    assert status == "stop_fail_missing_backward_repair_report"
```

- [ ] **Step 2: Implement orchestrator**

Create `run_m1_measured_loop.py` with these public functions:

- `run_all(dry_run_capture: bool = False) -> dict[str, Any]`: execute Gate1, Gate2, Gate3, Gate4, and conditionally Gate5.
- `classify_completion(summary: dict[str, Any]) -> str`: return `completed_gate5_formal_selector`, `blocked_on_acquisition_with_repair_report`, or a stop/fail status.
- `main(argv: list[str] | None = None) -> int`: run the orchestrator from CLI.

Execution order:

1. `m1_workload_resolver.py`
2. `m1_ncu_capture_dispatcher.py`
3. `m1_measured_feature_extractor.py`
4. `m1_selector_eligibility.py`
5. `pka_m1_selector.py` only when Gate4 allows

Stop rules:

- If Gate4 blocks due to `<3`, require repair report and return blocked-on-acquisition status.
- If Gate4 allows, run Gate5 and require all formal artifacts.
- If mixed timing or invalid feature table, return stop/fail.

- [ ] **Step 3: Run orchestration tests**

Run:

```bash
pytest -q experiments/baseline_diagnosis/test_l1_regression.py -k "m1_completion"
```

Expected: PASS.

- [ ] **Step 4: Run dry-run orchestration**

Run:

```bash
python experiments/baseline_diagnosis/run_m1_measured_loop.py --dry-run-capture
```

Expected: Completes Gate1-Gate4 and either blocks with repair report or, if test fixtures make Gate5 possible, runs Gate5.

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline_diagnosis/run_m1_measured_loop.py experiments/baseline_diagnosis/test_l1_regression.py
git commit -m "feat: orchestrate M1 measured loop gates"
```

### Task 9: Run Full Verification and Finalize

**Files:**
- No new files unless tests reveal necessary fixes.

- [ ] **Step 1: Run unit tests**

```bash
pytest -q experiments/baseline_diagnosis/tests/test_m1_workload_resolver.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_ncu_capture_dispatcher.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_measured_feature_extractor.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_selector_eligibility.py
pytest -q experiments/baseline_diagnosis/tests/test_m1_selector.py
```

Expected: all PASS.

- [ ] **Step 2: Run M0 regression**

```bash
python experiments/baseline_diagnosis/pka_m0_pipeline.py
pytest -q experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py
```

Expected: PASS and M0 artifacts remain `pka_m0_*`.

- [ ] **Step 3: Run L1 regression**

```bash
pytest -q experiments/baseline_diagnosis/test_l1_regression.py
```

Expected: PASS.

- [ ] **Step 4: Run M1 loop**

```bash
python experiments/baseline_diagnosis/run_m1_measured_loop.py
```

Expected one of:

- `completed_gate5_formal_selector` with four M1 formal artifacts.
- `blocked_on_acquisition_with_repair_report` with Gate1-Gate4 artifacts and complete repair report.

- [ ] **Step 5: Inspect artifact isolation**

Run:

```bash
find artifacts/a_line/l1 -maxdepth 1 -type f | sort | rg 'pka_m0|pka_pca|pka_kmeans|representative_anchor|pka_compression|m1_'
```

Expected:

- M0 artifacts exist only under `pka_m0_*`.
- M1 artifacts, if Gate5 ran, use formal non-M0 paths.

- [ ] **Step 6: Commit final fixes**

If verification required fixes:

```bash
git add <changed-files>
git commit -m "fix: complete M1 measured loop verification"
```

If no fixes were needed, do not create an empty commit.

## RLCR Stop-Hook Checklist

The RLCR must STOP/FAIL if any of these are true:

- Not all L1 P0 entries have Gate1 outcome.
- Gate2 ignores a Gate1 resolved record.
- Gate3 ignores a Gate2 eligible capture job.
- `pka_feature_table_l1.json` contains incomplete 12D rows.
- `measured_rows < 3` and no complete `m1_backward_repair_report_l1.json`.
- `measured_rows < 3` and Gate5 artifacts are written as if successful.
- Gate5 reads full feature table metadata instead of Gate4 selector projection.
- Gate5 writes or overwrites `pka_m0_*`.
- M0 tests fail after selector core extraction.
- Evaluation claims simulator accuracy or measured speedup.

## Notes for RLCR

- Prefer dry-run Gate2 tests for deterministic CI behavior.
- Real NCU capture may be environment-dependent. A correct blocked-on-acquisition result is acceptable only if Gate4 repair report is complete.
- Do not add B-line consumption to this plan.
- Do not weaken measured-only semantics to get Gate5 running.
- Do not replace strict Gate4 timing policy with weighted coverage fallback for mixed units.
