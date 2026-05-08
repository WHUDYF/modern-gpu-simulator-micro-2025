from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m1_ncu_capture_dispatcher as dispatcher
from shared_acquisition import metric_available_in_query, selected_metric_records


def test_gate2_dedups_commands_and_never_uses_set_full(monkeypatch, tmp_path):
    resolution_path = tmp_path / "resolution.json"
    resolution_path.write_text(json.dumps([
        {
            "manifest_entry_id": "L1_A",
            "resolution_status": "resolved",
            "workload_id": "w",
            "kernel_or_case": "ka",
            "resolved_run_command": [sys.executable, "-c", "print(1)"],
            "working_directory": str(tmp_path),
            "capture_timeout_seconds": 5,
        },
        {
            "manifest_entry_id": "L1_B",
            "resolution_status": "resolved",
            "workload_id": "w",
            "kernel_or_case": "kb",
            "resolved_run_command": [sys.executable, "-c", "print(1)"],
            "working_directory": str(tmp_path),
            "capture_timeout_seconds": 5,
        },
    ]))
    monkeypatch.setattr(dispatcher, "RESOLUTION_PATH", resolution_path)
    monkeypatch.setattr(dispatcher, "ATTEMPTS_PATH", tmp_path / "attempts.json")
    monkeypatch.setattr(dispatcher, "GAP_PATH", tmp_path / "gaps.json")
    monkeypatch.setattr(dispatcher, "QUERY_PATH", tmp_path / "query.json")
    monkeypatch.setattr(dispatcher, "RESOLUTION_TABLE_PATH", tmp_path / "resolution_table.json")
    monkeypatch.setattr(dispatcher, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(dispatcher, "_write_query_artifacts", lambda: selected_metric_records())
    attempts, gaps = dispatcher.dispatch(dry_run=True)
    assert len(attempts) == 1
    assert attempts[0]["consuming_manifest_entry_ids"] == ["L1_A", "L1_B"]
    assert "--set" not in attempts[0]["ncu_capture_command"]
    assert "--metrics" in attempts[0]["ncu_capture_command"]
    assert "launch_grid_size" not in ",".join(attempts[0]["selected_metrics"])
    assert "gpu__time_duration.sum" in attempts[0]["selected_metrics"]
    assert "sm__cycles_elapsed.sum" in attempts[0]["selected_metrics"]
    assert attempts[0]["gate3_eligible"] is False
    assert gaps[0]["capture_status"] == "dry_run_capture_skipped"
    attempts_doc = json.loads((tmp_path / "attempts.json").read_text())
    assert attempts_doc["artifact_name"] == "m1_ncu_capture_attempts_l1"
    assert attempts_doc["summary"]["capture_job_count"] == 1
    command_doc = json.loads((tmp_path / "results" / attempts[0]["capture_job_id"] / "capture_command.json").read_text())
    assert command_doc["target_run_command"] == [sys.executable, "-c", "print(1)"]
    assert command_doc["selected_metrics"]
    env_doc = json.loads((tmp_path / "results" / attempts[0]["capture_job_id"] / "capture_env_manifest.json").read_text())
    for key in ("gpu_name", "compute_capability", "driver_version", "cuda_version", "nsight_compute_version", "environment_signature", "capture_timestamp", "target_run_command", "ncu_capture_command", "selected_metrics", "output_csv_path"):
        assert key in env_doc


def test_gate2_rejects_nonempty_malformed_csv(tmp_path):
    csv_path = tmp_path / "capture.csv"
    csv_path.write_text("not,ncu,csv\n")
    status, eligible, reason = dispatcher._classify(0, "", csv_path)
    assert status == "malformed_ncu_csv"
    assert eligible is False
    assert reason == "missing_ncu_csv_header"


def test_gate2_timeout_with_valid_csv_remains_gate3_eligible(tmp_path):
    csv_path = tmp_path / "capture.csv"
    csv_path.write_text("==PROF==\nID,Kernel Name,Grid Size,Metric Name,Metric Value\n")
    status, eligible, reason = dispatcher._classify(None, "", csv_path, timed_out=True)
    assert status == "ncu_capture_timeout"
    assert eligible is True
    assert reason == "timeout_with_partial_csv"


def test_gate2_metric_query_matching_requires_exact_metric_name():
    query_text = "smsp__inst_executed.sum.per_cycle_active\nsmsp__inst_executed_op_global_ld.sum\n"
    assert not metric_available_in_query("smsp__inst_executed.sum", query_text)
    assert metric_available_in_query("smsp__inst_executed_op_global_ld.sum", query_text)


def test_gate2_timeout_with_malformed_csv_stays_timeout(tmp_path):
    csv_path = tmp_path / "capture.csv"
    csv_path.write_text("garbage\n")
    status, eligible, reason = dispatcher._classify(None, "", csv_path, timed_out=True)
    assert status == "ncu_capture_timeout"
    assert eligible is False
    assert reason == "timeout"


def test_gate2_classifies_permission_blocker(tmp_path):
    csv_path = tmp_path / "capture.csv"
    status, eligible, reason = dispatcher._classify(1, "ERR_NVGPUCTRPERM", csv_path)
    assert status == "permission_blocked"
    assert eligible is False
    assert reason == "ncu_permission_blocked"


def test_gate2_blocks_empty_metric_resolution(monkeypatch, tmp_path):
    resolution_path = tmp_path / "resolution.json"
    resolution_path.write_text(json.dumps([
        {
            "manifest_entry_id": "L1_A",
            "resolution_status": "resolved",
            "workload_id": "w",
            "kernel_or_case": "ka",
            "resolved_run_command": [sys.executable, "-c", "print(1)"],
            "working_directory": str(tmp_path),
            "capture_timeout_seconds": 5,
        }
    ]))
    empty_metric_rows = [
        {**row, "resolution_status": "unsupported", "selected_for_ncu_metrics": False}
        for row in selected_metric_records()
    ]
    monkeypatch.setattr(dispatcher, "RESOLUTION_PATH", resolution_path)
    monkeypatch.setattr(dispatcher, "ATTEMPTS_PATH", tmp_path / "attempts.json")
    monkeypatch.setattr(dispatcher, "GAP_PATH", tmp_path / "gaps.json")
    monkeypatch.setattr(dispatcher, "QUERY_PATH", tmp_path / "query.json")
    monkeypatch.setattr(dispatcher, "RESOLUTION_TABLE_PATH", tmp_path / "resolution_table.json")
    monkeypatch.setattr(dispatcher, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(dispatcher, "_write_query_artifacts", lambda: empty_metric_rows)
    attempts, gaps = dispatcher.dispatch(dry_run=False)
    assert attempts[0]["capture_status"] == "metric_resolution_blocked"
    assert attempts[0]["gate3_eligible"] is False
    assert gaps[0]["gap_reason"] == "selected_metrics_empty"


def test_gate2_dry_run_skips_even_when_metric_resolution_is_empty(monkeypatch, tmp_path):
    resolution_path = tmp_path / "resolution.json"
    resolution_path.write_text(json.dumps([
        {
            "manifest_entry_id": "L1_A",
            "resolution_status": "resolved",
            "workload_id": "w",
            "kernel_or_case": "ka",
            "resolved_run_command": [sys.executable, "-c", "print(1)"],
            "working_directory": str(tmp_path),
            "capture_timeout_seconds": 5,
        }
    ]))
    empty_metric_rows = [
        {**row, "resolution_status": "unsupported", "selected_for_ncu_metrics": False}
        for row in selected_metric_records()
    ]
    monkeypatch.setattr(dispatcher, "RESOLUTION_PATH", resolution_path)
    monkeypatch.setattr(dispatcher, "ATTEMPTS_PATH", tmp_path / "attempts.json")
    monkeypatch.setattr(dispatcher, "GAP_PATH", tmp_path / "gaps.json")
    monkeypatch.setattr(dispatcher, "QUERY_PATH", tmp_path / "query.json")
    monkeypatch.setattr(dispatcher, "RESOLUTION_TABLE_PATH", tmp_path / "resolution_table.json")
    monkeypatch.setattr(dispatcher, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(dispatcher, "_write_query_artifacts", lambda: empty_metric_rows)
    attempts, gaps = dispatcher.dispatch(dry_run=True)
    assert attempts[0]["capture_status"] == "dry_run_capture_skipped"
    assert attempts[0]["selected_metrics"] == []
    assert attempts[0]["gate3_eligible"] is False
    assert gaps[0]["gap_reason"] == "dry_run"
