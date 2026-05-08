from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m1_ncu_capture_dispatcher as dispatcher
from shared_acquisition import selected_metric_records


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
    assert attempts[0]["gate3_eligible"] is False
    assert gaps[0]["capture_status"] == "dry_run_capture_skipped"


def test_gate2_rejects_nonempty_malformed_csv(tmp_path):
    csv_path = tmp_path / "capture.csv"
    csv_path.write_text("not,ncu,csv\n")
    status, eligible, reason = dispatcher._classify(0, "", csv_path)
    assert status == "malformed_ncu_csv"
    assert eligible is False
    assert reason == "missing_ncu_csv_header"


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
    attempts, gaps = dispatcher.dispatch(dry_run=True)
    assert attempts[0]["capture_status"] == "metric_resolution_blocked"
    assert attempts[0]["gate3_eligible"] is False
    assert gaps[0]["gap_reason"] == "selected_metrics_empty"
