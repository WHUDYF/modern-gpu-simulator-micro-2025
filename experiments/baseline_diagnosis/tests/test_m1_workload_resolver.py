from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m1_workload_resolver as resolver


def test_format_command_preserves_args_placeholder_argv_boundaries(tmp_path):
    binary = tmp_path / "runner"

    command = resolver._format_command(
        ["{binary_path}", "{args}", "--tail"],
        binary,
        ["--foo", "bar baz"],
    )

    assert command == [str(binary), "--foo", "bar baz", "--tail"]


def test_gate1_resolves_each_p0_entry(monkeypatch, tmp_path):
    manifest = {
        "entries": [
            {"id": "L1_A", "priority": "P0", "benchmark_name": "ok", "kernel_or_case": "ok", "source_type": "local_microbench"},
            {"id": "L1_B", "priority": "P0", "benchmark_name": "missing", "kernel_or_case": "missing", "source_type": "local_microbench"},
        ]
    }
    registry = [
        {
            "workload_id": "ok",
            "binary_path": sys.executable,
            "build_command": None,
            "run_args": [],
            "run_command_template": ["{binary_path}", "-c", "print('ok')"],
            "working_directory": str(tmp_path),
            "smoke_timeout_seconds": 5,
            "capture_timeout_seconds": 10,
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "registry.json"
    manifest_path.write_text(json.dumps(manifest))
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setattr(resolver, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(resolver, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(resolver, "SMOKE_DIR", tmp_path / "smoke")
    resolved, gaps = resolver.resolve()
    assert [row["manifest_entry_id"] for row in resolved] == ["L1_A"]
    assert [row["manifest_entry_id"] for row in gaps] == ["L1_B"]
    assert resolved[0]["resolved_binary_path"] != "experiments/baseline_diagnosis/dispatch_ncu_capture.sh"
    assert resolved[0]["smoke_run"]["status"] == "passed"


def test_gate1_blocks_expected_output_regex_mismatch(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "registry.json"
    manifest_path.write_text(json.dumps({"entries": [
        {"id": "L1_A", "priority": "P0", "benchmark_name": "ok", "kernel_or_case": "ok", "source_type": "local_microbench"},
    ]}))
    registry_path.write_text(json.dumps([
        {
            "workload_id": "ok",
            "binary_path": sys.executable,
            "build_command": None,
            "run_args": [],
            "smoke_args": [],
            "expected_output_regex": "expected-token",
            "run_command_template": ["{binary_path}", "-c", "print('different')"],
            "working_directory": str(tmp_path),
            "smoke_timeout_seconds": 5,
            "capture_timeout_seconds": 10,
        }
    ]))
    monkeypatch.setattr(resolver, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(resolver, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(resolver, "SMOKE_DIR", tmp_path / "smoke")
    resolved, gaps = resolver.resolve()
    assert not resolved
    assert gaps[0]["gap_reason"] == "smoke_expected_output_regex_mismatch"
