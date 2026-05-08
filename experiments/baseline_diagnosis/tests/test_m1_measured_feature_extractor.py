from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m1_measured_feature_extractor as extractor
from shared_acquisition import FEATURE_ORDER, selected_metric_records


def _write_fixture_csv(path: Path, kernel_name: str = "my_kernel") -> None:
    rows = ["==PROF==", "ID,Kernel Name,Grid Size,Metric Name,Metric Value"]
    for metric in selected_metric_records():
        if not metric["selected_for_ncu_metrics"]:
            continue
        rows.append(f"0,{kernel_name},(2, 3, 1),{metric['actual_source_metric']},1")
    rows.append(f"0,{kernel_name},(2, 3, 1),gpu__time_duration.sum,123")
    path.write_text("\n".join(rows) + "\n")


def _write_env(path: Path) -> None:
    path.write_text(json.dumps({
        "gpu_name": "test-gpu",
        "compute_capability": "9.0",
        "driver_version": "test-driver",
        "cuda_version": "test-cuda",
        "nsight_compute_version": "test-ncu",
        "environment_signature": {"test": True},
        "capture_timestamp": "2026-05-08T00:00:00Z",
        "target_run_command": ["app"],
        "ncu_capture_command": ["ncu"],
        "selected_metrics": ["metric"],
        "output_csv_path": "capture.csv",
    }))


def test_gate3_extracts_complete_measured_row(monkeypatch, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    csv_path = job_dir / "capture.csv"
    env_path = job_dir / "capture_env_manifest.json"
    _write_fixture_csv(csv_path)
    _write_env(env_path)
    (job_dir / "selected_metrics.json").write_text(json.dumps(selected_metric_records()))
    attempts_path = tmp_path / "attempts.json"
    attempts_path.write_text(json.dumps([
        {
            "capture_job_id": "job",
            "gate3_eligible": True,
            "capture_status": "captured",
            "capture_exit_code": 0,
            "capture_stderr_path": "stderr.log",
            "capture_csv_path": str(csv_path),
            "environment_manifest_path": str(env_path),
            "consuming_manifest_entry_ids": ["L1_A"],
            "consuming_kernel_or_cases": ["my_kernel"],
            "consuming_manifest_entries": [
                {"manifest_entry_id": "L1_A", "source_type": "local_microbench", "benchmark_name": "bench", "kernel_or_case": "my_kernel"}
            ],
        }
    ]))
    monkeypatch.setattr(extractor, "ATTEMPTS_PATH", attempts_path)
    monkeypatch.setattr(extractor, "FEATURE_TABLE_PATH", tmp_path / "features.json")
    monkeypatch.setattr(extractor, "GAP_PATH", tmp_path / "gaps.json")
    monkeypatch.setattr(extractor, "FEATURE_AUDIT_PATH", tmp_path / "audit.json")
    monkeypatch.setattr(extractor, "JOIN_AUDIT_PATH", tmp_path / "join.json")
    features, gaps = extractor.extract()
    assert len(features) == 1
    assert not gaps
    assert features[0]["feature_mode"] == "pka_m1_measured"
    assert features[0]["dataset_level"] == "L1"
    assert features[0]["feature_status"] == "complete_measured"
    assert features[0]["capture_status"] == "captured"
    assert features[0]["duration_ns"] == 123.0
    assert features[0]["timing_basis"] == "duration_ns"
    assert set(features[0]["features"]) == set(FEATURE_ORDER)
    assert features[0]["features"]["num_thread_blocks"]["value"] == 6.0
    audit = json.loads((tmp_path / "audit.json").read_text())
    for key in (
        "total_consuming_manifest_entries",
        "gate3_eligible_capture_jobs",
        "parsed_capture_jobs",
        "measured_record_count",
        "gap_record_count",
        "complete_12d_count",
        "incomplete_12d_count",
        "feature_missing_counts",
        "gap_reason_counts",
    ):
        assert key in audit["summary"]
    for key in (
        "manifest_entry_id",
        "kernel_or_case",
        "capture_job_id",
        "kernel_invocation_id",
        "feature_status",
        "measured_features",
        "missing_features",
        "gap_reason",
    ):
        assert key in audit["entries"][0]
    assert audit["entries"][0]["feature_status"] == "complete_measured"
    assert set(audit["entries"][0]["measured_features"]) == set(FEATURE_ORDER)


def test_gate3_occurrence_join_and_nonzero_exit_provenance(monkeypatch, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    csv_path = job_dir / "capture.csv"
    env_path = job_dir / "capture_env_manifest.json"
    _write_env(env_path)
    rows = ["ID,Kernel Name,Grid Size,Metric Name,Metric Value"]
    for inv_id in ("0", "1"):
        for metric in selected_metric_records():
            if metric["selected_for_ncu_metrics"]:
                rows.append(f"{inv_id},repeat_kernel,(1, 1, 1),{metric['actual_source_metric']},2")
    csv_path.write_text("\n".join(rows) + "\n")
    (job_dir / "selected_metrics.json").write_text(json.dumps(selected_metric_records()))
    attempts_path = tmp_path / "attempts.json"
    attempts_path.write_text(json.dumps([
        {
            "capture_job_id": "job",
            "gate3_eligible": True,
            "capture_status": "capture_non_zero_exit_with_partial_csv",
            "capture_exit_code": 9,
            "capture_stderr_path": "stderr.log",
            "capture_csv_path": str(csv_path),
            "environment_manifest_path": str(env_path),
            "consuming_manifest_entry_ids": ["L1_A", "L1_B"],
            "consuming_kernel_or_cases": ["repeat_kernel", "repeat_kernel"],
            "consuming_manifest_entries": [
                {"manifest_entry_id": "L1_A", "source_type": "local_microbench", "benchmark_name": "bench", "kernel_or_case": "repeat_kernel"},
                {"manifest_entry_id": "L1_B", "source_type": "local_microbench", "benchmark_name": "bench", "kernel_or_case": "repeat_kernel"},
            ],
        }
    ]))
    monkeypatch.setattr(extractor, "ATTEMPTS_PATH", attempts_path)
    monkeypatch.setattr(extractor, "FEATURE_TABLE_PATH", tmp_path / "features.json")
    monkeypatch.setattr(extractor, "GAP_PATH", tmp_path / "gaps.json")
    monkeypatch.setattr(extractor, "FEATURE_AUDIT_PATH", tmp_path / "audit.json")
    monkeypatch.setattr(extractor, "JOIN_AUDIT_PATH", tmp_path / "join.json")
    features, gaps = extractor.extract()
    assert len(features) == 2
    assert not gaps
    assert all(row["capture_warning"] == "non_zero_exit" for row in features)
    assert all(row["capture_exit_code"] == 9 for row in features)
    join_rows = json.loads((tmp_path / "join.json").read_text())["entries"]
    assert [row["join_status"] for row in join_rows] == ["matched", "matched"]


def test_gate3_reports_empty_kernel_name(monkeypatch, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    csv_path = job_dir / "capture.csv"
    env_path = job_dir / "capture_env_manifest.json"
    _write_fixture_csv(csv_path, kernel_name="")
    _write_env(env_path)
    (job_dir / "selected_metrics.json").write_text(json.dumps(selected_metric_records()))
    attempts_path = tmp_path / "attempts.json"
    attempts_path.write_text(json.dumps([
        {
            "capture_job_id": "job",
            "gate3_eligible": True,
            "capture_status": "captured",
            "capture_csv_path": str(csv_path),
            "environment_manifest_path": str(env_path),
            "consuming_manifest_entry_ids": ["L1_A"],
            "consuming_kernel_or_cases": ["my_kernel"],
        }
    ]))
    monkeypatch.setattr(extractor, "ATTEMPTS_PATH", attempts_path)
    monkeypatch.setattr(extractor, "FEATURE_TABLE_PATH", tmp_path / "features.json")
    monkeypatch.setattr(extractor, "GAP_PATH", tmp_path / "gaps.json")
    monkeypatch.setattr(extractor, "FEATURE_AUDIT_PATH", tmp_path / "audit.json")
    monkeypatch.setattr(extractor, "JOIN_AUDIT_PATH", tmp_path / "join.json")
    features, gaps = extractor.extract()
    assert not features
    assert gaps[0]["gap_reason"] == "empty_kernel_name"


def test_gate3_reports_occurrence_mismatch(monkeypatch, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    csv_path = job_dir / "capture.csv"
    env_path = job_dir / "capture_env_manifest.json"
    _write_fixture_csv(csv_path, kernel_name="repeat_kernel")
    _write_env(env_path)
    (job_dir / "selected_metrics.json").write_text(json.dumps(selected_metric_records()))
    attempts_path = tmp_path / "attempts.json"
    attempts_path.write_text(json.dumps([
        {
            "capture_job_id": "job",
            "gate3_eligible": True,
            "capture_status": "captured",
            "capture_csv_path": str(csv_path),
            "environment_manifest_path": str(env_path),
            "consuming_manifest_entry_ids": ["L1_A", "L1_B"],
            "consuming_kernel_or_cases": ["repeat_kernel", "repeat_kernel"],
        }
    ]))
    monkeypatch.setattr(extractor, "ATTEMPTS_PATH", attempts_path)
    monkeypatch.setattr(extractor, "FEATURE_TABLE_PATH", tmp_path / "features.json")
    monkeypatch.setattr(extractor, "GAP_PATH", tmp_path / "gaps.json")
    monkeypatch.setattr(extractor, "FEATURE_AUDIT_PATH", tmp_path / "audit.json")
    monkeypatch.setattr(extractor, "JOIN_AUDIT_PATH", tmp_path / "join.json")
    features, gaps = extractor.extract()
    assert len(features) == 1
    assert gaps[0]["gap_reason"] == "occurrence_mismatch"


def test_gate3_rejects_missing_environment_manifest(monkeypatch, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    csv_path = job_dir / "capture.csv"
    _write_fixture_csv(csv_path)
    (job_dir / "selected_metrics.json").write_text(json.dumps(selected_metric_records()))
    attempts_path = tmp_path / "attempts.json"
    attempts_path.write_text(json.dumps({"attempts": [
        {
            "capture_job_id": "job",
            "gate3_eligible": True,
            "capture_status": "captured",
            "capture_csv_path": str(csv_path),
            "environment_manifest_path": str(job_dir / "missing_env.json"),
            "consuming_manifest_entry_ids": ["L1_A"],
            "consuming_kernel_or_cases": ["my_kernel"],
        }
    ]}))
    monkeypatch.setattr(extractor, "ATTEMPTS_PATH", attempts_path)
    monkeypatch.setattr(extractor, "FEATURE_TABLE_PATH", tmp_path / "features.json")
    monkeypatch.setattr(extractor, "GAP_PATH", tmp_path / "gaps.json")
    monkeypatch.setattr(extractor, "FEATURE_AUDIT_PATH", tmp_path / "audit.json")
    monkeypatch.setattr(extractor, "JOIN_AUDIT_PATH", tmp_path / "join.json")
    features, gaps = extractor.extract()
    assert not features
    assert gaps[0]["gap_reason"] == "env_manifest_missing"


def test_gate3_rejects_missing_selected_metrics(monkeypatch, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    csv_path = job_dir / "capture.csv"
    env_path = job_dir / "capture_env_manifest.json"
    _write_fixture_csv(csv_path)
    _write_env(env_path)
    attempts_path = tmp_path / "attempts.json"
    attempts_path.write_text(json.dumps({"attempts": [
        {
            "capture_job_id": "job",
            "gate3_eligible": True,
            "capture_status": "captured",
            "capture_csv_path": str(csv_path),
            "environment_manifest_path": str(env_path),
            "consuming_manifest_entry_ids": ["L1_A"],
            "consuming_kernel_or_cases": ["my_kernel"],
        }
    ]}))
    monkeypatch.setattr(extractor, "ATTEMPTS_PATH", attempts_path)
    monkeypatch.setattr(extractor, "FEATURE_TABLE_PATH", tmp_path / "features.json")
    monkeypatch.setattr(extractor, "GAP_PATH", tmp_path / "gaps.json")
    monkeypatch.setattr(extractor, "FEATURE_AUDIT_PATH", tmp_path / "audit.json")
    monkeypatch.setattr(extractor, "JOIN_AUDIT_PATH", tmp_path / "join.json")
    features, gaps = extractor.extract()
    assert not features
    assert gaps[0]["gap_reason"] == "selected_metrics_missing"
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["artifact_name"] == "pka_feature_audit_l1"
    assert audit["summary"]["gap_record_count"] == 1
    assert audit["summary"]["gap_reason_counts"]["selected_metrics_missing"] == 1
    assert audit["entries"][0]["status"] == "gap"
    assert audit["entries"][0]["gap_reason"] == "selected_metrics_missing"


def test_gate3_rejects_invalid_selected_metrics(monkeypatch, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    csv_path = job_dir / "capture.csv"
    env_path = job_dir / "capture_env_manifest.json"
    _write_fixture_csv(csv_path)
    _write_env(env_path)
    (job_dir / "selected_metrics.json").write_text(json.dumps([]))
    attempts_path = tmp_path / "attempts.json"
    attempts_path.write_text(json.dumps([
        {
            "capture_job_id": "job",
            "gate3_eligible": True,
            "capture_status": "captured",
            "capture_csv_path": str(csv_path),
            "environment_manifest_path": str(env_path),
            "consuming_manifest_entry_ids": ["L1_A"],
            "consuming_kernel_or_cases": ["my_kernel"],
        }
    ]))
    monkeypatch.setattr(extractor, "ATTEMPTS_PATH", attempts_path)
    monkeypatch.setattr(extractor, "FEATURE_TABLE_PATH", tmp_path / "features.json")
    monkeypatch.setattr(extractor, "GAP_PATH", tmp_path / "gaps.json")
    monkeypatch.setattr(extractor, "FEATURE_AUDIT_PATH", tmp_path / "audit.json")
    monkeypatch.setattr(extractor, "JOIN_AUDIT_PATH", tmp_path / "join.json")
    features, gaps = extractor.extract()
    assert not features
    assert gaps[0]["gap_reason"] == "selected_metrics_invalid"


def test_gate3_reports_metric_not_in_selected_allowlist(monkeypatch, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    csv_path = job_dir / "capture.csv"
    env_path = job_dir / "capture_env_manifest.json"
    _write_fixture_csv(csv_path)
    _write_env(env_path)
    unsupported = [
        {**row, "resolution_status": "unsupported", "selected_for_ncu_metrics": False}
        for row in selected_metric_records()
    ]
    (job_dir / "selected_metrics.json").write_text(json.dumps(unsupported))
    attempts_path = tmp_path / "attempts.json"
    attempts_path.write_text(json.dumps([
        {
            "capture_job_id": "job",
            "gate3_eligible": True,
            "capture_status": "captured",
            "capture_csv_path": str(csv_path),
            "environment_manifest_path": str(env_path),
            "consuming_manifest_entry_ids": ["L1_A"],
            "consuming_kernel_or_cases": ["my_kernel"],
        }
    ]))
    monkeypatch.setattr(extractor, "ATTEMPTS_PATH", attempts_path)
    monkeypatch.setattr(extractor, "FEATURE_TABLE_PATH", tmp_path / "features.json")
    monkeypatch.setattr(extractor, "GAP_PATH", tmp_path / "gaps.json")
    monkeypatch.setattr(extractor, "FEATURE_AUDIT_PATH", tmp_path / "audit.json")
    monkeypatch.setattr(extractor, "JOIN_AUDIT_PATH", tmp_path / "join.json")
    features, gaps = extractor.extract()
    assert not features
    assert gaps[0]["gap_reason"] == "metric_not_in_selected_allowlist"
