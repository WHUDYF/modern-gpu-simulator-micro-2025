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
    path.write_text("\n".join(rows) + "\n")


def test_gate3_extracts_complete_measured_row(monkeypatch, tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    csv_path = job_dir / "capture.csv"
    _write_fixture_csv(csv_path)
    (job_dir / "selected_metrics.json").write_text(json.dumps(selected_metric_records()))
    attempts_path = tmp_path / "attempts.json"
    attempts_path.write_text(json.dumps([
        {
            "capture_job_id": "job",
            "gate3_eligible": True,
            "capture_status": "captured",
            "capture_csv_path": str(csv_path),
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
    assert len(features) == 1
    assert not gaps
    assert features[0]["feature_mode"] == "pka_m1_measured"
    assert set(features[0]["features"]) == set(FEATURE_ORDER)
    assert features[0]["features"]["num_thread_blocks"]["value"] == 6.0

