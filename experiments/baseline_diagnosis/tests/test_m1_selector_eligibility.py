from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m1_selector_eligibility as gate4
from shared_acquisition import FEATURE_ORDER


def _feature_row(idx: int) -> dict:
    return {
        "record_id": f"r{idx}",
        "manifest_id": f"L1_{idx}",
        "kernel_invocation_id": f"k{idx}#1",
        "feature_mode": "pka_m1_measured",
        "features": {
            name: {
                "value": float(idx + 1),
                "status": "measured",
                "canonical_metric": f"canon_{name}",
                "actual_source_metric": f"metric_{name}",
                "provenance": {"source": "fixture"},
            }
            for name in FEATURE_ORDER
        },
    }


def test_gate4_blocks_insufficient_rows_and_writes_repair(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"entries": [
        {"id": "L1_0", "priority": "P0"},
        {"id": "L1_1", "priority": "P0"},
        {"id": "L1_2", "priority": "P0"},
    ]}))
    feature_path = tmp_path / "features.json"
    feature_path.write_text(json.dumps([_feature_row(0)]))
    gap_path = tmp_path / "acq_gap.json"
    gap_path.write_text(json.dumps([{"manifest_entry_id": "L1_1", "gap_reason": "missing_required_metrics"}]))
    monkeypatch.setattr(gate4, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(gate4, "FEATURE_TABLE_PATH", feature_path)
    monkeypatch.setattr(gate4, "ACQ_GAP_PATH", gap_path)
    monkeypatch.setattr(gate4, "RESOLUTION_GAP_PATH", tmp_path / "res_gap.json")
    monkeypatch.setattr(gate4, "CAPTURE_GAP_PATH", tmp_path / "cap_gap.json")
    monkeypatch.setattr(gate4, "ELIGIBILITY_PATH", tmp_path / "elig.json")
    monkeypatch.setattr(gate4, "SELECTOR_INPUT_PATH", tmp_path / "selector_input.json")
    monkeypatch.setattr(gate4, "REPAIR_JSON_PATH", tmp_path / "repair.json")
    monkeypatch.setattr(gate4, "REPAIR_MD_PATH", tmp_path / "repair.md")
    eligibility = gate4.evaluate()
    assert eligibility["selector_eligibility_state"] == "selector_blocked_insufficient_measured_records"
    assert eligibility["gate5_allowed"] is False
    assert eligibility["feature_table_preflight"]["status"] == "passed"
    assert eligibility["timing_check"]["weight_mode"] == "member_count_fallback"
    assert "insufficient_measured_records" in eligibility["blocking_reasons"]
    repair = json.loads((tmp_path / "repair.json").read_text())
    assert len(repair["entries"]) == 3
    assert repair["entries"][1]["entry_status"] == "blocked"
    assert repair["entries"][1]["gate3_status"] == "blocked"
    assert repair["entries"][1]["repair_action_type"] == "code_fix_required"
    assert repair["summary"]["total_p0_entries"] == 3
