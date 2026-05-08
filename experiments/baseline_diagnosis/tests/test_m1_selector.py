from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pka_m1_selector as selector
from shared_acquisition import FEATURE_ORDER


def _record(idx: int) -> dict:
    return {
        "record_id": f"r{idx}",
        "kernel_invocation_id": f"k{idx}#1",
        "feature_mode": "pka_m1_measured",
        "weight_input": {"weight_mode": "member_count_fallback", "value": 1.0},
        "features": {
            name: {"value": float(idx + 1), "status": "measured"}
            for name in FEATURE_ORDER
        },
    }


def test_gate5_consumes_projection_and_writes_formal_artifacts(monkeypatch, tmp_path):
    elig = tmp_path / "elig.json"
    selector_input = tmp_path / "selector_input.json"
    elig.write_text(json.dumps({
        "gate5_allowed": True,
        "selector_eligibility_state": "selector_ready",
        "weight_mode": "member_count_fallback",
        "timing_unit": None,
    }))
    selector_input.write_text(json.dumps([_record(0), _record(1), _record(2)]))
    monkeypatch.setattr(selector, "ELIGIBILITY_PATH", elig)
    monkeypatch.setattr(selector, "SELECTOR_INPUT_PATH", selector_input)
    monkeypatch.setattr(selector, "ARTIFACT_DIR", tmp_path)
    outputs = selector.run()
    assert outputs["projection"]["method"] == "numpy_svd"
    assert outputs["clusters"]["method"] == "deterministic_farthest_first_kmeans"
    assert outputs["anchors"]["forbidden_field_audit"]["status"] == "passed"
    assert outputs["evaluation"]["weight_mode"] == "member_count_fallback"
    assert outputs["evaluation"]["metric_scope"] == "structural_only_not_simulator_accuracy"
    assert (tmp_path / "pka_pca_projection_l1.json").exists()
    assert (tmp_path / "pka_kmeans_clusters_l1.json").exists()
    assert (tmp_path / "representative_anchor_table_l1.json").exists()
    assert (tmp_path / "pka_compression_evaluation_l1.json").exists()


def test_gate5_honors_timing_weight_contract(monkeypatch, tmp_path):
    elig = tmp_path / "elig.json"
    selector_input = tmp_path / "selector_input.json"
    records = [_record(0), _record(1), _record(2)]
    for idx, row in enumerate(records):
        row["weight_input"] = {"weight_mode": "timing_weight", "timing_unit": "duration_ns", "value": float(idx + 1)}
    elig.write_text(json.dumps({
        "gate5_allowed": True,
        "selector_eligibility_state": "selector_ready",
        "weight_mode": "timing_weight",
        "timing_unit": "duration_ns",
    }))
    selector_input.write_text(json.dumps(records))
    monkeypatch.setattr(selector, "ELIGIBILITY_PATH", elig)
    monkeypatch.setattr(selector, "SELECTOR_INPUT_PATH", selector_input)
    monkeypatch.setattr(selector, "ARTIFACT_DIR", tmp_path)
    outputs = selector.run()
    assert outputs["evaluation"]["weight_mode"] == "timing_weight"
    assert outputs["evaluation"]["timing_unit"] == "duration_ns"
    assert outputs["evaluation"]["weighted_coverage"] == 1.0
