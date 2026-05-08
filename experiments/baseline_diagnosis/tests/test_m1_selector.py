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
    elig.write_text(json.dumps({"gate5_allowed": True, "selector_eligibility_state": "selector_ready"}))
    selector_input.write_text(json.dumps([_record(0), _record(1), _record(2)]))
    monkeypatch.setattr(selector, "ELIGIBILITY_PATH", elig)
    monkeypatch.setattr(selector, "SELECTOR_INPUT_PATH", selector_input)
    monkeypatch.setattr(selector, "ARTIFACT_DIR", tmp_path)
    outputs = selector.run()
    assert outputs["projection"]["pca"]["method"] == "numpy_svd"
    assert outputs["clusters"]["kmeans"]["method"] == "deterministic_farthest_first"
    assert (tmp_path / "pka_pca_projection_l1.json").exists()
    assert (tmp_path / "pka_kmeans_clusters_l1.json").exists()
    assert (tmp_path / "representative_anchor_table_l1.json").exists()
    assert (tmp_path / "pka_compression_evaluation_l1.json").exists()

