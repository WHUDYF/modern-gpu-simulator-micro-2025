"""Gate 5 formal M1 selector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pka_selector_core import build_outputs
from shared_acquisition import ARTIFACT_DIR, file_hash, read_json, stable_hash, write_json

ELIGIBILITY_PATH = ARTIFACT_DIR / "m1_selector_eligibility_l1.json"
SELECTOR_INPUT_PATH = ARTIFACT_DIR / "m1_selector_input_l1.json"

FORBIDDEN = {"kernel_name", "source_path", "expected_behavior_axis", "family", "regime", "shape_hint", "trace_order"}


def run() -> dict:
    eligibility = read_json(ELIGIBILITY_PATH, {})
    if not eligibility.get("gate5_allowed"):
        raise SystemExit(f"Gate5 blocked: {eligibility.get('selector_eligibility_state', 'missing_eligibility')}")
    records = read_json(SELECTOR_INPUT_PATH, [])
    if len(records) < 3:
        raise SystemExit("Gate5 blocked: fewer than 3 selector records")
    for row in records:
        forbidden = sorted(FORBIDDEN & set(row))
        if forbidden:
            raise SystemExit(f"Gate5 blocked: forbidden selector fields {forbidden}")
        if row.get("feature_mode") != "pka_m1_measured":
            raise SystemExit("Gate5 blocked: non-M1 feature_mode")
    outputs = build_outputs(records, mode="pka_m1_measured", feature_mode="pka_m1_measured")
    selector_hash = file_hash(SELECTOR_INPUT_PATH)
    eligibility_hash = file_hash(ELIGIBILITY_PATH)
    for key in ("projection", "clusters", "evaluation"):
        outputs[key]["input_selector_projection_hash"] = selector_hash
        outputs[key]["gate4_eligibility_hash"] = eligibility_hash
    for row in outputs["anchors"]:
        row["input_selector_projection_hash"] = selector_hash
        row["gate4_eligibility_hash"] = eligibility_hash
    write_json(ARTIFACT_DIR / "pka_pca_projection_l1.json", outputs["projection"])
    write_json(ARTIFACT_DIR / "pka_kmeans_clusters_l1.json", outputs["clusters"])
    write_json(ARTIFACT_DIR / "representative_anchor_table_l1.json", outputs["anchors"])
    write_json(ARTIFACT_DIR / "pka_compression_evaluation_l1.json", outputs["evaluation"])
    return outputs


def main() -> int:
    run()
    print("Gate5 formal M1 selector complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

