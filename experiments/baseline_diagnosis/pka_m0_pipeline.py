"""M0 wrapper around the shared deterministic PKA selector core."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pka_selector_core import build_outputs
from shared_acquisition import ARTIFACT_DIR, REPO_ROOT, stable_hash, write_json

FIXTURE_PATH = REPO_ROOT / "experiments" / "baseline_diagnosis" / "fixtures" / "pka_m0_feature_table_l1.json"


def run(input_path: Path = FIXTURE_PATH, output_dir: Path = ARTIFACT_DIR) -> dict:
    records = json.loads(input_path.read_text())
    outputs = build_outputs(records, mode="pka_m0_fixture", feature_mode="pka_m0_fixture")
    input_hash = stable_hash(records)
    outputs["projection"]["input_feature_table_hash"] = input_hash
    outputs["clusters"]["input_feature_table_hash"] = input_hash
    outputs["evaluation"]["input_feature_table_hash"] = input_hash
    write_json(output_dir / "pka_m0_pca_projection_l1.json", outputs["projection"])
    write_json(output_dir / "pka_m0_kmeans_clusters_l1.json", outputs["clusters"])
    write_json(output_dir / "pka_m0_representative_anchor_table_l1.json", outputs["anchors"])
    write_json(output_dir / "pka_m0_compression_evaluation_l1.json", outputs["evaluation"])
    return outputs


def main() -> int:
    run()
    print("M0 selector pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

