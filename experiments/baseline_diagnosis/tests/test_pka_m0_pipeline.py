from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pka_m0_pipeline


def test_m0_pipeline_writes_deterministic_artifacts(tmp_path):
    outputs = pka_m0_pipeline.run(output_dir=tmp_path)
    assert outputs["projection"]["pca"]["method"] == "numpy_svd"
    assert outputs["clusters"]["kmeans"]["method"] == "deterministic_farthest_first"
    assert (tmp_path / "pka_m0_pca_projection_l1.json").exists()
    assert (tmp_path / "pka_m0_kmeans_clusters_l1.json").exists()
    assert (tmp_path / "pka_m0_representative_anchor_table_l1.json").exists()
    assert (tmp_path / "pka_m0_compression_evaluation_l1.json").exists()
    anchors = json.loads((tmp_path / "pka_m0_representative_anchor_table_l1.json").read_text())
    assert anchors
    assert all(row["representative_selection"] == "nearest_centroid_real_record" for row in anchors)

