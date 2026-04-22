import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.backend_pipeline.backend_builder import build_anchor_table, build_backend_outputs, load_full_features  # noqa: E402


INPUT = REPO_ROOT / "experiments" / "mini_transformer" / "mini_transformer_v4_full.json"
SCRIPT = ROOT / "build_backend_outputs.py"


def test_anchor_table_aggregates_gemm_tiled_real_counts():
    anchors = build_anchor_table(load_full_features(INPUT))
    gemm = next(anchor for anchor in anchors if anchor["rep_kernel_id"] == "A1")
    assert gemm["kernel_name"] == "gemm_tiled"
    assert gemm["coverage_count"] == 7
    assert gemm["coverage_weight"] == 0.5
    assert gemm["canonical_status"] == "stable"


def test_backend_outputs_keep_current_canonical_absorptions():
    outputs = build_backend_outputs(load_full_features(INPUT))
    families = {row["family_id"]: row for row in outputs["family_table"]}
    regimes = {row["regime_id"]: row for row in outputs["regime_table"]}
    assert "A5" in families["F2_reduction_normalize"]["member_rep_kernels"]
    assert families["F2_reduction_normalize"]["canonical_status"] == "absorbed-with-review"
    assert regimes["R4_layernorm_reduction"]["canonical_status"] == "review-needed"
    assert "A6" in families["F4_elementwise_fusion"]["member_rep_kernels"]
    assert regimes["R6_residual_elementwise"]["resource_signature"] == "dram-dominated elementwise path"


def test_cli_writes_expected_backend_artifacts(tmp_path):
    subprocess.run([sys.executable, str(SCRIPT), "--input", str(INPUT), "--output-dir", str(tmp_path)], check=True)
    expected = {"backend_anchor_table_v1.json", "backend_family_table_v1.json", "backend_regime_table_v1.json", "backend_priority_lane_table_v1.json", "backend_validation_worksheet_v1.json", "backend_writeback_map_v1.json"}
    assert expected == {path.name for path in tmp_path.iterdir()}
    family_table = json.loads((tmp_path / "backend_family_table_v1.json").read_text())
    assert family_table[0]["family_id"] == "F1_dense_tiled"


def test_no_priority_family_baseline_uses_stable_non_importance_order():
    outputs = build_backend_outputs(load_full_features(INPUT))
    rows = [
        row
        for row in outputs["priority_lane_table"]
        if row["object_level"] == "family" and row["priority_source"] == "no-priority"
    ]
    rows.sort(key=lambda row: row["priority_rank"])
    assert [row["family_id"] for row in rows] == [
        "F1_dense_tiled",
        "F2_reduction_normalize",
        "F3_streaming_aggregation",
        "F4_elementwise_fusion",
    ]


def test_no_priority_regime_baseline_uses_original_order():
    outputs = build_backend_outputs(load_full_features(INPUT))
    rows = [
        row
        for row in outputs["priority_lane_table"]
        if row["object_level"] == "regime" and row["priority_source"] == "no-priority"
    ]
    rows.sort(key=lambda row: row["priority_rank"])
    assert [row["regime_id"] for row in rows] == [
        "R1_projection_dense",
        "R2_attention_score_dense",
        "R3_softmax_reduction",
        "R4_layernorm_reduction",
        "R5_context_streaming",
        "R6_residual_elementwise",
    ]
