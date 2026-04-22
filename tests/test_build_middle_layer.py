from pathlib import Path
import sys
import tempfile

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.baseline_diagnosis.build_middle_layer import (
    DEFAULT_RULE_CONFIG,
    build_middle_layer_artifacts,
    write_middle_layer_artifacts,
)


def test_build_middle_layer_artifacts_has_expected_counts():
    bundle = build_middle_layer_artifacts(REPO_ROOT)

    assert len(bundle["anchors"]) == 9
    assert len(bundle["families"]) == 4
    assert len(bundle["regimes"]) == 9
    assert len(bundle["lanes"]) == 9


def test_rule_config_exists_and_is_family_centered_yaml():
    assert DEFAULT_RULE_CONFIG.exists()

    config = yaml.safe_load(DEFAULT_RULE_CONFIG.read_text())

    assert config["workload"] == "mini_transformer_v4"
    assert config["rule_config_version"] == "v1"
    assert len(config["families"]) == 4
    assert all("anchors" in family for family in config["families"])
    assert all("regimes" in family for family in config["families"])
    assert all("decision_weight_factors" in family for family in config["families"])


def test_anchor_builder_splits_dense_kernel_into_multiple_context_aware_anchors():
    bundle = build_middle_layer_artifacts(REPO_ROOT)
    dense_anchors = [anchor for anchor in bundle["anchors"] if anchor["kernel_name"] == "gemm_tiled"]

    assert [anchor["anchor_id"] for anchor in dense_anchors] == [
        "A1_qkv_projection_dense_48x32",
        "A5_output_projection_dense_48x32",
        "A8_ffn_expand_dense_192x32",
        "A9_ffn_contract_dense_48x32",
    ]

    softmax_anchor = next(anchor for anchor in bundle["anchors"] if anchor["anchor_id"] == "A3_softmax_reduce_24x1")
    assert softmax_anchor["ape_lookup_key"] == "softmax_kernel|(6144, 1, 1)|(256, 1, 1)"
    assert softmax_anchor["ape_elapsed_cycles_ape"] is not None

    shared_shape_dense = {
        anchor["anchor_id"]: anchor
        for anchor in dense_anchors
        if anchor["anchor_id"] in {
            "A1_qkv_projection_dense_48x32",
            "A5_output_projection_dense_48x32",
            "A9_ffn_contract_dense_48x32",
        }
    }
    assert all(anchor["ape_lookup_key"] is None for anchor in shared_shape_dense.values())
    assert all(anchor["ape_evidence_status"] == "shared_across_anchors" for anchor in shared_shape_dense.values())


def test_middle_layer_mappings_are_internally_consistent():
    bundle = build_middle_layer_artifacts(REPO_ROOT)
    anchor_ids = {anchor["anchor_id"] for anchor in bundle["anchors"]}
    family_ids = {family["family_id"] for family in bundle["families"]}
    regime_ids = {regime["regime_id"] for regime in bundle["regimes"]}

    for family in bundle["families"]:
        assert set(family["input_anchor_ids"]).issubset(anchor_ids)
        assert "decision_weight_factors" in family
        assert "coverage_weight" in family
        assert "time_weight" in family
        assert "decision_weight" in family
        assert set(family["weight_source"]) == {"coverage", "time", "decision"}

    for regime in bundle["regimes"]:
        assert regime["family_id"] in family_ids
        assert set(regime["source_anchor_ids"]).issubset(anchor_ids)
        assert "decision_weight_factors" in regime
        assert "coverage_weight" in regime
        assert "time_weight" in regime
        assert "local_decision_weight" in regime
        assert set(regime["weight_source"]) == {"coverage", "time", "decision"}
        assert regime["simulator_lane_id"].startswith("L")

    for lane in bundle["lanes"]:
        assert lane["target_regime_id"] in regime_ids
        assert lane["target_family_id"] in family_ids


def test_observed_anchor_ratios_sum_to_one():
    bundle = build_middle_layer_artifacts(REPO_ROOT)
    coverage_sum = sum(anchor["observed_coverage_ratio"] for anchor in bundle["anchors"])
    time_sum = sum(anchor["observed_time_ratio"] for anchor in bundle["anchors"])

    assert abs(coverage_sum - 1.0) < 1e-4
    assert abs(time_sum - 1.0) < 1e-4


def test_metadata_records_rule_config_path():
    bundle = build_middle_layer_artifacts(REPO_ROOT)

    assert bundle["metadata"]["rule_config_path"] == "docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml"


def test_custom_rule_config_path_does_not_crash_metadata_generation():
    relative_repo_root = Path(".")
    relative_rule_config = Path("docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml")

    bundle = build_middle_layer_artifacts(relative_repo_root, relative_rule_config)

    assert bundle["metadata"]["rule_config_path"] == "docs/family_criteria/mini_transformer_v4/mini_transformer_middle_layer_rules_v1_2026-04-22.yaml"


def test_invalid_rule_config_kernel_coverage_raises():
    config = yaml.safe_load(DEFAULT_RULE_CONFIG.read_text())
    config["families"][0]["anchors"][0]["kernel_ids"] = [1, 2, 3]

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_config_path = Path(tmpdir) / "bad_rules.yaml"
        bad_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        try:
            build_middle_layer_artifacts(REPO_ROOT, bad_config_path)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected ValueError for missing kernel coverage")

    assert "missing kernel ids" in message


def test_invalid_rule_config_mechanism_evidence_raises():
    config = yaml.safe_load(DEFAULT_RULE_CONFIG.read_text())
    config["families"][0]["anchors"][0]["expected_squash_segments"] = [99]

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_config_path = Path(tmpdir) / "bad_rules.yaml"
        bad_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        try:
            build_middle_layer_artifacts(REPO_ROOT, bad_config_path)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid squash evidence")

    assert "squash segments mismatch" in message


def test_importance_scoring_sheet_and_writeback_records_exist_and_align():
    bundle = build_middle_layer_artifacts(REPO_ROOT)

    scoring_rows = bundle["importance_scoring_sheet"]
    writeback_rows = bundle["writeback_lane_to_regime"]

    assert len(scoring_rows) == len(bundle["families"]) + len(bundle["regimes"])
    assert len(writeback_rows) == len(bundle["lanes"])

    valid_weight_sources = {"measured", "derived", "provisional", "placeholder"}
    family_by_id = {family["family_id"]: family for family in bundle["families"]}
    regime_by_id = {regime["regime_id"]: regime for regime in bundle["regimes"]}
    for row in scoring_rows:
        assert "coverage_weight" in row
        assert "time_weight" in row
        assert "decision_weight" in row
        assert "family_importance_score" in row
        assert "local_decision_weight" in row
        assert "regime_priority_score" in row
        assert set(row["weight_source"].values()).issubset(valid_weight_sources)
        if row["object_level"] == "family":
            family = family_by_id[row["object_id"]]
            assert row["family_importance_score"] == family["importance_score"]
            assert row["local_decision_weight"] is None
            assert row["regime_priority_score"] is None
        else:
            regime = regime_by_id[row["object_id"]]
            assert row["family_importance_score"] == regime["family_importance_score"]
            assert row["local_decision_weight"] == regime["local_decision_weight"]
            assert row["regime_priority_score"] == regime["regime_priority_score"]

    lane_by_id = {lane["lane_id"]: lane for lane in bundle["lanes"]}
    regime_ids = set(regime_by_id)
    for row in writeback_rows:
        assert row["lane_id"] in lane_by_id
        assert row["target_regime_id"] in regime_ids
        assert row["writeback_chain"]["lane_to_regime"] == row["target_regime_id"]
        assert row["writeback_chain"]["regime_to_family"] == row["target_family_id"]


def test_write_middle_layer_artifacts_creates_expected_files(tmp_path):
    bundle = build_middle_layer_artifacts(REPO_ROOT)

    write_middle_layer_artifacts(bundle, tmp_path)

    expected = {
        "anchors.json",
        "families.json",
        "regimes.json",
        "lanes.json",
        "bundle.json",
        "anchors.md",
        "families.md",
        "regimes.md",
        "lanes.md",
        "importance_scoring_sheet.json",
        "importance_scoring_sheet.md",
        "writeback_lane_to_regime.json",
        "writeback_lane_to_regime.md",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
