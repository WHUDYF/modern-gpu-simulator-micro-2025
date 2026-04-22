from pathlib import Path
import sys

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


def test_middle_layer_mappings_are_internally_consistent():
    bundle = build_middle_layer_artifacts(REPO_ROOT)
    anchor_ids = {anchor["anchor_id"] for anchor in bundle["anchors"]}
    family_ids = {family["family_id"] for family in bundle["families"]}
    regime_ids = {regime["regime_id"] for regime in bundle["regimes"]}

    for family in bundle["families"]:
        assert set(family["input_anchor_ids"]).issubset(anchor_ids)
        assert "decision_weight_factors" in family

    for regime in bundle["regimes"]:
        assert regime["family_id"] in family_ids
        assert set(regime["source_anchor_ids"]).issubset(anchor_ids)
        assert "decision_weight_factors" in regime
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
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
