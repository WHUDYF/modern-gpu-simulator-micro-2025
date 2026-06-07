import copy

import pytest

from experiments.gcl_phase_b.correctness import (
    evaluate_gate7_correctness,
    evaluate_gate7_correctness_from_artifacts,
)
from experiments.gcl_phase_b.pipeline import run_embedding_export
from experiments.gcl_phase_b.selector import select_phase_b_representatives
from experiments.gcl_phase_b.utils import hash_without
from tests.gcl_resnet50.formal_chain import build_artifact_shape_tensors
from tests.gcl_resnet50.real_chain import run_real_nondegenerate_gate1_to_gate7_artifacts


def _gate6_artifacts():
    return {
        "artifact_type": "gcl_resnet50_gate6_selector_artifacts",
        "artifact_version": "gate6_selector_artifacts_v1",
        "source_embedding_table_hash": "table-hash",
        "k_selection_report": {"selected_k": 2, "mode": "silhouette_k"},
        "kmeans_cluster_assignment_table": {
            "assignments": [
                {"record_id": "a", "kernel_invocation_id": "k0", "cluster_id": 0},
                {"record_id": "b", "kernel_invocation_id": "k1", "cluster_id": 0},
                {"record_id": "c", "kernel_invocation_id": "k2", "cluster_id": 1},
            ]
        },
        "representative_anchor_table": {
            "anchors": [
                {
                    "cluster_id": 0,
                    "representative_record_id": "a",
                    "kernel_invocation_id": "k0",
                    "distance_to_centroid": 0.1,
                },
                {
                    "cluster_id": 1,
                    "representative_record_id": "c",
                    "kernel_invocation_id": "k2",
                    "distance_to_centroid": 0.0,
                },
            ]
        },
        "cluster_family_evidence_report": {
            "family_labels_used_for_clustering": False,
            "members": [
                {"record_id": "a", "cluster_id": 0, "family": "conv", "weight": 0.4},
                {"record_id": "b", "cluster_id": 0, "family": "conv", "weight": 0.3},
                {"record_id": "c", "cluster_id": 1, "family": "bn_relu", "weight": 0.3},
            ],
            "clusters": [
                {"cluster_id": 0, "majority_family": "conv", "purity": 1.0, "weight": 0.7},
                {"cluster_id": 1, "majority_family": "bn_relu", "purity": 1.0, "weight": 0.3},
            ],
        },
        "selector_manifest_hash": "selector-hash",
    }


def test_gate7_records_embedding_geometry_metrics():
    report = evaluate_gate7_correctness(
        _gate6_artifacts(),
        embedding_geometry={
            "silhouette": 0.5,
            "davies_bouldin": 0.8,
            "calinski_harabasz": 10.0,
            "intra_distance_mean": 0.2,
            "inter_distance_mean": 1.0,
        },
    )

    assert report["artifact_type"] == "gcl_resnet50_gate7_cluster_correctness_manifest"
    assert report["artifact_version"] == "gate7_cluster_correctness_manifest_v1"
    assert report["threshold_policy"] == "report_only_v1"
    assert report["claim_status"] == "quantified_no_correctness_claim"
    assert report["threshold_claim_status"] == "not_set_until_real_resnet50_baseline"
    assert report["suggested_min_silhouette_score"] is None
    assert report["suggested_min_weighted_cluster_purity"] is None
    assert report["suggested_max_global_weighted_mape"] is None
    assert report["suggested_min_assignment_stability_ari"] is None
    for field in [
        "source_gate6_selector_manifest_hash",
        "source_cluster_assignment_table_hash",
        "source_representative_anchor_table_hash",
        "source_embedding_table_hash",
        "metric_source_manifest_hash",
        "family_label_source_hash",
        "structural_summary_source_hash",
        "embedding_quality_report_hash",
        "family_alignment_report_hash",
        "representative_quality_report_hash",
        "metric_error_report_hash",
        "stability_report_hash",
        "gate7_cluster_correctness_manifest_hash",
    ]:
        assert field in report
    assert report["embedding_geometry_metrics"]["inter_intra_ratio"] == 5.0
    assert report["stability_report"]["stability_status"] == "single_run_not_evaluated"


def test_gate7_rejects_debug_selector_artifacts():
    artifacts = _gate6_artifacts()
    artifacts["artifact_status"] = "debug_not_formal"

    with pytest.raises(ValueError, match="debug"):
        evaluate_gate7_correctness(artifacts)


def test_gate7_records_family_and_representative_quality():
    report = evaluate_gate7_correctness(_gate6_artifacts())

    assert report["family_alignment_metrics"]["cluster_purity"] == 1.0
    assert report["family_alignment_metrics"]["weighted_purity"] == 1.0
    assert report["family_alignment_metrics"]["ari"] == 1.0
    assert report["family_alignment_metrics"]["nmi"] == 1.0
    assert report["representative_quality_metrics"]["representative_p95_distance"] == 0.1
    assert report["representative_quality_metrics"]["high_weight_outlier_count"] == 0


def test_gate7_computes_partial_family_alignment_metrics_from_member_labels():
    artifacts = _gate6_artifacts()
    artifacts["cluster_family_evidence_report"]["members"] = [
        {"record_id": "a", "cluster_id": 0, "family": "conv", "weight": 0.4},
        {"record_id": "b", "cluster_id": 0, "family": "bn_relu", "weight": 0.3},
        {"record_id": "c", "cluster_id": 1, "family": "conv", "weight": 0.3},
    ]

    report = evaluate_gate7_correctness(artifacts)

    assert report["family_alignment_metrics"]["ari"] == -0.5
    assert report["family_alignment_metrics"]["nmi"] > 0.0
    assert report["family_alignment_metrics"]["homogeneity"] > 0.0
    assert report["family_alignment_metrics"]["completeness"] > 0.0
    assert report["family_alignment_metrics"]["v_measure"] > 0.0
    assert report["family_alignment_metrics"]["mixed_family_cluster_count"] == 1


def test_gate7_marks_family_evidence_unavailable_when_all_member_labels_missing():
    artifacts = _gate6_artifacts()
    artifacts["cluster_family_evidence_report"]["members"] = [
        {"record_id": "a", "cluster_id": 0, "family": None, "weight": 0.4},
        {"record_id": "b", "cluster_id": 0, "family": "", "weight": 0.3},
        {"record_id": "c", "cluster_id": 1, "weight": 0.3},
    ]

    report = evaluate_gate7_correctness(artifacts)
    metrics = report["family_alignment_metrics"]

    assert metrics["family_evidence_status"] == "unavailable"
    assert metrics["family_alignment_claim_status"] == "no_family_claim"
    assert metrics["ari"] is None
    assert metrics["nmi"] is None
    assert metrics["unlabeled_record_count"] == 3


def test_gate7_marks_family_evidence_unavailable_when_member_labels_partial():
    artifacts = _gate6_artifacts()
    artifacts["cluster_family_evidence_report"]["members"] = [
        {"record_id": "a", "cluster_id": 0, "family": "conv", "weight": 0.4},
        {"record_id": "b", "cluster_id": 0, "family": None, "weight": 0.3},
        {"record_id": "c", "cluster_id": 1, "family": "bn_relu", "weight": 0.3},
    ]

    report = evaluate_gate7_correctness(artifacts)
    metrics = report["family_alignment_metrics"]

    assert metrics["family_evidence_status"] == "unavailable"
    assert metrics["family_alignment_claim_status"] == "no_family_claim"
    assert metrics["ari"] is None
    assert metrics["nmi"] is None
    assert metrics["unlabeled_record_count"] == 1


def test_gate7_does_not_modify_gate6_assignments():
    artifacts = _gate6_artifacts()
    before = copy.deepcopy(artifacts["kmeans_cluster_assignment_table"]["assignments"])

    report = evaluate_gate7_correctness(artifacts)

    assert artifacts["kmeans_cluster_assignment_table"]["assignments"] == before
    assert report["source_assignment_hash"]


def test_gate7_records_metric_error_reports():
    report = evaluate_gate7_correctness(
        _gate6_artifacts(),
        metric_rows=[
            {"cluster_id": 0, "measured": 100.0, "predicted": 90.0, "weight": 0.7},
            {"cluster_id": 1, "measured": 50.0, "predicted": 55.0, "weight": 0.3},
        ],
    )

    assert report["metric_error_report"]["global_weighted_mape"] == 0.1
    assert report["metric_error_report"]["cluster_max_relative_error"] == {"0": 0.1, "1": 0.1}
    assert report["metric_error_report"]["global_p95_relative_error"] == 0.1
    assert report["metric_error_report"]["global_max_relative_error"] == 0.1
    assert report["metric_error_report"]["cluster_metric_correlation"]["0"] is None
    assert report["metric_error_report"]["high_error_member_count"] == 0
    assert report["metric_error_report"]["high_weight_bad_cluster_count"] == 0


def test_gate7_metric_error_reports_correlation_and_high_error_counts():
    report = evaluate_gate7_correctness(
        _gate6_artifacts(),
        metric_rows=[
            {"cluster_id": 0, "measured": 100.0, "predicted": 100.0, "weight": 0.6},
            {"cluster_id": 0, "measured": 200.0, "predicted": 300.0, "weight": 0.9},
            {"cluster_id": 0, "measured": 300.0, "predicted": 450.0, "weight": 0.2},
            {"cluster_id": 1, "measured": 50.0, "predicted": 55.0, "weight": 0.1},
        ],
    )

    metric_report = report["metric_error_report"]
    assert metric_report["cluster_max_relative_error"]["0"] == 0.5
    assert metric_report["cluster_metric_correlation"]["0"] > 0.9
    assert metric_report["cluster_metric_rank_correlation"]["0"] == 1.0
    assert metric_report["high_error_member_count"] == 2
    assert metric_report["high_weight_high_error_member_count"] == 1
    assert metric_report["bad_cluster_count"] == 1


def test_gate7_metric_error_skips_incomplete_rows_without_crashing():
    report = evaluate_gate7_correctness(
        _gate6_artifacts(),
        metric_rows=[
            {"cluster_id": 0, "measured": 100.0, "predicted": 90.0, "weight": 1.0},
            {"cluster_id": 0, "measured": 200.0, "weight": 2.0},
            {"cluster_id": 1, "predicted": 50.0, "weight": 3.0},
        ],
    )

    metric_report = report["metric_error_report"]
    assert metric_report["status"] == "partial_metric_missing"
    assert metric_report["metric_claim_status"] == "unavailable"
    assert metric_report["complete_row_count"] == 1
    assert metric_report["missing_metric_row_count"] == 2
    assert metric_report["missing_metric_rows"] == [
        {"row_index": 1, "cluster_id": "0", "missing_fields": ["predicted"]},
        {"row_index": 2, "cluster_id": "1", "missing_fields": ["measured"]},
    ]
    assert metric_report["global_weighted_mape"] == 0.1


def test_gate7_metric_error_marks_all_incomplete_rows_unavailable():
    report = evaluate_gate7_correctness(
        _gate6_artifacts(),
        metric_rows=[
            {"cluster_id": 0, "measured": 100.0},
            {"cluster_id": 1, "predicted": 50.0},
        ],
    )

    metric_report = report["metric_error_report"]
    assert metric_report["status"] == "metric_source_missing"
    assert metric_report["metric_claim_status"] == "unavailable"
    assert metric_report["complete_row_count"] == 0
    assert metric_report["missing_metric_row_count"] == 2
    assert metric_report["global_weighted_mape"] is None


def test_gate7_weighted_purity_uses_cluster_weights_not_cluster_count():
    artifacts = _gate6_artifacts()
    artifacts["cluster_family_evidence_report"]["clusters"] = [
        {"cluster_id": 0, "majority_family": "conv", "purity": 0.5, "weight": 0.9},
        {"cluster_id": 1, "majority_family": "bn_relu", "purity": 1.0, "weight": 0.1},
    ]

    report = evaluate_gate7_correctness(artifacts)

    assert report["family_alignment_metrics"]["cluster_purity"] == 0.75
    assert report["family_alignment_metrics"]["weighted_purity"] == 0.55


def test_gate7_reports_metric_unit_conflict():
    report = evaluate_gate7_correctness(
        _gate6_artifacts(),
        metric_rows=[
            {"cluster_id": 0, "measured": 100.0, "predicted": 90.0, "unit": "cycles"},
            {"cluster_id": 1, "measured": 50.0, "predicted": 55.0, "unit": "ms"},
        ],
    )

    assert report["metric_error_report"]["status"] == "metric_unit_conflict"


def test_gate7_rejects_stability_claim_from_single_run():
    with pytest.raises(ValueError, match="single-run"):
        evaluate_gate7_correctness(_gate6_artifacts(), stability_claim="stable")


def test_gate7_rejects_artifact_shape_embedding_chain_as_formal_evidence(tmp_path):
    table, _training = run_embedding_export(build_artifact_shape_tensors(tmp_path), tmp_path)
    selector_artifacts = select_phase_b_representatives(table, seed=11, allow_debug=True)

    with pytest.raises(ValueError, match="debug"):
        evaluate_gate7_correctness_from_artifacts(
            selector_artifacts=selector_artifacts,
            embedding_table=table,
        )


def test_gate7_records_embedding_geometry_metrics_from_real_root_gate6(tmp_path):
    chain = run_real_nondegenerate_gate1_to_gate7_artifacts(tmp_path / "real_chain")

    report = evaluate_gate7_correctness_from_artifacts(
        selector_artifacts=chain["selector_artifacts"],
        embedding_table=chain["embedding_table"],
    )

    assert report["source_gate5_embedding_table_hash"] == (
        chain["embedding_table"]["kernel_embedding_table_hash"]
    )
    assert report["source_embedding_table_hash"] == (
        chain["embedding_table"]["kernel_embedding_table_hash"]
    )
    assert report["source_gate6_selector_manifest_hash"] == (
        chain["selector_artifacts"]["selector_manifest_hash"]
    )
    assert report["source_cluster_assignment_table_hash"]
    assert report["source_representative_anchor_table_hash"]
    assert report["embedding_quality_report_hash"]
    assert report["embedding_geometry_metrics"]["silhouette"] is not None
    assert report["embedding_geometry_metrics"]["davies_bouldin"] is not None
    assert report["embedding_geometry_metrics"]["calinski_harabasz"] is not None


def test_gate7_from_artifacts_hash_matches_final_manifest_payload(tmp_path):
    chain = run_real_nondegenerate_gate1_to_gate7_artifacts(tmp_path / "real_chain")

    report = evaluate_gate7_correctness_from_artifacts(
        selector_artifacts=chain["selector_artifacts"],
        embedding_table=chain["embedding_table"],
    )

    expected_hash = hash_without(
        report,
        "gate7_cluster_correctness_manifest_hash",
        "gate7_correctness_manifest_hash",
    )
    assert report["gate7_cluster_correctness_manifest_hash"] == expected_hash
    assert report["gate7_correctness_manifest_hash"] == expected_hash


def test_gate7_records_family_representative_metric_and_stability_from_real_root(tmp_path):
    chain = run_real_nondegenerate_gate1_to_gate7_artifacts(tmp_path / "real_chain")

    report = evaluate_gate7_correctness_from_artifacts(
        selector_artifacts=chain["selector_artifacts"],
        embedding_table=chain["embedding_table"],
        metric_rows=[
            {
                "cluster_id": 0,
                "measured": 100.0,
                "predicted": 95.0,
                "weight": 1.0,
                "unit": "cycles",
            }
        ],
    )

    assert set(report["family_alignment_metrics"]) == {
        "ari",
        "completeness",
        "cluster_purity",
        "family_alignment_claim_status",
        "family_evidence_status",
        "family_to_cluster_coverage",
        "high_weight_mixed_family_cluster_count",
        "homogeneity",
        "mixed_family_cluster_count",
        "nmi",
        "unlabeled_record_count",
        "v_measure",
        "weighted_purity",
    }
    assert report["family_alignment_metrics"]["cluster_purity"] is not None
    assert report["family_alignment_metrics"]["weighted_purity"] is not None
    assert report["family_alignment_metrics"]["ari"] is not None
    assert report["family_alignment_metrics"]["nmi"] is not None
    assert report["family_alignment_metrics"]["family_alignment_claim_status"] == "reported"
    assert report["representative_quality_metrics"]["representative_p95_distance"] is not None
    assert report["representative_quality_metrics"]["cluster_reports"]
    for cluster_report in report["representative_quality_metrics"]["cluster_reports"]:
        assert "coverage_weight_sum" in cluster_report
        assert "mean_distance_to_representative" in cluster_report
        assert "p95_distance_to_representative" in cluster_report
        assert "max_distance_to_representative" in cluster_report
        assert "representative_rank_to_centroid" in cluster_report
        assert "outlier_member_ratio" in cluster_report
    assert report["metric_error_report"]["status"] == "reported"
    assert report["metric_error_report"]["global_weighted_mape"] == 0.05
    assert report["metric_error_report"]["global_max_relative_error"] == 0.05
    assert report["stability_report"]["stability_status"] == "single_run_not_evaluated"
