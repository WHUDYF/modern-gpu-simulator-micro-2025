import copy

import pytest

from experiments.gcl_phase_b.correctness import (
    evaluate_gate7_correctness,
    evaluate_gate7_correctness_from_artifacts,
)
from experiments.gcl_phase_b.pipeline import run_embedding_export
from experiments.gcl_phase_b.selector import select_phase_b_representatives
from tests.gcl_resnet50.formal_chain import build_formal_tensors


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

    assert report["artifact_type"] == "gcl_resnet50_gate7_correctness_manifest"
    assert report["threshold_policy"] == "report_only_v1"
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
    assert report["representative_quality_metrics"]["representative_p95_distance"] == 0.1
    assert report["representative_quality_metrics"]["high_weight_outlier_count"] == 0


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
    assert report["metric_error_report"]["high_weight_bad_cluster_count"] == 0


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


def test_gate7_computes_geometry_from_gate5_embedding_and_gate6_assignments(tmp_path):
    table, _training = run_embedding_export(build_formal_tensors(tmp_path), tmp_path)
    selector_artifacts = select_phase_b_representatives(table, seed=11)

    report = evaluate_gate7_correctness_from_artifacts(
        selector_artifacts=selector_artifacts,
        embedding_table=table,
    )

    assert report["source_gate5_embedding_table_hash"] == table["kernel_embedding_table_hash"]
    geometry = report["embedding_geometry_metrics"]
    assert geometry["silhouette"] is not None
    assert geometry["intra_distance_mean"] is not None
    assert geometry["inter_distance_mean"] is not None
