"""Gate 7 report-only correctness evaluation for GCL ResNet-50 clusters."""

from __future__ import annotations

from statistics import median
from typing import Any

from .utils import stable_hash


def evaluate_gate7_correctness(
    selector_artifacts: dict[str, Any],
    *,
    embedding_geometry: dict[str, float] | None = None,
    metric_rows: list[dict[str, Any]] | None = None,
    stability_claim: str | None = None,
) -> dict[str, Any]:
    if selector_artifacts.get("artifact_status") == "debug_not_formal":
        raise ValueError("debug selector artifacts cannot enter Gate7 formal evaluation")
    if selector_artifacts.get("artifact_type") != "gcl_resnet50_gate6_selector_artifacts":
        raise ValueError("Gate7 requires Gate6 selector artifacts")
    if stability_claim == "stable":
        raise ValueError("single-run evaluation cannot claim stable cluster assignments")
    assignments = list(selector_artifacts["kmeans_cluster_assignment_table"]["assignments"])
    anchors = list(selector_artifacts["representative_anchor_table"]["anchors"])
    family_report = selector_artifacts.get("cluster_family_evidence_report", {})
    report = {
        "artifact_type": "gcl_resnet50_gate7_correctness_manifest",
        "artifact_version": "gate7_correctness_manifest_v1",
        "threshold_policy": "report_only_v1",
        "source_gate6_selector_manifest_hash": selector_artifacts.get("selector_manifest_hash"),
        "source_assignment_hash": stable_hash(assignments),
        "embedding_geometry_metrics": _embedding_geometry_metrics(embedding_geometry or {}),
        "family_alignment_metrics": _family_alignment_metrics(family_report),
        "representative_quality_metrics": _representative_quality_metrics(anchors),
        "metric_error_report": _metric_error_report(metric_rows or []),
        "stability_report": {
            "stability_status": "single_run_not_evaluated",
            "assignment_stability_ari": None,
            "k_stability": None,
            "centroid_drift": None,
            "representative_stability_rate": None,
        },
        "assignment_count": len(assignments),
    }
    report["gate7_correctness_manifest_hash"] = stable_hash(report)
    return report


def _embedding_geometry_metrics(values: dict[str, float]) -> dict[str, float | None]:
    intra = _float_or_none(values.get("intra_distance_mean"))
    inter = _float_or_none(values.get("inter_distance_mean"))
    ratio = None
    if intra is not None and inter is not None and intra != 0:
        ratio = round(inter / intra, 8)
    return {
        "silhouette": _float_or_none(values.get("silhouette")),
        "davies_bouldin": _float_or_none(values.get("davies_bouldin")),
        "calinski_harabasz": _float_or_none(values.get("calinski_harabasz")),
        "intra_distance_mean": intra,
        "inter_distance_mean": inter,
        "inter_intra_ratio": ratio,
    }


def _family_alignment_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    clusters = report.get("clusters", [])
    if not clusters:
        return {
            "cluster_purity": None,
            "weighted_purity": None,
            "ari": None,
            "nmi": None,
        }
    purities = [float(cluster.get("purity", 0.0)) for cluster in clusters]
    weights = [float(cluster.get("weight", 1.0)) for cluster in clusters]
    total_weight = sum(weights) or 1.0
    return {
        "cluster_purity": round(sum(purities) / len(purities), 8),
        "weighted_purity": round(
            sum(purity * weight for purity, weight in zip(purities, weights)) / total_weight,
            8,
        ),
        "ari": report.get("ari"),
        "nmi": report.get("nmi"),
    }


def _representative_quality_metrics(anchors: list[dict[str, Any]]) -> dict[str, float | int | None]:
    distances = sorted(float(anchor.get("distance_to_centroid", 0.0)) for anchor in anchors)
    if not distances:
        return {
            "representative_p95_distance": None,
            "outlier_ratio": None,
            "high_weight_outlier_count": 0,
        }
    p95_index = min(len(distances) - 1, int(round((len(distances) - 1) * 0.95)))
    p95 = distances[p95_index]
    threshold = max(p95, median(distances) * 3.0)
    outliers = [distance for distance in distances if distance > threshold]
    return {
        "representative_p95_distance": round(p95, 8),
        "outlier_ratio": round(len(outliers) / len(distances), 8),
        "high_weight_outlier_count": 0,
    }


def _metric_error_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "status": "not_provided",
            "cluster_weighted_mape": {},
            "cluster_p95_relative_error": {},
            "global_weighted_mape": None,
            "high_weight_bad_cluster_count": 0,
        }
    units = {row.get("unit") for row in rows if row.get("unit") is not None}
    if len(units) > 1:
        return {
            "status": "metric_unit_conflict",
            "cluster_weighted_mape": {},
            "cluster_p95_relative_error": {},
            "global_weighted_mape": None,
            "high_weight_bad_cluster_count": 0,
        }
    weighted_errors = []
    total_weight = 0.0
    cluster_errors: dict[str, list[float]] = {}
    for row in rows:
        measured = float(row["measured"])
        predicted = float(row["predicted"])
        weight = float(row.get("weight", 1.0))
        error = abs(predicted - measured) / measured if measured else 0.0
        cluster_id = str(row["cluster_id"])
        cluster_errors.setdefault(cluster_id, []).append(error)
        weighted_errors.append(error * weight)
        total_weight += weight
    cluster_mape = {
        cluster_id: round(sum(errors) / len(errors), 8)
        for cluster_id, errors in cluster_errors.items()
    }
    cluster_p95 = {
        cluster_id: round(sorted(errors)[min(len(errors) - 1, int(round((len(errors) - 1) * 0.95)))], 8)
        for cluster_id, errors in cluster_errors.items()
    }
    return {
        "status": "reported",
        "cluster_weighted_mape": cluster_mape,
        "cluster_p95_relative_error": cluster_p95,
        "global_weighted_mape": round(sum(weighted_errors) / (total_weight or 1.0), 8),
        "high_weight_bad_cluster_count": 0,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
