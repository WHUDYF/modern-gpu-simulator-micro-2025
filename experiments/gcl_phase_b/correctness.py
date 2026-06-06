"""Gate 7 report-only correctness evaluation for GCL ResNet-50 clusters."""

from __future__ import annotations

from statistics import median
from typing import Any

import numpy as np

from .embedding_export import validate_phase_b_embedding_table
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
        "source_gate5_embedding_table_hash": None,
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


def evaluate_gate7_correctness_from_artifacts(
    *,
    selector_artifacts: dict[str, Any],
    embedding_table: dict[str, Any],
    metric_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_phase_b_embedding_table(embedding_table)
    if selector_artifacts.get("source_embedding_table_hash") != embedding_table.get(
        "kernel_embedding_table_hash"
    ):
        raise ValueError("Gate7 selector artifacts must match Gate5 embedding table")
    geometry = _compute_embedding_geometry(selector_artifacts, embedding_table)
    report = evaluate_gate7_correctness(
        selector_artifacts,
        embedding_geometry=geometry,
        metric_rows=metric_rows,
    )
    report["source_gate5_embedding_table_hash"] = embedding_table["kernel_embedding_table_hash"]
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


def _compute_embedding_geometry(
    selector_artifacts: dict[str, Any],
    embedding_table: dict[str, Any],
) -> dict[str, float]:
    rows_by_id = {row["record_id"]: row for row in embedding_table["embeddings"]}
    assignments = selector_artifacts["kmeans_cluster_assignment_table"]["assignments"]
    vectors = []
    labels = []
    for assignment in assignments:
        row = rows_by_id.get(assignment["record_id"])
        if row is None:
            raise ValueError("Gate7 assignment references missing embedding row")
        vectors.append(row["kernel_embedding"])
        labels.append(int(assignment["cluster_id"]))
    matrix = np.asarray(vectors, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    distances = _pairwise_distances(matrix)
    intra_distances = []
    inter_distances = []
    for i in range(len(label_array)):
        for j in range(i + 1, len(label_array)):
            if label_array[i] == label_array[j]:
                intra_distances.append(float(distances[i, j]))
            else:
                inter_distances.append(float(distances[i, j]))
    return {
        "silhouette": _mean_silhouette(distances, label_array),
        "davies_bouldin": _davies_bouldin(matrix, label_array),
        "calinski_harabasz": _calinski_harabasz(matrix, label_array),
        "intra_distance_mean": float(np.mean(intra_distances)) if intra_distances else 0.0,
        "inter_distance_mean": float(np.mean(inter_distances)) if inter_distances else 0.0,
    }


def _pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    diffs = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def _mean_silhouette(distances: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2 or len(labels) < 2:
        return 0.0
    scores = []
    for index, label in enumerate(labels):
        same = [j for j, other in enumerate(labels) if other == label and j != index]
        other_labels = sorted(set(labels.tolist()) - {int(label)})
        a = float(np.mean([distances[index, j] for j in same])) if same else 0.0
        b = min(
            float(np.mean([distances[index, j] for j, other in enumerate(labels) if other == other_label]))
            for other_label in other_labels
        )
        denom = max(a, b)
        scores.append(0.0 if denom == 0.0 else (b - a) / denom)
    return round(float(np.mean(scores)), 8)


def _davies_bouldin(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2:
        return 0.0
    centroids = {label: matrix[labels == label].mean(axis=0) for label in unique}
    scatters = {
        label: float(np.mean(np.linalg.norm(matrix[labels == label] - centroids[label], axis=1)))
        for label in unique
    }
    values = []
    for label in unique:
        ratios = []
        for other in unique:
            if other == label:
                continue
            distance = float(np.linalg.norm(centroids[label] - centroids[other]))
            ratios.append(0.0 if distance == 0.0 else (scatters[label] + scatters[other]) / distance)
        values.append(max(ratios))
    return round(float(np.mean(values)), 8)


def _calinski_harabasz(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique = sorted(set(labels.tolist()))
    n_samples = len(labels)
    k = len(unique)
    if k < 2 or n_samples <= k:
        return 0.0
    global_mean = matrix.mean(axis=0)
    between = 0.0
    within = 0.0
    for label in unique:
        cluster = matrix[labels == label]
        centroid = cluster.mean(axis=0)
        between += len(cluster) * float(np.sum((centroid - global_mean) ** 2))
        within += float(np.sum((cluster - centroid) ** 2))
    if within == 0.0:
        return 0.0
    return round(float((between / (k - 1)) / (within / (n_samples - k))), 8)
