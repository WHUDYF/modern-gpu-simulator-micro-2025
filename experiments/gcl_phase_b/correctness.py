"""Gate 7 report-only correctness evaluation for GCL ResNet-50 clusters."""

from __future__ import annotations

from statistics import median
from typing import Any

import numpy as np

from .embedding_export import validate_phase_b_embedding_table
from .utils import stable_hash

GATE7_CLUSTER_CORRECTNESS_FILENAME = "gate7_cluster_correctness_manifest.json"
GATE7_CLUSTER_CORRECTNESS_TYPE = "gcl_resnet50_gate7_cluster_correctness_manifest"
GATE7_CLUSTER_CORRECTNESS_VERSION = "gate7_cluster_correctness_manifest_v1"
GATE7_REPORT_ONLY_CLAIM_STATUS = "quantified_no_correctness_claim"
GATE7_REPORT_FILENAMES = {
    "embedding_quality_report": "cluster_embedding_quality_report.json",
    "family_alignment_report": "cluster_family_alignment_report.json",
    "representative_quality_report": "representative_quality_report.json",
    "metric_error_report": "cluster_metric_error_report.json",
    "stability_report": "cluster_stability_report.json",
}


def evaluate_gate7_correctness(
    selector_artifacts: dict[str, Any],
    *,
    embedding_geometry: dict[str, float] | None = None,
    representative_quality: dict[str, Any] | None = None,
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
    embedding_metrics = _embedding_geometry_metrics(embedding_geometry or {})
    family_metrics = _family_alignment_metrics(family_report)
    representative_metrics = representative_quality or _representative_quality_metrics(anchors)
    metric_report = _metric_error_report(metric_rows or [])
    stability_report = {
        "stability_status": "single_run_not_evaluated",
        "assignment_stability_ari": None,
        "k_stability": None,
        "centroid_drift": None,
        "representative_stability_rate": None,
    }
    report_artifacts = build_gate7_report_artifacts(
        embedding_quality=embedding_metrics,
        family_alignment=family_metrics,
        representative_quality=representative_metrics,
        metric_error=metric_report,
        stability=stability_report,
    )
    report = {
        "artifact_type": GATE7_CLUSTER_CORRECTNESS_TYPE,
        "artifact_version": GATE7_CLUSTER_CORRECTNESS_VERSION,
        "threshold_policy": "report_only_v1",
        "threshold_claim_status": "not_set_until_real_resnet50_baseline",
        "suggested_min_silhouette_score": None,
        "suggested_min_weighted_cluster_purity": None,
        "suggested_max_global_weighted_mape": None,
        "suggested_min_assignment_stability_ari": None,
        "claim_status": GATE7_REPORT_ONLY_CLAIM_STATUS,
        "source_gate6_selector_manifest_hash": selector_artifacts.get("selector_manifest_hash"),
        "source_cluster_assignment_table_hash": stable_hash(
            selector_artifacts["kmeans_cluster_assignment_table"]
        ),
        "source_representative_anchor_table_hash": stable_hash(
            selector_artifacts["representative_anchor_table"]
        ),
        "source_embedding_table_hash": selector_artifacts.get("source_embedding_table_hash"),
        "source_gate5_embedding_table_hash": None,
        "metric_source_manifest_hash": stable_hash(
            {
                "artifact_type": "gcl_resnet50_gate7_metric_source_manifest",
                "metric_rows": metric_rows or [],
            }
        )
        if metric_rows is not None
        else None,
        "family_label_source_hash": stable_hash(family_report) if family_report else None,
        "structural_summary_source_hash": stable_hash(
            selector_artifacts.get("structural_evaluation_artifacts", {})
        ),
        "source_assignment_hash": stable_hash(assignments),
        "embedding_geometry_metrics": embedding_metrics,
        "family_alignment_metrics": family_metrics,
        "representative_quality_metrics": representative_metrics,
        "metric_error_report": metric_report,
        "stability_report": stability_report,
        "embedding_quality_report_hash": report_artifacts["embedding_quality_report"][
            "report_hash"
        ],
        "family_alignment_report_hash": report_artifacts["family_alignment_report"][
            "report_hash"
        ],
        "representative_quality_report_hash": report_artifacts["representative_quality_report"][
            "report_hash"
        ],
        "metric_error_report_hash": report_artifacts["metric_error_report"]["report_hash"],
        "stability_report_hash": report_artifacts["stability_report"]["report_hash"],
        "gate7_report_artifacts": report_artifacts,
        "assignment_count": len(assignments),
    }
    report["gate7_cluster_correctness_manifest_hash"] = stable_hash(report)
    report["gate7_correctness_manifest_hash"] = report["gate7_cluster_correctness_manifest_hash"]
    return report


def build_gate7_report_artifacts(
    *,
    embedding_quality: dict[str, Any],
    family_alignment: dict[str, Any],
    representative_quality: dict[str, Any],
    metric_error: dict[str, Any],
    stability: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "embedding_quality_report": _gate7_report_artifact(
            "gcl_resnet50_cluster_embedding_quality_report",
            embedding_quality,
        ),
        "family_alignment_report": _gate7_report_artifact(
            "gcl_resnet50_cluster_family_alignment_report",
            family_alignment,
        ),
        "representative_quality_report": _gate7_report_artifact(
            "gcl_resnet50_representative_quality_report",
            representative_quality,
        ),
        "metric_error_report": _gate7_report_artifact(
            "gcl_resnet50_cluster_metric_error_report",
            metric_error,
        ),
        "stability_report": _gate7_report_artifact(
            "gcl_resnet50_cluster_stability_report",
            stability,
        ),
    }


def _gate7_report_artifact(artifact_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    artifact = {
        "artifact_type": artifact_type,
        "artifact_version": "gate7_report_artifact_v1",
        "report_payload": payload,
    }
    artifact["report_hash"] = stable_hash(artifact)
    return artifact


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
    representative_quality = _compute_representative_quality(selector_artifacts, embedding_table)
    report = evaluate_gate7_correctness(
        selector_artifacts,
        embedding_geometry=geometry,
        representative_quality=representative_quality,
        metric_rows=metric_rows,
    )
    report["source_gate5_embedding_table_hash"] = embedding_table["kernel_embedding_table_hash"]
    report["source_embedding_table_hash"] = embedding_table["kernel_embedding_table_hash"]
    report["gate7_cluster_correctness_manifest_hash"] = stable_hash(
        {key: value for key, value in report.items() if key != "gate7_correctness_manifest_hash"}
    )
    report["gate7_correctness_manifest_hash"] = report["gate7_cluster_correctness_manifest_hash"]
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


def _family_alignment_metrics(report: dict[str, Any]) -> dict[str, Any]:
    clusters = report.get("clusters", [])
    empty = {
        "family_evidence_status": "unavailable",
        "family_alignment_claim_status": "no_family_claim",
        "cluster_purity": None,
        "weighted_purity": None,
        "ari": None,
        "nmi": None,
        "homogeneity": None,
        "completeness": None,
        "v_measure": None,
        "family_to_cluster_coverage": {},
        "mixed_family_cluster_count": 0,
        "high_weight_mixed_family_cluster_count": 0,
        "unlabeled_record_count": 0,
    }
    if not clusters:
        return empty
    purities = [float(cluster.get("purity", 0.0)) for cluster in clusters]
    weights = [float(cluster.get("weight", 1.0)) for cluster in clusters]
    total_weight = sum(weights) or 1.0
    members = list(report.get("members", []))
    labeled_members = [member for member in members if member.get("family")]
    cluster_labels = [int(member["cluster_id"]) for member in labeled_members]
    family_labels = [str(member["family"]) for member in labeled_members]
    unlabeled = len([member for member in members if not member.get("family")])
    ari = nmi = homogeneity = completeness = v_measure = None
    coverage: dict[str, Any] = {}
    mixed_family_cluster_count = 0
    high_weight_mixed_family_cluster_count = 0
    has_complete_member_labels = bool(members) and unlabeled == 0 and len(family_labels) == len(members)
    if has_complete_member_labels:
        ari = _adjusted_rand_index(family_labels, cluster_labels)
        nmi, homogeneity, completeness, v_measure = _normalized_mutual_information(
            family_labels,
            cluster_labels,
        )
        coverage = _family_to_cluster_coverage(members)
        mixed_clusters = _mixed_family_clusters(members)
        mixed_family_cluster_count = len(mixed_clusters)
        cluster_weight_by_id = {
            int(cluster["cluster_id"]): float(cluster.get("weight", 0.0)) for cluster in clusters
        }
        high_weight_mixed_family_cluster_count = sum(
            1 for cluster_id in mixed_clusters if cluster_weight_by_id.get(cluster_id, 0.0) >= 0.5
        )
    family_evidence_status = "available" if has_complete_member_labels else "unavailable"
    family_alignment_claim_status = (
        "reported" if has_complete_member_labels else "no_family_claim"
    )
    return {
        "family_evidence_status": family_evidence_status,
        "family_alignment_claim_status": family_alignment_claim_status,
        "cluster_purity": round(sum(purities) / len(purities), 8),
        "weighted_purity": round(
            sum(purity * weight for purity, weight in zip(purities, weights)) / total_weight,
            8,
        ),
        "ari": ari,
        "nmi": nmi,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "family_to_cluster_coverage": coverage,
        "mixed_family_cluster_count": mixed_family_cluster_count,
        "high_weight_mixed_family_cluster_count": high_weight_mixed_family_cluster_count,
        "unlabeled_record_count": unlabeled,
    }


def _representative_quality_metrics(anchors: list[dict[str, Any]]) -> dict[str, float | int | None]:
    distances = sorted(float(anchor.get("distance_to_centroid", 0.0)) for anchor in anchors)
    if not distances:
        return {
            "representative_p95_distance": None,
            "outlier_ratio": None,
            "high_weight_outlier_count": 0,
            "representative_quality_status": "not_applicable_no_anchors",
            "cluster_reports": [],
        }
    p95_index = min(len(distances) - 1, int(round((len(distances) - 1) * 0.95)))
    p95 = distances[p95_index]
    threshold = max(p95, median(distances) * 3.0)
    outliers = [distance for distance in distances if distance > threshold]
    return {
        "representative_p95_distance": round(p95, 8),
        "outlier_ratio": round(len(outliers) / len(distances), 8),
        "high_weight_outlier_count": 0,
        "representative_quality_status": "reported_anchor_centroid_only",
        "cluster_reports": [],
    }


def _metric_error_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "status": "not_provided",
            "metric_claim_status": "unavailable",
            "complete_row_count": 0,
            "missing_metric_row_count": 0,
            "missing_metric_rows": [],
            "cluster_weighted_mape": {},
            "cluster_p95_relative_error": {},
            "cluster_max_relative_error": {},
            "cluster_metric_correlation": {},
            "cluster_metric_rank_correlation": {},
            "global_weighted_mape": None,
            "global_p95_relative_error": None,
            "global_max_relative_error": None,
            "bad_cluster_count": 0,
            "high_error_member_count": 0,
            "high_weight_high_error_member_count": 0,
            "high_weight_bad_cluster_count": 0,
        }
    units = {row.get("unit") for row in rows if row.get("unit") is not None}
    if len(units) > 1:
        return {
            "status": "metric_unit_conflict",
            "metric_claim_status": "unavailable",
            "complete_row_count": 0,
            "missing_metric_row_count": 0,
            "missing_metric_rows": [],
            "cluster_weighted_mape": {},
            "cluster_p95_relative_error": {},
            "cluster_max_relative_error": {},
            "cluster_metric_correlation": {},
            "cluster_metric_rank_correlation": {},
            "global_weighted_mape": None,
            "global_p95_relative_error": None,
            "global_max_relative_error": None,
            "bad_cluster_count": 0,
            "high_error_member_count": 0,
            "high_weight_high_error_member_count": 0,
            "high_weight_bad_cluster_count": 0,
        }
    weighted_error_sum = 0.0
    total_weight = 0.0
    cluster_rows: dict[str, list[dict[str, float]]] = {}
    all_errors = []
    all_weights = []
    missing_metric_rows = []
    complete_rows = []
    required_fields = ("cluster_id", "measured", "predicted")
    for row_index, row in enumerate(rows):
        missing_fields = [field for field in required_fields if row.get(field) is None]
        if missing_fields:
            missing_metric_rows.append(
                {
                    "row_index": row_index,
                    "cluster_id": str(row.get("cluster_id", "unknown")),
                    "missing_fields": missing_fields,
                }
            )
        else:
            complete_rows.append(row)
    if not complete_rows:
        return {
            "status": "metric_source_missing",
            "metric_claim_status": "unavailable",
            "complete_row_count": 0,
            "missing_metric_row_count": len(missing_metric_rows),
            "missing_metric_rows": missing_metric_rows,
            "cluster_weighted_mape": {},
            "cluster_p95_relative_error": {},
            "cluster_max_relative_error": {},
            "cluster_metric_correlation": {},
            "cluster_metric_rank_correlation": {},
            "global_weighted_mape": None,
            "global_p95_relative_error": None,
            "global_max_relative_error": None,
            "bad_cluster_count": 0,
            "high_error_member_count": 0,
            "high_weight_high_error_member_count": 0,
            "high_weight_bad_cluster_count": 0,
        }
    for row in complete_rows:
        measured = float(row["measured"])
        predicted = float(row["predicted"])
        weight = float(row.get("weight", 1.0))
        error = abs(predicted - measured) / max(abs(measured), 1e-9)
        cluster_id = str(row["cluster_id"])
        cluster_rows.setdefault(cluster_id, []).append(
            {
                "measured": measured,
                "predicted": predicted,
                "weight": weight,
                "error": error,
            }
        )
        all_errors.append(error)
        all_weights.append(weight)
        weighted_error_sum += error * weight
        total_weight += weight
    high_error_threshold = 0.2
    high_weight_threshold = _high_weight_threshold(all_weights)
    cluster_mape = {
        cluster_id: round(
            sum(item["error"] * item["weight"] for item in items)
            / (sum(item["weight"] for item in items) or 1.0),
            8,
        )
        for cluster_id, items in cluster_rows.items()
    }
    cluster_p95 = {
        cluster_id: round(_percentile([item["error"] for item in items], 0.95), 8)
        for cluster_id, items in cluster_rows.items()
    }
    cluster_max = {
        cluster_id: round(max(item["error"] for item in items), 8)
        for cluster_id, items in cluster_rows.items()
    }
    cluster_corr = {
        cluster_id: _correlation(
            [item["measured"] for item in items],
            [item["predicted"] for item in items],
        )
        for cluster_id, items in cluster_rows.items()
    }
    cluster_rank_corr = {
        cluster_id: _rank_correlation(
            [item["measured"] for item in items],
            [item["predicted"] for item in items],
        )
        for cluster_id, items in cluster_rows.items()
    }
    bad_clusters = {
        cluster_id
        for cluster_id, items in cluster_rows.items()
        if any(item["error"] > high_error_threshold for item in items)
    }
    cluster_weight_sum = {
        cluster_id: sum(item["weight"] for item in items) for cluster_id, items in cluster_rows.items()
    }
    return {
        "status": "partial_metric_missing" if missing_metric_rows else "reported",
        "metric_claim_status": "unavailable" if missing_metric_rows else "reported",
        "complete_row_count": len(complete_rows),
        "missing_metric_row_count": len(missing_metric_rows),
        "missing_metric_rows": missing_metric_rows,
        "cluster_weighted_mape": cluster_mape,
        "cluster_p95_relative_error": cluster_p95,
        "cluster_max_relative_error": cluster_max,
        "cluster_metric_correlation": cluster_corr,
        "cluster_metric_rank_correlation": cluster_rank_corr,
        "global_weighted_mape": round(weighted_error_sum / (total_weight or 1.0), 8),
        "global_p95_relative_error": round(_percentile(all_errors, 0.95), 8),
        "global_max_relative_error": round(max(all_errors), 8),
        "bad_cluster_count": len(bad_clusters),
        "high_error_member_count": sum(1 for error in all_errors if error > high_error_threshold),
        "high_weight_high_error_member_count": sum(
            1
            for item in [item for items in cluster_rows.values() for item in items]
            if item["error"] > high_error_threshold and item["weight"] >= high_weight_threshold
        ),
        "high_weight_bad_cluster_count": sum(
            1
            for cluster_id in bad_clusters
            if cluster_weight_sum.get(cluster_id, 0.0) >= _high_weight_threshold(
                list(cluster_weight_sum.values())
            )
        ),
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


def _compute_representative_quality(
    selector_artifacts: dict[str, Any],
    embedding_table: dict[str, Any],
) -> dict[str, Any]:
    rows_by_id = {row["record_id"]: row for row in embedding_table["embeddings"]}
    assignments = selector_artifacts["kmeans_cluster_assignment_table"]["assignments"]
    anchors = selector_artifacts["representative_anchor_table"]["anchors"]
    anchors_by_cluster = {int(anchor["cluster_id"]): anchor for anchor in anchors}
    assignment_by_cluster: dict[int, list[dict[str, Any]]] = {}
    for assignment in assignments:
        assignment_by_cluster.setdefault(int(assignment["cluster_id"]), []).append(assignment)
    cluster_reports = []
    all_distances = []
    high_weight_outlier_count = 0
    for cluster_id, cluster_assignments in sorted(assignment_by_cluster.items()):
        anchor = anchors_by_cluster.get(cluster_id)
        if anchor is None:
            raise ValueError("Gate7 assignment references cluster without representative anchor")
        representative_record_id = anchor["representative_record_id"]
        representative_row = rows_by_id.get(representative_record_id)
        if representative_row is None:
            raise ValueError("Gate7 representative anchor references missing embedding row")
        representative_vector = np.asarray(representative_row["kernel_embedding"], dtype=np.float64)
        member_entries = []
        member_vectors = []
        for assignment in cluster_assignments:
            row = rows_by_id.get(assignment["record_id"])
            if row is None:
                raise ValueError("Gate7 assignment references missing embedding row")
            vector = np.asarray(row["kernel_embedding"], dtype=np.float64)
            distance = float(np.linalg.norm(vector - representative_vector))
            weight = _row_weight(row)
            member_entries.append(
                {
                    "record_id": row["record_id"],
                    "kernel_invocation_id": row["kernel_invocation_id"],
                    "distance_to_representative": distance,
                    "weight": weight,
                }
            )
            member_vectors.append(vector)
        distances = [entry["distance_to_representative"] for entry in member_entries]
        weights = [entry["weight"] for entry in member_entries]
        centroid = np.asarray(member_vectors, dtype=np.float64).mean(axis=0)
        centroid_distances = [
            float(np.linalg.norm(vector - centroid)) for vector in member_vectors
        ]
        sorted_centroid = sorted(
            zip(centroid_distances, [entry["record_id"] for entry in member_entries]),
            key=lambda item: (item[0], item[1]),
        )
        representative_rank = next(
            index + 1
            for index, (_distance, record_id) in enumerate(sorted_centroid)
            if record_id == representative_record_id
        )
        mean_distance = float(np.mean(distances)) if distances else 0.0
        std_distance = float(np.std(distances)) if distances else 0.0
        outlier_threshold = mean_distance + 2.0 * std_distance
        high_weight_threshold = _high_weight_threshold(weights)
        outlier_members = [
            entry for entry in member_entries if entry["distance_to_representative"] > outlier_threshold
        ]
        high_weight_outliers = [
            entry for entry in outlier_members if entry["weight"] >= high_weight_threshold
        ]
        high_weight_outlier_count += len(high_weight_outliers)
        all_distances.extend(distances)
        cluster_reports.append(
            {
                "cluster_id": cluster_id,
                "representative_record_id": representative_record_id,
                "member_count": len(member_entries),
                "coverage_weight_sum": round(sum(weights), 8),
                "mean_distance_to_representative": round(mean_distance, 8),
                "p95_distance_to_representative": round(_percentile(distances, 0.95), 8),
                "max_distance_to_representative": round(max(distances) if distances else 0.0, 8),
                "representative_rank_to_centroid": representative_rank,
                "outlier_member_ratio": round(
                    len(outlier_members) / len(member_entries) if member_entries else 0.0,
                    8,
                ),
                "high_weight_outlier_count": len(high_weight_outliers),
                "members": [
                    {
                        **entry,
                        "distance_to_representative": round(
                            float(entry["distance_to_representative"]),
                            8,
                        ),
                        "weight": round(float(entry["weight"]), 8),
                    }
                    for entry in member_entries
                ],
            }
        )
    representative_p95 = _percentile(all_distances, 0.95) if all_distances else None
    outlier_total = sum(report["outlier_member_ratio"] * report["member_count"] for report in cluster_reports)
    member_total = sum(report["member_count"] for report in cluster_reports) or 1
    status = (
        "weak_representative"
        if any(report["outlier_member_ratio"] > 0.0 for report in cluster_reports)
        else "reported"
    )
    return {
        "representative_quality_status": status,
        "representative_p95_distance": (
            round(float(representative_p95), 8) if representative_p95 is not None else None
        ),
        "outlier_ratio": round(float(outlier_total) / float(member_total), 8),
        "high_weight_outlier_count": high_weight_outlier_count,
        "cluster_reports": cluster_reports,
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


def _adjusted_rand_index(labels_true: list[Any], labels_pred: list[Any]) -> float:
    if len(labels_true) != len(labels_pred):
        raise ValueError("ARI requires equal-length label arrays")
    n = len(labels_true)
    if n < 2:
        return 1.0
    contingency = _contingency(labels_true, labels_pred)
    sum_comb = sum(_comb2(count) for row in contingency.values() for count in row.values())
    true_counts: dict[Any, int] = {}
    pred_counts: dict[Any, int] = {}
    for true, pred in zip(labels_true, labels_pred):
        true_counts[true] = true_counts.get(true, 0) + 1
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    sum_true = sum(_comb2(count) for count in true_counts.values())
    sum_pred = sum(_comb2(count) for count in pred_counts.values())
    total = _comb2(n)
    if total == 0:
        return 1.0
    expected = sum_true * sum_pred / total
    max_index = (sum_true + sum_pred) / 2.0
    denominator = max_index - expected
    if denominator == 0.0:
        return 1.0 if sum_comb == max_index else 0.0
    return round(float((sum_comb - expected) / denominator), 8)


def _normalized_mutual_information(
    labels_true: list[Any],
    labels_pred: list[Any],
) -> tuple[float, float, float, float]:
    n = len(labels_true)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    contingency = _contingency(labels_true, labels_pred)
    true_counts: dict[Any, int] = {}
    pred_counts: dict[Any, int] = {}
    for true, pred in zip(labels_true, labels_pred):
        true_counts[true] = true_counts.get(true, 0) + 1
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    mutual_info = 0.0
    for true, row in contingency.items():
        for pred, count in row.items():
            mutual_info += (count / n) * np.log((count * n) / (true_counts[true] * pred_counts[pred]))
    entropy_true = _entropy(true_counts.values(), n)
    entropy_pred = _entropy(pred_counts.values(), n)
    nmi = 1.0 if entropy_true == 0.0 and entropy_pred == 0.0 else 0.0
    if entropy_true > 0.0 or entropy_pred > 0.0:
        denominator = (entropy_true + entropy_pred) / 2.0
        nmi = 0.0 if denominator == 0.0 else mutual_info / denominator
    homogeneity = 1.0 if entropy_true == 0.0 else mutual_info / entropy_true
    completeness = 1.0 if entropy_pred == 0.0 else mutual_info / entropy_pred
    v_measure = (
        0.0
        if homogeneity + completeness == 0.0
        else 2.0 * homogeneity * completeness / (homogeneity + completeness)
    )
    return (
        round(float(nmi), 8),
        round(float(homogeneity), 8),
        round(float(completeness), 8),
        round(float(v_measure), 8),
    )


def _contingency(labels_true: list[Any], labels_pred: list[Any]) -> dict[Any, dict[Any, int]]:
    table: dict[Any, dict[Any, int]] = {}
    for true, pred in zip(labels_true, labels_pred):
        row = table.setdefault(true, {})
        row[pred] = row.get(pred, 0) + 1
    return table


def _comb2(count: int) -> float:
    return count * (count - 1) / 2.0


def _entropy(counts: Any, total: int) -> float:
    entropy = 0.0
    for count in counts:
        if count:
            probability = count / total
            entropy -= probability * np.log(probability)
    return float(entropy)


def _family_to_cluster_coverage(members: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, dict[int, float]] = {}
    for member in members:
        family = member.get("family")
        if not family:
            continue
        cluster_id = int(member["cluster_id"])
        weight = float(member.get("weight", 1.0))
        clusters = by_family.setdefault(str(family), {})
        clusters[cluster_id] = clusters.get(cluster_id, 0.0) + weight
    coverage = {}
    for family, clusters in sorted(by_family.items()):
        total_weight = sum(clusters.values()) or 1.0
        primary_cluster, primary_weight = sorted(
            clusters.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        coverage[family] = {
            "primary_cluster_id": primary_cluster,
            "primary_cluster_weight": round(primary_weight, 8),
            "coverage_ratio": round(primary_weight / total_weight, 8),
            "cluster_count": len(clusters),
        }
    return coverage


def _mixed_family_clusters(members: list[dict[str, Any]]) -> set[int]:
    families_by_cluster: dict[int, set[str]] = {}
    for member in members:
        if member.get("family"):
            families_by_cluster.setdefault(int(member["cluster_id"]), set()).add(str(member["family"]))
    return {
        cluster_id for cluster_id, families in families_by_cluster.items() if len(families) > 1
    }


def _row_weight(row: dict[str, Any]) -> float:
    weight_input = row.get("weight_input", {})
    for key in ("runtime_weight", "measured_runtime_weight", "node_count"):
        value = weight_input.get(key)
        if value is not None:
            weight = float(value)
            if weight > 0.0:
                return weight
    return 1.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def _correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 8)


def _rank_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2:
        return None
    return _correlation(_average_ranks(x_values), _average_ranks(y_values))


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[index][1]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[indexed[position][0]] = average_rank
        index = end + 1
    return ranks


def _high_weight_threshold(weights: list[float]) -> float:
    if not weights:
        return 0.0
    return float(np.mean(weights))
