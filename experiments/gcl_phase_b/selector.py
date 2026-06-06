"""M0 selector integration for Phase B embedding tables."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.gcl_phase_a.selector import choose_silhouette_k, zscore_normalize
from experiments.gcl_phase_a.utils import hash_without
from .embedding_export import EMBEDDING_DIM, REPRESENTATION_MODE, validate_phase_b_embedding_table


def _embedding_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    return table["embeddings"]


def _embedding_table_hash(table: dict[str, Any]) -> str:
    return table["kernel_embedding_table_hash"]


def select_phase_b_representatives(table: dict[str, Any], seed: int = 20260602) -> dict[str, Any]:
    _validate_gate6_input_table(table)
    for row in table.get("embeddings", []):
        if row.get("resource_blocked"):
            raise ValueError("resource-blocked embedding rows cannot enter M0 selector")
    validate_phase_b_embedding_table(table)
    rows = _embedding_rows(table)
    if len(rows) == 1:
        row = rows[0]
        artifact = {
            "artifact_type": "gcl_resnet50_gate6_selector_artifacts",
            "artifact_version": "gate6_selector_artifacts_v1",
            "representation_mode": REPRESENTATION_MODE,
            "embedding_normalization_report": {
                "artifact_type": "gcl_resnet50_embedding_normalization_report",
                "normalization_policy": "engineering_default_z_score",
                "paper_defined": False,
                "embedding_dim": table["embedding_dim"],
                "input_fields": ["kernel_embedding"],
            },
            "k_selection_report": {
                "artifact_type": "gcl_resnet50_k_selection_report",
                "mode": "silhouette_k",
                "selected_k": 1,
                "selected_score": 0.0,
                "candidates": [{"k": 1, "score": 0.0}],
                "fallback_reason": "single_embedding_batch",
            },
            "kmeans_cluster_assignment_table": {
                "artifact_type": "gcl_resnet50_kmeans_cluster_assignment_table",
                "algorithm": "deterministic_kmeans",
                "assignments": [
                    {
                        "record_id": row["record_id"],
                        "kernel_invocation_id": row["kernel_invocation_id"],
                        "cluster_id": 0,
                    }
                ],
            },
            "representative_anchor_table": {
                "artifact_type": "gcl_resnet50_representative_anchor_table",
                "anchors": [
                    {
                        "cluster_id": 0,
                        "representative_record_id": row["record_id"],
                        "kernel_invocation_id": row["kernel_invocation_id"],
                        "distance_to_centroid": 0.0,
                    }
                ],
            },
            "cluster_family_evidence_report": {
                "artifact_type": "gcl_resnet50_cluster_family_evidence_report",
                "family_labels_used_for_clustering": False,
                "evidence_mode": "post_clustering_only",
                "clusters": [],
            },
            "structural_evaluation_artifacts": {
                "row_count": len(rows),
                "cluster_count": 1,
                "anchor_count": 1,
                "seed": seed,
            },
            "source_embedding_table_hash": _embedding_table_hash(table),
        }
        artifact["selector_manifest_hash"] = hash_without(artifact, "selector_manifest_hash")
        validate_gate6_selector_artifacts(artifact)
        return artifact
    return _select_representatives(table, seed=seed)


def _select_representatives(table: dict[str, Any], seed: int) -> dict[str, Any]:
    rows = _embedding_rows(table)
    matrix = np.asarray([row["kernel_embedding"] for row in rows], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM:
        raise ValueError("embedding_dim mismatch")
    normalized = zscore_normalize(matrix)
    silhouette = choose_silhouette_k(normalized)
    labels = silhouette["labels"]
    centroids = silhouette["centroids"]
    assignments = []
    anchors = []
    for row, cluster_id in zip(rows, labels):
        assignments.append(
            {
                "record_id": row["record_id"],
                "kernel_invocation_id": row["kernel_invocation_id"],
                "cluster_id": int(cluster_id),
            }
        )
    for cluster_id in sorted(set(labels.tolist())):
        member_indices = np.flatnonzero(labels == cluster_id)
        distances = np.linalg.norm(normalized[member_indices] - centroids[cluster_id], axis=1)
        selected_index = int(member_indices[int(np.argmin(distances))])
        selected_row = rows[selected_index]
        anchors.append(
            {
                "cluster_id": int(cluster_id),
                "representative_record_id": selected_row["record_id"],
                "kernel_invocation_id": selected_row["kernel_invocation_id"],
                "distance_to_centroid": round(float(np.min(distances)), 8),
            }
        )
    artifact = {
        "artifact_type": "gcl_resnet50_gate6_selector_artifacts",
        "artifact_version": "gate6_selector_artifacts_v1",
        "representation_mode": REPRESENTATION_MODE,
        "embedding_normalization_report": {
            "artifact_type": "gcl_resnet50_embedding_normalization_report",
            "normalization_policy": "engineering_default_z_score",
            "paper_defined": False,
            "embedding_dim": EMBEDDING_DIM,
            "input_fields": ["kernel_embedding"],
        },
        "k_selection_report": {
            "artifact_type": "gcl_resnet50_k_selection_report",
            "mode": "silhouette_k",
            "selected_k": silhouette["selected_k"],
            "selected_score": round(float(silhouette["score"]), 8),
            "candidates": [
                {"k": item["k"], "score": round(float(item["score"]), 8)}
                for item in silhouette["candidates"]
            ],
        },
        "kmeans_cluster_assignment_table": {
            "artifact_type": "gcl_resnet50_kmeans_cluster_assignment_table",
            "algorithm": "deterministic_kmeans",
            "assignments": assignments,
        },
        "representative_anchor_table": {
            "artifact_type": "gcl_resnet50_representative_anchor_table",
            "anchors": anchors,
        },
        "cluster_family_evidence_report": {
            "artifact_type": "gcl_resnet50_cluster_family_evidence_report",
            "family_labels_used_for_clustering": False,
            "evidence_mode": "post_clustering_only",
            "clusters": [],
        },
        "structural_evaluation_artifacts": {
            "row_count": len(rows),
            "cluster_count": int(silhouette["selected_k"]),
            "anchor_count": len(anchors),
            "seed": seed,
        },
        "source_embedding_table_hash": _embedding_table_hash(table),
    }
    artifact["selector_manifest_hash"] = hash_without(artifact, "selector_manifest_hash")
    validate_gate6_selector_artifacts(artifact)
    return artifact


def _validate_gate6_input_table(table: dict[str, Any]) -> None:
    if table.get("artifact_status") == "debug_not_formal":
        raise ValueError("Gate6 selector requires formal embedding table")
    if table.get("family_labels_used_for_clustering") is True:
        raise ValueError("family labels cannot guide Gate6 clustering")
    forbidden = {"kernel_name", "family_label", "runtime", "graph_size", "weight_input"}
    fields = set(table.get("clustering_input_fields", ["kernel_embedding"]))
    blocked = sorted(fields.intersection(forbidden))
    if blocked:
        raise ValueError(f"forbidden clustering field: {blocked}")
    for row in table.get("embeddings", []):
        if row.get("source_view") == "augmented":
            raise ValueError("selector embedding must come from canonical non-augmented graph")
        if row.get("embedding_dim") != EMBEDDING_DIM or len(row.get("kernel_embedding", [])) != EMBEDDING_DIM:
            raise ValueError("Gate6 selector requires 256-dimensional canonical embeddings")


def validate_gate6_selector_artifacts(artifact: dict[str, Any]) -> None:
    if artifact.get("artifact_type") != "gcl_resnet50_gate6_selector_artifacts":
        raise ValueError("unexpected Gate6 selector artifact_type")
    if artifact.get("artifact_version") != "gate6_selector_artifacts_v1":
        raise ValueError("unexpected Gate6 selector artifact_version")
    normalization = artifact.get("embedding_normalization_report", {})
    if normalization.get("normalization_policy") != "engineering_default_z_score":
        raise ValueError("unexpected Gate6 normalization_policy")
    if normalization.get("paper_defined") is not False:
        raise ValueError("Gate6 normalization paper_defined must be false")
    if normalization.get("input_fields") != ["kernel_embedding"]:
        raise ValueError("Gate6 normalization must only use kernel_embedding")
    if artifact.get("k_selection_report", {}).get("mode") != "silhouette_k":
        raise ValueError("Gate6 K selection must use silhouette_k")
    assignments = artifact.get("kmeans_cluster_assignment_table", {}).get("assignments")
    if not assignments:
        raise ValueError("Gate6 assignments must be non-empty")
    anchors = artifact.get("representative_anchor_table", {}).get("anchors")
    if not anchors:
        raise ValueError("Gate6 anchors must be non-empty")
    family_report = artifact.get("cluster_family_evidence_report", {})
    if family_report.get("family_labels_used_for_clustering") is not False:
        raise ValueError("family evidence must be post-clustering only")
    if artifact.get("selector_manifest_hash") != hash_without(artifact, "selector_manifest_hash"):
        raise ValueError("selector_manifest_hash is not reproducible")
