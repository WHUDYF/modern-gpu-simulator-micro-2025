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
    for row in table.get("embeddings", []):
        if row.get("resource_blocked"):
            raise ValueError("resource-blocked embedding rows cannot enter M0 selector")
    validate_phase_b_embedding_table(table)
    rows = _embedding_rows(table)
    if len(rows) == 1:
        row = rows[0]
        artifact = {
            "artifact_type": "gcl_m0_selector_artifacts",
            "representation_mode": REPRESENTATION_MODE,
            "normalization": {
                "mode": "z_score",
                "embedding_dim": table["embedding_dim"],
            },
            "silhouette_report": {
                "mode": "silhouette_k",
                "selected_k": 1,
                "selected_score": 0.0,
                "candidates": [{"k": 1, "score": 0.0}],
                "fallback_reason": "single_embedding_batch",
            },
            "cluster_assignments": [
                {
                    "record_id": row["record_id"],
                    "kernel_invocation_id": row["kernel_invocation_id"],
                    "cluster_id": 0,
                }
            ],
            "representative_anchor_table": [
                {
                    "cluster_id": 0,
                    "representative_record_id": row["record_id"],
                    "kernel_invocation_id": row["kernel_invocation_id"],
                    "distance_to_centroid": 0.0,
                }
            ],
            "structural_evaluation_artifacts": {
                "row_count": len(rows),
                "cluster_count": 1,
                "anchor_count": 1,
                "seed": seed,
            },
            "source_embedding_table_hash": _embedding_table_hash(table),
        }
        artifact["selector_manifest_hash"] = hash_without(artifact, "selector_manifest_hash")
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
        "artifact_type": "gcl_m0_selector_artifacts",
        "representation_mode": REPRESENTATION_MODE,
        "normalization": {
            "mode": "z_score",
            "embedding_dim": EMBEDDING_DIM,
        },
        "silhouette_report": {
            "mode": "silhouette_k",
            "selected_k": silhouette["selected_k"],
            "selected_score": round(float(silhouette["score"]), 8),
            "candidates": [
                {"k": item["k"], "score": round(float(item["score"]), 8)}
                for item in silhouette["candidates"]
            ],
        },
        "cluster_assignments": assignments,
        "representative_anchor_table": anchors,
        "structural_evaluation_artifacts": {
            "row_count": len(rows),
            "cluster_count": int(silhouette["selected_k"]),
            "anchor_count": len(anchors),
            "seed": seed,
        },
        "source_embedding_table_hash": _embedding_table_hash(table),
    }
    artifact["selector_manifest_hash"] = hash_without(artifact, "selector_manifest_hash")
    return artifact
