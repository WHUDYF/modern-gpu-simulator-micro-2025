"""Deterministic M0 selector over GCL Phase A kernel embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from .embedding_export import EMBEDDING_DIM, REPRESENTATION_MODE, validate_embedding_table
from .utils import hash_without


def _embedding_matrix(table: dict[str, Any]) -> np.ndarray:
    validate_embedding_table(table)
    matrix = np.asarray([row["embedding"] for row in table["rows"]], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM:
        raise ValueError("embedding_dim mismatch")
    return matrix


def zscore_normalize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (matrix - mean) / std


def _initial_centroids(matrix: np.ndarray, k: int) -> np.ndarray:
    centroids = [matrix[0]]
    selected_indices = {0}
    while len(centroids) < k:
        distances = np.min(
            np.stack([np.linalg.norm(matrix - centroid, axis=1) for centroid in centroids], axis=1),
            axis=1,
        )
        distances[list(selected_indices)] = -1.0
        next_index = int(np.argmax(distances))
        if distances[next_index] <= 0.0:
            break
        selected_indices.add(next_index)
        centroids.append(matrix[next_index])
    return np.asarray(centroids, dtype=np.float64)


def deterministic_kmeans(matrix: np.ndarray, k: int, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if k < 1 or k > matrix.shape[0]:
        raise ValueError("invalid k for deterministic K-Means")
    centroids = _initial_centroids(matrix, k)
    k = centroids.shape[0]
    labels = np.zeros(matrix.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        distances = np.stack([np.linalg.norm(matrix - centroid, axis=1) for centroid in centroids], axis=1)
        new_labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for cluster_id in range(k):
            members = matrix[new_labels == cluster_id]
            if members.size:
                new_centroids[cluster_id] = members.mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids):
            break
        labels = new_labels
        centroids = new_centroids
    return labels, centroids


def silhouette_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = sorted(set(labels.tolist()))
    if len(unique_labels) < 2 or len(unique_labels) >= matrix.shape[0]:
        return 0.0
    scores = []
    for idx, point in enumerate(matrix):
        same_cluster = matrix[labels == labels[idx]]
        if same_cluster.shape[0] <= 1:
            a_distance = 0.0
        else:
            same_indices = np.flatnonzero(labels == labels[idx])
            peer_indices = same_indices[same_indices != idx]
            a_distance = float(np.mean(np.linalg.norm(matrix[peer_indices] - point, axis=1)))
        b_distance = min(
            float(np.mean(np.linalg.norm(matrix[labels == other_label] - point, axis=1)))
            for other_label in unique_labels
            if other_label != labels[idx]
        )
        denominator = max(a_distance, b_distance)
        scores.append(0.0 if denominator == 0.0 else (b_distance - a_distance) / denominator)
    return float(np.mean(scores))


def choose_silhouette_k(matrix: np.ndarray, k_min: int = 2, k_max: int | None = None) -> dict[str, Any]:
    if matrix.shape[0] < 2:
        raise ValueError("silhouette_k requires at least two embeddings")
    unique_count = np.unique(matrix, axis=0).shape[0]
    if unique_count == 1:
        labels = np.zeros(matrix.shape[0], dtype=np.int64)
        centroids = matrix[:1].copy()
        return {
            "selected_k": 1,
            "score": 0.0,
            "candidates": [{"k": 1, "score": 0.0}],
            "labels": labels,
            "centroids": centroids,
        }
    upper = min(k_max or 6, matrix.shape[0] - 1)
    upper = min(upper, unique_count)
    if upper < k_min:
        upper = k_min
    candidates = []
    for k in range(k_min, upper + 1):
        labels, centroids = deterministic_kmeans(matrix, k)
        candidates.append(
            {
                "k": k,
                "score": silhouette_score(matrix, labels),
                "labels": labels,
                "centroids": centroids,
            }
        )
    best = max(candidates, key=lambda candidate: (candidate["score"], -candidate["k"]))
    return {
        "selected_k": best["k"],
        "score": best["score"],
        "candidates": [{"k": item["k"], "score": item["score"]} for item in candidates],
        "labels": best["labels"],
        "centroids": best["centroids"],
    }


def select_representatives(table: dict[str, Any], seed: int = 20260602) -> dict[str, Any]:
    if table.get("row_count", 0) == 0:
        raise ValueError("embedding table row count must be greater than zero")
    if table.get("embedding_dim") != EMBEDDING_DIM:
        raise ValueError("embedding_dim mismatch")
    matrix = _embedding_matrix(table)
    normalized = zscore_normalize(matrix)
    silhouette = choose_silhouette_k(normalized)
    labels = silhouette["labels"]
    centroids = silhouette["centroids"]
    assignments = []
    anchors = []
    for row, cluster_id in zip(table["rows"], labels):
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
        selected_row = table["rows"][selected_index]
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
            "row_count": table["row_count"],
            "cluster_count": int(silhouette["selected_k"]),
            "anchor_count": len(anchors),
            "seed": seed,
        },
        "source_embedding_table_hash": table["embedding_table_hash"],
    }
    artifact["selector_manifest_hash"] = hash_without(artifact, "selector_manifest_hash")
    return artifact
