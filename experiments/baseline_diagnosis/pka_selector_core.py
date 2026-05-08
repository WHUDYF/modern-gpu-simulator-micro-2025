"""Deterministic selector core shared by PKA M0 and M1."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from shared_acquisition import COUNT_FEATURES, FEATURE_ORDER, RATIO_FEATURES, stable_hash


def validate_selector_records(records: list[dict[str, Any]], expected_feature_mode: str | None = None) -> None:
    if len(records) < 2:
        raise ValueError("selector requires at least two records")
    seen = set()
    for rec in records:
        record_id = rec.get("record_id") or rec.get("kernel_invocation_id")
        if not record_id:
            raise ValueError("selector record missing record_id/kernel_invocation_id")
        if record_id in seen:
            raise ValueError(f"duplicate selector record_id: {record_id}")
        seen.add(record_id)
        if expected_feature_mode and rec.get("feature_mode") != expected_feature_mode:
            raise ValueError(f"{record_id}: feature_mode {rec.get('feature_mode')} != {expected_feature_mode}")
        features = rec.get("features", {})
        for feature_name in FEATURE_ORDER:
            feature = features.get(feature_name)
            if not isinstance(feature, dict):
                raise ValueError(f"{record_id}: missing feature {feature_name}")
            if feature.get("status") != "measured":
                raise ValueError(f"{record_id}: feature {feature_name} is not measured")
            if not isinstance(feature.get("value"), (int, float)):
                raise ValueError(f"{record_id}: feature {feature_name} value is not numeric")


def _record_id(rec: dict[str, Any]) -> str:
    return str(rec.get("record_id") or rec.get("kernel_invocation_id"))


def build_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for rec in records:
        row = []
        for feature_name in FEATURE_ORDER:
            value = float(rec["features"][feature_name]["value"])
            if feature_name in COUNT_FEATURES:
                value = math.log1p(max(value, 0.0))
            elif feature_name in RATIO_FEATURES:
                value = min(max(value, 0.0), 1.0)
            row.append(value)
        rows.append(row)
    return np.array(rows, dtype=float)


def preprocess(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    zero_std = [FEATURE_ORDER[i] for i, std in enumerate(stds) if std == 0]
    safe_stds = np.where(stds == 0, 1.0, stds)
    normalized = (matrix - means) / safe_stds
    return normalized, {
        "feature_order": FEATURE_ORDER,
        "log1p_features": sorted(COUNT_FEATURES),
        "ratio_features": sorted(RATIO_FEATURES),
        "mean": means.tolist(),
        "std_deviation": safe_stds.tolist(),
        "zero_std_features": zero_std,
    }


def run_pca(matrix: np.ndarray, max_components: int = 3) -> tuple[np.ndarray, dict[str, Any]]:
    if matrix.shape[0] < 2:
        raise ValueError("pca requires at least two rows")
    centered = matrix - matrix.mean(axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(s ** 2))
    if total <= 0:
        raise ValueError("pca_degenerate_input")
    n_components = min(max_components, matrix.shape[0] - 1, matrix.shape[1], len(s))
    projection = u[:, :n_components] * s[:n_components]
    explained = ((s[:n_components] ** 2) / total).tolist()
    return projection, {
        "method": "numpy_svd",
        "n_components": n_components,
        "explained_variance_ratio": explained,
        "total_explained_variance": float(sum(explained)),
        "component_matrix": vt[:n_components, :].tolist(),
    }


def _dist2(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.dot(diff, diff))


def _choose_k(n_records: int) -> int:
    if n_records < 2:
        return n_records
    return max(2, min(n_records, math.ceil(math.sqrt(n_records))))


def farthest_first_kmeans(
    projection: np.ndarray,
    record_ids: list[str],
    k: int | None = None,
    max_iter: int = 300,
) -> tuple[list[int], list[list[float]], dict[str, Any]]:
    n_records = len(record_ids)
    if n_records != len(projection):
        raise ValueError("record_ids length does not match projection")
    k = _choose_k(n_records) if k is None else max(1, min(k, n_records))
    first = min(range(n_records), key=lambda idx: record_ids[idx])
    center_indices = [first]
    while len(center_indices) < k:
        candidates = []
        for idx in range(n_records):
            if idx in center_indices:
                continue
            nearest = min(_dist2(projection[idx], projection[c]) for c in center_indices)
            candidates.append((-nearest, record_ids[idx], idx))
        center_indices.append(sorted(candidates)[0][2])
    centers = projection[center_indices].copy()
    assignments = [0] * n_records
    iterations = 0
    for iterations in range(1, max_iter + 1):
        changed = False
        for idx in range(n_records):
            best = min(
                range(k),
                key=lambda c: (_dist2(projection[idx], centers[c]), record_ids[idx], c),
            )
            if assignments[idx] != best:
                assignments[idx] = best
                changed = True
        new_centers = centers.copy()
        for c in range(k):
            members = [idx for idx, assigned in enumerate(assignments) if assigned == c]
            if members:
                new_centers[c] = projection[members].mean(axis=0)
        centers = new_centers
        if not changed:
            break
    return assignments, centers.tolist(), {
        "method": "deterministic_farthest_first",
        "k": k,
        "initial_center_record_ids": [record_ids[idx] for idx in center_indices],
        "max_iter": max_iter,
        "iterations": iterations,
        "distance": "squared_euclidean_in_pca_space",
    }


def build_outputs(records: list[dict[str, Any]], mode: str, feature_mode: str) -> dict[str, Any]:
    validate_selector_records(records, expected_feature_mode=feature_mode)
    record_ids = [_record_id(rec) for rec in records]
    matrix = build_matrix(records)
    normalized, preprocessing = preprocess(matrix)
    projection, pca_meta = run_pca(normalized)
    assignments, centers, kmeans_meta = farthest_first_kmeans(projection, record_ids)

    clusters: dict[int, list[int]] = {}
    for idx, cluster_id in enumerate(assignments):
        clusters.setdefault(cluster_id, []).append(idx)

    projection_rows = []
    for idx, rec in enumerate(records):
        projection_rows.append({
            "mode": mode,
            "feature_mode": feature_mode,
            "record_id": record_ids[idx],
            "kernel_invocation_id": rec.get("kernel_invocation_id"),
            "coordinates": projection[idx].tolist(),
        })

    cluster_rows = []
    anchor_rows = []
    for ordinal, cluster_id in enumerate(sorted(clusters), 1):
        member_indices = clusters[cluster_id]
        center = np.array(centers[cluster_id])
        representative_idx = min(
            member_indices,
            key=lambda idx: (_dist2(projection[idx], center), record_ids[idx]),
        )
        member_ids = [record_ids[idx] for idx in member_indices]
        cluster_name = f"{mode}-cluster-{ordinal}"
        for idx in member_indices:
            cluster_rows.append({
                "mode": mode,
                "cluster_id": cluster_name,
                "record_id": record_ids[idx],
                "kernel_invocation_id": records[idx].get("kernel_invocation_id"),
                "distance_to_centroid": _dist2(projection[idx], center),
            })
        anchor_rows.append({
            "mode": mode,
            "feature_mode": feature_mode,
            "cluster_id": cluster_name,
            "rep_record_id": record_ids[representative_idx],
            "rep_kernel_id": record_ids[representative_idx],
            "member_record_ids": member_ids,
            "member_invocations": [records[idx].get("kernel_invocation_id") for idx in member_indices],
            "coverage_count": len(member_indices),
            "coverage_weight": len(member_indices) / len(records),
            "representative_selection": "nearest_centroid_real_record",
            "distance_metadata": "squared_euclidean_in_pca_space",
        })

    evaluation = {
        "mode": mode,
        "feature_mode": feature_mode,
        "compression_ratio": len(records) / max(1, len(anchor_rows)),
        "coverage_count": len(records),
        "anchor_count": len(anchor_rows),
        "weighted_coverage": sum(row["coverage_weight"] for row in anchor_rows),
        "weight_mode": "member_count_fallback",
        "pca": pca_meta,
        "kmeans": kmeans_meta,
        "cluster_feature_variance": _cluster_variance(projection, assignments),
    }
    replay_hash = stable_hash({
        "mode": mode,
        "record_ids": record_ids,
        "assignments": assignments,
        "anchors": anchor_rows,
    })
    for obj in (pca_meta, kmeans_meta, evaluation):
        obj["deterministic_replay_hash"] = replay_hash
    return {
        "projection": {
            "mode": mode,
            "feature_mode": feature_mode,
            "preprocessing": preprocessing,
            "pca": pca_meta,
            "records": projection_rows,
        },
        "clusters": {
            "mode": mode,
            "feature_mode": feature_mode,
            "kmeans": kmeans_meta,
            "records": cluster_rows,
        },
        "anchors": anchor_rows,
        "evaluation": evaluation,
        "deterministic_replay_hash": replay_hash,
    }


def _cluster_variance(projection: np.ndarray, assignments: list[int]) -> dict[str, float]:
    result = {}
    for cluster_id in sorted(set(assignments)):
        members = projection[[idx for idx, assigned in enumerate(assignments) if assigned == cluster_id]]
        result[str(cluster_id)] = float(np.var(members)) if len(members) else 0.0
    return result

