"""M0 selector integration for Phase B embedding tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiments.gcl_phase_a.selector import choose_silhouette_k, zscore_normalize
from experiments.gcl_phase_a.utils import hash_without
from .utils import read_json
from .embedding_export import EMBEDDING_DIM, REPRESENTATION_MODE, validate_phase_b_embedding_table


def _embedding_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    return table["embeddings"]


def _embedding_table_hash(table: dict[str, Any]) -> str:
    return table["kernel_embedding_table_hash"]


def _selector_status_fields(table: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_status": table.get("artifact_status", "debug_not_formal"),
        "formal_input_eligible": table.get("formal_input_eligible", False),
    }


def select_phase_b_representatives(
    table: dict[str, Any],
    seed: int = 20260602,
    allow_debug: bool = False,
    lineage_bundle: dict[str, Any] | None = None,
    gate5_manifests: dict[str, dict[str, Any]] | None = None,
    gate5_artifact_root: Path | str | None = None,
) -> dict[str, Any]:
    for row in table.get("embeddings", []):
        if row.get("resource_blocked"):
            raise ValueError("resource-blocked embedding rows cannot enter M0 selector")
    _validate_gate6_input_table(
        table,
        allow_debug=allow_debug,
        lineage_bundle=lineage_bundle,
        gate5_manifests=gate5_manifests,
        gate5_artifact_root=gate5_artifact_root,
    )
    validate_phase_b_embedding_table(table)
    rows = _embedding_rows(table)
    if len(rows) == 1:
        row = rows[0]
        artifact = {
            "artifact_type": "gcl_resnet50_gate6_selector_artifacts",
            "artifact_version": "gate6_selector_artifacts_v1",
            **_selector_status_fields(table),
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
                "clusters": _post_clustering_family_evidence([row], [0]),
                "members": _post_clustering_family_members([row], [0]),
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
        **_selector_status_fields(table),
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
            "clusters": _post_clustering_family_evidence(rows, labels.tolist()),
            "members": _post_clustering_family_members(rows, labels.tolist()),
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


def _post_clustering_family_evidence(
    rows: list[dict[str, Any]],
    labels: list[int],
) -> list[dict[str, Any]]:
    row_weights = [_family_row_weight(row) for row in rows]
    total_weight = sum(row_weights) or 1.0
    by_cluster: dict[int, dict[str, float]] = {}
    cluster_member_counts: dict[int, int] = {}
    for row, label in zip(rows, labels):
        family = _post_clustering_family_label(row)
        row_weight = _family_row_weight(row)
        cluster_id = int(label)
        cluster_counts = by_cluster.setdefault(cluster_id, {})
        cluster_counts[family] = cluster_counts.get(family, 0.0) + row_weight
        cluster_member_counts[cluster_id] = cluster_member_counts.get(cluster_id, 0) + 1
    clusters = []
    for cluster_id, weights_by_family in sorted(by_cluster.items()):
        majority_family, majority_weight = sorted(
            weights_by_family.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        cluster_weight = sum(weights_by_family.values())
        family_member_counts: dict[str, int] = {}
        for row, label in zip(rows, labels):
            if int(label) == cluster_id:
                family = _post_clustering_family_label(row)
                family_member_counts[family] = family_member_counts.get(family, 0) + 1
        majority_count = sorted(
            family_member_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][1]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "majority_family": majority_family,
                "purity": round(float(majority_weight) / float(cluster_weight), 8),
                "weight": round(float(cluster_weight) / float(total_weight), 8),
                "member_count": cluster_member_counts[cluster_id],
                "majority_member_count": majority_count,
                "raw_weight": round(float(cluster_weight), 8),
                "weight_source": "embedding_row_weight_input_node_count",
                "evidence_source": "post_clustering_embedding_metadata",
            }
        )
    return clusters


def _post_clustering_family_members(
    rows: list[dict[str, Any]],
    labels: list[int],
) -> list[dict[str, Any]]:
    return [
        {
            "record_id": row["record_id"],
            "kernel_invocation_id": row["kernel_invocation_id"],
            "cluster_id": int(label),
            "family": _post_clustering_family_label(row),
            "weight": round(float(_family_row_weight(row)), 8),
            "evidence_source": "post_clustering_embedding_metadata",
        }
        for row, label in zip(rows, labels)
    ]


def _post_clustering_family_label(row: dict[str, Any]) -> str:
    if row.get("trace_family"):
        return str(row["trace_family"])
    return ""


def _family_row_weight(row: dict[str, Any]) -> float:
    weight_input = row.get("weight_input", {})
    for key in ("runtime_weight", "measured_runtime_weight", "node_count"):
        value = weight_input.get(key)
        if value is not None:
            weight = float(value)
            if weight > 0.0:
                return weight
    return 1.0


def _validate_gate6_input_table(
    table: dict[str, Any],
    allow_debug: bool = False,
    lineage_bundle: dict[str, Any] | None = None,
    gate5_manifests: dict[str, dict[str, Any]] | None = None,
    gate5_artifact_root: Path | str | None = None,
) -> None:
    if table.get("artifact_status") != "formal" and not allow_debug:
        raise ValueError("Gate6 selector requires formal embedding table")
    required_provenance = {
        "artifact_status",
        "formal_input_eligible",
        "workload_id",
        "execution_mode",
        "trace_source",
        "input_scope",
        "scheduler_metadata_source",
    }
    missing = required_provenance.difference(table)
    if missing and not allow_debug:
        raise ValueError(f"Gate6 selector requires formal provenance fields: {sorted(missing)}")
    if table.get("formal_input_eligible") is not True and not allow_debug:
        raise ValueError("Gate6 selector requires formal input eligible embedding table")
    if not allow_debug and not table.get("gate5_lineage_hash"):
        raise ValueError("Gate5 lineage is required for formal Gate6 selector input")
    if not allow_debug:
        lineage = table.get("gate5_lineage")
        if not isinstance(lineage, dict):
            raise ValueError("Gate5 lineage is required for formal Gate6 selector input")
        required_lineage = {
            "source_graph_tensor_bundle_hash",
            "training_run_manifest_hash",
            "checkpoint_manifest_hash",
            "readout_manifest_bundle_hash",
            "embedding_export_report_hash",
        }
        missing_lineage = required_lineage.difference(lineage)
        if missing_lineage:
            raise ValueError(f"Gate5 lineage missing required fields: {sorted(missing_lineage)}")
        _validate_persisted_gate5_lineage_bundle(
            table,
            lineage_bundle=lineage_bundle,
            gate5_manifests=gate5_manifests,
            gate5_artifact_root=gate5_artifact_root,
        )
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


def _validate_persisted_gate5_lineage_bundle(
    table: dict[str, Any],
    *,
    lineage_bundle: dict[str, Any] | None,
    gate5_manifests: dict[str, dict[str, Any]] | None,
    gate5_artifact_root: Path | str | None,
) -> None:
    if gate5_artifact_root is None:
        raise ValueError("persisted Gate5 artifact root is required for formal Gate6 selector input")
    persisted_artifacts = _load_persisted_gate5_artifacts(Path(gate5_artifact_root))
    if gate5_manifests is not None:
        raise ValueError("formal Gate6 selector must load persisted Gate5 manifests from disk")
    disk_lineage_bundle = persisted_artifacts["lineage_bundle"]
    if lineage_bundle is not None and lineage_bundle != disk_lineage_bundle:
        raise ValueError("caller Gate5 lineage bundle does not match persisted artifact root")
    lineage_bundle = disk_lineage_bundle
    if lineage_bundle.get("artifact_type") != "gcl_resnet50_gate5_lineage_bundle":
        raise ValueError("persisted Gate5 lineage bundle artifact_type mismatch")
    if lineage_bundle.get("lineage") != table.get("gate5_lineage"):
        raise ValueError("persisted Gate5 lineage bundle does not match embedding table lineage")
    if lineage_bundle.get("gate5_lineage_bundle_hash") != table.get("gate5_lineage_bundle_hash"):
        raise ValueError("persisted Gate5 lineage bundle hash does not match embedding table")
    if lineage_bundle.get("gate5_lineage_bundle_hash") != hash_without(
        lineage_bundle, "gate5_lineage_bundle_hash"
    ):
        raise ValueError("persisted Gate5 lineage bundle hash is not reproducible")
    persisted_hashes = lineage_bundle.get("persisted_manifest_hashes")
    if not isinstance(persisted_hashes, dict):
        raise ValueError("persisted Gate5 manifest hashes are required for formal Gate6 selector input")
    expected = {
        "training_run_manifest_hash": table["gate5_lineage"]["training_run_manifest_hash"],
        "checkpoint_manifest_hash": table["gate5_lineage"]["checkpoint_manifest_hash"],
        "readout_manifest_bundle_hash": table["gate5_lineage"]["readout_manifest_bundle_hash"],
        "embedding_export_report_hash": table["gate5_lineage"]["embedding_export_report_hash"],
    }
    if persisted_hashes != expected:
        raise ValueError("persisted Gate5 manifest hashes do not match embedding table lineage")
    _validate_gate5_manifest_objects(table, persisted_artifacts["manifests"])


def _load_persisted_gate5_artifacts(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError("persisted Gate5 artifact root is required for formal Gate6 selector input")
    filenames = {
        "lineage_bundle": "gate5_lineage_bundle.json",
        "training_run_manifest": "rgcn_training_run_manifest.json",
        "checkpoint_manifest": "rgcn_checkpoint_manifest.json",
        "readout_manifest_bundle": "readout_manifest.json",
        "embedding_export_report": "embedding_export_report.json",
    }
    loaded = {}
    for key, filename in filenames.items():
        path = root / filename
        if not path.is_file():
            raise ValueError(f"persisted Gate5 artifact root missing {filename}")
        loaded[key] = read_json(path)
    return {
        "lineage_bundle": loaded["lineage_bundle"],
        "manifests": {
            "training_run_manifest": loaded["training_run_manifest"],
            "checkpoint_manifest": loaded["checkpoint_manifest"],
            "readout_manifest_bundle": loaded["readout_manifest_bundle"],
            "embedding_export_report": loaded["embedding_export_report"],
        },
    }


def _validate_gate5_manifest_objects(
    table: dict[str, Any],
    gate5_manifests: dict[str, dict[str, Any]] | None,
) -> None:
    if not gate5_manifests:
        raise ValueError("persisted Gate5 manifest objects are required for formal Gate6 selector input")
    required = {
        "training_run_manifest",
        "checkpoint_manifest",
        "readout_manifest_bundle",
        "embedding_export_report",
    }
    missing = required.difference(gate5_manifests)
    if missing:
        raise ValueError(f"persisted Gate5 manifest objects missing fields: {sorted(missing)}")
    lineage = table["gate5_lineage"]
    recomputed = {
        "training_run_manifest_hash": hash_without(
            gate5_manifests["training_run_manifest"],
            "training_run_manifest_hash",
        ),
        "checkpoint_manifest_hash": _checkpoint_manifest_hash(gate5_manifests["checkpoint_manifest"]),
        "readout_manifest_bundle_hash": hash_without(
            gate5_manifests["readout_manifest_bundle"],
            "readout_manifest_bundle_hash",
        ),
        "embedding_export_report_hash": hash_without(
            gate5_manifests["embedding_export_report"],
            "embedding_export_report_hash",
        ),
    }
    if recomputed != {
        "training_run_manifest_hash": lineage["training_run_manifest_hash"],
        "checkpoint_manifest_hash": lineage["checkpoint_manifest_hash"],
        "readout_manifest_bundle_hash": lineage["readout_manifest_bundle_hash"],
        "embedding_export_report_hash": lineage["embedding_export_report_hash"],
    }:
        raise ValueError("persisted Gate5 manifest objects do not match embedding table lineage")


def _checkpoint_manifest_hash(manifest: dict[str, Any]) -> str:
    hash_field = (
        "rgcn_checkpoint_manifest_hash"
        if "rgcn_checkpoint_manifest_hash" in manifest
        else "checkpoint_manifest_hash"
    )
    return hash_without(manifest, hash_field)


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
