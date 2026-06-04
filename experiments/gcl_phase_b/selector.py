"""M0 selector integration for Phase B embedding tables."""

from __future__ import annotations

from typing import Any

from experiments.gcl_phase_a.embedding_export import REPRESENTATION_MODE, validate_embedding_table
from experiments.gcl_phase_a.selector import select_representatives
from experiments.gcl_phase_a.utils import hash_without


def select_phase_b_representatives(table: dict[str, Any], seed: int = 20260602) -> dict[str, Any]:
    for row in table.get("rows", []):
        if row.get("resource_blocked"):
            raise ValueError("resource-blocked embedding rows cannot enter M0 selector")
    validate_embedding_table(table)
    if table["row_count"] == 1:
        row = table["rows"][0]
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
                "row_count": table["row_count"],
                "cluster_count": 1,
                "anchor_count": 1,
                "seed": seed,
            },
            "source_embedding_table_hash": table["embedding_table_hash"],
        }
        artifact["selector_manifest_hash"] = hash_without(artifact, "selector_manifest_hash")
        return artifact
    return select_representatives(table, seed=seed)
