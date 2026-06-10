"""M0-compatible kernel embedding table export for GCL Phase A."""

from __future__ import annotations

from typing import Any

import numpy as np

from .rgcn import require_torch
from .tensorizer import validate_tensor_artifact
from .utils import hash_without, stable_hash

REPRESENTATION_MODE = "gcl_phase_a_rgcn_canonical_kernel_embedding"
EMBEDDING_DIM = 256


def _KERNEL_EMBEDDING_DIM_GUARD() -> int:
    return 256


def export_embedding_table(
    tensors: list[dict[str, Any]],
    encoder,
    encoder_manifest: dict[str, Any],
) -> dict[str, Any]:
    torch = require_torch()
    if not tensors:
        raise ValueError("embedding export requires at least one tensor artifact")
    if "encoder_manifest_hash" not in encoder_manifest:
        encoder_manifest = dict(encoder_manifest)
        encoder_manifest["encoder_manifest_hash"] = stable_hash(encoder_manifest)
    rows = []
    encoder.eval()
    with torch.no_grad():
        for index, tensor in enumerate(tensors):
            if "augmentation_manifest" in tensor:
                raise ValueError("selector embedding must come from canonical non-augmented graph")
            validate_tensor_artifact(tensor)
            if encoder_manifest.get("partitioned_encoding") and hasattr(
                encoder,
                "encode_kernel_partitioned",
            ):
                kernel_embedding = (
                    encoder.encode_kernel_partitioned(tensor).detach().cpu().numpy()
                )
            else:
                kernel_embedding = encoder.encode_kernel(tensor).detach().cpu().numpy()
            row = _embedding_row(
                index=index,
                tensor=tensor,
                embedding=kernel_embedding,
                encoder_manifest_hash=encoder_manifest["encoder_manifest_hash"],
            )
            rows.append(row)
    table = {
        "artifact_type": "gcl_kernel_embedding_table",
        "representation_mode": REPRESENTATION_MODE,
        "embedding_dim": EMBEDDING_DIM,
        "row_count": len(rows),
        "rows": rows,
        "encoder_manifest_hash": encoder_manifest["encoder_manifest_hash"],
    }
    table["embedding_table_hash"] = hash_without(table, "embedding_table_hash")
    validate_embedding_table(table)
    return table


def _embedding_row(
    index: int,
    tensor: dict[str, Any],
    embedding: np.ndarray,
    encoder_manifest_hash: str,
) -> dict[str, Any]:
    if embedding.shape != (_KERNEL_EMBEDDING_DIM_GUARD(),):
        raise ValueError("M0 selector embedding must be the 256-dimensional encoder readout")
    rounded_embedding = [round(float(value), 8) for value in embedding.tolist()]
    row = {
        "record_id": f"gcl_embedding:{index:04d}",
        "kernel_invocation_id": tensor["kernel_invocation_id"],
        "representation_mode": REPRESENTATION_MODE,
        "embedding_dim": EMBEDDING_DIM,
        "embedding": rounded_embedding,
        "source_graph_hash": tensor["input_graph_hash"],
        "encoder_manifest_hash": encoder_manifest_hash,
        "weight_input": {
            "graph_id": tensor["graph_id"],
            "node_count": tensor["graph_batch_metadata"]["node_count"],
            "edge_count": tensor["graph_batch_metadata"]["edge_count"],
        },
    }
    row["embedding_hash"] = hash_without(row, "embedding_hash")
    return row


def validate_embedding_table(table: dict[str, Any]) -> None:
    if table.get("representation_mode") != REPRESENTATION_MODE:
        raise ValueError("unexpected representation_mode")
    if table.get("embedding_dim") != EMBEDDING_DIM:
        raise ValueError("embedding table must use 256-dimensional kernel embeddings")
    rows = table.get("rows")
    if not rows:
        raise ValueError("embedding table must contain rows")
    for row in rows:
        required = {
            "record_id",
            "kernel_invocation_id",
            "representation_mode",
            "embedding_dim",
            "embedding",
            "source_graph_hash",
            "encoder_manifest_hash",
            "embedding_hash",
            "weight_input",
        }
        missing = required.difference(row)
        if missing:
            raise ValueError(f"embedding row missing required fields: {sorted(missing)}")
        if row["representation_mode"] != REPRESENTATION_MODE:
            raise ValueError("embedding row representation_mode mismatch")
        if row["embedding_dim"] != EMBEDDING_DIM:
            raise ValueError("projection output is not a valid M0 selector embedding")
        if len(row["embedding"]) != EMBEDDING_DIM:
            raise ValueError("embedding vector length must be 256")
        if row["embedding_hash"] != hash_without(row, "embedding_hash"):
            raise ValueError("embedding_hash is not reproducible")
    if table["row_count"] != len(rows):
        raise ValueError("embedding table row_count mismatch")
    if table["embedding_table_hash"] != hash_without(table, "embedding_table_hash"):
        raise ValueError("embedding_table_hash is not reproducible")
