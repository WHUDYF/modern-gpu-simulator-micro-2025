"""Gate 5 embedding export for Phase B tensors using CTA-aware readout."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.gcl_phase_a.rgcn import require_torch
from experiments.gcl_phase_a.utils import hash_without, stable_hash

from .readout import build_readout_manifest
from .tensorizer import validate_phase_b_tensor_artifact

REPRESENTATION_MODE = "gcl_resnet50_rgcn_selected_sm_kernel_embedding"
EMBEDDING_DIM = 256


def export_phase_b_embedding_table(
    tensors: list[dict[str, Any]],
    encoder,
    encoder_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = require_torch()
    if not tensors:
        raise ValueError("embedding export requires at least one tensor artifact")
    if "encoder_manifest_hash" not in encoder_manifest:
        encoder_manifest = dict(encoder_manifest)
        encoder_manifest["encoder_manifest_hash"] = stable_hash(encoder_manifest)
    rows = []
    readout_manifests = []
    encoder.eval()
    with torch.no_grad():
        for index, tensor in enumerate(tensors):
            if "augmentation_manifest" in tensor:
                raise ValueError("selector embedding must come from canonical non-augmented graph")
            validate_phase_b_tensor_artifact(tensor)
            node_features = torch.as_tensor(tensor["node_features"], dtype=torch.float32)
            edge_index = torch.as_tensor(tensor["edge_index"], dtype=torch.long)
            edge_type = torch.as_tensor(tensor["edge_type"], dtype=torch.long)
            node_embeddings = encoder(node_features, edge_index, edge_type)
            readout_manifest, kernel_embedding = build_readout_manifest(tensor, node_embeddings)
            row = _embedding_row(
                index=index,
                tensor=tensor,
                embedding=kernel_embedding.detach().cpu().numpy(),
                encoder_manifest_hash=encoder_manifest["encoder_manifest_hash"],
                readout_manifest_hash=readout_manifest["readout_manifest_hash"],
            )
            rows.append(row)
            readout_manifests.append(readout_manifest)
    table = {
        "artifact_type": "gcl_kernel_embedding_table",
        "representation_mode": REPRESENTATION_MODE,
        "embedding_dim": EMBEDDING_DIM,
        "row_count": len(rows),
        "rows": rows,
        "encoder_manifest_hash": encoder_manifest["encoder_manifest_hash"],
    }
    table["embedding_table_hash"] = hash_without(table, "embedding_table_hash")
    validate_phase_b_embedding_table(table)
    readout_bundle = {
        "artifact_type": "gcl_phase_b_readout_manifest_bundle",
        "manifests": readout_manifests,
    }
    readout_bundle["readout_manifest_bundle_hash"] = hash_without(
        readout_bundle, "readout_manifest_bundle_hash"
    )
    return table, readout_bundle


def _embedding_row(
    index: int,
    tensor: dict[str, Any],
    embedding: np.ndarray,
    encoder_manifest_hash: str,
    readout_manifest_hash: str,
) -> dict[str, Any]:
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError("M0 selector embedding must be the 256-dimensional encoder readout")
    rounded_embedding = [round(float(value), 8) for value in embedding.tolist()]
    row = {
        "record_id": f"gcl_embedding:{index:04d}",
        "kernel_invocation_id": tensor["kernel_invocation_id"],
        "representation_mode": REPRESENTATION_MODE,
        "input_representation_mode": tensor["representation_mode"],
        "pseudo_node_mode": tensor["pseudo_node_mode"],
        "paper_reproduction_mode": tensor["paper_reproduction_mode"],
        "embedding_dim": EMBEDDING_DIM,
        "embedding": rounded_embedding,
        "source_graph_hash": tensor["input_graph_hash"],
        "encoder_manifest_hash": encoder_manifest_hash,
        "readout_manifest_hash": readout_manifest_hash,
        "weight_input": {
            "graph_id": tensor["graph_id"],
            "node_count": tensor["graph_batch_metadata"]["node_count"],
            "edge_count": tensor["graph_batch_metadata"]["edge_count"],
            "readout_hierarchy": "node_to_warp_to_cta_to_selected_sm_to_kernel",
        },
    }
    row["embedding_hash"] = hash_without(row, "embedding_hash")
    return row


def validate_phase_b_embedding_table(table: dict[str, Any]) -> None:
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
            "input_representation_mode",
            "pseudo_node_mode",
            "paper_reproduction_mode",
            "embedding_dim",
            "embedding",
            "source_graph_hash",
            "encoder_manifest_hash",
            "readout_manifest_hash",
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
        if row["weight_input"].get("readout_hierarchy") != (
            "node_to_warp_to_cta_to_selected_sm_to_kernel"
        ):
            raise ValueError("embedding row must record CTA-aware readout hierarchy")
        if row["embedding_hash"] != hash_without(row, "embedding_hash"):
            raise ValueError("embedding_hash is not reproducible")
    if table["row_count"] != len(rows):
        raise ValueError("embedding table row_count mismatch")
    if table["embedding_table_hash"] != hash_without(table, "embedding_table_hash"):
        raise ValueError("embedding_table_hash is not reproducible")
