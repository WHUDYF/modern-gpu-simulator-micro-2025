"""Hierarchical readout manifest for Phase B tensors."""

from __future__ import annotations

from typing import Any

from .tensorizer import validate_phase_b_tensor_artifact
from .utils import hash_without


def build_readout_manifest(tensor: dict[str, Any], node_embeddings) -> tuple[dict[str, Any], Any]:
    validate_phase_b_tensor_artifact(tensor)
    if node_embeddings.shape[0] != tensor["node_features"].shape[0]:
        raise ValueError("node embedding count must match tensor node count")
    warp_rows = []
    warp_embeddings = []
    torch = __import__("torch")
    for partition_id, indices in sorted(tensor["warp_partitions"].items()):
        if not indices:
            raise ValueError("warp partition must not be empty")
        index_tensor = torch.tensor(indices, dtype=torch.long, device=node_embeddings.device)
        warp_embedding = node_embeddings.index_select(0, index_tensor).mean(dim=0)
        warp_embeddings.append(warp_embedding)
        warp_rows.append(
            {
                "partition_id": partition_id,
                "node_count_used": len(indices),
                "pooling_method": "mean",
                "warp_embedding_dim": int(warp_embedding.shape[0]),
            }
        )
    if not warp_embeddings:
        raise ValueError("at least one warp partition is required")
    kernel_embedding = torch.stack(warp_embeddings, dim=0).mean(dim=0)
    manifest = {
        "artifact_type": "gcl_phase_b_readout_manifest",
        "graph_id": tensor["graph_id"],
        "kernel_invocation_id": tensor["kernel_invocation_id"],
        "input_graph_hash": tensor["input_graph_hash"],
        "warps": warp_rows,
        "kernel": {
            "warp_count_used": len(warp_rows),
            "pooling_method": "average",
            "kernel_embedding_dim": int(kernel_embedding.shape[0]),
        },
    }
    manifest["readout_manifest_hash"] = hash_without(manifest, "readout_manifest_hash")
    validate_readout_manifest(manifest, tensor)
    return manifest, kernel_embedding


def validate_readout_manifest(manifest: dict[str, Any], tensor: dict[str, Any]) -> None:
    required = {"warps", "kernel", "input_graph_hash", "readout_manifest_hash"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"readout manifest missing required fields: {sorted(missing)}")
    if manifest["input_graph_hash"] != tensor["input_graph_hash"]:
        raise ValueError("readout manifest input_graph_hash mismatch")
    if len(manifest["warps"]) != len(tensor["warp_partitions"]):
        raise ValueError("readout manifest warp count mismatch")
    if manifest["kernel"].get("pooling_method") != "average":
        raise ValueError("kernel readout must use average pooling")
    if manifest["kernel"].get("kernel_embedding_dim") != 256:
        raise ValueError("kernel embedding dimension must be 256")
    for row in manifest["warps"]:
        if row.get("pooling_method") != "mean":
            raise ValueError("warp readout must use mean pooling")
        if row.get("node_count_used", 0) <= 0:
            raise ValueError("warp readout cannot use empty warp partition")
    if manifest["readout_manifest_hash"] != hash_without(manifest, "readout_manifest_hash"):
        raise ValueError("readout_manifest_hash is not reproducible")
