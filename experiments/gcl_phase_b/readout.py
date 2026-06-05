"""Hierarchical readout manifest for Phase B tensors."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .tensorizer import validate_phase_b_tensor_artifact
from .utils import hash_without

READOUT_HIERARCHY = "node_to_warp_to_cta_to_selected_sm_to_kernel"


def build_readout_manifest(tensor: dict[str, Any], node_embeddings) -> tuple[dict[str, Any], Any]:
    validate_phase_b_tensor_artifact(tensor)
    if node_embeddings.shape[0] != tensor["node_features"].shape[0]:
        raise ValueError("node embedding count must match tensor node count")
    warp_rows = []
    warp_embeddings_by_cta = defaultdict(list)
    torch = __import__("torch")
    for partition_id, indices in sorted(tensor["warp_partitions"].items()):
        if not indices:
            raise ValueError("warp partition must not be empty")
        partition_tensor = tensor["warp_partition_tensors"][partition_id]
        cta_id = partition_tensor["cta_id"]
        index_tensor = torch.tensor(indices, dtype=torch.long, device=node_embeddings.device)
        warp_embedding = node_embeddings.index_select(0, index_tensor).mean(dim=0)
        warp_embeddings_by_cta[cta_id].append(warp_embedding)
        warp_rows.append(
            {
                "partition_id": partition_id,
                "cta_id": cta_id,
                "warp_id": partition_tensor["warp_id"],
                "node_count_used": len(indices),
                "pooling_method": "mean",
                "warp_embedding_dim": int(warp_embedding.shape[0]),
            }
        )
    if not warp_rows:
        raise ValueError("at least one warp partition is required")
    cta_rows = []
    cta_embeddings = []
    for cta_id, warp_embeddings in sorted(warp_embeddings_by_cta.items()):
        cta_embedding = torch.stack(warp_embeddings, dim=0).mean(dim=0)
        cta_embeddings.append(cta_embedding)
        cta_rows.append(
            {
                "cta_id": cta_id,
                "warp_count_used": len(warp_embeddings),
                "pooling_method": "average",
                "cta_embedding_dim": int(cta_embedding.shape[0]),
            }
        )
    selected_sm_embedding = torch.stack(cta_embeddings, dim=0).mean(dim=0)
    kernel_embedding = selected_sm_embedding
    kernel_embedding_hash = hash_without(
        {
            "kernel_embedding": [
                round(float(value), 8)
                for value in kernel_embedding.detach().cpu().tolist()
            ]
        }
    )
    manifest = {
        "artifact_type": "gcl_phase_b_readout_manifest",
        "graph_id": tensor["graph_id"],
        "kernel_invocation_id": tensor["kernel_invocation_id"],
        "input_graph_hash": tensor["input_graph_hash"],
        "readout_hierarchy": READOUT_HIERARCHY,
        "warps": warp_rows,
        "ctas": cta_rows,
        "selected_sm": {
            "cta_count_used": len(cta_rows),
            "pooling_method": "average",
            "selected_sm_embedding_dim": int(selected_sm_embedding.shape[0]),
        },
        "kernel": {
            "kernel_embedding_source": "selected_sm_embedding",
            "pooling_method": "identity",
            "kernel_embedding_dim": int(kernel_embedding.shape[0]),
            "kernel_embedding_hash": kernel_embedding_hash,
        },
    }
    manifest["readout_manifest_hash"] = hash_without(manifest, "readout_manifest_hash")
    validate_readout_manifest(manifest, tensor)
    return manifest, kernel_embedding


def validate_readout_manifest(manifest: dict[str, Any], tensor: dict[str, Any]) -> None:
    required = {
        "readout_hierarchy",
        "warps",
        "ctas",
        "selected_sm",
        "kernel",
        "input_graph_hash",
        "readout_manifest_hash",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"readout manifest missing required fields: {sorted(missing)}")
    if manifest["readout_hierarchy"] != READOUT_HIERARCHY:
        raise ValueError("unexpected readout_hierarchy")
    if manifest["input_graph_hash"] != tensor["input_graph_hash"]:
        raise ValueError("readout manifest input_graph_hash mismatch")
    if len(manifest["warps"]) != len(tensor["warp_partitions"]):
        raise ValueError("readout manifest warp count mismatch")
    expected_partitions = tensor["warp_partitions"]
    expected_partition_tensors = tensor["warp_partition_tensors"]
    seen_partition_ids = set()
    warp_count_by_cta: dict[str, int] = defaultdict(int)
    for row in manifest["warps"]:
        required_warp = {
            "partition_id",
            "cta_id",
            "warp_id",
            "node_count_used",
            "pooling_method",
            "warp_embedding_dim",
        }
        missing_warp = required_warp.difference(row)
        if missing_warp:
            raise ValueError(f"readout warp row missing required fields: {sorted(missing_warp)}")
        partition_id = row["partition_id"]
        if partition_id not in expected_partitions:
            raise ValueError("readout manifest references unknown partition_id")
        expected_partition = expected_partition_tensors[partition_id]
        if row["cta_id"] != expected_partition["cta_id"]:
            raise ValueError("readout manifest cta_id mismatch")
        if row["warp_id"] != expected_partition["warp_id"]:
            raise ValueError("readout manifest warp_id mismatch")
        if partition_id in seen_partition_ids:
            raise ValueError("readout manifest contains duplicate partition_id")
        seen_partition_ids.add(partition_id)
        warp_count_by_cta[row["cta_id"]] += 1
        if row.get("pooling_method") != "mean":
            raise ValueError("warp readout must use mean pooling")
        if row["node_count_used"] != len(expected_partitions[partition_id]):
            raise ValueError("readout node_count_used mismatch")
        if row["node_count_used"] <= 0:
            raise ValueError("warp readout cannot use empty warp partition")
        if row["warp_embedding_dim"] != 256:
            raise ValueError("readout warp_embedding_dim must be 256")
    if seen_partition_ids != set(expected_partitions):
        raise ValueError("readout manifest partition_id set mismatch")
    seen_cta_ids = set()
    for row in manifest["ctas"]:
        required_cta = {"cta_id", "warp_count_used", "pooling_method", "cta_embedding_dim"}
        missing_cta = required_cta.difference(row)
        if missing_cta:
            raise ValueError(f"readout CTA row missing required fields: {sorted(missing_cta)}")
        cta_id = row["cta_id"]
        if cta_id not in warp_count_by_cta:
            raise ValueError("readout manifest references unknown cta_id")
        if cta_id in seen_cta_ids:
            raise ValueError("readout manifest contains duplicate cta_id")
        seen_cta_ids.add(cta_id)
        if row["warp_count_used"] != warp_count_by_cta[cta_id]:
            raise ValueError("readout CTA warp_count_used mismatch")
        if row["pooling_method"] != "average":
            raise ValueError("CTA readout must use average pooling")
        if row["cta_embedding_dim"] != 256:
            raise ValueError("readout cta_embedding_dim must be 256")
    if seen_cta_ids != set(warp_count_by_cta):
        raise ValueError("readout manifest cta_id set mismatch")
    selected_sm = manifest["selected_sm"]
    if selected_sm.get("cta_count_used") != len(warp_count_by_cta):
        raise ValueError("selected SM readout cta_count_used mismatch")
    if selected_sm.get("pooling_method") != "average":
        raise ValueError("selected SM readout must use average pooling")
    if selected_sm.get("selected_sm_embedding_dim") != 256:
        raise ValueError("selected SM embedding dimension must be 256")
    if manifest["kernel"].get("kernel_embedding_source") != "selected_sm_embedding":
        raise ValueError("kernel_embedding_source must be selected_sm_embedding")
    if manifest["kernel"].get("pooling_method") != "identity":
        raise ValueError("kernel readout must use identity pooling from selected SM")
    if manifest["kernel"].get("kernel_embedding_dim") != 256:
        raise ValueError("kernel embedding dimension must be 256")
    if not manifest["kernel"].get("kernel_embedding_hash"):
        raise ValueError("kernel_embedding_hash is required")
    if manifest["readout_manifest_hash"] != hash_without(manifest, "readout_manifest_hash"):
        raise ValueError("readout_manifest_hash is not reproducible")
