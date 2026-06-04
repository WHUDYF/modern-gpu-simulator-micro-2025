"""Minimal contrastive training smoke for GCL Phase A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .rgcn import MinimalRGCNEncoder, ProjectionHead, model_config, require_torch
from .tensorizer import FEATURE_WIDTH, validate_tensor_artifact
from .utils import hash_without


def validate_training_inputs(tensors: list[dict[str, Any]]) -> None:
    if not tensors:
        raise ValueError("training requires at least one tensor artifact")
    for tensor in tensors:
        for indices in tensor["warp_partitions"].values():
            if not indices:
                raise ValueError("warp partition must not be empty")
        validate_tensor_artifact(tensor)
        if tensor["node_features"].shape[1] != FEATURE_WIDTH:
            raise ValueError("RGCN training requires feature width 64")


def _filter_edges(edge_index: np.ndarray, edge_type: np.ndarray, keep_indices: np.ndarray):
    old_to_new = {int(old): new for new, old in enumerate(keep_indices.tolist())}
    kept_edges = []
    kept_types = []
    for column, relation in zip(edge_index.T, edge_type):
        source, target = int(column[0]), int(column[1])
        if source in old_to_new and target in old_to_new:
            kept_edges.append([old_to_new[source], old_to_new[target]])
            kept_types.append(int(relation))
    if kept_edges:
        return np.asarray(kept_edges, dtype=np.int64).T, np.asarray(kept_types, dtype=np.int64)
    return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)


def augment_tensor(
    tensor: dict[str, Any],
    seed: int,
    node_drop_rate: float = 0.15,
    edge_drop_rate: float = 0.15,
    noise_sigma: float = 0.01,
) -> tuple[dict[str, Any], int]:
    rng = np.random.default_rng(seed)
    node_features = np.asarray(tensor["node_features"], dtype=np.float32).copy()
    node_types = tensor["node_types"]
    keep_mask = np.ones(node_features.shape[0], dtype=bool)
    for idx, node_type in enumerate(node_types):
        if node_type != "instruction" and rng.random() < node_drop_rate:
            keep_mask[idx] = False
    keep_indices = np.flatnonzero(keep_mask)
    if keep_indices.size == 0:
        raise ValueError("augmentation removed every node")
    old_to_new = {int(old): new for new, old in enumerate(keep_indices.tolist())}
    warp_partitions = {
        warp_id: [old_to_new[idx] for idx in indices if idx in old_to_new]
        for warp_id, indices in tensor["warp_partitions"].items()
    }
    retry_count = 0
    if any(not indices for indices in warp_partitions.values()):
        raise ValueError("augmentation produced empty warp partition")
    node_features = node_features[keep_indices]
    node_types = [node_types[index] for index in keep_indices]
    node_features += rng.normal(0.0, noise_sigma, size=node_features.shape).astype(np.float32)
    for index, node_type in enumerate(node_types):
        if node_type in {"register_version", "input_variable", "unknown_variable"}:
            node_features[index, 40:64] = 0.0
        elif node_type == "pseudo":
            node_features[index, 16:64] = 0.0
    edge_index, edge_type = _filter_edges(tensor["edge_index"], tensor["edge_type"], keep_indices)
    if edge_type.size:
        edge_keep = rng.random(edge_type.shape[0]) >= edge_drop_rate
        edge_index = edge_index[:, edge_keep]
        edge_type = edge_type[edge_keep]
    augmented = dict(tensor)
    augmented["node_features"] = node_features
    augmented["node_types"] = node_types
    augmented["edge_index"] = edge_index
    augmented["edge_type"] = edge_type
    augmented["warp_partitions"] = warp_partitions
    augmented["augmentation_manifest"] = {
        "node_drop_rate": node_drop_rate,
        "edge_drop_rate": edge_drop_rate,
        "noise_sigma": noise_sigma,
        "seed": seed,
    }
    return augmented, retry_count


def _augment_with_retry(tensor: dict[str, Any], seed: int, max_retries: int = 3):
    retry_count = 0
    for attempt in range(max_retries + 1):
        try:
            augmented, inner_retries = augment_tensor(tensor, seed + attempt)
            return augmented, retry_count + inner_retries
        except ValueError as exc:
            if "empty warp partition" not in str(exc) and "removed every node" not in str(exc):
                raise
            retry_count += 1
    raise ValueError(f"augmentation produced empty warp partition after {retry_count} retries")


def _encode_batch(encoder, tensors: list[dict[str, Any]]):
    embeddings = [encoder.encode_kernel(tensor) for tensor in tensors]
    return require_torch().stack(embeddings, dim=0)


def info_nce_loss(projections_a, projections_b, temperature: float = 0.2):
    torch = require_torch()
    import torch.nn.functional as F

    a = F.normalize(projections_a, dim=1)
    b = F.normalize(projections_b, dim=1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.shape[0], dtype=torch.long, device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def train_minimal_contrastive(
    tensors: list[dict[str, Any]],
    out_dir: Path,
    seed: int = 20260602,
) -> dict[str, Any]:
    torch = require_torch()
    validate_training_inputs(tensors)
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = MinimalRGCNEncoder()
    projection_head = ProjectionHead()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()), lr=0.005
    )
    views_a = []
    views_b = []
    retry_count = 0
    for index, tensor in enumerate(tensors):
        view_a, retries_a = _augment_with_retry(tensor, seed + index * 2)
        view_b, retries_b = _augment_with_retry(tensor, seed + index * 2 + 1)
        views_a.append(view_a)
        views_b.append(view_b)
        retry_count += retries_a + retries_b

    encoder.train()
    projection_head.train()
    embeddings_a = _encode_batch(encoder, views_a)
    embeddings_b = _encode_batch(encoder, views_b)
    projections_a = projection_head(embeddings_a)
    projections_b = projection_head(embeddings_b)
    loss = info_nce_loss(projections_a, projections_b)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "rgcn_checkpoint.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "projection_head": projection_head.state_dict(),
            "model_config": model_config(),
            "seed": seed,
        },
        checkpoint_path,
    )
    checkpoint_hash = hash_without({"checkpoint_bytes": checkpoint_path.read_bytes().hex()})
    source_tensor_hashes = [tensor["tensor_hash"] for tensor in tensors]
    encoder_manifest = {
        "encoder_name": "minimal_phase_a_rgcn",
        "encoder_version": 1,
        "model_config": model_config(),
        "source_tensor_hashes": source_tensor_hashes,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
    }
    encoder_manifest["encoder_manifest_hash"] = hash_without(
        encoder_manifest, "encoder_manifest_hash", "checkpoint_path"
    )
    return {
        "training_mode": "minimal_rgcn_contrastive_smoke",
        "loss": float(loss.detach().cpu().item()),
        "optimizer_step_count": 1,
        "augmentation_retry_count": retry_count,
        "kernel_embedding_shape": list(embeddings_a.shape),
        "projection_output_shape": list(projections_a.shape),
        "checkpoint_manifest": encoder_manifest,
        "encoder": encoder,
        "projection_head": projection_head,
    }
