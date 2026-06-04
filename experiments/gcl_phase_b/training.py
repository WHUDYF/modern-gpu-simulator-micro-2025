"""Phase B training-view helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.gcl_phase_a.train import augment_tensor

from .tensorizer import validate_phase_b_tensor_artifact
from .utils import hash_without


def _view_hash(view: dict[str, Any]) -> str:
    payload = dict(view)
    manifest = dict(payload.get("phase_b_augmentation_manifest", {}))
    manifest.pop("augmentation_manifest_hash", None)
    manifest.pop("view_hash", None)
    payload["phase_b_augmentation_manifest"] = manifest
    payload["node_features"] = np.asarray(payload["node_features"], dtype=np.float32).tolist()
    payload["edge_index"] = np.asarray(payload["edge_index"], dtype=np.int64).tolist()
    payload["edge_type"] = np.asarray(payload["edge_type"], dtype=np.int64).tolist()
    return hash_without(payload, "tensor_hash")


def _manifest(
    tensor: dict[str, Any],
    view: dict[str, Any],
    view_id: str,
    seed: int,
    retry_count: int,
) -> dict[str, Any]:
    dropped_node_count = int(tensor["node_features"].shape[0] - view["node_features"].shape[0])
    dropped_edge_count = int(tensor["edge_type"].shape[0] - view["edge_type"].shape[0])
    manifest = {
        "artifact_type": "gcl_phase_b_augmentation_manifest",
        "input_graph_hash": tensor["input_graph_hash"],
        "random_seed": seed,
        "view_id": view_id,
        "augmentation_types": ["node_dropping", "edge_dropping", "feature_noise_injection"],
        "rates": {"node_drop_rate": 0.15, "edge_drop_rate": 0.15},
        "dropped_node_count": dropped_node_count,
        "dropped_edge_count": dropped_edge_count,
        "feature_noise_std": 0.01,
        "retry_count": retry_count,
    }
    view["phase_b_augmentation_manifest"] = manifest
    manifest["view_hash"] = _view_hash(view)
    manifest["augmentation_manifest_hash"] = hash_without(manifest, "augmentation_manifest_hash")
    return manifest


def create_augmented_training_views(tensor: dict[str, Any], seed: int = 20260602):
    validate_phase_b_tensor_artifact(tensor)
    view_a, retries_a = augment_tensor(tensor, seed)
    view_b, retries_b = augment_tensor(tensor, seed + 1)
    view_a = dict(view_a)
    view_b = dict(view_b)
    view_a.pop("graph_hash", None)
    view_b.pop("graph_hash", None)
    _manifest(tensor, view_a, "A", seed, retries_a)
    _manifest(tensor, view_b, "B", seed + 1, retries_b)
    return view_a, view_b
