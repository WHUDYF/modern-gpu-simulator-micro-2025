"""Tensorization for strict GCL-Sampler Phase A node features."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .graph_builder import EDGE_RELATION_SCHEMA, validate_graph_artifact
from .utils import hash_without, stable_hash

FEATURE_WIDTH = 64
NODE_FEATURE_SCHEMA_NAME = "gcl_m2_phase_a_paper_node_feature_v1"
PAPER_REPRODUCTION_MODE = "strict_gcl_sampler_node_features"
INSTRUCTION_FEATURE_COMBINE = "concat_opcode63_normalized_pc1"
TENSORIZER_VERSION = "gcl_phase_a_tensorizer_v1"
PADDING_POLICY = "strict_zero_padding"
MISSING_VALUE_POLICY = "missing numeric values become 0.0"
VARIABLE_NODE_TYPES = {"register_version", "input_variable", "unknown_variable"}


def node_feature_schema() -> dict[str, Any]:
    return {
        "schema_name": NODE_FEATURE_SCHEMA_NAME,
        "schema_version": 1,
        "feature_width": FEATURE_WIDTH,
        "node_type_layouts": {
            "instruction": [
                {
                    "block_name": "opcode_token_embedding",
                    "start_index": 0,
                    "end_index": 63,
                    "block_kind": "learned_embedding",
                    "source_fields": ["opcode"],
                    "normalization": "none",
                    "default_value": 0.0,
                    "trainable": True,
                    "paper_defined": True,
                },
                {
                    "block_name": "normalized_pc",
                    "start_index": 63,
                    "end_index": 64,
                    "block_kind": "fixed_numeric",
                    "source_fields": ["pc"],
                    "normalization": "graph_minmax",
                    "default_value": 0.0,
                    "trainable": False,
                    "paper_defined": True,
                },
            ],
            "variable": [
                {
                    "block_name": "variable_token_embedding",
                    "start_index": 0,
                    "end_index": 32,
                    "block_kind": "learned_embedding",
                    "source_fields": ["token"],
                    "normalization": "none",
                    "default_value": 0.0,
                    "trainable": True,
                    "paper_defined": True,
                },
                {
                    "block_name": "dynamic_value_statistics",
                    "start_index": 32,
                    "end_index": 40,
                    "block_kind": "fixed_numeric",
                    "source_fields": ["observed_dynamic_values"],
                    "normalization": "bounded_tanh",
                    "default_value": 0.0,
                    "trainable": False,
                    "paper_defined": True,
                },
                {
                    "block_name": "variable_zero_padding",
                    "start_index": 40,
                    "end_index": 64,
                    "block_kind": "zero_padding",
                    "source_fields": [],
                    "normalization": "none",
                    "default_value": 0.0,
                    "trainable": False,
                    "paper_defined": True,
                },
            ],
            "pseudo": [
                {
                    "block_name": "pseudo_token_embedding",
                    "start_index": 0,
                    "end_index": 16,
                    "block_kind": "learned_embedding",
                    "source_fields": ["pseudo_kind"],
                    "normalization": "none",
                    "default_value": 0.0,
                    "trainable": True,
                    "paper_defined": True,
                },
                {
                    "block_name": "pseudo_zero_padding",
                    "start_index": 16,
                    "end_index": 64,
                    "block_kind": "zero_padding",
                    "source_fields": [],
                    "normalization": "none",
                    "default_value": 0.0,
                    "trainable": False,
                    "paper_defined": True,
                },
            ],
        },
        "embedding_blocks": [
            "opcode_token_embedding",
            "variable_token_embedding",
            "pseudo_token_embedding",
        ],
        "numeric_feature_blocks": ["normalized_pc", "dynamic_value_statistics"],
        "padding_blocks": ["variable_zero_padding", "pseudo_zero_padding"],
        "instruction_feature_mode": "opcode_embedding_plus_pc",
        "instruction_feature_combine": INSTRUCTION_FEATURE_COMBINE,
        "normalization_policy": "pc graph minmax; dynamic stats bounded_tanh",
        "missing_value_policy": MISSING_VALUE_POLICY,
        "paper_reproduction_mode": PAPER_REPRODUCTION_MODE,
    }


def _token_embedding(token: str, dim: int) -> np.ndarray:
    seed = int(stable_hash({"token": token, "dim": dim})[:16], 16)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.05, size=dim).astype(np.float32)


def _dynamic_value_statistics(values: list[float]) -> np.ndarray:
    if not values:
        return np.zeros(8, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    median = float(np.median(arr))
    minimum = float(np.min(arr))
    maximum = float(np.max(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    skew = 0.0
    if std > 1e-8:
        skew = float(np.mean(((arr - mean) / std) ** 3))
    stats = np.asarray([mean, std, median, minimum, maximum, p25, p75, skew], dtype=np.float32)
    return np.tanh(stats / 32.0).astype(np.float32)


def _normalized_pc(node: dict[str, Any], pc_min: int, pc_max: int) -> float:
    if pc_max == pc_min:
        return 0.0
    return float((node["pc"] - pc_min) / (pc_max - pc_min))


def tensorize_graph(graph: dict[str, Any]) -> dict[str, Any]:
    validate_graph_artifact(graph)
    nodes = graph["nodes"]
    node_index = {node["node_id"]: idx for idx, node in enumerate(nodes)}
    pcs = [node["pc"] for node in nodes if node["node_type"] == "instruction"]
    pc_min = min(pcs)
    pc_max = max(pcs)
    features = np.zeros((len(nodes), FEATURE_WIDTH), dtype=np.float32)

    for idx, node in enumerate(nodes):
        if node["node_type"] == "instruction":
            features[idx, 0:63] = _token_embedding(node["opcode"], 63)
            features[idx, 63] = _normalized_pc(node, pc_min, pc_max)
        elif node["node_type"] in VARIABLE_NODE_TYPES:
            features[idx, 0:32] = _token_embedding(node["token"], 32)
            features[idx, 32:40] = _dynamic_value_statistics(node.get("observed_dynamic_values", []))
        elif node["node_type"] == "pseudo":
            features[idx, 0:16] = _token_embedding(node["pseudo_kind"], 16)
        else:
            raise ValueError(f"unsupported node_type: {node['node_type']}")

    edge_pairs = [[node_index[edge["source"]], node_index[edge["target"]]] for edge in graph["edges"]]
    edge_index = (
        np.asarray(edge_pairs, dtype=np.int64).T
        if edge_pairs
        else np.empty((2, 0), dtype=np.int64)
    )
    edge_type = np.asarray(
        [graph["edge_relation_schema"][edge["relation"]] for edge in graph["edges"]],
        dtype=np.int64,
    )
    tensor = {
        "artifact_type": "graph_tensor",
        "graph_id": graph["graph_id"],
        "kernel_invocation_id": graph["kernel_invocation_id"],
        "input_graph_hash": graph["graph_hash"],
        "tensorizer_version": TENSORIZER_VERSION,
        "node_feature_schema": node_feature_schema(),
        "edge_relation_schema": graph["edge_relation_schema"],
        "feature_width": FEATURE_WIDTH,
        "padding_policy": PADDING_POLICY,
        "missing_value_policy": MISSING_VALUE_POLICY,
        "node_ids": [node["node_id"] for node in nodes],
        "node_types": [node["node_type"] for node in nodes],
        "node_features": features,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "warp_partitions": {
            warp_id: [node_index[node_id] for node_id in node_ids]
            for warp_id, node_ids in graph["warp_partitions"].items()
        },
        "graph_batch_metadata": {
            "graph_id": graph["graph_id"],
            "kernel_invocation_id": graph["kernel_invocation_id"],
            "node_count": len(nodes),
            "edge_count": len(graph["edges"]),
        },
    }
    tensor["tensor_hash"] = _tensor_hash(tensor)
    validate_tensor_artifact(tensor)
    return tensor


def tensorize_graphs(graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tensorize_graph(graph) for graph in graphs]


def _serializable_tensor(tensor: dict[str, Any]) -> dict[str, Any]:
    payload = dict(tensor)
    payload["node_features"] = np.asarray(payload["node_features"], dtype=np.float32).tolist()
    payload["edge_index"] = np.asarray(payload["edge_index"]).tolist()
    payload["edge_type"] = np.asarray(payload["edge_type"]).tolist()
    return payload


def _tensor_hash(tensor: dict[str, Any]) -> str:
    return hash_without(_serializable_tensor(tensor), "tensor_hash")


def tensor_to_jsonable(tensor: dict[str, Any]) -> dict[str, Any]:
    return _serializable_tensor(tensor)


def validate_tensor_artifact(tensor: dict[str, Any]) -> None:
    required = {
        "input_graph_hash",
        "tensorizer_version",
        "edge_relation_schema",
        "node_feature_schema",
        "feature_width",
        "padding_policy",
        "missing_value_policy",
        "node_features",
        "edge_index",
        "edge_type",
        "warp_partitions",
        "graph_batch_metadata",
        "tensor_hash",
    }
    missing = required.difference(tensor)
    if missing:
        raise ValueError(f"tensor artifact missing required fields: {sorted(missing)}")
    schema = tensor["node_feature_schema"]
    if schema.get("schema_name") != NODE_FEATURE_SCHEMA_NAME:
        raise ValueError("unexpected node_feature_schema")
    if tensor.get("tensorizer_version") != TENSORIZER_VERSION:
        raise ValueError("unexpected tensorizer_version")
    if tensor.get("feature_width") != FEATURE_WIDTH:
        raise ValueError("feature_width must be 64")
    if tensor.get("padding_policy") != PADDING_POLICY:
        raise ValueError("padding_policy must be strict_zero_padding")
    if tensor.get("missing_value_policy") != MISSING_VALUE_POLICY:
        raise ValueError("unexpected missing_value_policy")
    if schema.get("paper_reproduction_mode") != PAPER_REPRODUCTION_MODE:
        raise ValueError("unexpected paper_reproduction_mode")
    if schema.get("instruction_feature_combine") != INSTRUCTION_FEATURE_COMBINE:
        raise ValueError("instruction_feature_combine must be concat_opcode63_normalized_pc1")

    features = np.asarray(tensor["node_features"])
    if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
        raise ValueError("node_features must have shape [node_count, 64]")
    edge_index = np.asarray(tensor["edge_index"])
    edge_type = np.asarray(tensor["edge_type"])
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edge_count]")
    if edge_index.shape[1] != edge_type.shape[0]:
        raise ValueError("edge_index and edge_type length mismatch")
    for idx, node_type in enumerate(tensor.get("node_types", [])):
        if node_type in VARIABLE_NODE_TYPES and not np.allclose(features[idx, 40:64], 0.0):
            raise ValueError("variable zero padding must remain zero")
        if node_type == "pseudo" and not np.allclose(features[idx, 16:64], 0.0):
            raise ValueError("pseudo zero padding must remain zero")
    if not np.isfinite(features).all():
        raise ValueError("node_features must be finite")
    if "trace_index" in schema.get("instruction_feature_combine", ""):
        raise ValueError("trace index positional encoding is not allowed in Phase A")
    if tensor["tensor_hash"] != _tensor_hash(tensor):
        raise ValueError("tensor_hash is not reproducible")


def tensor_from_jsonable(tensor: dict[str, Any]) -> dict[str, Any]:
    restored = dict(tensor)
    restored["node_features"] = np.asarray(restored["node_features"], dtype=np.float32)
    restored["edge_index"] = np.asarray(restored["edge_index"], dtype=np.int64)
    restored["edge_type"] = np.asarray(restored["edge_type"], dtype=np.int64)
    validate_tensor_artifact(restored)
    return restored
