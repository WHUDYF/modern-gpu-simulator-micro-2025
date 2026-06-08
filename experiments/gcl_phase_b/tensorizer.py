"""Tensorization for GCL Phase B representative-SM canonical graphs."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.gcl_phase_a.tensorizer import (
    FEATURE_WIDTH,
    MISSING_VALUE_POLICY,
    NODE_FEATURE_SCHEMA_NAME,
    PADDING_POLICY,
    PAPER_REPRODUCTION_MODE,
    _dynamic_value_statistics,
    _normalized_pc,
    _token_embedding,
    node_feature_schema,
)

from .graph_builder import EDGE_RELATION_SCHEMA, VARIABLE_NODE_TYPES, validate_phase_b_graph_artifact
from .utils import hash_without

PHASE_B_TENSORIZER_VERSION = "gcl_phase_b_tensorizer_v1"
FUNCTIONAL_FIRST_PAPER_MODE = "functional_first_no_pseudo_node_not_strict_reproduction"


def tensorize_phase_b_graphs(graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not graphs:
        raise ValueError("Phase B tensorization requires graphs")
    return [tensorize_phase_b_graph(graph) for graph in graphs]


def tensorize_phase_b_graph(graph: dict[str, Any]) -> dict[str, Any]:
    validate_phase_b_graph_artifact(graph)
    nodes = graph["nodes"]
    node_index = {node["node_id"]: index for index, node in enumerate(nodes)}
    pcs = [node["pc"] for node in nodes if node["node_type"] == "instruction"]
    pc_min = min(pcs)
    pc_max = max(pcs)
    features = np.zeros((len(nodes), FEATURE_WIDTH), dtype=np.float32)

    for index, node in enumerate(nodes):
        if node["node_type"] == "instruction":
            features[index, 0:63] = _token_embedding(node["opcode"], 63)
            features[index, 63] = _normalized_pc(node, pc_min, pc_max)
        elif node["node_type"] in VARIABLE_NODE_TYPES:
            features[index, 0:32] = _token_embedding(node["token"], 32)
            features[index, 32:40] = _dynamic_value_statistics(node.get("observed_dynamic_values", []))
        elif node["node_type"] == "pseudo":
            features[index, 0:16] = _token_embedding(node["pseudo_kind"], 16)
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
    edge_offset_by_id = {edge["edge_id"]: offset for offset, edge in enumerate(graph["edges"])}
    warp_partitions = {
        partition_id: [node_index[node_id] for node_id in partition["node_ids"]]
        for partition_id, partition in graph["warp_partitions"].items()
    }
    warp_partition_tensors = {
        partition_id: {
            "partition_id": partition_id,
            "cta_id": partition["cta_id"],
            "warp_id": partition["warp_id"],
            "node_indices": [node_index[node_id] for node_id in partition["node_ids"]],
            "edge_indices": _partition_edge_indices(partition, edge_offset_by_id),
            "instruction_count": partition["instruction_count"],
        }
        for partition_id, partition in graph["warp_partitions"].items()
    }
    tensor = {
        "artifact_type": "phase_b_graph_tensor",
        "graph_id": graph["graph_id"],
        "kernel_invocation_id": graph["kernel_invocation_id"],
        "input_graph_hash": graph["graph_hash"],
        "tensorizer_version": PHASE_B_TENSORIZER_VERSION,
        "phase_b_tensorizer_version": PHASE_B_TENSORIZER_VERSION,
        "node_feature_schema": node_feature_schema(),
        "edge_relation_schema": EDGE_RELATION_SCHEMA,
        "feature_width": FEATURE_WIDTH,
        "representation_mode": _representation_mode(graph),
        "pseudo_node_mode": _pseudo_node_mode(graph),
        "paper_reproduction_mode": _paper_reproduction_mode(graph),
        "padding_policy": PADDING_POLICY,
        "missing_value_policy": MISSING_VALUE_POLICY,
        "node_ids": [node["node_id"] for node in nodes],
        "node_types": [node["node_type"] for node in nodes],
        "node_features": features,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "warp_partitions": warp_partitions,
        "warp_partition_tensors": warp_partition_tensors,
        "graph_batch_metadata": {
            "graph_id": graph["graph_id"],
            "kernel_invocation_id": graph["kernel_invocation_id"],
            "artifact_status": graph.get("artifact_status", "formal"),
            "formal_input_eligible": graph.get("formal_input_eligible", True),
            "workload_id": graph.get("workload_id"),
            "execution_mode": graph.get("execution_mode"),
            "trace_source": graph.get("trace_source"),
            "input_scope": graph.get("input_scope"),
            "scheduler_metadata_source": graph.get("scheduler_metadata_source"),
            "collection_scope": graph["collection_scope"],
            "trace_family": graph.get("trace_family"),
            "selected_sm": graph["selected_sm"],
            "node_count": len(nodes),
            "edge_count": len(graph["edges"]),
            "warp_count": len(graph["warp_partitions"]),
            "source_graph_hash": graph["graph_hash"],
        },
    }
    tensor["tensor_hash"] = _tensor_hash(tensor)
    validate_phase_b_tensor_artifact(tensor)
    return tensor


def _pseudo_node_mode(graph: dict[str, Any]) -> str:
    pseudo_nodes = [node for node in graph["nodes"] if node["node_type"] == "pseudo"]
    if not pseudo_nodes:
        return "no_pseudo_node"
    if all(node.get("pseudo_kind") == "mem_ref" for node in pseudo_nodes):
        return "mem_ref_only"
    raise ValueError("unsupported pseudo node mode")


def _representation_mode(graph: dict[str, Any]) -> str:
    mode = _pseudo_node_mode(graph)
    if mode == "mem_ref_only":
        return "gcl_resnet50_mem_ref_only"
    if mode == "no_pseudo_node":
        return "gcl_resnet50_no_pseudo_node"
    raise ValueError("unsupported representation mode")


def _paper_reproduction_mode(graph: dict[str, Any]) -> str:
    mode = _pseudo_node_mode(graph)
    if mode == "mem_ref_only":
        return PAPER_REPRODUCTION_MODE
    if mode == "no_pseudo_node":
        return FUNCTIONAL_FIRST_PAPER_MODE
    raise ValueError("unsupported paper reproduction mode")


def _partition_edge_indices(
    partition: dict[str, Any],
    edge_offset_by_id: dict[str, int],
) -> list[int]:
    return [edge_offset_by_id[edge_id] for edge_id in partition["edge_ids"]]


def _serializable_tensor(tensor: dict[str, Any]) -> dict[str, Any]:
    payload = dict(tensor)
    payload["node_features"] = np.asarray(payload["node_features"], dtype=np.float32).tolist()
    payload["edge_index"] = np.asarray(payload["edge_index"], dtype=np.int64).tolist()
    payload["edge_type"] = np.asarray(payload["edge_type"], dtype=np.int64).tolist()
    return payload


def _tensor_hash(tensor: dict[str, Any]) -> str:
    return hash_without(_serializable_tensor(tensor), "tensor_hash")


def tensor_to_jsonable(tensor: dict[str, Any]) -> dict[str, Any]:
    return _serializable_tensor(tensor)


def tensor_from_jsonable(tensor: dict[str, Any]) -> dict[str, Any]:
    restored = dict(tensor)
    restored["node_features"] = np.asarray(restored["node_features"], dtype=np.float32)
    restored["edge_index"] = np.asarray(restored["edge_index"], dtype=np.int64)
    restored["edge_type"] = np.asarray(restored["edge_type"], dtype=np.int64)
    validate_phase_b_tensor_artifact(restored)
    return restored


def validate_phase_b_tensor_artifact(tensor: dict[str, Any]) -> None:
    required = {
        "input_graph_hash",
        "tensorizer_version",
        "phase_b_tensorizer_version",
        "edge_relation_schema",
        "node_feature_schema",
        "feature_width",
        "representation_mode",
        "pseudo_node_mode",
        "paper_reproduction_mode",
        "padding_policy",
        "missing_value_policy",
        "node_features",
        "edge_index",
        "edge_type",
        "warp_partitions",
        "warp_partition_tensors",
        "graph_batch_metadata",
        "tensor_hash",
    }
    missing = required.difference(tensor)
    if missing:
        raise ValueError(f"Phase B tensor artifact missing required fields: {sorted(missing)}")
    if tensor["phase_b_tensorizer_version"] != PHASE_B_TENSORIZER_VERSION:
        raise ValueError("unexpected phase_b_tensorizer_version")
    if tensor["tensorizer_version"] != PHASE_B_TENSORIZER_VERSION:
        raise ValueError("unexpected tensorizer_version")
    schema = tensor["node_feature_schema"]
    if schema.get("schema_name") != NODE_FEATURE_SCHEMA_NAME:
        raise ValueError("unexpected node_feature_schema")
    if schema.get("paper_reproduction_mode") != PAPER_REPRODUCTION_MODE:
        raise ValueError("unexpected paper_reproduction_mode")
    if tensor["feature_width"] != FEATURE_WIDTH:
        raise ValueError("feature_width must be 64")
    if tensor["representation_mode"] not in {
        "gcl_resnet50_mem_ref_only",
        "gcl_resnet50_no_pseudo_node",
    }:
        raise ValueError("unsupported representation_mode")
    if tensor["pseudo_node_mode"] not in {"mem_ref_only", "no_pseudo_node"}:
        raise ValueError("unsupported pseudo_node_mode")
    if tensor["representation_mode"] == "gcl_resnet50_mem_ref_only" and (
        tensor["paper_reproduction_mode"] != PAPER_REPRODUCTION_MODE
    ):
        raise ValueError("unexpected paper_reproduction_mode")
    if tensor["representation_mode"] == "gcl_resnet50_no_pseudo_node" and (
        tensor["paper_reproduction_mode"] != FUNCTIONAL_FIRST_PAPER_MODE
    ):
        raise ValueError("no-pseudo mode must not be marked strict paper reproduction")
    if tensor["padding_policy"] != PADDING_POLICY:
        raise ValueError("padding_policy must be strict_zero_padding")
    if tensor["missing_value_policy"] != MISSING_VALUE_POLICY:
        raise ValueError("unexpected missing_value_policy")

    features = np.asarray(tensor["node_features"])
    edge_index = np.asarray(tensor["edge_index"])
    edge_type = np.asarray(tensor["edge_type"])
    if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
        raise ValueError("node_features must have shape [node_count, 64]")
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edge_count]")
    if edge_index.shape[1] != edge_type.shape[0]:
        raise ValueError("edge_index and edge_type length mismatch")
    if not np.isfinite(features).all():
        raise ValueError("node_features must be finite")
    for index, node_type in enumerate(tensor.get("node_types", [])):
        if node_type in VARIABLE_NODE_TYPES and not np.allclose(features[index, 40:64], 0.0):
            raise ValueError("variable zero padding must remain zero")
        if node_type == "pseudo" and not np.allclose(features[index, 16:64], 0.0):
            raise ValueError("pseudo zero padding must remain zero")
    node_count = features.shape[0]
    for partition_id, indices in tensor["warp_partitions"].items():
        if not indices:
            raise ValueError(f"warp partition {partition_id} must contain node indices")
        if any(index < 0 or index >= node_count for index in indices):
            raise ValueError("warp partition contains invalid node index")
    for partition in tensor["warp_partition_tensors"].values():
        node_indices = partition.get("node_indices", [])
        if not node_indices:
            raise ValueError("warp partition tensor must contain node indices")
        if any(index < 0 or index >= node_count for index in node_indices):
            raise ValueError("warp partition tensor contains invalid node index")
        edge_indices = partition.get("edge_indices", [])
        if any(index < 0 or index >= edge_type.shape[0] for index in edge_indices):
            raise ValueError("warp partition tensor contains invalid edge index")
    if tensor["tensor_hash"] != _tensor_hash(tensor):
        raise ValueError("tensor_hash is not reproducible")
