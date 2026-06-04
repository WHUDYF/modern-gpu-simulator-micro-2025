"""Canonical graph construction for GCL Phase A."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .trace_fixture import COLLECTION_SCOPE, validate_trace_fixture
from .utils import hash_without

EDGE_RELATION_SCHEMA = {
    "control_flow": 0,
    "data_source": 1,
    "data_destination": 2,
}
GRAPH_HASH_EXCLUDED_FIELDS = ("graph_hash", "source_fixture_hash")
VARIABLE_NODE_TYPES = {"register_version", "input_variable", "unknown_variable"}


def _instruction_id(entry: dict[str, Any]) -> str:
    return f"i:w{entry['warp_id']}:t{entry['trace_index']}"


def _variable_id(token: str) -> str:
    return f"{_semantic_node_type(token)}:{token}"


def _mem_ref_id(entry: dict[str, Any]) -> str:
    return f"p:mem_ref:w{entry['warp_id']}:t{entry['trace_index']}"


def _semantic_node_type(token: str) -> str:
    if token.startswith(("R", "P")):
        return "register_version"
    if token.startswith("input:"):
        return "input_variable"
    return "unknown_variable"


def _is_memory_opcode(opcode: str) -> bool:
    return opcode.startswith("LDG") or opcode.startswith("STG")


def _is_address_token(token: str) -> bool:
    return token.startswith("R")


def _is_raw_register_token(token: str) -> bool:
    return token.startswith(("R", "P")) and ".v" not in token


def _versioned_register_token(token: str, warp_id: int, version: int) -> str:
    if _is_raw_register_token(token):
        return f"{token}.v{version}.w{warp_id}"
    return token


def _warp_scoped_variable_token(token: str, warp_id: int) -> str:
    if token.startswith(("input:", "unknown:")) and ".w" not in token:
        return f"{token}.w{warp_id}"
    return token


def _resolve_source_token(
    token: str,
    warp_id: int,
    register_versions: dict[str, int],
) -> str:
    if not _is_raw_register_token(token):
        return _warp_scoped_variable_token(token, warp_id)
    version = register_versions.get(token, 0)
    return _versioned_register_token(token, warp_id, version)


def _resolve_destination_token(
    token: str,
    warp_id: int,
    register_versions: dict[str, int],
) -> str:
    if not _is_raw_register_token(token):
        return token
    version = register_versions.get(token, 0) + 1
    register_versions[token] = version
    return _versioned_register_token(token, warp_id, version)


def _address_source_positions(entry: dict[str, Any]) -> set[int]:
    if not _is_memory_opcode(entry["opcode"]):
        return set()
    for index, token in enumerate(entry["source_operands"]):
        if _is_address_token(token):
            return {index}
    return set()


def _add_variable_node(
    nodes: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    token: str,
    values: list[float],
) -> str:
    node_id = _variable_id(token)
    if node_id not in node_by_id:
        node = {
            "node_id": node_id,
            "node_type": _semantic_node_type(token),
            "token": token,
            "observed_dynamic_values": [],
        }
        node_by_id[node_id] = node
        nodes.append(node)
    node_by_id[node_id]["observed_dynamic_values"].extend(values)
    return node_id


def _edge(source: str, target: str, relation: str) -> dict[str, str]:
    return {"source": source, "target": target, "relation": relation}


def build_canonical_graphs(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    validate_trace_fixture(fixture)
    return [build_canonical_graph(record, fixture["fixture_hash"]) for record in fixture["records"]]


def build_canonical_graph(record: dict[str, Any], fixture_hash: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    warp_partitions: dict[str, list[str]] = {}

    for warp in record["warps"]:
        warp_key = str(warp["warp_id"])
        instruction_ids: list[str] = []
        previous_instruction_id: str | None = None
        register_versions: dict[str, int] = {}
        for entry in warp["entries"]:
            instruction_id = _instruction_id(entry)
            instruction_node = {
                "node_id": instruction_id,
                "node_type": "instruction",
                "opcode": entry["opcode"],
                "pc": entry["pc"],
                "warp_id": entry["warp_id"],
                "trace_index": entry["trace_index"],
                "active_mask": entry["active_mask"],
                "source_entry_hash": entry["source_entry_hash"],
            }
            address_positions = _address_source_positions(entry)
            node_by_id[instruction_id] = instruction_node
            nodes.append(instruction_node)
            instruction_ids.append(instruction_id)

            if previous_instruction_id is not None:
                edges.append(_edge(previous_instruction_id, instruction_id, "control_flow"))
            previous_instruction_id = instruction_id

            observed_values = entry.get("observed_dynamic_values", [])
            for source_index, token in enumerate(entry["source_operands"]):
                resolved_token = _resolve_source_token(token, entry["warp_id"], register_versions)
                variable_id = _add_variable_node(nodes, node_by_id, resolved_token, observed_values)
                if source_index in address_positions:
                    instruction_node["address_source_node_id"] = variable_id
                    mem_ref_id = _mem_ref_id(entry)
                    if mem_ref_id not in node_by_id:
                        mem_node = {
                            "node_id": mem_ref_id,
                            "node_type": "pseudo",
                            "pseudo_kind": "mem_ref",
                            "warp_id": entry["warp_id"],
                            "trace_index": entry["trace_index"],
                            "source_entry_hash": entry["source_entry_hash"],
                        }
                        node_by_id[mem_ref_id] = mem_node
                        nodes.append(mem_node)
                    edges.append(_edge(variable_id, mem_ref_id, "data_source"))
                    edges.append(_edge(mem_ref_id, instruction_id, "data_source"))
                else:
                    edges.append(_edge(variable_id, instruction_id, "data_source"))

            for token in entry["destination_operands"]:
                resolved_token = _resolve_destination_token(token, entry["warp_id"], register_versions)
                variable_id = _add_variable_node(nodes, node_by_id, resolved_token, observed_values)
                edges.append(_edge(instruction_id, variable_id, "data_destination"))

        warp_partitions[warp_key] = instruction_ids

    graph = {
        "artifact_type": "canonical_graph",
        "graph_id": f"graph:{record['kernel_invocation_id']}",
        "kernel_invocation_id": record["kernel_invocation_id"],
        "trace_family": record["trace_family"],
        "collection_scope": COLLECTION_SCOPE,
        "source_fixture_hash": fixture_hash,
        "nodes": nodes,
        "edges": edges,
        "edge_relation_schema": EDGE_RELATION_SCHEMA,
        "warp_partitions": warp_partitions,
        "graph_summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "instruction_node_count": sum(1 for node in nodes if node["node_type"] == "instruction"),
            "variable_node_count": sum(1 for node in nodes if node["node_type"] in VARIABLE_NODE_TYPES),
            "pseudo_node_count": sum(1 for node in nodes if node["node_type"] == "pseudo"),
        },
    }
    graph["graph_hash"] = hash_without(graph, *GRAPH_HASH_EXCLUDED_FIELDS)
    validate_graph_artifact(graph)
    return graph


def validate_graph_artifact(graph: dict[str, Any]) -> None:
    required = {
        "graph_id",
        "kernel_invocation_id",
        "collection_scope",
        "nodes",
        "edges",
        "warp_partitions",
        "graph_summary",
        "graph_hash",
    }
    missing = required.difference(graph)
    if missing:
        raise ValueError(f"graph artifact missing required fields: {sorted(missing)}")
    if graph["collection_scope"] != COLLECTION_SCOPE:
        raise ValueError("graph collection_scope must be selected_warps_fixture")
    node_by_id = {node["node_id"]: node for node in graph["nodes"]}
    if len(node_by_id) != len(graph["nodes"]):
        raise ValueError("node_id values must be unique")
    for edge in graph["edges"]:
        if edge["source"] not in node_by_id or edge["target"] not in node_by_id:
            raise ValueError("edge references unknown node")
        if edge["relation"] not in EDGE_RELATION_SCHEMA:
            raise ValueError("edge relation is not in schema")
        source_node = node_by_id[edge["source"]]
        target_node = node_by_id[edge["target"]]
        source_type = source_node["node_type"]
        target_type = target_node["node_type"]
        if edge["relation"] == "control_flow":
            if source_type != "instruction" or target_type != "instruction":
                raise ValueError("control_flow edges must connect instruction nodes only")
            if source_node["warp_id"] != target_node["warp_id"]:
                raise ValueError("control_flow edges must stay within a warp")
            if target_node["trace_index"] != source_node["trace_index"] + 1:
                raise ValueError("control_flow edges must connect consecutive instruction nodes")
    if not graph["warp_partitions"]:
        raise ValueError("warp_partitions must not be empty")

    observed_instruction_order: dict[str, list[int]] = defaultdict(list)
    for node in graph["nodes"]:
        if node["node_type"] == "instruction":
            observed_instruction_order[str(node["warp_id"])].append(node["trace_index"])
    for warp_id, trace_indices in observed_instruction_order.items():
        if trace_indices != sorted(trace_indices):
            raise ValueError(f"ordering violation in warp {warp_id}")
        expected_node_ids = [
            node["node_id"]
            for node in graph["nodes"]
            if node["node_type"] == "instruction" and str(node["warp_id"]) == warp_id
        ]
        if graph["warp_partitions"].get(warp_id) != expected_node_ids:
            raise ValueError("warp_partitions must match instruction node order")
        control_edges = {
            (edge["source"], edge["target"])
            for edge in graph["edges"]
            if edge["relation"] == "control_flow"
        }
        for source_id, target_id in zip(expected_node_ids, expected_node_ids[1:]):
            if (source_id, target_id) not in control_edges:
                raise ValueError("missing consecutive control_flow edge")

    data_edges = {
        (edge["source"], edge["target"])
        for edge in graph["edges"]
        if edge["relation"] == "data_source"
    }
    for memory_node in (
        node for node in graph["nodes"] if node["node_type"] == "instruction" and _is_memory_opcode(node["opcode"])
    ):
        mem_ref_id = f"p:mem_ref:w{memory_node['warp_id']}:t{memory_node['trace_index']}"
        expected_address_source = memory_node.get("address_source_node_id")
        if not expected_address_source:
            raise ValueError("memory instructions require address_source_node_id metadata")
        if mem_ref_id not in node_by_id:
            raise ValueError("memory instructions require mem_ref pseudo nodes")
        if (mem_ref_id, memory_node["node_id"]) not in data_edges:
            raise ValueError("memory instructions must receive data-flow from mem_ref pseudo nodes")
        if expected_address_source not in node_by_id:
            raise ValueError("memory instructions require known address source node")
        if node_by_id[expected_address_source]["node_type"] != "register_version":
            raise ValueError("memory instructions require register_version address source")
        if (expected_address_source, mem_ref_id) not in data_edges:
            raise ValueError("memory instructions require exact address variable to mem_ref data-flow")
    for node in graph["nodes"]:
        if node["node_type"] == "pseudo" and node.get("pseudo_kind") != "mem_ref":
            raise ValueError("Phase A only supports mem_ref pseudo nodes")
    if graph["graph_hash"] != hash_without(graph, *GRAPH_HASH_EXCLUDED_FIELDS):
        raise ValueError("graph_hash is not reproducible")
