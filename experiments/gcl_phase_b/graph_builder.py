"""Per-warp canonical graph construction for GCL Phase B."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .trace_fixture import COLLECTION_SCOPE
from .utils import hash_without

EDGE_RELATION_SCHEMA = {
    "control_flow": 0,
    "data_source": 1,
    "data_destination": 2,
}
GRAPH_HASH_EXCLUDED_FIELDS = ("graph_hash",)
VARIABLE_NODE_TYPES = {"register_version", "input_variable", "unknown_variable"}


def _is_raw_register_token(token: str) -> bool:
    return token.startswith(("R", "P")) and ".v" not in token


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


def _versioned_register_token(token: str, warp_id: int, version: int) -> str:
    if _is_raw_register_token(token):
        return f"{token}.v{version}.w{warp_id}"
    return token


def _warp_scoped_variable_token(token: str, partition_id: str) -> str:
    if token.startswith(("input:", "unknown:")) and ".wp" not in token:
        safe_partition = partition_id.replace(":", "_")
        return f"{token}.wp{safe_partition}"
    return token


def _resolve_source_token(
    token: str,
    warp_id: int,
    partition_id: str,
    register_versions: dict[str, int],
) -> str:
    if not _is_raw_register_token(token):
        return _warp_scoped_variable_token(token, partition_id)
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


def _instruction_id(partition_id: str, trace_index: int) -> str:
    return f"i:wp{partition_id}:t{trace_index}"


def _variable_id(partition_id: str, token: str) -> str:
    return f"v:wp{partition_id}:{_semantic_node_type(token)}:{token}"


def _mem_ref_id(partition_id: str, trace_index: int) -> str:
    return f"p:wp{partition_id}:mem_ref:t{trace_index}"


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
    partition_id: str,
    cta_id: str,
    warp_id: int,
    token: str,
    values: list[float],
) -> str:
    node_id = _variable_id(partition_id, token)
    if node_id not in node_by_id:
        node = {
            "node_id": node_id,
            "node_type": _semantic_node_type(token),
            "token": token,
            "cta_id": cta_id,
            "warp_id": warp_id,
            "warp_partition_id": partition_id,
            "observed_dynamic_values": [],
        }
        node_by_id[node_id] = node
        nodes.append(node)
    node_by_id[node_id]["observed_dynamic_values"].extend(values)
    return node_id


def _edge(edge_id: str, source: str, target: str, relation: str, partition_id: str) -> dict[str, str]:
    return {
        "edge_id": edge_id,
        "source": source,
        "target": target,
        "relation": relation,
        "warp_partition_id": partition_id,
    }


def _append_edge(
    edges: list[dict[str, str]],
    source: str,
    target: str,
    relation: str,
    partition_id: str,
) -> None:
    edge_id = f"e:{len(edges):05d}"
    edges.append(_edge(edge_id, source, target, relation, partition_id))


def build_phase_b_graphs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("Phase B graph construction requires trace records")
    return [build_phase_b_graph(record) for record in records]


def build_phase_b_graph(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("collection_scope") != COLLECTION_SCOPE:
        raise ValueError("Phase B graph record must use representative-SM scope")

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    partitions: dict[str, dict[str, Any]] = {}

    for warp in sorted(record["warps"], key=lambda item: item["warp_partition_id"]):
        partition_id = warp["warp_partition_id"]
        cta_id = warp["cta_id"]
        warp_id = warp["warp_id"]
        entries = sorted(warp["entries"], key=lambda entry: entry["trace_index"])
        if not entries:
            raise ValueError("non-empty warp partition requires trace entries")
        partition_node_ids: list[str] = []
        partition_edge_ids: list[str] = []
        instruction_ids: list[str] = []
        previous_instruction_id: str | None = None
        register_versions: dict[str, int] = {}

        for entry in entries:
            instruction_id = _instruction_id(partition_id, entry["trace_index"])
            instruction_node = {
                "node_id": instruction_id,
                "node_type": "instruction",
                "opcode": entry["opcode"],
                "pc": entry["pc"],
                "cta_id": cta_id,
                "warp_id": warp_id,
                "warp_partition_id": partition_id,
                "trace_index": entry["trace_index"],
                "active_mask": entry["active_mask"],
                "source_entry_hash": entry["source_entry_hash"],
            }
            node_by_id[instruction_id] = instruction_node
            nodes.append(instruction_node)
            partition_node_ids.append(instruction_id)
            instruction_ids.append(instruction_id)

            if previous_instruction_id is not None:
                before = len(edges)
                _append_edge(edges, previous_instruction_id, instruction_id, "control_flow", partition_id)
                partition_edge_ids.extend(edge["edge_id"] for edge in edges[before:])
            previous_instruction_id = instruction_id

            address_positions = _address_source_positions(entry)
            observed_values = entry.get("observed_dynamic_values", [])
            for source_index, token in enumerate(entry["source_operands"]):
                resolved = _resolve_source_token(token, warp_id, partition_id, register_versions)
                variable_id = _add_variable_node(
                    nodes, node_by_id, partition_id, cta_id, warp_id, resolved, observed_values
                )
                if variable_id not in partition_node_ids:
                    partition_node_ids.append(variable_id)
                before = len(edges)
                if source_index in address_positions:
                    instruction_node["address_source_node_id"] = variable_id
                    mem_ref_id = _mem_ref_id(partition_id, entry["trace_index"])
                    if mem_ref_id not in node_by_id:
                        mem_node = {
                            "node_id": mem_ref_id,
                            "node_type": "pseudo",
                            "pseudo_kind": "mem_ref",
                            "cta_id": cta_id,
                            "warp_id": warp_id,
                            "warp_partition_id": partition_id,
                            "trace_index": entry["trace_index"],
                            "source_entry_hash": entry["source_entry_hash"],
                        }
                        node_by_id[mem_ref_id] = mem_node
                        nodes.append(mem_node)
                    if mem_ref_id not in partition_node_ids:
                        partition_node_ids.append(mem_ref_id)
                    _append_edge(edges, variable_id, mem_ref_id, "data_source", partition_id)
                    _append_edge(edges, mem_ref_id, instruction_id, "data_source", partition_id)
                else:
                    _append_edge(edges, variable_id, instruction_id, "data_source", partition_id)
                partition_edge_ids.extend(edge["edge_id"] for edge in edges[before:])

            for token in entry["destination_operands"]:
                resolved = _resolve_destination_token(token, warp_id, register_versions)
                variable_id = _add_variable_node(
                    nodes, node_by_id, partition_id, cta_id, warp_id, resolved, observed_values
                )
                if variable_id not in partition_node_ids:
                    partition_node_ids.append(variable_id)
                before = len(edges)
                _append_edge(edges, instruction_id, variable_id, "data_destination", partition_id)
                partition_edge_ids.extend(edge["edge_id"] for edge in edges[before:])

        partitions[partition_id] = {
            "partition_id": partition_id,
            "cta_id": cta_id,
            "warp_id": warp_id,
            "node_ids": partition_node_ids,
            "edge_ids": partition_edge_ids,
            "instruction_node_ids": instruction_ids,
            "instruction_count": len(instruction_ids),
            "node_count": len(partition_node_ids),
            "edge_count": len(partition_edge_ids),
            "first_trace_index": entries[0]["trace_index"],
            "last_trace_index": entries[-1]["trace_index"],
        }

    node_type_counts = Counter(node["node_type"] for node in nodes)
    edge_type_counts = Counter(edge["relation"] for edge in edges)
    graph = {
        "artifact_type": "phase_b_canonical_graph",
        "graph_id": f"phase_b_graph:{record['kernel_invocation_id']}",
        "kernel_invocation_id": record["kernel_invocation_id"],
        "trace_family": record["trace_family"],
        "collection_scope": COLLECTION_SCOPE,
        "selected_sm": record["selected_sm"],
        "included_cta_ids": record["included_cta_ids"],
        "selected_sm_policy_report_hash": record["selected_sm_policy_report_hash"],
        "source_trace_hash": record["source_trace_hash"],
        "nodes": nodes,
        "edges": edges,
        "edge_relation_schema": EDGE_RELATION_SCHEMA,
        "warp_partitions": partitions,
        "graph_summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "instruction_node_count": node_type_counts["instruction"],
            "variable_node_count": sum(node_type_counts[node_type] for node_type in VARIABLE_NODE_TYPES),
            "pseudo_node_count": node_type_counts["pseudo"],
            "warp_count": len(partitions),
            "node_type_counts": dict(sorted(node_type_counts.items())),
            "edge_type_counts": dict(sorted(edge_type_counts.items())),
        },
    }
    graph["graph_hash"] = hash_without(graph, *GRAPH_HASH_EXCLUDED_FIELDS)
    validate_phase_b_graph_artifact(graph)
    return graph


def validate_phase_b_graph_artifact(graph: dict[str, Any]) -> None:
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
        raise ValueError(f"Phase B graph missing required fields: {sorted(missing)}")
    if graph["collection_scope"] != COLLECTION_SCOPE:
        raise ValueError("Phase B graph collection_scope must be single_representative_sm_all_ctas")
    node_by_id = {node["node_id"]: node for node in graph["nodes"]}
    if len(node_by_id) != len(graph["nodes"]):
        raise ValueError("node_id values must be unique")
    for edge in graph["edges"]:
        if edge["source"] not in node_by_id or edge["target"] not in node_by_id:
            raise ValueError("edge references unknown node")
        if edge["relation"] not in EDGE_RELATION_SCHEMA:
            raise ValueError("edge relation is not in schema")
        source = node_by_id[edge["source"]]
        target = node_by_id[edge["target"]]
        if source["warp_partition_id"] != target["warp_partition_id"]:
            if edge["relation"] == "control_flow":
                raise ValueError("control_flow edges must not cross warp partitions")
            raise ValueError("edges must stay inside a warp partition")
        if edge["warp_partition_id"] != source["warp_partition_id"]:
            raise ValueError("edge partition metadata mismatch")
        if edge["relation"] == "control_flow":
            if source["node_type"] != "instruction" or target["node_type"] != "instruction":
                raise ValueError("control_flow edges must connect instruction nodes")
            if source["warp_partition_id"] != target["warp_partition_id"]:
                raise ValueError("control_flow edges must not cross warp partitions")
    if any("edge_id" not in edge for edge in graph["edges"]):
        raise ValueError("edge_id is required for every edge")
    edge_by_id = {edge["edge_id"]: edge for edge in graph["edges"]}
    if len(edge_by_id) != len(graph["edges"]):
        raise ValueError("edge_id values must be unique")

    if not graph["warp_partitions"]:
        raise ValueError("warp_partitions must not be empty")
    node_partition_hits: defaultdict[str, int] = defaultdict(int)
    for partition_id, partition in graph["warp_partitions"].items():
        required_partition = {
            "warp_id",
            "cta_id",
            "node_ids",
            "edge_ids",
            "instruction_count",
            "node_count",
            "edge_count",
            "first_trace_index",
            "last_trace_index",
        }
        missing_partition = required_partition.difference(partition)
        if missing_partition:
            raise ValueError(f"warp partition missing required fields: {sorted(missing_partition)}")
        if partition["instruction_count"] <= 0:
            raise ValueError("non-empty warp partition must contain instruction nodes")
        if partition["node_count"] != len(partition["node_ids"]):
            raise ValueError("partition node_count mismatch")
        if partition["edge_count"] != len(partition["edge_ids"]):
            raise ValueError("partition edge_count mismatch")
        for node_id in partition["node_ids"]:
            if node_id not in node_by_id:
                raise ValueError("partition references unknown node")
            if node_by_id[node_id]["warp_partition_id"] != partition_id:
                raise ValueError("partition node metadata mismatch")
            node_partition_hits[node_id] += 1
        for edge_id in partition["edge_ids"]:
            if edge_id not in edge_by_id:
                raise ValueError("partition references unknown edge")
            if edge_by_id[edge_id]["warp_partition_id"] != partition_id:
                raise ValueError("partition edge metadata mismatch")
    if set(node_partition_hits) != set(node_by_id) or any(count != 1 for count in node_partition_hits.values()):
        raise ValueError("each graph node must belong to exactly one warp partition")

    observed_order: dict[str, list[int]] = defaultdict(list)
    for node in graph["nodes"]:
        if node["node_type"] == "instruction":
            observed_order[node["warp_partition_id"]].append(node["trace_index"])
    control_edges = {
        (edge["source"], edge["target"])
        for edge in graph["edges"]
        if edge["relation"] == "control_flow"
    }
    expected_control_edges = set()
    for partition_id, trace_indices in observed_order.items():
        if trace_indices != sorted(trace_indices):
            raise ValueError(f"ordering violation in warp partition {partition_id}")
        instruction_ids = [
            node["node_id"]
            for node in graph["nodes"]
            if node["node_type"] == "instruction" and node["warp_partition_id"] == partition_id
        ]
        for source_id, target_id in zip(instruction_ids, instruction_ids[1:]):
            expected_control_edges.add((source_id, target_id))
            if (source_id, target_id) not in control_edges:
                raise ValueError("missing consecutive control_flow edge")
    if control_edges != expected_control_edges:
        raise ValueError("control_flow edges must connect only consecutive instruction nodes")

    if graph["graph_summary"]["node_count"] != len(graph["nodes"]):
        raise ValueError("graph_summary node_count mismatch")
    if graph["graph_summary"]["edge_count"] != len(graph["edges"]):
        raise ValueError("graph_summary edge_count mismatch")
    if graph["graph_hash"] != hash_without(graph, *GRAPH_HASH_EXCLUDED_FIELDS):
        raise ValueError("graph_hash is not reproducible")
