import copy

import pytest

from experiments.gcl_phase_b.graph_builder import (
    GRAPH_HASH_EXCLUDED_FIELDS,
    build_phase_b_graphs,
    validate_phase_b_graph_artifact,
)
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records
from experiments.gcl_phase_b.utils import hash_without


def _graph():
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    return build_phase_b_graphs(records)[0]


def test_builds_per_warp_graphs_then_kernel_union():
    graph = _graph()

    validate_phase_b_graph_artifact(graph)
    assert graph["collection_scope"] == "single_representative_sm_all_ctas"
    assert {
        "graph_id",
        "kernel_invocation_id",
        "nodes",
        "edges",
        "warp_partitions",
        "graph_summary",
        "graph_hash",
    }.issubset(graph)
    assert set(graph["warp_partitions"]) == {"1:0", "1:1", "2:0", "2:1"}


def test_graph_builder_orders_multi_digit_warp_partitions_numerically():
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    record = copy.deepcopy(records[0])
    two = copy.deepcopy(record["warps"][0])
    ten = copy.deepcopy(record["warps"][1])
    two["warp_partition_id"] = "2:0"
    two["cta_id"] = "cta_2"
    two["warp_id"] = 0
    ten["warp_partition_id"] = "10:0"
    ten["cta_id"] = "cta_10"
    ten["warp_id"] = 0
    record["warps"] = [ten, two]

    graph = build_phase_b_graphs([record])[0]
    instruction_partitions = [
        node["warp_partition_id"]
        for node in graph["nodes"]
        if node["node_type"] == "instruction"
    ]

    assert instruction_partitions[:4] == ["2:0"] * 4
    assert instruction_partitions[4:8] == ["10:0"] * 4


def test_graph_builder_parses_bracketed_address_operands_to_base_registers():
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    record = copy.deepcopy(records[0])
    warp = record["warps"][0]
    warp["entries"] = [
        {
            **warp["entries"][0],
            "opcode": "LDG.E.64.SYS",
            "source_operands": ["[R4+0x10]"],
            "destination_operands": ["R8"],
            "trace_index": 0,
        },
        {
            **warp["entries"][1],
            "opcode": "LDG.E.64.SYS",
            "source_operands": ["[UR12+0x20]"],
            "destination_operands": ["R9"],
            "trace_index": 1,
        },
    ]
    record["warps"] = [warp]

    graph = build_phase_b_graphs([record])[0]
    register_tokens = {
        node["token"]
        for node in graph["nodes"]
        if node["node_type"] == "register_version"
    }

    assert "R4.v0.w0" in register_tokens
    assert "UR12.v0.w0" in register_tokens
    assert all("[R4+0x10]" not in token for token in register_tokens)
    assert all("[UR12+0x20]" not in token for token in register_tokens)


def test_graph_validator_rejects_cross_warp_control_flow():
    graph = _graph()
    instruction_nodes = [node for node in graph["nodes"] if node["node_type"] == "instruction"]
    first = next(node for node in instruction_nodes if node["warp_partition_id"] == "1:0")
    other = next(node for node in instruction_nodes if node["warp_partition_id"] == "2:0")
    graph["edges"].append({"source": first["node_id"], "target": other["node_id"], "relation": "control_flow"})

    with pytest.raises(ValueError, match="control_flow"):
        validate_phase_b_graph_artifact(graph)


def test_graph_validator_rejects_non_consecutive_control_flow_shortcut():
    graph = _graph()
    partition = graph["warp_partitions"]["1:0"]
    instruction_ids = partition["instruction_node_ids"]
    graph["edges"].append(
        {
            "edge_id": "e:shortcut",
            "source": instruction_ids[0],
            "target": instruction_ids[2],
            "relation": "control_flow",
            "warp_partition_id": "1:0",
        }
    )
    partition["edge_ids"].append("e:shortcut")
    partition["edge_count"] += 1
    graph["graph_summary"]["edge_count"] += 1

    with pytest.raises(ValueError, match="control_flow"):
        validate_phase_b_graph_artifact(graph)


def test_graph_validator_rejects_missing_trace_index_in_warp_partition():
    graph = _graph()
    mutated = copy.deepcopy(graph)
    instruction_nodes = [
        node
        for node in mutated["nodes"]
        if node["node_type"] == "instruction" and node["warp_partition_id"] == "1:0"
    ]
    instruction_nodes[-1]["trace_index"] += 2

    with pytest.raises(ValueError, match="non-consecutive trace_index"):
        validate_phase_b_graph_artifact(mutated)


def test_graph_validator_rejects_duplicate_trace_index_in_warp_partition():
    graph = _graph()
    mutated = copy.deepcopy(graph)
    instruction_nodes = [
        node
        for node in mutated["nodes"]
        if node["node_type"] == "instruction" and node["warp_partition_id"] == "1:0"
    ]
    instruction_nodes[1]["trace_index"] = instruction_nodes[0]["trace_index"]

    with pytest.raises(ValueError, match="duplicate trace_index"):
        validate_phase_b_graph_artifact(mutated)


def test_warp_partitions_are_complete_and_replayable():
    graph = _graph()

    validate_phase_b_graph_artifact(graph)
    seen = []
    for partition in graph["warp_partitions"].values():
        assert {
            "warp_id",
            "cta_id",
            "node_ids",
            "edge_ids",
            "instruction_count",
            "node_count",
            "edge_count",
            "first_trace_index",
            "last_trace_index",
        }.issubset(partition)
        seen.extend(partition["node_ids"])
    assert sorted(seen) == sorted(node["node_id"] for node in graph["nodes"])


def test_graph_validator_rejects_edge_missing_from_warp_partition():
    graph = _graph()
    mutated = copy.deepcopy(graph)
    partition_id, partition = next(iter(mutated["warp_partitions"].items()))
    edge_id = partition["edge_ids"].pop()
    partition["edge_count"] -= 1
    mutated["graph_hash"] = hash_without(mutated, *GRAPH_HASH_EXCLUDED_FIELDS)
    assert any(edge["edge_id"] == edge_id for edge in mutated["edges"])

    with pytest.raises(ValueError, match="edge.*warp partition"):
        validate_phase_b_graph_artifact(mutated)


def test_graph_validator_rejects_duplicate_partition_node():
    graph = _graph()
    mutated = copy.deepcopy(graph)
    first_key, second_key = list(mutated["warp_partitions"])[:2]
    duplicated_node = mutated["warp_partitions"][first_key]["node_ids"][0]
    mutated["warp_partitions"][second_key]["node_ids"].append(duplicated_node)

    with pytest.raises(ValueError, match="partition"):
        validate_phase_b_graph_artifact(mutated)


def test_graph_validator_rejects_stale_memory_address_source_edges():
    graph = _graph()
    mutated = copy.deepcopy(graph)
    memory_node = next(
        node
        for node in mutated["nodes"]
        if node["node_type"] == "instruction" and node["opcode"].startswith("LDG")
    )
    replacement = next(
        node
        for node in mutated["nodes"]
        if node["node_type"] == "register_version"
        and node["warp_partition_id"] == memory_node["warp_partition_id"]
        and node["node_id"] != memory_node["address_source_node_id"]
    )
    memory_node["address_source_node_id"] = replacement["node_id"]
    mutated["graph_hash"] = hash_without(mutated, "graph_hash")

    with pytest.raises(ValueError, match="address"):
        validate_phase_b_graph_artifact(mutated)
