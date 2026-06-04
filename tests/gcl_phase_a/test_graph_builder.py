import copy

import pytest

from experiments.gcl_phase_a.graph_builder import (
    build_canonical_graph,
    build_canonical_graphs,
    validate_graph_artifact,
)
from experiments.gcl_phase_a.trace_fixture import build_controlled_trace_fixture
from experiments.gcl_phase_a.utils import hash_without


def _raw_record(entries):
    return {
        "kernel_invocation_id": "raw_kernel",
        "trace_family": "raw_register_contract",
        "collection_scope": "selected_warps_fixture",
        "warps": [{"warp_id": 0, "entries": entries}],
    }


def _multi_warp_record(warp_entries):
    return {
        "kernel_invocation_id": "raw_kernel",
        "trace_family": "raw_register_contract",
        "collection_scope": "selected_warps_fixture",
        "warps": [
            {"warp_id": warp_id, "entries": entries}
            for warp_id, entries in enumerate(warp_entries)
        ],
    }


def _entry(index, opcode, dests, srcs):
    return {
        "kernel_invocation_id": "raw_kernel",
        "trace_family": "raw_register_contract",
        "collection_scope": "selected_warps_fixture",
        "warp_id": 0,
        "trace_index": index,
        "pc": 0x1000 + index * 8,
        "opcode": opcode,
        "active_mask": "0xffffffff",
        "destination_operands": dests,
        "source_operands": srcs,
        "observed_dynamic_values": [float(index), float(index + 1)],
        "source_entry_hash": f"raw_hash_{index}",
    }


def test_builds_expected_graph_count():
    fixture = build_controlled_trace_fixture()

    graphs = build_canonical_graphs(fixture)
    repeated_graphs = build_canonical_graphs(fixture)

    assert len(graphs) == 12
    for graph, repeated in zip(graphs, repeated_graphs):
        validate_graph_artifact(graph)
        assert graph["graph_hash"] == repeated["graph_hash"]
        assert {
            "graph_id",
            "kernel_invocation_id",
            "collection_scope",
            "nodes",
            "edges",
            "warp_partitions",
            "graph_summary",
            "graph_hash",
        }.issubset(graph)


def test_graph_hash_is_stable_across_seed_only_fixture_reruns():
    first_graphs = build_canonical_graphs(build_controlled_trace_fixture(seed=1))
    second_graphs = build_canonical_graphs(build_controlled_trace_fixture(seed=2))

    assert [graph["graph_hash"] for graph in first_graphs] == [
        graph["graph_hash"] for graph in second_graphs
    ]


def test_graph_validator_detects_unsorted_instruction_order():
    fixture = build_controlled_trace_fixture()
    record = copy.deepcopy(fixture["records"][0])
    record["warps"][0]["entries"][0], record["warps"][0]["entries"][1] = (
        record["warps"][0]["entries"][1],
        record["warps"][0]["entries"][0],
    )

    with pytest.raises(ValueError, match="ordering violation|consecutive"):
        build_canonical_graph(record, fixture["fixture_hash"])


def test_graph_validator_rejects_missing_warp_partitions():
    graph = build_canonical_graphs(build_controlled_trace_fixture())[0]
    del graph["warp_partitions"]

    with pytest.raises(ValueError, match="missing required fields"):
        validate_graph_artifact(graph)


def test_mem_ref_is_data_flow_only():
    graph = build_canonical_graphs(build_controlled_trace_fixture())[0]
    node_by_id = {node["node_id"]: node for node in graph["nodes"]}
    mem_ref_nodes = [
        node for node in graph["nodes"] if node["node_type"] == "pseudo" and node["pseudo_kind"] == "mem_ref"
    ]

    assert mem_ref_nodes
    for edge in graph["edges"]:
        source_type = node_by_id[edge["source"]]["node_type"]
        target_type = node_by_id[edge["target"]]["node_type"]
        if edge["relation"] == "control_flow":
            assert source_type == "instruction"
            assert target_type == "instruction"
        assert not (edge["relation"] == "control_flow" and "mem_ref" in (edge["source"], edge["target"]))

    assert any(
        node_by_id[edge["source"]]["node_type"] == "register_version"
        and node_by_id[edge["target"]]["node_type"] == "pseudo"
        and edge["relation"] == "data_source"
        for edge in graph["edges"]
    )
    assert any(
        node_by_id[edge["source"]]["node_type"] == "pseudo"
        and node_by_id[edge["target"]]["node_type"] == "instruction"
        and edge["relation"] == "data_source"
        for edge in graph["edges"]
    )


def test_raw_register_reuse_creates_distinct_register_versions():
    graph = build_canonical_graph(
        _raw_record(
            [
                _entry(0, "MOV", ["R4"], ["input:a"]),
                _entry(1, "FADD", ["R5"], ["R4", "input:b"]),
                _entry(2, "FADD", ["R4"], ["R5", "input:c"]),
                _entry(3, "STG.E.64.SYS", [], ["R20", "R4"]),
            ]
        ),
        "fixture_hash",
    )
    node_by_id = {node["node_id"]: node for node in graph["nodes"]}
    r4_nodes = [
        node for node in graph["nodes"] if node["node_type"] == "register_version" and node["token"].startswith("R4.")
    ]
    stg_instruction = next(
        node for node in graph["nodes"] if node["node_type"] == "instruction" and node["opcode"].startswith("STG")
    )

    assert {node["token"] for node in r4_nodes} == {"R4.v1.w0", "R4.v2.w0"}
    assert {
        edge["source"]
        for edge in graph["edges"]
        if edge["target"] == stg_instruction["node_id"] and edge["relation"] == "data_source"
    } & {"register_version:R4.v2.w0"}
    assert "register_version:R4.v1.w0" in node_by_id
    assert "register_version:R4.v2.w0" in node_by_id


def test_live_in_raw_register_uses_version_zero_before_first_local_write():
    graph = build_canonical_graph(
        _raw_record(
            [
                _entry(0, "FADD", ["R5"], ["R4", "input:b"]),
                _entry(1, "MOV", ["R4"], ["input:c"]),
            ]
        ),
        "fixture_hash",
    )
    tokens = {
        node["token"]
        for node in graph["nodes"]
        if node["node_type"] == "register_version" and node["token"].startswith("R4.")
    }

    assert tokens == {"R4.v0.w0", "R4.v1.w0"}


def test_input_variable_nodes_are_partitioned_per_warp():
    graph = build_canonical_graph(
        _multi_warp_record(
            [
                [_entry(0, "MOV", ["R4"], ["input:shared"])],
                [{**_entry(0, "MOV", ["R4"], ["input:shared"]), "warp_id": 1}],
            ]
        ),
        "fixture_hash",
    )
    tokens = {
        node["token"]
        for node in graph["nodes"]
        if node["node_type"] == "input_variable"
    }

    assert tokens == {"input:shared.w0", "input:shared.w1"}


def test_memory_address_role_does_not_require_raddr_name():
    graph = build_canonical_graph(
        _raw_record(
            [
                _entry(0, "LDG.E.64.SYS", ["R8"], ["R14"]),
                _entry(1, "STG.E.64.SYS", [], ["R22", "R8"]),
            ]
        ),
        "fixture_hash",
    )
    node_by_id = {node["node_id"]: node for node in graph["nodes"]}
    mem_ref_ids = {node["node_id"] for node in graph["nodes"] if node["node_type"] == "pseudo"}
    address_to_mem_ref = [
        edge
        for edge in graph["edges"]
        if edge["relation"] == "data_source"
        and edge["target"] in mem_ref_ids
        and node_by_id[edge["source"]]["node_type"] == "register_version"
    ]

    assert len(mem_ref_ids) == 2
    assert {node_by_id[edge["source"]]["token"] for edge in address_to_mem_ref} == {
        "R14.v0.w0",
        "R22.v0.w0",
    }


def test_predicate_operand_is_not_selected_as_memory_address_source():
    graph = build_canonical_graph(
        _raw_record(
            [
                _entry(0, "LDG.E.64.SYS", ["R8"], ["P0", "R14"]),
            ]
        ),
        "fixture_hash",
    )
    node_by_id = {node["node_id"]: node for node in graph["nodes"]}
    mem_ref_id = next(node["node_id"] for node in graph["nodes"] if node["node_type"] == "pseudo")
    address_edges = [
        edge
        for edge in graph["edges"]
        if edge["relation"] == "data_source" and edge["target"] == mem_ref_id
    ]

    assert [node_by_id[edge["source"]]["token"] for edge in address_edges] == ["R14.v0.w0"]


def test_graph_validator_rejects_wrong_raw_address_register_with_recomputed_hash():
    graph = build_canonical_graph(
        _raw_record(
            [
                _entry(0, "LDG.E.64.SYS", ["R8"], ["R14"]),
                _entry(1, "STG.E.64.SYS", [], ["R22", "R8"]),
            ]
        ),
        "fixture_hash",
    )
    graph["edges"] = [
        edge
        for edge in graph["edges"]
        if not (edge["source"] == "register_version:R22.v0.w0" and edge["target"] == "p:mem_ref:w0:t1")
    ]
    graph["edges"].append(
        {"source": "register_version:R8.v1.w0", "target": "p:mem_ref:w0:t1", "relation": "data_source"}
    )
    graph["graph_hash"] = hash_without(graph, "graph_hash")

    with pytest.raises(ValueError, match="exact address"):
        validate_graph_artifact(graph)


def test_graph_uses_phase_a_semantic_node_taxonomy():
    graphs = build_canonical_graphs(build_controlled_trace_fixture())
    node_types = {node["node_type"] for graph in graphs for node in graph["nodes"]}

    assert {"instruction", "register_version", "input_variable", "unknown_variable", "pseudo"}.issubset(
        node_types
    )


def test_graph_validator_rejects_nonconsecutive_control_flow_with_recomputed_hash():
    graph = build_canonical_graphs(build_controlled_trace_fixture())[0]
    control_edge_index = next(
        index for index, edge in enumerate(graph["edges"]) if edge["relation"] == "control_flow"
    )
    graph["edges"][control_edge_index]["target"] = "i:w0:t2"
    graph["graph_hash"] = hash_without(graph, "graph_hash")

    with pytest.raises(ValueError, match="consecutive"):
        validate_graph_artifact(graph)


def test_graph_validator_rejects_missing_address_to_mem_ref_with_recomputed_hash():
    graph = build_canonical_graphs(build_controlled_trace_fixture())[0]
    graph["edges"] = [
        edge
        for edge in graph["edges"]
        if not (
            edge["relation"] == "data_source"
            and edge["target"].startswith("p:mem_ref")
            and edge["source"].startswith("register_version:Raddr")
        )
    ]
    graph["graph_hash"] = hash_without(graph, "graph_hash")

    with pytest.raises(ValueError, match="address variable"):
        validate_graph_artifact(graph)


def test_graph_validator_rejects_control_flow_pseudo_edge():
    graph = build_canonical_graphs(build_controlled_trace_fixture())[0]
    pseudo = next(node["node_id"] for node in graph["nodes"] if node["node_type"] == "pseudo")
    instruction = next(node["node_id"] for node in graph["nodes"] if node["node_type"] == "instruction")
    graph["edges"].append({"source": instruction, "target": pseudo, "relation": "control_flow"})

    with pytest.raises(ValueError, match="control_flow"):
        validate_graph_artifact(graph)


def test_graph_validator_rejects_missing_required_mem_ref():
    graph = build_canonical_graphs(build_controlled_trace_fixture())[0]
    graph["nodes"] = [node for node in graph["nodes"] if node["node_type"] != "pseudo"]
    graph["edges"] = [
        edge for edge in graph["edges"] if not edge["source"].startswith("p:") and not edge["target"].startswith("p:")
    ]

    with pytest.raises(ValueError, match="mem_ref"):
        validate_graph_artifact(graph)
