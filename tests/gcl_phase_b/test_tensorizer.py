import pytest

from experiments.gcl_phase_a.tensorizer import NODE_FEATURE_SCHEMA_NAME, PAPER_REPRODUCTION_MODE
from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.tensorizer import (
    FUNCTIONAL_FIRST_PAPER_MODE,
    _partition_edge_indices,
    _tensor_hash,
    tensorize_phase_b_graphs,
    validate_phase_b_tensor_artifact,
)
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records


def _tensor():
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    graph = build_phase_b_graphs(records)[0]
    return tensorize_phase_b_graphs([graph])[0]


def test_phase_b_tensorization_reuses_phase_a_strict_schema():
    tensor = _tensor()

    validate_phase_b_tensor_artifact(tensor)
    assert tensor["node_feature_schema"]["schema_name"] == NODE_FEATURE_SCHEMA_NAME
    assert tensor["node_feature_schema"]["paper_reproduction_mode"] == PAPER_REPRODUCTION_MODE
    assert tensor["feature_width"] == 64
    assert tensor["phase_b_tensorizer_version"] == "gcl_phase_b_tensorizer_v1"


def test_phase_b_tensor_records_resnet50_representation_mode():
    tensor = _tensor()

    validate_phase_b_tensor_artifact(tensor)
    assert tensor["representation_mode"] == "gcl_resnet50_mem_ref_only"
    assert tensor["pseudo_node_mode"] == "mem_ref_only"
    assert tensor["paper_reproduction_mode"] == PAPER_REPRODUCTION_MODE


def test_phase_b_tensor_records_no_pseudo_mode_as_non_strict():
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    graph = build_phase_b_graphs(records)[0]
    pseudo_ids = {node["node_id"] for node in graph["nodes"] if node["node_type"] == "pseudo"}
    graph["nodes"] = [node for node in graph["nodes"] if node["node_id"] not in pseudo_ids]
    graph["edges"] = [
        edge
        for edge in graph["edges"]
        if edge["source"] not in pseudo_ids and edge["target"] not in pseudo_ids
    ]
    for partition in graph["warp_partitions"].values():
        partition["node_ids"] = [node_id for node_id in partition["node_ids"] if node_id not in pseudo_ids]
        partition["edge_ids"] = [
            edge["edge_id"]
            for edge in graph["edges"]
            if edge["warp_partition_id"] == partition["partition_id"]
        ]
        partition["node_count"] = len(partition["node_ids"])
        partition["edge_count"] = len(partition["edge_ids"])
    graph["graph_summary"]["node_count"] = len(graph["nodes"])
    graph["graph_summary"]["edge_count"] = len(graph["edges"])
    graph["graph_summary"]["node_type_counts"]["pseudo"] = 0
    graph["pseudo_node_mode"] = "no_pseudo_node"
    graph["graph_hash"] = "stale"
    from experiments.gcl_phase_b.utils import hash_without

    graph["graph_hash"] = hash_without(graph, "graph_hash")

    tensor = tensorize_phase_b_graphs([graph])[0]

    validate_phase_b_tensor_artifact(tensor)
    assert tensor["representation_mode"] == "gcl_resnet50_no_pseudo_node"
    assert tensor["pseudo_node_mode"] == "no_pseudo_node"
    assert tensor["paper_reproduction_mode"] == FUNCTIONAL_FIRST_PAPER_MODE


def test_tensor_bundle_contains_warp_partition_tensors():
    tensor = _tensor()

    assert {
        "node_features",
        "edge_index",
        "edge_type",
        "warp_partitions",
        "warp_partition_tensors",
        "graph_batch_metadata",
    }.issubset(tensor)
    assert tensor["edge_index"].shape[0] == 2
    assert tensor["edge_index"].shape[1] == tensor["edge_type"].shape[0]
    for partition in tensor["warp_partition_tensors"].values():
        assert partition["node_indices"]
        assert all(0 <= index < tensor["node_features"].shape[0] for index in partition["node_indices"])


def test_partition_edge_indices_use_precomputed_edge_lookup():
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    graph = build_phase_b_graphs(records)[0]
    edge_offset_by_id = {edge["edge_id"]: offset for offset, edge in enumerate(graph["edges"])}

    partition = graph["warp_partitions"]["1:0"]
    edge_indices = _partition_edge_indices(partition, edge_offset_by_id)

    assert edge_indices == [edge_offset_by_id[edge_id] for edge_id in partition["edge_ids"]]


def test_phase_b_tensor_validator_rejects_bad_partition_index():
    tensor = _tensor()
    first_key = next(iter(tensor["warp_partition_tensors"]))
    tensor["warp_partition_tensors"][first_key]["node_indices"].append(tensor["node_features"].shape[0] + 1)

    with pytest.raises(ValueError, match="node index"):
        validate_phase_b_tensor_artifact(tensor)


def test_phase_b_tensor_validator_rejects_bad_partition_edge_index():
    tensor = _tensor()
    first_key = next(iter(tensor["warp_partition_tensors"]))
    tensor["warp_partition_tensors"][first_key]["edge_indices"].append(tensor["edge_type"].shape[0] + 1)
    tensor["tensor_hash"] = _tensor_hash(tensor)

    with pytest.raises(ValueError, match="edge index"):
        validate_phase_b_tensor_artifact(tensor)
