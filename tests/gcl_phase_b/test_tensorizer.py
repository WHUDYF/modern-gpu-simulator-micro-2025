import pytest

from experiments.gcl_phase_a.tensorizer import NODE_FEATURE_SCHEMA_NAME, PAPER_REPRODUCTION_MODE
from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs, validate_phase_b_tensor_artifact
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


def test_phase_b_tensor_validator_rejects_bad_partition_index():
    tensor = _tensor()
    first_key = next(iter(tensor["warp_partition_tensors"]))
    tensor["warp_partition_tensors"][first_key]["node_indices"].append(tensor["node_features"].shape[0] + 1)

    with pytest.raises(ValueError, match="node index"):
        validate_phase_b_tensor_artifact(tensor)
