import copy

import numpy as np
import pytest

from experiments.gcl_phase_a.graph_builder import build_canonical_graphs
from experiments.gcl_phase_a.tensorizer import (
    FEATURE_WIDTH,
    NODE_FEATURE_SCHEMA_NAME,
    PAPER_REPRODUCTION_MODE,
    tensorize_graph,
    validate_tensor_artifact,
)
from experiments.gcl_phase_a.trace_fixture import build_controlled_trace_fixture
from experiments.gcl_phase_a.utils import hash_without


def _tensor():
    graph = build_canonical_graphs(build_controlled_trace_fixture())[0]
    return tensorize_graph(graph)


def test_node_features_are_64_wide():
    tensor = _tensor()
    features = tensor["node_features"]

    assert features.shape[1] == FEATURE_WIDTH
    assert tensor["node_feature_schema"]["schema_name"] == NODE_FEATURE_SCHEMA_NAME
    assert tensor["node_feature_schema"]["paper_reproduction_mode"] == PAPER_REPRODUCTION_MODE
    assert tensor["node_feature_schema"]["instruction_feature_combine"] == "concat_opcode63_normalized_pc1"
    assert tensor["tensorizer_version"] == "gcl_phase_a_tensorizer_v1"
    assert tensor["feature_width"] == 64
    assert tensor["padding_policy"] == "strict_zero_padding"
    assert tensor["missing_value_policy"] == "missing numeric values become 0.0"

    for idx, node_type in enumerate(tensor["node_types"]):
        if node_type == "instruction":
            assert np.isfinite(features[idx, 0:63]).all()
            assert 0.0 <= features[idx, 63] <= 1.0
        if node_type in {"register_version", "input_variable", "unknown_variable"}:
            assert np.isfinite(features[idx, 0:40]).all()
            assert np.allclose(features[idx, 40:64], 0.0)
        if node_type == "pseudo":
            assert np.isfinite(features[idx, 0:16]).all()
            assert np.allclose(features[idx, 16:64], 0.0)


def test_tensor_validator_rejects_nonzero_padding_and_trace_index_mode():
    tensor = _tensor()
    variable_idx = tensor["node_types"].index("register_version")
    tensor["node_features"][variable_idx, 40] = 1.0

    with pytest.raises(ValueError, match="variable zero padding"):
        validate_tensor_artifact(tensor)

    tensor = _tensor()
    tensor["node_feature_schema"]["instruction_feature_combine"] = "trace_index_positional_encoding"

    with pytest.raises(ValueError, match="instruction_feature_combine"):
        validate_tensor_artifact(tensor)


def test_rgcn_inputs_are_complete():
    graph = build_canonical_graphs(build_controlled_trace_fixture())[0]
    tensor = tensorize_graph(graph)
    repeated = tensorize_graph(graph)

    validate_tensor_artifact(tensor)
    assert {"node_features", "edge_index", "edge_type", "warp_partitions", "graph_batch_metadata"}.issubset(
        tensor
    )
    assert tensor["edge_index"].shape[0] == 2
    assert tensor["edge_index"].shape[1] == tensor["edge_type"].shape[0]
    assert tensor["input_graph_hash"] == graph["graph_hash"]
    assert tensor["tensor_hash"] == repeated["tensor_hash"]


def test_tensorize_graph_preserves_empty_edge_shape_for_single_instruction_graph():
    graph = {
        "artifact_type": "canonical_graph",
        "graph_id": "graph:single_instruction",
        "kernel_invocation_id": "single_instruction",
        "trace_family": "minimal",
        "collection_scope": "selected_warps_fixture",
        "source_fixture_hash": "fixture",
        "nodes": [
            {
                "node_id": "i:w0:t0",
                "node_type": "instruction",
                "opcode": "NOP",
                "pc": 4096,
                "warp_id": 0,
                "trace_index": 0,
                "active_mask": 0xFFFFFFFF,
                "source_entry_hash": "entry",
            }
        ],
        "edges": [],
        "edge_relation_schema": {"control_flow": 0, "data_source": 1, "data_destination": 2},
        "warp_partitions": {"0": ["i:w0:t0"]},
        "graph_summary": {
            "node_count": 1,
            "edge_count": 0,
            "instruction_node_count": 1,
            "variable_node_count": 0,
            "pseudo_node_count": 0,
        },
    }
    graph["graph_hash"] = hash_without(graph, "graph_hash", "source_fixture_hash")

    tensor = tensorize_graph(graph)

    assert tensor["edge_index"].shape == (2, 0)
    assert tensor["edge_type"].shape == (0,)


def test_tensor_validator_rejects_missing_schema_and_edge_mismatch():
    tensor = _tensor()
    del tensor["input_graph_hash"]

    with pytest.raises(ValueError, match="missing required fields"):
        validate_tensor_artifact(tensor)

    tensor = _tensor()
    tensor["edge_type"] = tensor["edge_type"][:-1]

    with pytest.raises(ValueError, match="length mismatch"):
        validate_tensor_artifact(tensor)


@pytest.mark.parametrize(
    "field",
    ["tensorizer_version", "feature_width", "padding_policy", "missing_value_policy"],
)
def test_tensor_validator_rejects_missing_top_level_manifest_fields(field):
    tensor = _tensor()
    del tensor[field]

    with pytest.raises(ValueError, match="missing required fields"):
        validate_tensor_artifact(tensor)
