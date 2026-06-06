from pathlib import Path

import pytest

from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_debug_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs, validate_phase_b_tensor_artifact
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records
from tests.gcl_resnet50.formal_chain import build_artifact_shape_tensors

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def _debug_tensors():
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)
    manifest, _reports, _preview = build_representative_sm_manifest_from_bundle(bundle)
    graphs = build_phase_b_graphs(build_phase_b_trace_records(manifest))
    return tensorize_phase_b_graphs(graphs)


def test_gate4_outputs_rgcn_tensor_bundle_shape_for_debug_smoke_graph():
    tensors = _debug_tensors()

    assert tensors
    for tensor in tensors:
        assert tensor["feature_width"] == 64
        assert tensor["node_features"].shape[1] == 64
        assert tensor["edge_index"].shape[0] == 2
        assert tensor["edge_index"].shape[1] == tensor["edge_type"].shape[0]
        assert tensor["warp_partition_tensors"]


def test_gate4_rejects_invalid_feature_width_or_partition_metadata():
    tensor = _debug_tensors()[0]
    tensor["feature_width"] = 63

    with pytest.raises(ValueError, match="feature_width"):
        validate_phase_b_tensor_artifact(tensor)


def test_gate4_tensorizes_artifact_shape_graphs_without_formal_claim(tmp_path):
    tensors = build_artifact_shape_tensors(tmp_path)

    assert tensors
    for tensor in tensors:
        validate_phase_b_tensor_artifact(tensor)
        assert tensor["feature_width"] == 64
        assert tensor["graph_batch_metadata"]["artifact_status"] == "debug_not_formal"
        assert tensor["graph_batch_metadata"]["formal_input_eligible"] is False
        assert tensor["graph_batch_metadata"]["trace_source"] == "synthetic_protobuf_artifact_shape"
        assert tensor["node_features"].shape[1] == 64
        assert tensor["edge_index"].shape[0] == 2
