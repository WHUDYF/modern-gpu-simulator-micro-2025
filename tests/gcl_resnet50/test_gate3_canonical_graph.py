from pathlib import Path

from experiments.gcl_phase_b.graph_builder import VARIABLE_NODE_TYPES, build_phase_b_graphs
from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_debug_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records
from gcl_resnet50.formal_chain import build_artifact_shape_graphs
from gcl_resnet50.real_chain import build_real_graphs

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def test_gate3_debug_manifest_cannot_be_claimed_as_formal_graph_source():
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)
    manifest, _reports, _preview = build_representative_sm_manifest_from_bundle(bundle)

    records = build_phase_b_trace_records(manifest)
    graphs = build_phase_b_graphs(records)

    assert manifest["artifact_status"] == "debug_not_formal"
    assert graphs
    assert all(graph["collection_scope"] == "single_representative_sm_all_ctas" for graph in graphs)


def test_gate3_uses_allowed_node_and_edge_types_for_debug_smoke_graph():
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)
    manifest, _reports, _preview = build_representative_sm_manifest_from_bundle(bundle)
    graphs = build_phase_b_graphs(build_phase_b_trace_records(manifest))

    allowed_nodes = {"instruction", "pseudo", *VARIABLE_NODE_TYPES}
    allowed_edges = {"control_flow", "data_source", "data_destination"}
    for graph in graphs:
        assert {node["node_type"] for node in graph["nodes"]}.issubset(allowed_nodes)
        assert {edge["relation"] for edge in graph["edges"]}.issubset(allowed_edges)


def test_gate3_builds_artifact_shape_canonical_graphs_without_formal_claim(tmp_path):
    manifest, graphs = build_artifact_shape_graphs(tmp_path)

    assert manifest["artifact_status"] == "debug_not_formal"
    assert graphs
    allowed_nodes = {"instruction", "pseudo", *VARIABLE_NODE_TYPES}
    allowed_edges = {"control_flow", "data_source", "data_destination"}
    for graph in graphs:
        assert graph["artifact_status"] == "debug_not_formal"
        assert graph["formal_input_eligible"] is False
        assert graph["trace_source"] == "synthetic_protobuf_artifact_shape"
        assert {node["node_type"] for node in graph["nodes"]}.issubset(allowed_nodes)
        assert {edge["relation"] for edge in graph["edges"]}.issubset(allowed_edges)


def test_gate3_builds_canonical_graphs_from_real_resnet50_manifest():
    manifest, reports, preview, graphs = build_real_graphs()

    assert manifest["artifact_status"] == "formal"
    assert reports["reports"]
    assert preview["invocations"]
    assert graphs
    allowed_nodes = {"instruction", "pseudo", *VARIABLE_NODE_TYPES}
    allowed_edges = {"control_flow", "data_source", "data_destination"}
    for graph in graphs:
        assert graph["artifact_status"] == "formal"
        assert graph["formal_input_eligible"] is True
        assert graph["trace_source"] == "nvbit"
        assert graph["collection_scope"] == "single_representative_sm_all_ctas"
        assert graph["nodes"]
        assert graph["edges"]
        assert graph["warp_partitions"]
        assert {node["node_type"] for node in graph["nodes"]}.issubset(allowed_nodes)
        assert {edge["relation"] for edge in graph["edges"]}.issubset(allowed_edges)
