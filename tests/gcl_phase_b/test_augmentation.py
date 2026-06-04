from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs
from experiments.gcl_phase_b.training import create_augmented_training_views
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records


def test_augmentation_manifests_reference_canonical_graph_without_overwrite():
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    graph = build_phase_b_graphs(records)[0]
    tensor = tensorize_phase_b_graphs([graph])[0]

    view_a, view_b = create_augmented_training_views(tensor, seed=17)

    assert graph["graph_hash"] == tensor["input_graph_hash"]
    for view in (view_a, view_b):
        manifest = view["phase_b_augmentation_manifest"]
        assert manifest["input_graph_hash"] == graph["graph_hash"]
        assert manifest["augmentation_manifest_hash"]
        assert manifest["view_hash"]
        assert manifest["view_id"] in {"A", "B"}
        assert "graph_hash" not in view
