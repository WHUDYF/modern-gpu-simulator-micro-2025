from experiments.gcl_phase_a.graph_builder import build_canonical_graphs
from experiments.gcl_phase_a.pipeline import run_pipeline
from experiments.gcl_phase_a.tensorizer import tensor_to_jsonable, tensorize_graphs
from experiments.gcl_phase_a.trace_fixture import build_controlled_trace_fixture
from experiments.gcl_phase_a.utils import hash_without


def test_phase_a_artifacts_are_replayable(tmp_path):
    first = run_pipeline(tmp_path / "first")
    second = run_pipeline(tmp_path / "second")

    assert first["hashes"]["trace_fixture_hash"] == second["hashes"]["trace_fixture_hash"]
    assert first["hashes"]["graph_hashes"] == second["hashes"]["graph_hashes"]
    assert first["hashes"]["tensor_hashes"] == second["hashes"]["tensor_hashes"]
    assert first["hashes"]["encoder_manifest_hash"] == second["hashes"]["encoder_manifest_hash"]
    assert first["hashes"]["embedding_table_hash"] == second["hashes"]["embedding_table_hash"]
    assert first["hashes"]["selector_manifest_hash"] == second["hashes"]["selector_manifest_hash"]


def test_fixture_change_changes_graph_and_tensor_hashes():
    fixture = build_controlled_trace_fixture()
    changed = build_controlled_trace_fixture()
    entry = changed["records"][0]["warps"][0]["entries"][0]
    entry["observed_dynamic_values"][0] += 1.0
    entry["source_entry_hash"] = hash_without(entry, "source_entry_hash")
    changed["fixture_hash"] = hash_without(changed, "fixture_hash")

    original_graphs = build_canonical_graphs(fixture)
    changed_graphs = build_canonical_graphs(changed)
    original_tensors = tensorize_graphs(original_graphs)
    changed_tensors = tensorize_graphs(changed_graphs)

    assert original_graphs[0]["graph_hash"] != changed_graphs[0]["graph_hash"]
    assert original_tensors[0]["tensor_hash"] != changed_tensors[0]["tensor_hash"]


def test_tensorizer_schema_change_changes_schema_hash_surface():
    fixture = build_controlled_trace_fixture()
    graph = build_canonical_graphs(fixture)[0]
    tensor = tensorize_graphs([graph])[0]
    changed = tensor_to_jsonable(tensor)
    changed["node_feature_schema"]["schema_version"] = 2

    assert tensor["tensor_hash"] != hash_without(changed, "tensor_hash")
