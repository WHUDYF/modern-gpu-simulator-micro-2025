from pathlib import Path

from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate5

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def test_resnet50_gate1_5_pipeline_hashes_are_replayable(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    manifest_a = run_resnet50_gate1_to_gate5(FIXTURE_ROOT, out_a, seed=20260606)
    manifest_b = run_resnet50_gate1_to_gate5(FIXTURE_ROOT, out_b, seed=20260606)

    assert manifest_a["hashes"]["adapter_bundle_hash"] == manifest_b["hashes"]["adapter_bundle_hash"]
    assert manifest_a["hashes"]["trace_manifest_hash"] == manifest_b["hashes"]["trace_manifest_hash"]
    assert (
        manifest_a["hashes"]["canonical_graph_bundle_hash"]
        == manifest_b["hashes"]["canonical_graph_bundle_hash"]
    )
    assert (
        manifest_a["hashes"]["graph_tensor_bundle_hash"]
        == manifest_b["hashes"]["graph_tensor_bundle_hash"]
    )
    assert manifest_a["hashes"]["embedding_table_hash"] == manifest_b["hashes"]["embedding_table_hash"]
