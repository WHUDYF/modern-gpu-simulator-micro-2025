from pathlib import Path

from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate5

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def test_resnet50_gate_pipeline_stops_at_gate5_embedding_table(tmp_path):
    out_dir = tmp_path / "gate1_5"

    manifest = run_resnet50_gate1_to_gate5(FIXTURE_ROOT, out_dir, seed=20260606)

    assert manifest["final_gate"] == "gate5"
    assert (out_dir / "resnet50_trace_adapter_bundle.json").exists()
    assert (out_dir / "representative_sm_trace_manifest.json").exists()
    assert (out_dir / "canonical_graph_bundle.json").exists()
    assert (out_dir / "graph_tensor_bundle.json").exists()
    assert (out_dir / "kernel_embedding_table.json").exists()
    assert not (out_dir / "selector_artifacts.json").exists()
    assert (out_dir / "readout_manifest.json").exists()


def test_resnet50_gate_pipeline_outputs_embedding_rows_for_each_kernel(tmp_path):
    out_dir = tmp_path / "gate1_5_rows"

    manifest = run_resnet50_gate1_to_gate5(FIXTURE_ROOT, out_dir, seed=20260606)

    import json

    table = json.loads((out_dir / "kernel_embedding_table.json").read_text())
    assert table["embedding_dim"] == 256
    assert table["row_count"] == 2
    assert len(table["rows"]) == 2
    assert all("readout_manifest_hash" in row for row in table["rows"])
    assert all(
        row["weight_input"]["readout_hierarchy"] == "node_to_warp_to_cta_to_selected_sm_to_kernel"
        for row in table["rows"]
    )
    assert manifest["hashes"]["embedding_table_hash"] == table["embedding_table_hash"]
