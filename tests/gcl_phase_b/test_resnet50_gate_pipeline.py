from pathlib import Path
import json
import shutil

from experiments.gcl_phase_b.embedding_export import READOUT_HIERARCHY
from experiments.gcl_phase_b.resnet50_gate0 import record_resnet50_gate0_trace_acquisition
from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate7

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def _formal_gate0_root(tmp_path):
    root = tmp_path / "formal_gate0"
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "dynamic_trace.pb").write_bytes(b"formal-protobuf-placeholder")
    threadblocks_dir = root / "threadblocks"
    threadblocks_dir.mkdir()
    shutil.copy(root / "threadblocks.json", threadblocks_dir / "threadblocks.json")
    record_resnet50_gate0_trace_acquisition(root)
    return root


def test_resnet50_gate_pipeline_reaches_gate7_correctness(tmp_path):
    out_dir = tmp_path / "gate1_7"

    manifest = run_resnet50_gate1_to_gate7(_formal_gate0_root(tmp_path), out_dir, seed=20260606)

    assert manifest["final_gate"] == "gate7"
    assert (out_dir / "resnet50_trace_adapter_bundle.json").exists()
    assert (out_dir / "representative_sm_trace_manifest.json").exists()
    assert (out_dir / "canonical_graph_bundle.json").exists()
    assert (out_dir / "graph_tensor_bundle.json").exists()
    assert (out_dir / "rgcn_training_run_manifest.json").exists()
    assert (out_dir / "rgcn_checkpoint_manifest.json").exists()
    assert (out_dir / "kernel_embedding_table.json").exists()
    assert (out_dir / "selector_artifacts.json").exists()
    assert (out_dir / "gate7_correctness_manifest.json").exists()
    assert (out_dir / "readout_manifest.json").exists()


def test_resnet50_gate_pipeline_outputs_embedding_rows_for_each_kernel(tmp_path):
    out_dir = tmp_path / "gate1_5_rows"

    manifest = run_resnet50_gate1_to_gate7(_formal_gate0_root(tmp_path), out_dir, seed=20260606)

    table = json.loads((out_dir / "kernel_embedding_table.json").read_text())
    graph_tensor_bundle = json.loads((out_dir / "graph_tensor_bundle.json").read_text())
    assert table["embedding_dim"] == 256
    assert table["artifact_type"] == "gcl_resnet50_kernel_embedding_table"
    assert table["artifact_version"] == "gate5_kernel_embedding_table_v1"
    assert table["source_graph_tensor_bundle_hash"] == graph_tensor_bundle["graph_tensor_bundle_hash"]
    assert table["readout_hierarchy"] == READOUT_HIERARCHY
    assert len(table["embeddings"]) == 2
    assert all("readout_manifest_hash" in row for row in table["embeddings"])
    assert all(len(row["kernel_embedding"]) == 256 for row in table["embeddings"])
    assert all(row["graph_id"] for row in table["embeddings"])
    assert all(row["source_tensor_hash"] for row in table["embeddings"])
    assert all(row["collection_scope"] == "single_representative_sm_all_ctas" for row in table["embeddings"])
    assert all(row["selected_sm"] is not None for row in table["embeddings"])
    assert all(
        row["weight_input"]["readout_hierarchy"] == READOUT_HIERARCHY
        for row in table["embeddings"]
    )
    assert manifest["hashes"]["embedding_table_hash"] == table["kernel_embedding_table_hash"]


def test_resnet50_gate_pipeline_outputs_formal_training_manifests(tmp_path):
    out_dir = tmp_path / "gate1_5_training_manifest"

    run_resnet50_gate1_to_gate7(_formal_gate0_root(tmp_path), out_dir, seed=20260606)

    graph_tensor_bundle = json.loads((out_dir / "graph_tensor_bundle.json").read_text())
    table = json.loads((out_dir / "kernel_embedding_table.json").read_text())
    training_manifest = json.loads((out_dir / "rgcn_training_run_manifest.json").read_text())
    checkpoint_manifest = json.loads((out_dir / "rgcn_checkpoint_manifest.json").read_text())

    assert training_manifest["artifact_type"] == "gcl_resnet50_rgcn_training_run_manifest"
    assert training_manifest["artifact_version"] == "gate5_rgcn_training_run_manifest_v1"
    assert training_manifest["source_graph_tensor_bundle_hash"] == graph_tensor_bundle[
        "graph_tensor_bundle_hash"
    ]
    assert training_manifest["readout_hierarchy"] == READOUT_HIERARCHY
    assert training_manifest["random_seed"] == 20260606
    assert training_manifest["train_graph_count"] == 2
    assert training_manifest["training_status"] == "formal_gate5_complete"
    assert training_manifest["best_checkpoint_hash"] == table["checkpoint_hash"]
    assert training_manifest["training_run_manifest_hash"]

    assert checkpoint_manifest["artifact_type"] == "gcl_resnet50_rgcn_checkpoint_manifest"
    assert checkpoint_manifest["artifact_version"] == "gate5_rgcn_checkpoint_manifest_v1"
    assert checkpoint_manifest["encoder_manifest_hash"] == table["encoder_manifest_hash"]
    assert checkpoint_manifest["checkpoint_hash"] == table["checkpoint_hash"]
    assert checkpoint_manifest["checkpoint_created_from_training_run_manifest_hash"] == (
        training_manifest["training_run_manifest_hash"]
    )
