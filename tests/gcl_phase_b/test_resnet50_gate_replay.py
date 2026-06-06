from pathlib import Path
import shutil

from experiments.gcl_phase_b.resnet50_gate0 import record_resnet50_gate0_trace_acquisition
from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate5

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def _formal_gate0_root(tmp_path, name):
    root = tmp_path / name
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "dynamic_trace.pb").write_bytes(b"formal-protobuf-placeholder")
    threadblocks_dir = root / "threadblocks"
    threadblocks_dir.mkdir()
    shutil.copy(root / "threadblocks.json", threadblocks_dir / "threadblocks.json")
    record_resnet50_gate0_trace_acquisition(root)
    return root


def test_resnet50_gate1_5_pipeline_hashes_are_replayable(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    manifest_a = run_resnet50_gate1_to_gate5(
        _formal_gate0_root(tmp_path, "formal_a"), out_a, seed=20260606
    )
    manifest_b = run_resnet50_gate1_to_gate5(
        _formal_gate0_root(tmp_path, "formal_b"), out_b, seed=20260606
    )

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
