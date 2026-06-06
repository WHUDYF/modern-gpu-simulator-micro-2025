from pathlib import Path
import json

from experiments.gcl_phase_b.resnet50_gate0 import (
    record_resnet50_gate0_trace_acquisition,
    write_resnet50_gate0_blocker_report,
)
from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate7
from experiments.gcl_phase_b.utils import read_json
from tests.gcl_resnet50.formal_fixture import write_minimal_formal_resnet50_root


def _blocked_gate0_root(tmp_path):
    root = tmp_path / "blocked_gate0"
    root.mkdir()
    write_resnet50_gate0_blocker_report(
        root,
        reason="real ResNet-50 NVBit trace is not available",
        missing_requirements=["dynamic_trace.pb", "threadblocks/"],
    )
    return root


def test_resnet50_gate_pipeline_stops_at_gate0_blocker(tmp_path):
    out_dir = tmp_path / "gate0_blocked"

    manifest = run_resnet50_gate1_to_gate7(_blocked_gate0_root(tmp_path), out_dir, seed=20260606)

    assert manifest["final_gate"] == "gate0_blocked"
    assert manifest["artifact_status"] == "formal_blocked"
    assert manifest["formal_input_eligible"] is False
    assert (out_dir / "gate0_trace_acquisition_blocker_report.json").exists()
    assert not (out_dir / "resnet50_trace_adapter_bundle.json").exists()
    assert not (out_dir / "kernel_embedding_table.json").exists()


def test_resnet50_gate_pipeline_blocker_manifest_is_replayable(tmp_path):
    out_dir = tmp_path / "gate0_blocked_replay"

    manifest = run_resnet50_gate1_to_gate7(_blocked_gate0_root(tmp_path), out_dir, seed=20260606)

    stored = json.loads((out_dir / "gate1_7_pipeline_manifest.json").read_text())
    assert stored["pipeline_manifest_hash"] == manifest["pipeline_manifest_hash"]


def test_resnet50_gate_pipeline_wires_gate7_gate8_and_gate9_from_formal_artifacts(tmp_path):
    root = write_minimal_formal_resnet50_root(tmp_path / "formal_trace")
    record_resnet50_gate0_trace_acquisition(root)
    out_dir = tmp_path / "formal_out"

    manifest = run_resnet50_gate1_to_gate7(root, out_dir, seed=20260606)

    assert manifest["final_gate"] == "gate9_report_only"
    gate7 = read_json(out_dir / "gate7_correctness_manifest.json")
    gate8 = read_json(out_dir / "gate8_tuning_vector_proposal.json")
    gate9 = read_json(out_dir / "gate9_sampled_vs_full_evaluation.json")
    assert gate7["embedding_geometry_metrics"]["silhouette"] is not None
    assert gate7["source_gate5_embedding_table_hash"]
    assert gate8["extension_label"] == "our_extension_not_original_gcl_sampler"
    assert gate9["claim_status"] == "baseline_missing_no_speedup_or_accuracy_claim"
