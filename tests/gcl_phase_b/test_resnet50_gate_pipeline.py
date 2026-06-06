from pathlib import Path
import json

from experiments.gcl_phase_b.resnet50_gate0 import (
    record_resnet50_gate0_trace_acquisition,
    write_resnet50_gate0_blocker_report,
)
from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate7
from experiments.gcl_phase_b.utils import read_json, write_json
from tests.gcl_resnet50.formal_fixture import write_minimal_artifact_shape_resnet50_root


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


def test_resnet50_gate_pipeline_rejects_synthetic_artifact_shape_as_formal_root(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "artifact_shape_trace")

    try:
        record_resnet50_gate0_trace_acquisition(root)
    except ValueError as exc:
        assert "synthetic artifact-shape" in str(exc)
    else:
        raise AssertionError("synthetic artifact-shape root must not produce formal Gate0")


def test_resnet50_gate_pipeline_uses_baselines_when_available(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "contract_trace",
        evidence_scope="real_resnet50_nvbit_collection",
    )
    record_resnet50_gate0_trace_acquisition(root)
    out_dir = tmp_path / "baseline_out"
    baseline_path = tmp_path / "baselines.json"
    write_json(
        baseline_path,
        {
            "metric_rows": [
                {
                    "cluster_id": 0,
                    "measured": 100.0,
                    "predicted": 95.0,
                    "weight": 1.0,
                    "unit": "cycles",
                }
            ],
            "sampled_metrics": {"cycles": 95.0},
            "full_baseline_metrics": {"cycles": 100.0},
            "measured_baseline_metrics": {"cycles": 100.0},
        },
    )

    manifest = run_resnet50_gate1_to_gate7(
        root,
        out_dir,
        seed=20260606,
        baseline_artifacts_path=baseline_path,
    )

    assert manifest["final_gate"] == "gate9_evaluated"
    gate7 = read_json(out_dir / "gate7_correctness_manifest.json")
    gate9 = read_json(out_dir / "gate9_sampled_vs_full_evaluation.json")
    assert gate7["metric_error_report"]["status"] == "reported"
    assert gate7["metric_error_report"]["global_weighted_mape"] == 0.05
    assert gate9["full_vs_sampled_simulation_report"]["cycles"]["relative_error"] == 0.05
    assert gate9["sampled_speedup_report"]["cycles_speedup"] > 1.0
