from pathlib import Path
import json

from experiments.gcl_phase_b.resnet50_gate0 import (
    record_resnet50_gate0_trace_acquisition,
    write_resnet50_gate0_blocker_report,
)
from experiments.gcl_phase_b.resnet50_gate_pipeline import (
    GATE1_7_PIPELINE_MANIFEST_FILENAME,
    run_resnet50_gate1_to_gate7,
)
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

    stored = json.loads((out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME).read_text())
    assert stored["pipeline_manifest_hash"] == manifest["pipeline_manifest_hash"]
    assert not (out_dir / "gate1_5_pipeline_manifest.json").exists()


def test_resnet50_gate_pipeline_rejects_synthetic_artifact_shape_as_formal_root(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "artifact_shape_trace")

    try:
        record_resnet50_gate0_trace_acquisition(root)
    except ValueError as exc:
        assert "real NVBit runtime artifact origin" in str(exc)
    else:
        raise AssertionError("synthetic artifact-shape root must not produce formal Gate0")


def test_resnet50_gate_pipeline_keeps_baseline_artifacts_blocked_without_real_gate0(tmp_path):
    root = _blocked_gate0_root(tmp_path)
    baseline_path = tmp_path / "baseline_artifacts.json"
    baseline_path.write_text(
        json.dumps(
            {
                "metric_rows": [
                    {
                        "cluster_id": 0,
                        "measured": 100.0,
                        "predicted": 95.0,
                        "weight": 2.0,
                        "unit": "cycles",
                    }
                ],
                "sampled_metrics": {"cycles": 95.0},
                "full_baseline_metrics": {"cycles": 100.0},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "pipeline_out"

    manifest = run_resnet50_gate1_to_gate7(
        root,
        out_dir,
        seed=20260606,
        baseline_artifacts_path=baseline_path,
    )

    assert manifest["final_gate"] == "gate0_blocked"
    assert (out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME).exists()
    assert not (out_dir / "gate7_correctness_manifest.json").exists()
    assert not (out_dir / "gate9_sampled_vs_full_evaluation.json").exists()
