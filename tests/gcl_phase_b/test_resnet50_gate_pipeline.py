from pathlib import Path
import json

import pytest

from experiments.baseline_diagnosis.proto_gen import trace_pb2
from experiments.gcl_phase_b.resnet50_gate0 import (
    ResNet50NvbitAcquisitionConfig,
    acquire_resnet50_gate0_trace,
    record_resnet50_gate0_trace_acquisition,
    write_resnet50_gate0_blocker_report,
)
from experiments.gcl_phase_b.resnet50_gate_pipeline import (
    GATE1_7_PIPELINE_MANIFEST_FILENAME,
    run_resnet50_gate1_to_gate7,
)
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


def _contract_style_root(root):
    write_minimal_artifact_shape_resnet50_root(
        root,
        evidence_scope="real_resnet50_nvbit_collection",
    )
    trace = trace_pb2.Trace()
    trace.ParseFromString((root / "dynamic_trace.pb").read_bytes())
    trace.name = "resnet50_contract_trace"
    (root / "dynamic_trace.pb").write_bytes(trace.SerializeToString())
    scheduler = read_json(root / "scheduler_metadata.json")
    scheduler["artifact_type"] = "resnet50_scheduler_metadata_real_nvbit_contract"
    write_json(root / "scheduler_metadata.json", scheduler)
    enhanced = read_json(root / "enhanced_execution_info.json")
    enhanced["artifact_type"] = "resnet50_enhanced_execution_info_real_nvbit_contract"
    write_json(root / "enhanced_execution_info.json", enhanced)
    evidence = read_json(root / "nvbit_collection_evidence.json")
    evidence["runner_invocation"] = ["python", "collect_real_resnet50_trace.py"]
    write_json(root / "nvbit_collection_evidence.json", evidence)


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
        assert "synthetic artifact-shape" in str(exc)
    else:
        raise AssertionError("synthetic artifact-shape root must not produce formal Gate0")


def test_resnet50_gate_pipeline_propagates_baseline_artifacts_in_debug_report_path(tmp_path):
    root = tmp_path / "formal_shape_trace"

    def runner(command, *, cwd, env):
        assert env["GCL_RESNET50_TRACE_OUT"] == str(root)
        _contract_style_root(root)
        return {"returncode": 0, "stdout": "ok", "stderr": ""}

    acquire_resnet50_gate0_trace(
        ResNet50NvbitAcquisitionConfig(
            output_root=root,
            workload_command=["python", "run_resnet50.py"],
            nvbit_tool_path=Path("/opt/nvbit/tools/trace_tool.so"),
            working_directory=tmp_path,
        ),
        runner=runner,
    )
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

    assert manifest["final_gate"] == "gate9_evaluated"
    assert (out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME).exists()
    gate7 = read_json(out_dir / "gate7_correctness_manifest.json")
    assert gate7["metric_error_report"]["status"] == "reported"
    assert gate7["metric_error_report"]["global_weighted_mape"] == 0.05
    gate9 = read_json(out_dir / "gate9_sampled_vs_full_evaluation.json")
    assert gate9["sampled_speedup_report"]["cycles_speedup"] == pytest.approx(1.05263158)
    assert gate9["sampled_error_report"]["cycles_relative_error"] == 0.05
