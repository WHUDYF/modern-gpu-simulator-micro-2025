import json
import shutil
from pathlib import Path

import pytest

from experiments.gcl_phase_b.resnet50_gate0 import (
    ResNet50NvbitAcquisitionConfig,
    acquire_resnet50_gate0_trace,
    record_resnet50_gate0_trace_acquisition,
    write_resnet50_gate0_blocker_report,
)
from tests.gcl_resnet50.formal_fixture import write_minimal_artifact_shape_resnet50_root

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def _fixture_backed_root(tmp_path):
    root = tmp_path / "fixture_backed"
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "dynamic_trace.pb").write_bytes(b"formal-protobuf-placeholder")
    threadblocks_dir = root / "threadblocks"
    threadblocks_dir.mkdir()
    shutil.copy(root / "threadblocks.json", threadblocks_dir / "threadblocks.json")
    return root


def test_gate0_writes_blocker_when_real_resnet50_nvbit_collection_is_unavailable(tmp_path):
    root = tmp_path / "missing_real_trace"
    root.mkdir()

    report = write_resnet50_gate0_blocker_report(
        root,
        reason="real ResNet-50 NVBit trace has not been collected in this workspace",
        missing_requirements=[
            "dynamic_trace.pb",
            "threadblocks/",
            "enhanced_execution_info.json",
            "scheduler_metadata.json",
            "stats.csv",
            "nvbit_collection_evidence.json",
        ],
    )

    assert report["artifact_type"] == "gcl_resnet50_gate0_trace_acquisition_blocker_report"
    assert report["artifact_status"] == "formal_blocked"
    assert report["formal_input_eligible"] is False
    assert report["blocked_gate"] == "gate0"
    assert "dynamic_trace.pb" in report["missing_requirements"]
    assert (root / "gate0_trace_acquisition_blocker_report.json").exists()


def test_gate0_rejects_fixture_backed_placeholder_root(tmp_path):
    root = _fixture_backed_root(tmp_path)
    (root / "nvbit_collection_evidence.json").write_text(
        json.dumps(
            {
                "artifact_status": "formal_collection_evidence",
                "workload_id": "resnet50",
                "execution_mode": "real_trace",
                "trace_source": "nvbit",
                "input_scope": "full_resnet50_inference_trace",
                "scheduler_metadata_source": "real_nvbit_smid",
                "collection_status": "completed",
                "fixture_backed": True,
                "nvbit_loaded": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixture-backed"):
        record_resnet50_gate0_trace_acquisition(root)


def test_gate0_rejects_missing_real_smid_metadata(tmp_path):
    root = _fixture_backed_root(tmp_path)
    (root / "nvbit_collection_evidence.json").write_text(
        json.dumps(
            {
                "artifact_status": "formal_collection_evidence",
                "workload_id": "resnet50",
                "execution_mode": "real_trace",
                "trace_source": "nvbit",
                "input_scope": "full_resnet50_inference_trace",
                "scheduler_metadata_source": "real_nvbit_smid",
                "collection_status": "completed",
                "fixture_backed": False,
                "nvbit_loaded": True,
            }
        ),
        encoding="utf-8",
    )
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    scheduler["scheduler_metadata_source"] = "file_order_fallback"
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")

    with pytest.raises(ValueError, match="real_nvbit_smid"):
        record_resnet50_gate0_trace_acquisition(root)


def test_gate0_acquisition_runner_rejects_synthetic_artifact_shape_output(tmp_path):
    root = tmp_path / "formal_trace"
    executed = []

    def runner(command, *, cwd, env):
        executed.append({"command": command, "cwd": cwd, "env": env})
        write_minimal_artifact_shape_resnet50_root(root)
        return {"returncode": 0, "stdout": "ok", "stderr": ""}

    config = ResNet50NvbitAcquisitionConfig(
        output_root=root,
        workload_command=["python", "run_resnet50.py"],
        nvbit_tool_path=Path("/opt/nvbit/tools/trace_tool.so"),
        working_directory=tmp_path,
        environment={"CUDA_VISIBLE_DEVICES": "0"},
    )

    with pytest.raises(ValueError, match="synthetic artifact-shape"):
        acquire_resnet50_gate0_trace(config, runner=runner)

    assert executed
    assert executed[0]["command"] == ["python", "run_resnet50.py"]
    assert executed[0]["env"]["LD_PRELOAD"] == "/opt/nvbit/tools/trace_tool.so"
    assert executed[0]["env"]["GCL_RESNET50_TRACE_OUT"] == str(root)
    assert not (root / "gate0_trace_acquisition_manifest.json").exists()


def test_gate0_rejects_synthetic_helper_even_if_scope_claims_real_collection(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "spoofed_trace",
        evidence_scope="real_resnet50_nvbit_collection",
    )

    with pytest.raises(ValueError, match="collector attestation"):
        record_resnet50_gate0_trace_acquisition(root)
