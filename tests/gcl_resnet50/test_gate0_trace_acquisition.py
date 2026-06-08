import json
import shutil
from pathlib import Path

import pytest

from experiments.gcl_phase_b.resnet50_gate0 import (
    COLLECTOR_PRODUCER,
    ResNet50NvbitAcquisitionConfig,
    acquire_resnet50_gate0_trace,
    record_resnet50_gate0_trace_acquisition,
    write_resnet50_gate0_blocker_report,
)
from experiments.gcl_phase_b.utils import hash_without, read_json, write_json
from gcl_resnet50.formal_fixture import write_minimal_artifact_shape_resnet50_root

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

    with pytest.raises(ValueError, match="real NVBit runtime artifact origin"):
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

    with pytest.raises(ValueError, match="real NVBit runtime artifact origin"):
        acquire_resnet50_gate0_trace(config, runner=runner)

    assert executed
    assert executed[0]["command"] == ["python", "run_resnet50.py"]
    assert executed[0]["env"]["LD_PRELOAD"] == "/opt/nvbit/tools/trace_tool.so"
    assert executed[0]["env"]["GCL_RESNET50_TRACE_OUT"] == str(root)
    assert not (root / "gate0_trace_acquisition_manifest.json").exists()


def test_gate0_dynamic_trace_shape_check_handles_real_trace_gpu_device_map(tmp_path):
    from experiments.baseline_diagnosis.proto_gen import trace_pb2
    from experiments.gcl_phase_b.resnet50_gate0 import _is_artifact_shape_dynamic_trace

    trace = trace_pb2.Trace()
    trace.nvbit_version = "real-nvbit"
    kernel = trace.gpu_device[0].streams[0].kernels.add()
    kernel.name = "real_resnet50_conv2d_kernel"
    kernel.function_unique_id = 9999
    path = tmp_path / "dynamic_trace.pb"
    path.write_bytes(trace.SerializeToString())

    assert _is_artifact_shape_dynamic_trace(path) is False


def test_gate0_acquisition_runner_sets_nvbit_trace_folder_environment(tmp_path):
    root = tmp_path / "formal_trace"
    executed = []

    def runner(command, *, cwd, env):
        executed.append({"command": command, "cwd": cwd, "env": env})
        return {"returncode": 0, "stdout": "ok", "stderr": ""}

    config = ResNet50NvbitAcquisitionConfig(
        output_root=root,
        workload_command=["python", "run_resnet50.py"],
        nvbit_tool_path=Path("/opt/nvbit/tools/tracer_tool.so"),
        working_directory=tmp_path,
        environment={"CUDA_VISIBLE_DEVICES": "0"},
    )

    with pytest.raises(FileNotFoundError, match="missing Gate0 source artifact"):
        acquire_resnet50_gate0_trace(config, runner=runner)

    env = executed[0]["env"]
    assert env["USER_DEFINED_FOLDERS"] == "1"
    assert env["TRACES_FOLDER"] == str(root)
    assert env["CUDA_INJECTION64_PATH"] == "/opt/nvbit/tools/tracer_tool.so"
    assert env["LD_PRELOAD"] == "/opt/nvbit/tools/tracer_tool.so"


def test_gate0_acquisition_runner_records_evidence_from_real_artifact_contract(tmp_path):
    root = tmp_path / "formal_trace"

    def runner(command, *, cwd, env):
        root.mkdir(parents=True, exist_ok=True)
        (root / "dynamic_trace.pb").write_bytes(b"real-nvbit-dynamic-trace")
        threadblocks = root / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
        threadblocks.mkdir(parents=True)
        (threadblocks / "d_0_s_0_k_1_0,0,0.pb").write_bytes(b"real-threadblock")
        extra_info = root / "extra_info"
        extra_info.mkdir()
        write_json(
            extra_info / "enhanced_execution_info.json",
            {"artifact_type": "real_nvbit_enhanced_execution_info", "instructions": []},
        )
        write_json(
            root / "scheduler_metadata.json",
            {
                "artifact_type": "gcl_real_trace_scheduler_metadata",
                "artifact_version": "resnet50_scheduler_metadata_v1",
                "scheduler_metadata_source": "real_nvbit_smid",
                "source": "nvbit_tracer",
                "kernel_invocations": [
                    {
                        "kernel_invocation_id": "d_0_s_0_k_1",
                        "kernel_id": 1,
                        "cta_records": [
                            {
                                "cta_id": "0,0,0",
                                "sm_id": 3,
                                "first_seen_order": 10,
                                "last_seen_order": 12,
                                "warp_ids": [0, 1],
                                "trace_entry_count": 3,
                            }
                        ],
                    }
                ],
            },
        )
        (root / "stats.csv").write_text(
            "device_id, stream_id, kernel id, kernel mangled name\n"
            "0, 0, kernel-1.trace, real_kernel\n",
            encoding="utf-8",
        )
        write_json(
            root / "nvbit_collection_evidence.json",
            {
                "artifact_status": "formal_collection_evidence",
                "workload_id": "resnet50",
                "execution_mode": "real_trace",
                "trace_source": "nvbit",
                "input_scope": "full_resnet50_inference_trace",
                "scheduler_metadata_source": "real_nvbit_smid",
                "collection_status": "completed",
                "fixture_backed": False,
                "collector_artifact_origin": "real_nvbit_runtime",
                "evidence_scope": "real_resnet50_nvbit_collection",
                "nvbit_loaded": True,
                "collector_session_id_from_env": env[
                    "GCL_RESNET50_COLLECTOR_SESSION_ID"
                ],
            },
        )
        return {"returncode": 0, "stdout": "NVBit Loaded", "stderr": ""}

    config = ResNet50NvbitAcquisitionConfig(
        output_root=root,
        workload_command=["python", "run_resnet50.py"],
        nvbit_tool_path=Path("/opt/nvbit/tools/tracer_tool.so"),
        working_directory=tmp_path,
    )

    manifest = acquire_resnet50_gate0_trace(config, runner=runner)

    evidence = read_json(root / "nvbit_collection_evidence.json")
    assert evidence["collector_artifact_origin"] == "real_nvbit_runtime"
    assert evidence["fixture_backed"] is False
    assert evidence["nvbit_loaded"] is True
    assert evidence["collector_session_id_from_env"] == evidence["collector_session_id"]
    assert manifest["formal_input_eligible"] is True
    assert (root / "nvbit_collector_attestation.json").exists()


def test_gate0_acquisition_synthesized_evidence_records_current_collector_session(tmp_path):
    root = tmp_path / "formal_trace_synthesized_evidence"

    def runner(command, *, cwd, env):
        root.mkdir(parents=True, exist_ok=True)
        (root / "dynamic_trace.pb").write_bytes(b"real-nvbit-dynamic-trace")
        threadblocks = root / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
        threadblocks.mkdir(parents=True)
        (threadblocks / "d_0_s_0_k_1_0,0,0.pb").write_bytes(b"real-threadblock")
        extra_info = root / "extra_info"
        extra_info.mkdir()
        write_json(
            extra_info / "enhanced_execution_info.json",
            {"artifact_type": "real_nvbit_enhanced_execution_info", "instructions": []},
        )
        write_json(
            root / "scheduler_metadata.json",
            {
                "artifact_type": "gcl_real_trace_scheduler_metadata",
                "artifact_version": "resnet50_scheduler_metadata_v1",
                "scheduler_metadata_source": "real_nvbit_smid",
                "source": "nvbit_tracer",
                "kernel_invocations": [
                    {
                        "kernel_invocation_id": "d_0_s_0_k_1",
                        "kernel_id": 1,
                        "cta_records": [
                            {
                                "cta_id": "0,0,0",
                                "sm_id": 3,
                                "first_seen_order": 10,
                                "last_seen_order": 12,
                                "warp_ids": [0, 1],
                                "trace_entry_count": 3,
                            }
                        ],
                    }
                ],
            },
        )
        (root / "stats.csv").write_text(
            "device_id, stream_id, kernel id, kernel mangled name\n"
            "0, 0, kernel-1.trace, real_kernel\n",
            encoding="utf-8",
        )
        return {"returncode": 0, "stdout": "NVBit Loaded", "stderr": ""}

    config = ResNet50NvbitAcquisitionConfig(
        output_root=root,
        workload_command=["python", "run_resnet50.py"],
        nvbit_tool_path=Path("/opt/nvbit/tools/tracer_tool.so"),
        working_directory=tmp_path,
    )

    manifest = acquire_resnet50_gate0_trace(config, runner=runner)

    evidence = read_json(root / "nvbit_collection_evidence.json")
    assert evidence["collector_session_id_from_env"] == evidence["collector_session_id"]
    assert manifest["formal_input_eligible"] is True


def test_gate0_acquisition_retry_rebinds_existing_evidence_to_current_session(tmp_path):
    root = tmp_path / "formal_trace_retry"
    session_ids = []

    def runner(command, *, cwd, env):
        session_ids.append(env["GCL_RESNET50_COLLECTOR_SESSION_ID"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "dynamic_trace.pb").write_bytes(b"real-nvbit-dynamic-trace")
        threadblocks = root / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
        threadblocks.mkdir(parents=True, exist_ok=True)
        (threadblocks / "d_0_s_0_k_1_0,0,0.pb").write_bytes(b"real-threadblock")
        extra_info = root / "extra_info"
        extra_info.mkdir(exist_ok=True)
        write_json(
            extra_info / "enhanced_execution_info.json",
            {"artifact_type": "real_nvbit_enhanced_execution_info", "instructions": []},
        )
        write_json(
            root / "scheduler_metadata.json",
            {
                "artifact_type": "gcl_real_trace_scheduler_metadata",
                "artifact_version": "resnet50_scheduler_metadata_v1",
                "scheduler_metadata_source": "real_nvbit_smid",
                "source": "nvbit_tracer",
                "kernel_invocations": [
                    {
                        "kernel_invocation_id": "d_0_s_0_k_1",
                        "kernel_id": 1,
                        "cta_records": [
                            {
                                "cta_id": "0,0,0",
                                "sm_id": 3,
                                "first_seen_order": 10,
                                "last_seen_order": 12,
                                "warp_ids": [0, 1],
                                "trace_entry_count": 3,
                            }
                        ],
                    }
                ],
            },
        )
        (root / "stats.csv").write_text(
            "device_id, stream_id, kernel id, kernel mangled name\n"
            "0, 0, kernel-1.trace, real_kernel\n",
            encoding="utf-8",
        )
        evidence_path = root / "nvbit_collection_evidence.json"
        if not evidence_path.exists():
            write_json(
                evidence_path,
                {
                    "artifact_status": "formal_collection_evidence",
                    "workload_id": "resnet50",
                    "execution_mode": "real_trace",
                    "trace_source": "nvbit",
                    "input_scope": "full_resnet50_inference_trace",
                    "scheduler_metadata_source": "real_nvbit_smid",
                    "collection_status": "completed",
                    "fixture_backed": False,
                    "collector_artifact_origin": "real_nvbit_runtime",
                    "evidence_scope": "real_resnet50_nvbit_collection",
                    "nvbit_loaded": True,
                    "collector_session_id_from_env": env[
                        "GCL_RESNET50_COLLECTOR_SESSION_ID"
                    ],
                },
            )
        return {"returncode": 0, "stdout": "NVBit Loaded", "stderr": ""}

    config = ResNet50NvbitAcquisitionConfig(
        output_root=root,
        workload_command=["python", "run_resnet50.py"],
        nvbit_tool_path=Path("/opt/nvbit/tools/tracer_tool.so"),
        working_directory=tmp_path,
    )

    acquire_resnet50_gate0_trace(config, runner=runner)
    manifest = acquire_resnet50_gate0_trace(config, runner=runner)

    evidence = read_json(root / "nvbit_collection_evidence.json")
    assert session_ids[0] != session_ids[1]
    assert evidence["collector_session_id_from_env"] == session_ids[1]
    assert evidence["collector_session_id"] == session_ids[1]
    assert manifest["formal_input_eligible"] is True


def test_gate0_recording_accepts_persisted_acquisition_artifacts_after_restart(tmp_path):
    root = tmp_path / "formal_trace_restart"

    def runner(command, *, cwd, env):
        root.mkdir(parents=True, exist_ok=True)
        (root / "dynamic_trace.pb").write_bytes(b"real-nvbit-dynamic-trace")
        threadblocks = root / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
        threadblocks.mkdir(parents=True)
        (threadblocks / "d_0_s_0_k_1_0,0,0.pb").write_bytes(b"real-threadblock")
        write_json(
            root / "enhanced_execution_info.json",
            {"artifact_type": "real_nvbit_enhanced_execution_info", "instructions": []},
        )
        write_json(
            root / "scheduler_metadata.json",
            {
                "artifact_type": "gcl_real_trace_scheduler_metadata",
                "artifact_version": "resnet50_scheduler_metadata_v1",
                "scheduler_metadata_source": "real_nvbit_smid",
                "source": "nvbit_tracer",
                "kernel_invocations": [
                    {
                        "kernel_invocation_id": "d_0_s_0_k_1",
                        "kernel_id": 1,
                        "cta_records": [
                            {
                                "cta_id": "0,0,0",
                                "sm_id": 3,
                                "first_seen_order": 10,
                                "last_seen_order": 12,
                                "warp_ids": [0, 1],
                                "trace_entry_count": 3,
                            }
                        ],
                    }
                ],
            },
        )
        (root / "stats.csv").write_text(
            "device_id, stream_id, kernel id, kernel mangled name\n"
            "0, 0, kernel-1.trace, real_kernel\n",
            encoding="utf-8",
        )
        write_json(
            root / "nvbit_collection_evidence.json",
            {
                "artifact_status": "formal_collection_evidence",
                "workload_id": "resnet50",
                "execution_mode": "real_trace",
                "trace_source": "nvbit",
                "input_scope": "full_resnet50_inference_trace",
                "scheduler_metadata_source": "real_nvbit_smid",
                "collection_status": "completed",
                "fixture_backed": False,
                "collector_artifact_origin": "real_nvbit_runtime",
                "evidence_scope": "real_resnet50_nvbit_collection",
                "nvbit_loaded": True,
                "collector_session_id_from_env": env[
                    "GCL_RESNET50_COLLECTOR_SESSION_ID"
                ],
            },
        )
        return {"returncode": 0, "stdout": "NVBit Loaded", "stderr": ""}

    config = ResNet50NvbitAcquisitionConfig(
        output_root=root,
        workload_command=["python", "run_resnet50.py"],
        nvbit_tool_path=Path("/opt/nvbit/tools/tracer_tool.so"),
        working_directory=tmp_path,
    )

    first_manifest = acquire_resnet50_gate0_trace(config, runner=runner)
    (root / "gate0_trace_acquisition_manifest.json").unlink()

    replay_manifest = record_resnet50_gate0_trace_acquisition(root)

    assert replay_manifest["gate0_manifest_hash"] == first_manifest["gate0_manifest_hash"]


def test_gate0_rejects_synthetic_helper_even_if_scope_claims_real_collection(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "spoofed_trace",
        evidence_scope="real_resnet50_nvbit_collection",
    )

    with pytest.raises(ValueError, match="real NVBit runtime artifact origin"):
        record_resnet50_gate0_trace_acquisition(root)


def test_gate0_rejects_handwritten_evidence_hash_without_persisted_attestation(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "spoofed_trace",
        evidence_scope="real_resnet50_nvbit_collection",
    )
    evidence_path = root / "nvbit_collection_evidence.json"
    evidence = read_json(evidence_path)
    evidence["collector_attestation_hash"] = "self-declared"
    write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="real NVBit runtime artifact origin"):
        record_resnet50_gate0_trace_acquisition(root)


def test_gate0_rejects_handwritten_matching_attestation_on_synthetic_root(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "spoofed_trace",
        evidence_scope="real_resnet50_nvbit_collection",
    )
    from experiments.gcl_phase_b.resnet50_gate0 import _source_artifact_hashes

    attestation = {
        "artifact_type": "gcl_resnet50_nvbit_collector_attestation",
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "collection_status": "completed",
        "source_artifact_hashes": _source_artifact_hashes(root),
    }
    attestation["collector_attestation_hash"] = hash_without(
        attestation, "collector_attestation_hash"
    )
    write_json(root / "nvbit_collector_attestation.json", attestation)
    evidence_path = root / "nvbit_collection_evidence.json"
    evidence = read_json(evidence_path)
    evidence["collector_attestation_hash"] = attestation["collector_attestation_hash"]
    write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="real NVBit runtime artifact origin"):
        record_resnet50_gate0_trace_acquisition(root)


def test_gate0_rejects_handwritten_session_attestation_triplet_on_synthetic_root(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "spoofed_trace",
        evidence_scope="real_resnet50_nvbit_collection",
    )
    from experiments.gcl_phase_b.resnet50_gate0 import _source_artifact_hashes

    session = {
        "artifact_type": "gcl_resnet50_nvbit_collector_session",
        "artifact_version": "nvbit_collector_session_v1",
        "producer": COLLECTOR_PRODUCER,
        "collector_session_id": "handwritten-session",
        "workload_command": ["python", "run_resnet50.py"],
        "nvbit_tool_path": "/opt/nvbit/tools/trace_tool.so",
        "output_root": str(root),
        "created_unix_ns": 1,
    }
    session["collector_session_hash"] = hash_without(session, "collector_session_hash")
    write_json(root / ".nvbit_collector_session.json", session)
    attestation = {
        "artifact_type": "gcl_resnet50_nvbit_collector_attestation",
        "artifact_version": "nvbit_collector_attestation_v1",
        "producer": COLLECTOR_PRODUCER,
        "collector_session_id": session["collector_session_id"],
        "collector_session_hash": session["collector_session_hash"],
        "runner_returncode": 0,
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "collection_status": "completed",
        "source_artifact_hashes": _source_artifact_hashes(root),
    }
    attestation["collector_attestation_hash"] = hash_without(
        attestation,
        "collector_attestation_hash",
    )
    write_json(root / "nvbit_collector_attestation.json", attestation)
    evidence_path = root / "nvbit_collection_evidence.json"
    evidence = read_json(evidence_path)
    evidence["collector_producer"] = COLLECTOR_PRODUCER
    evidence["collector_session_id"] = session["collector_session_id"]
    evidence["collector_session_hash"] = session["collector_session_hash"]
    evidence["collector_attestation_hash"] = attestation["collector_attestation_hash"]
    write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="real NVBit runtime artifact origin"):
        record_resnet50_gate0_trace_acquisition(root)


def test_gate0_rejects_marker_rewritten_synthetic_root(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "marker_rewritten",
        evidence_scope="real_resnet50_nvbit_collection",
    )
    scheduler = read_json(root / "scheduler_metadata.json")
    scheduler["artifact_type"] = "resnet50_scheduler_metadata_real_nvbit_contract"
    write_json(root / "scheduler_metadata.json", scheduler)
    enhanced = read_json(root / "enhanced_execution_info.json")
    enhanced["artifact_type"] = "resnet50_enhanced_execution_info_real_nvbit_contract"
    write_json(root / "enhanced_execution_info.json", enhanced)
    evidence_path = root / "nvbit_collection_evidence.json"
    evidence = read_json(evidence_path)
    evidence["runner_invocation"] = ["python", "collect_real_resnet50_trace.py"]
    write_json(evidence_path, evidence)

    with pytest.raises(ValueError, match="real NVBit runtime artifact origin"):
        record_resnet50_gate0_trace_acquisition(root)


def test_gate0_acquisition_rejects_marker_rewritten_artifact_shape_runner_output(tmp_path):
    root = tmp_path / "rewritten_runner_output"

    def runner(command, *, cwd, env):
        write_minimal_artifact_shape_resnet50_root(
            root,
            evidence_scope="real_resnet50_nvbit_collection",
        )
        scheduler = read_json(root / "scheduler_metadata.json")
        scheduler["artifact_type"] = "gcl_real_trace_scheduler_metadata"
        scheduler["source"] = "nvbit_tracer"
        write_json(root / "scheduler_metadata.json", scheduler)
        enhanced = read_json(root / "enhanced_execution_info.json")
        enhanced["artifact_type"] = "real_nvbit_enhanced_execution_info"
        write_json(root / "enhanced_execution_info.json", enhanced)
        evidence = read_json(root / "nvbit_collection_evidence.json")
        evidence["collector_artifact_origin"] = "real_nvbit_runtime"
        evidence["runner_invocation"] = ["python", "collect_real_resnet50_trace.py"]
        evidence["collector_session_id_from_env"] = env[
            "GCL_RESNET50_COLLECTOR_SESSION_ID"
        ]
        write_json(root / "nvbit_collection_evidence.json", evidence)
        return {"returncode": 0, "stdout": "NVBit Loaded", "stderr": ""}

    config = ResNet50NvbitAcquisitionConfig(
        output_root=root,
        workload_command=["python", "collect_real_resnet50_trace.py"],
        nvbit_tool_path=Path("/opt/nvbit/tools/trace_tool.so"),
        working_directory=tmp_path,
    )

    with pytest.raises(ValueError, match="artifact-shape dynamic_trace protobuf"):
        acquire_resnet50_gate0_trace(config, runner=runner)

    assert not (root / "gate0_trace_acquisition_manifest.json").exists()


def test_gate0_rejects_self_authenticated_real_shaped_root_without_runner_session_binding(
    tmp_path,
):
    root = tmp_path / "self_authenticated_real_shape"
    root.mkdir()
    (root / "dynamic_trace.pb").write_bytes(b"looks-like-real-nvbit-dynamic-trace")
    threadblocks = root / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
    threadblocks.mkdir(parents=True)
    (threadblocks / "d_0_s_0_k_1_0,0,0.pb").write_bytes(b"looks-like-real-threadblock")
    write_json(
        root / "enhanced_execution_info.json",
        {"artifact_type": "real_nvbit_enhanced_execution_info", "instructions": []},
    )
    write_json(
        root / "scheduler_metadata.json",
        {
            "artifact_type": "gcl_real_trace_scheduler_metadata",
            "artifact_version": "resnet50_scheduler_metadata_v1",
            "scheduler_metadata_source": "real_nvbit_smid",
            "source": "nvbit_tracer",
            "kernel_invocations": [
                {
                    "kernel_invocation_id": "d_0_s_0_k_1",
                    "kernel_id": 1,
                    "cta_records": [
                        {
                            "cta_id": "0,0,0",
                            "sm_id": 3,
                            "first_seen_order": 10,
                            "last_seen_order": 12,
                            "warp_ids": [0, 1],
                            "trace_entry_count": 3,
                        }
                    ],
                }
            ],
        },
    )
    (root / "stats.csv").write_text(
        "device_id, stream_id, kernel id, kernel mangled name\n"
        "0, 0, kernel-1.trace, real_kernel\n",
        encoding="utf-8",
    )

    from experiments.gcl_phase_b.resnet50_gate0 import _source_artifact_hashes

    session = {
        "artifact_type": "gcl_resnet50_nvbit_collector_session",
        "artifact_version": "nvbit_collector_session_v1",
        "producer": COLLECTOR_PRODUCER,
        "collector_session_id": "handwritten-real-shaped-session",
        "workload_command": ["python", "run_resnet50.py"],
        "nvbit_tool_path": "/opt/nvbit/tools/trace_tool.so",
        "output_root": str(root),
        "created_unix_ns": 1,
    }
    session["collector_session_hash"] = hash_without(session, "collector_session_hash")
    write_json(root / ".nvbit_collector_session.json", session)
    attestation = {
        "artifact_type": "gcl_resnet50_nvbit_collector_attestation",
        "artifact_version": "nvbit_collector_attestation_v1",
        "producer": COLLECTOR_PRODUCER,
        "collector_session_id": session["collector_session_id"],
        "collector_session_hash": session["collector_session_hash"],
        "runner_returncode": 0,
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "collection_status": "completed",
        "source_artifact_hashes": _source_artifact_hashes(root),
    }
    attestation["collector_attestation_hash"] = hash_without(
        attestation,
        "collector_attestation_hash",
    )
    write_json(root / "nvbit_collector_attestation.json", attestation)
    evidence = {
        "artifact_status": "formal_collection_evidence",
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "collection_status": "completed",
        "fixture_backed": False,
        "collector_artifact_origin": "real_nvbit_runtime",
        "evidence_scope": "real_resnet50_nvbit_collection",
        "nvbit_loaded": True,
        "collector_producer": COLLECTOR_PRODUCER,
        "collector_session_id": session["collector_session_id"],
        "collector_session_hash": session["collector_session_hash"],
        "collector_attestation_hash": attestation["collector_attestation_hash"],
    }
    write_json(root / "nvbit_collection_evidence.json", evidence)

    with pytest.raises(ValueError, match="runner session environment"):
        record_resnet50_gate0_trace_acquisition(root)


def test_gate0_rejects_self_authenticated_triplet_with_forged_session_hash(
    tmp_path,
):
    root = tmp_path / "self_authenticated_forged_session_hash"
    root.mkdir()
    (root / "dynamic_trace.pb").write_bytes(b"looks-like-real-nvbit-dynamic-trace")
    threadblocks = root / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
    threadblocks.mkdir(parents=True)
    (threadblocks / "d_0_s_0_k_1_0,0,0.pb").write_bytes(b"looks-like-real-threadblock")
    write_json(
        root / "enhanced_execution_info.json",
        {"artifact_type": "real_nvbit_enhanced_execution_info", "instructions": []},
    )
    write_json(
        root / "scheduler_metadata.json",
        {
            "artifact_type": "gcl_real_trace_scheduler_metadata",
            "artifact_version": "resnet50_scheduler_metadata_v1",
            "scheduler_metadata_source": "real_nvbit_smid",
            "source": "nvbit_tracer",
            "kernel_invocations": [
                {
                    "kernel_invocation_id": "d_0_s_0_k_1",
                    "kernel_id": 1,
                    "cta_records": [
                        {
                            "cta_id": "0,0,0",
                            "sm_id": 3,
                            "first_seen_order": 10,
                            "last_seen_order": 12,
                            "warp_ids": [0, 1],
                            "trace_entry_count": 3,
                        }
                    ],
                }
            ],
        },
    )
    (root / "stats.csv").write_text(
        "device_id, stream_id, kernel id, kernel mangled name\n"
        "0, 0, kernel-1.trace, real_kernel\n",
        encoding="utf-8",
    )

    from experiments.gcl_phase_b.resnet50_gate0 import _source_artifact_hashes

    session = {
        "artifact_type": "gcl_resnet50_nvbit_collector_session",
        "artifact_version": "nvbit_collector_session_v1",
        "producer": COLLECTOR_PRODUCER,
        "collector_session_id": "handwritten-real-shaped-session",
        "workload_command": ["python", "run_resnet50.py"],
        "nvbit_tool_path": "/opt/nvbit/tools/trace_tool.so",
        "output_root": str(root),
        "created_unix_ns": 1,
        "collector_session_hash": "forged-session-hash",
    }
    write_json(root / ".nvbit_collector_session.json", session)
    attestation = {
        "artifact_type": "gcl_resnet50_nvbit_collector_attestation",
        "artifact_version": "nvbit_collector_attestation_v1",
        "producer": COLLECTOR_PRODUCER,
        "collector_session_id": session["collector_session_id"],
        "collector_session_hash": session["collector_session_hash"],
        "runner_returncode": 0,
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "collection_status": "completed",
        "source_artifact_hashes": _source_artifact_hashes(root),
    }
    attestation["collector_attestation_hash"] = hash_without(
        attestation,
        "collector_attestation_hash",
    )
    write_json(root / "nvbit_collector_attestation.json", attestation)
    evidence = {
        "artifact_status": "formal_collection_evidence",
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "collection_status": "completed",
        "fixture_backed": False,
        "collector_artifact_origin": "real_nvbit_runtime",
        "evidence_scope": "real_resnet50_nvbit_collection",
        "nvbit_loaded": True,
        "collector_producer": COLLECTOR_PRODUCER,
        "collector_session_id_from_env": session["collector_session_id"],
        "collector_session_id": session["collector_session_id"],
        "collector_session_hash": session["collector_session_hash"],
        "collector_attestation_hash": attestation["collector_attestation_hash"],
    }
    write_json(root / "nvbit_collection_evidence.json", evidence)

    with pytest.raises(ValueError, match="collector session hash"):
        record_resnet50_gate0_trace_acquisition(root)


def test_gate0_acquisition_runner_rejects_synthetic_trace_before_attestation(tmp_path):
    root = tmp_path / "formal_trace"
    executed = []

    def runner(command, *, cwd, env):
        executed.append({"command": command, "cwd": cwd, "env": env})
        write_minimal_artifact_shape_resnet50_root(
            root,
            evidence_scope="real_resnet50_nvbit_collection",
        )
        return {"returncode": 0, "stdout": "ok", "stderr": ""}

    config = ResNet50NvbitAcquisitionConfig(
        output_root=root,
        workload_command=["python", "run_resnet50.py"],
        nvbit_tool_path=Path("/opt/nvbit/tools/trace_tool.so"),
        working_directory=tmp_path,
        environment={"CUDA_VISIBLE_DEVICES": "0"},
    )

    with pytest.raises(ValueError, match="real NVBit runtime artifact origin"):
        acquire_resnet50_gate0_trace(config, runner=runner)

    assert executed
    assert not (root / "nvbit_collector_attestation.json").exists()
    assert not (root / "gate0_trace_acquisition_manifest.json").exists()
