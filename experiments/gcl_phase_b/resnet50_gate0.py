"""Gate 0 ResNet-50 NVBit trace acquisition manifest contract."""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .utils import hash_without, read_json, write_json

GATE0_MANIFEST_FILENAME = "gate0_trace_acquisition_manifest.json"
GATE0_BLOCKER_FILENAME = "gate0_trace_acquisition_blocker_report.json"
GATE0_ARTIFACT_TYPE = "gcl_resnet50_gate0_trace_acquisition_manifest"
GATE0_ARTIFACT_VERSION = "gate0_trace_acquisition_manifest_v1"
GATE0_BLOCKER_TYPE = "gcl_resnet50_gate0_trace_acquisition_blocker_report"
GATE0_BLOCKER_VERSION = "gate0_trace_acquisition_blocker_report_v1"
NVBIT_COLLECTION_EVIDENCE_FILENAME = "nvbit_collection_evidence.json"
NVBIT_COLLECTOR_ATTESTATION_FILENAME = "nvbit_collector_attestation.json"
NVBIT_COLLECTOR_SESSION_FILENAME = ".nvbit_collector_session.json"
COLLECTOR_PRODUCER = "acquire_resnet50_gate0_trace"
FORMAL_SOURCE_ARTIFACTS = {
    "dynamic_trace.pb": "file",
    "threadblocks/": "directory",
    "enhanced_execution_info.json": "file",
    "scheduler_metadata.json": "file",
    "stats.csv": "file",
}
_ACTIVE_COLLECTOR_SESSION_IDS: set[str] = set()


@dataclass(frozen=True)
class ResNet50NvbitAcquisitionConfig:
    output_root: Path
    workload_command: list[str]
    nvbit_tool_path: Path
    working_directory: Path | None = None
    environment: dict[str, str] = field(default_factory=dict)


RunnerResult = subprocess.CompletedProcess[str] | dict[str, Any]
Runner = Callable[..., RunnerResult]


def acquire_resnet50_gate0_trace(
    config: ResNet50NvbitAcquisitionConfig,
    *,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Run the configured ResNet-50 NVBit collection and record Gate 0."""

    if not config.workload_command:
        raise ValueError("ResNet-50 workload_command is required")
    if not str(config.nvbit_tool_path):
        raise ValueError("NVBit tool path is required")
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(config.environment)
    env["LD_PRELOAD"] = str(config.nvbit_tool_path)
    env["CUDA_INJECTION64_PATH"] = str(config.nvbit_tool_path)
    env["USER_DEFINED_FOLDERS"] = "1"
    env["TRACES_FOLDER"] = str(output_root)
    env["GCL_RESNET50_TRACE_OUT"] = str(output_root)
    session = _write_collector_session(output_root, config)
    env["GCL_RESNET50_COLLECTOR_SESSION_ID"] = session["collector_session_id"]
    run = runner or _subprocess_runner
    _ACTIVE_COLLECTOR_SESSION_IDS.add(session["collector_session_id"])
    try:
        result = run(
            list(config.workload_command),
            cwd=Path(config.working_directory) if config.working_directory else None,
            env=env,
        )
        returncode = _runner_returncode(result)
        if returncode != 0:
            raise RuntimeError(f"ResNet-50 NVBit acquisition failed with returncode {returncode}")
        _write_nvbit_collection_evidence(output_root, result)
        _write_collector_attestation(output_root, session, result)
        return record_resnet50_gate0_trace_acquisition(
            output_root,
            active_collector_session_id=session["collector_session_id"],
        )
    finally:
        _ACTIVE_COLLECTOR_SESSION_IDS.discard(session["collector_session_id"])


def record_resnet50_gate0_trace_acquisition(
    root: Path,
    *,
    active_collector_session_id: str | None = None,
) -> dict[str, Any]:
    """Record a formal Gate 0 manifest for an already collected NVBit trace root."""

    root = Path(root)
    evidence = _load_nvbit_collection_evidence(root)
    scheduler_metadata = read_json(root / "scheduler_metadata.json")
    if scheduler_metadata.get("scheduler_metadata_source") != "real_nvbit_smid":
        raise ValueError("scheduler_metadata_source must be real_nvbit_smid")
    _validate_scheduler_metadata_records(scheduler_metadata)
    _reject_fixture_backed_root(root)
    _reject_synthetic_artifact_shape_root(root, evidence, scheduler_metadata)
    source_hashes = _source_artifact_hashes(root)
    _validate_collector_attestation(
        root,
        evidence,
        source_hashes,
        active_collector_session_id=active_collector_session_id,
    )
    _validate_previous_gate0_manifest_for_revalidation(
        root,
        evidence,
        source_hashes,
        active_collector_session_id=active_collector_session_id,
    )
    manifest = {
        "artifact_type": GATE0_ARTIFACT_TYPE,
        "artifact_version": GATE0_ARTIFACT_VERSION,
        "artifact_status": "formal",
        "formal_input_eligible": True,
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "nvbit_collection_evidence_hash": hash_without(evidence),
        "source_artifact_hashes": source_hashes,
    }
    manifest["gate0_manifest_hash"] = hash_without(manifest, "gate0_manifest_hash")
    validate_gate0_trace_acquisition_manifest(manifest)
    write_json(root / GATE0_MANIFEST_FILENAME, manifest)
    return manifest


def _validate_previous_gate0_manifest_for_revalidation(
    root: Path,
    evidence: dict[str, Any],
    source_hashes: dict[str, str],
    *,
    active_collector_session_id: str | None,
) -> None:
    if active_collector_session_id is not None:
        return
    manifest_path = root / GATE0_MANIFEST_FILENAME
    if not manifest_path.is_file():
        _validate_first_time_gate0_recording_evidence(evidence)
        return
    manifest = read_json(manifest_path)
    _validate_existing_gate0_manifest(manifest, evidence, source_hashes)


def _validate_first_time_gate0_recording_evidence(evidence: dict[str, Any]) -> None:
    if evidence.get("collector_session_id_from_env") != evidence.get("collector_session_id"):
        raise ValueError(
            "existing formal Gate0 manifest is required for collector attestation revalidation"
        )
    if evidence.get("nvbit_banner_observed") is not True:
        raise ValueError(
            "existing formal Gate0 manifest is required for collector attestation revalidation"
        )


def _validate_existing_gate0_manifest(
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    source_hashes: dict[str, str],
) -> None:
    if manifest.get("artifact_type") != GATE0_ARTIFACT_TYPE:
        raise ValueError("existing formal Gate0 manifest artifact_type mismatch")
    if manifest.get("artifact_status") != "formal":
        raise ValueError("existing formal Gate0 manifest is required for revalidation")
    if manifest.get("formal_input_eligible") is not True:
        raise ValueError("existing formal Gate0 manifest must be formal input eligible")
    if manifest.get("source_artifact_hashes") != source_hashes:
        raise ValueError("existing formal Gate0 manifest source artifact hashes mismatch")
    if manifest.get("nvbit_collection_evidence_hash") != hash_without(evidence):
        raise ValueError("existing formal Gate0 manifest evidence hash mismatch")
    if manifest.get("gate0_manifest_hash") != hash_without(manifest, "gate0_manifest_hash"):
        raise ValueError("existing formal Gate0 manifest hash is not reproducible")


def write_resnet50_gate0_blocker_report(
    root: Path,
    *,
    reason: str,
    missing_requirements: list[str],
) -> dict[str, Any]:
    root = Path(root)
    report = {
        "artifact_type": GATE0_BLOCKER_TYPE,
        "artifact_version": GATE0_BLOCKER_VERSION,
        "artifact_status": "formal_blocked",
        "formal_input_eligible": False,
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "blocked_gate": "gate0",
        "reason": reason,
        "missing_requirements": list(missing_requirements),
        "available_artifacts": _available_gate0_artifacts(root),
    }
    report["gate0_blocker_report_hash"] = hash_without(report, "gate0_blocker_report_hash")
    write_json(root / GATE0_BLOCKER_FILENAME, report)
    return report


def validate_gate0_trace_acquisition_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("artifact_type") != GATE0_ARTIFACT_TYPE:
        raise ValueError("unexpected Gate0 artifact_type")
    if manifest.get("artifact_version") != GATE0_ARTIFACT_VERSION:
        raise ValueError("unexpected Gate0 artifact_version")
    if manifest.get("artifact_status") != "formal":
        raise ValueError("Gate0 manifest must be formal")
    if manifest.get("formal_input_eligible") is not True:
        raise ValueError("Gate0 manifest must be formal input eligible")
    required = {
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Gate0 {field} must be {expected}")
    hashes = manifest.get("source_artifact_hashes")
    if set(hashes or {}) != set(FORMAL_SOURCE_ARTIFACTS):
        raise ValueError("Gate0 source_artifact_hashes are incomplete")
    if manifest.get("gate0_manifest_hash") != hash_without(manifest, "gate0_manifest_hash"):
        raise ValueError("gate0_manifest_hash is not reproducible")
    if not manifest.get("nvbit_collection_evidence_hash"):
        raise ValueError("Gate0 manifest requires nvbit_collection_evidence_hash")


def load_gate0_trace_acquisition_manifest(root: Path) -> dict[str, Any]:
    path = Path(root) / GATE0_MANIFEST_FILENAME
    if not path.exists():
        raise ValueError("Gate0 formal acquisition manifest is required")
    manifest = read_json(path)
    validate_gate0_trace_acquisition_manifest(manifest)
    return manifest


def _load_nvbit_collection_evidence(root: Path) -> dict[str, Any]:
    path = root / NVBIT_COLLECTION_EVIDENCE_FILENAME
    if not path.exists():
        raise ValueError("real NVBit collection evidence is required for formal Gate0")
    evidence = read_json(path)
    _validate_nvbit_collection_evidence(evidence)
    return evidence


def _validate_nvbit_collection_evidence(evidence: dict[str, Any]) -> None:
    required = {
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "collection_status": "completed",
    }
    for field, expected in required.items():
        if evidence.get(field) != expected:
            raise ValueError(f"NVBit collection evidence {field} must be {expected}")
    if evidence.get("artifact_status") != "formal_collection_evidence":
        raise ValueError("NVBit collection evidence must be formal_collection_evidence")
    if evidence.get("fixture_backed") is not False:
        raise ValueError("fixture-backed roots cannot produce formal Gate0 manifests")
    if evidence.get("collector_artifact_origin") != "real_nvbit_runtime":
        raise ValueError("real NVBit runtime artifact origin is required for formal Gate0")
    evidence_scope = evidence.get("evidence_scope")
    if evidence_scope and evidence_scope != "real_resnet50_nvbit_collection":
        raise ValueError("synthetic artifact-shape roots cannot produce formal Gate0 manifests")
    if evidence.get("nvbit_loaded") is not True:
        raise ValueError("NVBit collection evidence must confirm nvbit_loaded")


def _validate_collector_attestation(
    root: Path,
    evidence: dict[str, Any],
    source_hashes: dict[str, str],
    *,
    active_collector_session_id: str | None,
) -> None:
    evidence_attestation_hash = evidence.get("collector_attestation_hash")
    if not evidence_attestation_hash:
        raise ValueError("collector attestation is required for formal Gate0")
    path = root / NVBIT_COLLECTOR_ATTESTATION_FILENAME
    if not path.is_file():
        raise ValueError("persisted collector attestation artifact is required for formal Gate0")
    attestation = read_json(path)
    if attestation.get("artifact_type") != "gcl_resnet50_nvbit_collector_attestation":
        raise ValueError("collector attestation artifact_type mismatch")
    if attestation.get("producer") != COLLECTOR_PRODUCER:
        raise ValueError("collector-produced attestation is required for formal Gate0")
    session_id = attestation.get("collector_session_id")
    if not session_id:
        raise ValueError("collector-produced attestation requires collector_session_id")
    if active_collector_session_id is not None and session_id != active_collector_session_id:
        raise ValueError("collector attestation is not bound to the active collector session")
    if (
        active_collector_session_id is not None
        and active_collector_session_id not in _ACTIVE_COLLECTOR_SESSION_IDS
    ):
        raise ValueError("active collector session is required for formal Gate0 recording")
    session_path = root / NVBIT_COLLECTOR_SESSION_FILENAME
    if not session_path.is_file():
        raise ValueError("collector-produced session artifact is required for formal Gate0")
    session = read_json(session_path)
    if session.get("producer") != COLLECTOR_PRODUCER:
        raise ValueError("collector-produced session artifact is required for formal Gate0")
    if session.get("collector_session_id") != session_id:
        raise ValueError("collector attestation session does not match producer session")
    session_hash = hash_without(session, "collector_session_hash")
    if session.get("collector_session_hash") != session_hash:
        raise ValueError("collector session hash is not reproducible")
    if attestation.get("collector_session_hash") != session_hash:
        raise ValueError("collector attestation does not match collector session hash")
    if evidence.get("collector_session_hash") != session_hash:
        raise ValueError("collector evidence does not match collector session hash")
    if (
        active_collector_session_id is not None
        and evidence.get("collector_session_id_from_env") != session_id
    ):
        raise ValueError("collector evidence must be bound to the runner session environment")
    if evidence.get("collector_session_id") != session_id:
        raise ValueError("collector attestation session does not match evidence")
    if evidence.get("collector_producer") != COLLECTOR_PRODUCER:
        raise ValueError("collector-produced evidence is required for formal Gate0")
    for field in (
        "workload_id",
        "execution_mode",
        "trace_source",
        "input_scope",
        "scheduler_metadata_source",
        "collection_status",
    ):
        if attestation.get(field) != evidence.get(field):
            raise ValueError(f"collector attestation {field} does not match evidence")
    if attestation.get("source_artifact_hashes") != source_hashes:
        raise ValueError("collector attestation source artifact hashes do not match Gate0 artifacts")
    actual_hash = hash_without(attestation, "collector_attestation_hash")
    if attestation.get("collector_attestation_hash") != actual_hash:
        raise ValueError("collector attestation hash is not reproducible")
    if evidence_attestation_hash != actual_hash:
        raise ValueError("collector attestation hash does not match evidence reference")


def _write_collector_session(
    output_root: Path,
    config: ResNet50NvbitAcquisitionConfig,
) -> dict[str, Any]:
    session = {
        "artifact_type": "gcl_resnet50_nvbit_collector_session",
        "artifact_version": "nvbit_collector_session_v1",
        "producer": COLLECTOR_PRODUCER,
        "collector_session_id": uuid.uuid4().hex,
        "workload_command": list(config.workload_command),
        "nvbit_tool_path": str(config.nvbit_tool_path),
        "output_root": str(output_root),
        "created_unix_ns": time.time_ns(),
    }
    session["collector_session_hash"] = hash_without(session, "collector_session_hash")
    write_json(output_root / NVBIT_COLLECTOR_SESSION_FILENAME, session)
    return session


def _write_collector_attestation(
    root: Path,
    session: dict[str, Any],
    result: RunnerResult,
) -> dict[str, Any]:
    evidence_path = root / NVBIT_COLLECTION_EVIDENCE_FILENAME
    if not evidence_path.is_file():
        raise ValueError("real NVBit collection evidence is required for formal Gate0")
    evidence = read_json(evidence_path)
    _validate_nvbit_collection_evidence(evidence)
    source_hashes = _source_artifact_hashes(root)
    attestation = {
        "artifact_type": "gcl_resnet50_nvbit_collector_attestation",
        "artifact_version": "nvbit_collector_attestation_v1",
        "producer": COLLECTOR_PRODUCER,
        "collector_session_id": session["collector_session_id"],
        "collector_session_hash": session["collector_session_hash"],
        "runner_returncode": _runner_returncode(result),
        "workload_id": evidence.get("workload_id"),
        "execution_mode": evidence.get("execution_mode"),
        "trace_source": evidence.get("trace_source"),
        "input_scope": evidence.get("input_scope"),
        "scheduler_metadata_source": evidence.get("scheduler_metadata_source"),
        "collection_status": evidence.get("collection_status"),
        "source_artifact_hashes": source_hashes,
    }
    attestation["collector_attestation_hash"] = hash_without(
        attestation, "collector_attestation_hash"
    )
    evidence["collector_producer"] = COLLECTOR_PRODUCER
    evidence["collector_session_id_from_env"] = session["collector_session_id"]
    evidence["collector_session_id"] = session["collector_session_id"]
    evidence["collector_session_hash"] = session["collector_session_hash"]
    evidence["collector_attestation_hash"] = attestation["collector_attestation_hash"]
    write_json(root / NVBIT_COLLECTOR_ATTESTATION_FILENAME, attestation)
    write_json(evidence_path, evidence)
    return attestation


def _write_nvbit_collection_evidence(root: Path, result: RunnerResult) -> dict[str, Any]:
    _source_artifact_hashes(root)
    evidence_path = root / NVBIT_COLLECTION_EVIDENCE_FILENAME
    if not evidence_path.is_file():
        raise ValueError("real NVBit collection evidence is required for formal Gate0")
    evidence = read_json(evidence_path)
    scheduler_metadata = read_json(root / "scheduler_metadata.json")
    if scheduler_metadata.get("scheduler_metadata_source") != "real_nvbit_smid":
        raise ValueError("scheduler_metadata_source must be real_nvbit_smid")
    _validate_scheduler_metadata_records(scheduler_metadata)
    _reject_synthetic_artifact_shape_root(root, evidence, scheduler_metadata)
    _validate_nvbit_collection_evidence(evidence)
    evidence["nvbit_banner_observed"] = _runner_output_contains(result, "nvbit")
    session_path = root / NVBIT_COLLECTOR_SESSION_FILENAME
    if session_path.is_file():
        session = read_json(session_path)
        evidence["collector_session_id_from_env"] = session.get("collector_session_id")
    write_json(evidence_path, evidence)
    return evidence


def _reject_fixture_backed_root(root: Path) -> None:
    fixture_files = {
        "dynamic_trace.json",
        "threadblocks.json",
        "enhanced_execution_info.json",
        "scheduler_metadata.json",
    }
    for filename in fixture_files:
        path = root / filename
        if not path.exists() or path.suffix != ".json":
            continue
        artifact_type = read_json(path).get("artifact_type", "")
        if "fixture" in artifact_type:
            raise ValueError("fixture-backed roots cannot produce formal Gate0 manifests")


def _reject_synthetic_artifact_shape_root(
    root: Path,
    evidence: dict[str, Any],
    scheduler_metadata: dict[str, Any],
) -> None:
    markers = []
    evidence_scope = evidence.get("evidence_scope")
    if evidence_scope == "synthetic_artifact_shape_unit_test_only":
        markers.append("synthetic evidence_scope")
    if evidence.get("runner_invocation") == ["python", "run_resnet50.py"]:
        markers.append("unit-test runner_invocation")
    if scheduler_metadata.get("artifact_type") == "resnet50_scheduler_metadata_nvbit":
        markers.append("artifact-shape scheduler_metadata")
    enhanced_path = root / "enhanced_execution_info.json"
    if enhanced_path.is_file():
        enhanced = read_json(enhanced_path)
        if enhanced.get("artifact_type") == "resnet50_enhanced_execution_info_nvbit":
            markers.append("artifact-shape enhanced_execution_info")
    dynamic_path = root / "dynamic_trace.pb"
    if dynamic_path.is_file() and b"resnet50_formal_unit_trace" in dynamic_path.read_bytes():
        markers.append("artifact-shape dynamic_trace")
    if dynamic_path.is_file() and _is_artifact_shape_dynamic_trace(dynamic_path):
        markers.append("artifact-shape dynamic_trace protobuf")
    if markers:
        raise ValueError(
            "synthetic artifact-shape roots cannot produce formal Gate0 manifests: "
            + ", ".join(sorted(markers))
        )


def _is_artifact_shape_dynamic_trace(path: Path) -> bool:
    try:
        from experiments.baseline_diagnosis.proto_gen import trace_pb2

        trace = trace_pb2.Trace()
        trace.ParseFromString(path.read_bytes())
    except Exception:
        return False
    if trace.nvbit_version == "unit-nvbit":
        return True
    for device in trace.gpu_device.values():
        for stream in device.streams.values():
            for kernel in stream.kernels:
                if kernel.name.startswith("resnet50_conv2d_fprop_tile") and (
                    kernel.function_unique_id in {1701, 1702}
                ):
                    return True
    return False


def _validate_scheduler_metadata_records(scheduler_metadata: dict[str, Any]) -> None:
    for invocation in scheduler_metadata.get("kernel_invocations", []):
        for record in invocation.get("cta_records", []):
            required = {
                "sm_id",
                "cta_id",
                "warp_ids",
                "first_seen_order",
                "last_seen_order",
                "trace_entry_count",
            }
            missing = required.difference(record)
            if missing:
                raise ValueError(f"scheduler metadata missing required fields: {sorted(missing)}")
            if int(record["first_seen_order"]) > int(record["last_seen_order"]):
                raise ValueError("scheduler metadata first_seen_order must be <= last_seen_order")
            if int(record["trace_entry_count"]) <= 0:
                raise ValueError("scheduler metadata trace_entry_count must be positive")
            if not record["warp_ids"]:
                raise ValueError("scheduler metadata warp_ids must be non-empty")


def _source_artifact_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for name, kind in FORMAL_SOURCE_ARTIFACTS.items():
        path = _resolve_gate0_artifact_path(root, name)
        if kind == "file":
            if not path.is_file():
                raise FileNotFoundError(f"missing Gate0 source artifact: {name}")
            hashes[name] = _file_hash(path)
        elif kind == "directory":
            if not path.is_dir():
                raise FileNotFoundError(f"missing Gate0 source artifact: {name}")
            hashes[name] = _directory_hash(path)
        else:
            raise ValueError(f"unsupported source artifact kind: {kind}")
    return hashes


def _available_gate0_artifacts(root: Path) -> list[str]:
    return [
        name
        for name in [*FORMAL_SOURCE_ARTIFACTS, NVBIT_COLLECTION_EVIDENCE_FILENAME]
        if _resolve_gate0_artifact_path(root, name).exists()
    ]


def _resolve_gate0_artifact_path(root: Path, name: str) -> Path:
    path = root / name.rstrip("/")
    if name == "enhanced_execution_info.json" and not path.exists():
        return root / "extra_info" / "enhanced_execution_info.json"
    return path


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _directory_hash(path: Path) -> str:
    entries = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        entries.append(
            {
                "relative_path": str(child.relative_to(path)),
                "sha256": _file_hash(child),
            }
        )
    return hash_without({"entries": entries})


def _subprocess_runner(command: list[str], *, cwd: Path | None, env: dict[str, str]) -> RunnerResult:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def _runner_returncode(result: RunnerResult) -> int:
    if isinstance(result, dict):
        return int(result.get("returncode", 0))
    return int(result.returncode)


def _runner_output_contains(result: RunnerResult, needle: str) -> bool:
    if isinstance(result, dict):
        output = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    else:
        output = f"{result.stdout or ''}\n{result.stderr or ''}"
    return needle.lower() in output.lower()
