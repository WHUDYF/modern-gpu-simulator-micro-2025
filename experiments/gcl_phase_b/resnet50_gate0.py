"""Gate 0 ResNet-50 NVBit trace acquisition manifest contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .utils import hash_without, read_json, write_json

GATE0_MANIFEST_FILENAME = "gate0_trace_acquisition_manifest.json"
GATE0_BLOCKER_FILENAME = "gate0_trace_acquisition_blocker_report.json"
GATE0_ARTIFACT_TYPE = "gcl_resnet50_gate0_trace_acquisition_manifest"
GATE0_ARTIFACT_VERSION = "gate0_trace_acquisition_manifest_v1"
GATE0_BLOCKER_TYPE = "gcl_resnet50_gate0_trace_acquisition_blocker_report"
GATE0_BLOCKER_VERSION = "gate0_trace_acquisition_blocker_report_v1"
NVBIT_COLLECTION_EVIDENCE_FILENAME = "nvbit_collection_evidence.json"
FORMAL_SOURCE_ARTIFACTS = {
    "dynamic_trace.pb": "file",
    "threadblocks/": "directory",
    "enhanced_execution_info.json": "file",
    "scheduler_metadata.json": "file",
    "stats.csv": "file",
}


def record_resnet50_gate0_trace_acquisition(root: Path) -> dict[str, Any]:
    """Record a formal Gate 0 manifest for an already collected NVBit trace root."""

    root = Path(root)
    evidence = _load_nvbit_collection_evidence(root)
    scheduler_metadata = read_json(root / "scheduler_metadata.json")
    if scheduler_metadata.get("scheduler_metadata_source") != "real_nvbit_smid":
        raise ValueError("scheduler_metadata_source must be real_nvbit_smid")
    _validate_scheduler_metadata_records(scheduler_metadata)
    _reject_fixture_backed_root(root)
    source_hashes = _source_artifact_hashes(root)
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
    if evidence.get("nvbit_loaded") is not True:
        raise ValueError("NVBit collection evidence must confirm nvbit_loaded")
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
        path = root / name.rstrip("/")
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
        if (root / name.rstrip("/")).exists()
    ]


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
