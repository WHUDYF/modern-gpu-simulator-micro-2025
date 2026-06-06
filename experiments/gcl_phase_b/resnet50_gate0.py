"""Gate 0 ResNet-50 NVBit trace acquisition manifest contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .utils import hash_without, read_json, write_json

GATE0_MANIFEST_FILENAME = "gate0_trace_acquisition_manifest.json"
GATE0_ARTIFACT_TYPE = "gcl_resnet50_gate0_trace_acquisition_manifest"
GATE0_ARTIFACT_VERSION = "gate0_trace_acquisition_manifest_v1"
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
    scheduler_metadata = read_json(root / "scheduler_metadata.json")
    if scheduler_metadata.get("scheduler_metadata_source") != "real_nvbit_smid":
        raise ValueError("scheduler_metadata_source must be real_nvbit_smid")
    _validate_scheduler_metadata_records(scheduler_metadata)
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
        "source_artifact_hashes": source_hashes,
    }
    manifest["gate0_manifest_hash"] = hash_without(manifest, "gate0_manifest_hash")
    validate_gate0_trace_acquisition_manifest(manifest)
    write_json(root / GATE0_MANIFEST_FILENAME, manifest)
    return manifest


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


def load_gate0_trace_acquisition_manifest(root: Path) -> dict[str, Any]:
    path = Path(root) / GATE0_MANIFEST_FILENAME
    if not path.exists():
        raise ValueError("Gate0 formal acquisition manifest is required")
    manifest = read_json(path)
    validate_gate0_trace_acquisition_manifest(manifest)
    return manifest


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
