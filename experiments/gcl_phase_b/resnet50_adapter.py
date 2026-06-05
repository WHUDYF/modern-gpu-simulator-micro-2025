"""ResNet-50 real-trace adapter for GCL Gate 1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import hash_without, read_json, write_json

ADAPTER_ARTIFACT_TYPE = "gcl_resnet50_trace_adapter_bundle"
ADAPTER_VERSION = "gate1_trace_adapter_v1"


@dataclass(frozen=True)
class ResNet50TraceSources:
    dynamic_trace: dict[str, Any]
    threadblocks: dict[str, Any]
    enhanced_execution_info: dict[str, Any]
    scheduler_metadata: dict[str, Any]
    stats_rows: list[dict[str, str]]


def load_resnet50_trace_sources(root: Path) -> ResNet50TraceSources:
    stats_path = root / "stats.csv"
    with stats_path.open(newline="", encoding="utf-8") as handle:
        stats_rows = list(csv.DictReader(handle))
    return ResNet50TraceSources(
        dynamic_trace=read_json(root / "dynamic_trace.json"),
        threadblocks=read_json(root / "threadblocks.json"),
        enhanced_execution_info=read_json(root / "enhanced_execution_info.json"),
        scheduler_metadata=read_json(root / "scheduler_metadata.json"),
        stats_rows=stats_rows,
    )


def build_resnet50_trace_adapter_bundle(root: Path) -> dict[str, Any]:
    sources = load_resnet50_trace_sources(root)
    if sources.scheduler_metadata.get("scheduler_metadata_source") != "real_nvbit_smid":
        raise ValueError("scheduler_metadata_source must be real_nvbit_smid")
    kernel_invocation_table = _kernel_invocation_table(sources.dynamic_trace)
    static_instruction_table = list(sources.enhanced_execution_info.get("instructions", []))
    invocation_lookup = _invocation_lookup(kernel_invocation_table)
    cta_scheduler_records = _cta_scheduler_records(
        sources.scheduler_metadata, invocation_lookup
    )
    per_warp_trace_records = _per_warp_trace_records(
        sources.threadblocks, invocation_lookup
    )
    bundle = {
        "artifact_type": ADAPTER_ARTIFACT_TYPE,
        "artifact_version": ADAPTER_VERSION,
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "source_artifact_hashes": {
            "dynamic_trace": hash_without(sources.dynamic_trace, "_hash"),
            "threadblocks": hash_without(sources.threadblocks, "_hash"),
            "enhanced_execution_info": hash_without(sources.enhanced_execution_info, "_hash"),
            "scheduler_metadata": hash_without(sources.scheduler_metadata, "_hash"),
        },
        "kernel_invocation_table": kernel_invocation_table,
        "static_instruction_table": static_instruction_table,
        "cta_scheduler_records": cta_scheduler_records,
        "per_warp_trace_records": per_warp_trace_records,
        "adapter_validation_report": {
            "status": "passed",
            "scheduler_metadata_complete": True,
            "errors": [],
        },
    }
    bundle["adapter_bundle_hash"] = hash_without(bundle, "adapter_bundle_hash")
    validate_resnet50_trace_adapter_bundle(bundle)
    return bundle


def _kernel_invocation_table(dynamic_trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for launch_order, row in enumerate(dynamic_trace.get("kernel_invocations", [])):
        rows.append(
            {
                "kernel_invocation_id": f"resnet50_k{launch_order:05d}",
                "kernel_id": row["kernel_id"],
                "kernel_name": row["kernel_name"],
                "function_unique_id": row["function_unique_id"],
                "device_id": row.get("device_id", 0),
                "stream_id": row.get("stream_id", 0),
                "launch_order": launch_order,
                "grid_dim": row["grid_dim"],
                "block_dim": row["block_dim"],
                "shared_memory_size": row.get("shared_memory_size", 0),
                "register_count": row.get("register_count", 0),
            }
        )
    return rows


def _invocation_lookup(kernel_invocation_table: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["kernel_invocation_id"]: row for row in kernel_invocation_table}
    by_launch_order = {int(row["launch_order"]): row for row in kernel_invocation_table}
    by_kernel_id: dict[int, list[dict[str, Any]]] = {}
    for row in kernel_invocation_table:
        by_kernel_id.setdefault(int(row["kernel_id"]), []).append(row)
    return {
        "by_id": by_id,
        "by_launch_order": by_launch_order,
        "by_kernel_id": by_kernel_id,
    }


def _resolve_kernel_invocation_id(record: dict[str, Any], lookup: dict[str, Any]) -> str:
    explicit = record.get("kernel_invocation_id")
    if explicit is not None:
        if explicit not in lookup["by_id"]:
            raise ValueError("raw record references unknown kernel_invocation_id")
        _validate_record_identity_consistency(record, lookup["by_id"][explicit])
        return explicit
    if "launch_order" in record:
        launch_order = int(record["launch_order"])
        if launch_order not in lookup["by_launch_order"]:
            raise ValueError("raw record references unknown launch_order")
        row = lookup["by_launch_order"][launch_order]
        _validate_record_identity_consistency(record, row)
        return row["kernel_invocation_id"]
    kernel_id = int(record["kernel_id"])
    candidates = lookup["by_kernel_id"].get(kernel_id, [])
    if len(candidates) == 1:
        return candidates[0]["kernel_invocation_id"]
    raise ValueError(
        "raw record for repeated kernel_id requires kernel_invocation_id or launch_order"
    )


def _validate_record_identity_consistency(record: dict[str, Any], row: dict[str, Any]) -> None:
    if "kernel_id" in record and int(record["kernel_id"]) != int(row["kernel_id"]):
        raise ValueError("raw record kernel_id does not match resolved kernel invocation")
    if "launch_order" in record and int(record["launch_order"]) != int(row["launch_order"]):
        raise ValueError("raw record launch_order does not match resolved kernel invocation")


def _cta_scheduler_records(
    scheduler_metadata: dict[str, Any],
    invocation_lookup: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for invocation in scheduler_metadata.get("kernel_invocations", []):
        kernel_id = invocation["kernel_id"]
        kernel_invocation_id = _resolve_kernel_invocation_id(invocation, invocation_lookup)
        for cta in invocation.get("cta_records", []):
            records.append(
                {
                    "kernel_invocation_id": kernel_invocation_id,
                    "kernel_id": kernel_id,
                    **cta,
                }
            )
    return records


def _per_warp_trace_records(
    threadblocks: dict[str, Any],
    invocation_lookup: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for record in threadblocks.get("threadblocks", []):
        kernel_invocation_id = _resolve_kernel_invocation_id(record, invocation_lookup)
        records.append(
            {
                "kernel_invocation_id": kernel_invocation_id,
                **record,
            }
        )
    return records


def validate_resnet50_trace_adapter_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("artifact_type") != ADAPTER_ARTIFACT_TYPE:
        raise ValueError("unexpected adapter artifact_type")
    if bundle.get("artifact_version") != ADAPTER_VERSION:
        raise ValueError("unexpected adapter artifact_version")
    if bundle.get("workload_id") != "resnet50":
        raise ValueError("workload_id must be resnet50")
    if bundle.get("execution_mode") != "real_trace":
        raise ValueError("execution_mode must be real_trace")
    if bundle.get("scheduler_metadata_source") != "real_nvbit_smid":
        raise ValueError("scheduler_metadata_source must be real_nvbit_smid")
    report = bundle.get("adapter_validation_report", {})
    if report.get("status") != "passed":
        raise ValueError("adapter validation report must be passed")
    if report.get("scheduler_metadata_complete") is not True:
        raise ValueError("scheduler metadata must be complete")
    if report.get("errors") != []:
        raise ValueError("adapter errors must be empty")
    if not bundle.get("kernel_invocation_table"):
        raise ValueError("kernel_invocation_table must be non-empty")
    if not bundle.get("static_instruction_table"):
        raise ValueError("static_instruction_table must be non-empty")
    if not bundle.get("cta_scheduler_records"):
        raise ValueError("cta_scheduler_records must be non-empty")
    if not bundle.get("per_warp_trace_records"):
        raise ValueError("per_warp_trace_records must be non-empty")
    _validate_invocation_provenance(bundle)
    if bundle.get("adapter_bundle_hash") != hash_without(bundle, "adapter_bundle_hash"):
        raise ValueError("adapter_bundle_hash is not reproducible")


def _validate_invocation_provenance(bundle: dict[str, Any]) -> None:
    invocation_ids = {row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]}
    scheduled_by_invocation: dict[str, int] = {invocation_id: 0 for invocation_id in invocation_ids}
    trace_records_by_invocation: dict[str, int] = {invocation_id: 0 for invocation_id in invocation_ids}
    seen_ctas: set[tuple[str, str]] = set()
    scheduler_by_cta: dict[tuple[str, str], dict[str, Any]] = {}
    for record in bundle["cta_scheduler_records"]:
        invocation_id = record.get("kernel_invocation_id")
        if invocation_id not in invocation_ids:
            raise ValueError("cta scheduler record references unknown kernel_invocation_id")
        cta_key = (invocation_id, record["cta_id"])
        if cta_key in seen_ctas:
            raise ValueError("duplicate cta scheduler record for kernel_invocation_id and cta_id")
        seen_ctas.add(cta_key)
        if int(record["first_seen_order"]) > int(record["last_seen_order"]):
            raise ValueError("cta scheduler first_seen_order must be <= last_seen_order")
        if record.get("trace_entry_count", 0) <= 0:
            raise ValueError("cta scheduler trace_entry_count must be positive")
        if not record.get("warp_ids"):
            raise ValueError("cta scheduler warp_ids must be non-empty")
        scheduled_by_invocation[invocation_id] += int(record["trace_entry_count"])
        scheduler_by_cta[cta_key] = record
    trace_by_cta: dict[tuple[str, str], dict[str, Any]] = {}
    for record in bundle["per_warp_trace_records"]:
        invocation_id = record.get("kernel_invocation_id")
        if invocation_id not in invocation_ids:
            raise ValueError("warp trace record references unknown kernel_invocation_id")
        cta_key = (invocation_id, record["cta_id"])
        aggregate = trace_by_cta.setdefault(cta_key, {"warp_ids": set(), "entry_count": 0})
        if record["warp_id"] in aggregate["warp_ids"]:
            raise ValueError("duplicate warp trace record for kernel_invocation_id, cta_id and warp_id")
        aggregate["warp_ids"].add(record["warp_id"])
        aggregate["entry_count"] += len(record.get("entries", []))
        trace_records_by_invocation[invocation_id] += len(record.get("entries", []))
    if set(scheduler_by_cta) != set(trace_by_cta):
        raise ValueError("scheduler CTA set must match warp trace CTA set")
    for cta_key, scheduler_record in scheduler_by_cta.items():
        trace_record = trace_by_cta[cta_key]
        if set(scheduler_record["warp_ids"]) != trace_record["warp_ids"]:
            raise ValueError("scheduler warp_ids must match traced warp IDs")
        if int(scheduler_record["trace_entry_count"]) != trace_record["entry_count"]:
            raise ValueError("scheduler trace_entry_count must match traced entry count")
    for invocation_id in invocation_ids:
        if scheduled_by_invocation[invocation_id] <= 0:
            raise ValueError("each kernel_invocation_id must retain scheduler records")
        if trace_records_by_invocation[invocation_id] <= 0:
            raise ValueError("each kernel_invocation_id must retain warp trace records")


def write_resnet50_trace_adapter_bundle(root: Path, out_path: Path) -> dict[str, Any]:
    bundle = build_resnet50_trace_adapter_bundle(root)
    write_json(out_path, bundle)
    return bundle
