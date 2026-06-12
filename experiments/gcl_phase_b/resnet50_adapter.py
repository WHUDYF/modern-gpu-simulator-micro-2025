"""ResNet-50 real-trace adapter for GCL Gate 1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.baseline_diagnosis.proto_gen import threadblock_pb2, trace_pb2

from .resnet50_gate0 import (
    load_gate0_trace_acquisition_manifest,
    validate_gate0_source_artifacts_match_manifest,
)
from .sm_selection import select_representative_sm
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
    gate0_manifest: dict[str, Any]


def load_resnet50_trace_sources(
    root: Path,
    *,
    invocation_limit: int | None = None,
    invocation_ids: list[str] | None = None,
) -> ResNet50TraceSources:
    if invocation_limit is not None and invocation_limit <= 0:
        raise ValueError("invocation_limit must be positive")
    gate0_manifest = load_gate0_trace_acquisition_manifest(root)
    validate_gate0_source_artifacts_match_manifest(root, gate0_manifest)
    stats_path = root / "stats.csv"
    with stats_path.open(newline="", encoding="utf-8") as handle:
        stats_rows = list(csv.DictReader(handle))
    enhanced_execution_info = read_json(_enhanced_execution_info_path(root))
    dynamic_trace = _load_dynamic_trace_pb(root / "dynamic_trace.pb")
    scheduler_metadata = read_json(root / "scheduler_metadata.json")
    if invocation_ids is not None:
        dynamic_trace = _filter_dynamic_trace_by_invocation_ids(
            dynamic_trace,
            set(invocation_ids),
        )
    if invocation_limit is not None:
        dynamic_trace = _limit_dynamic_trace_invocations(dynamic_trace, invocation_limit)
    if invocation_limit is not None or invocation_ids is not None:
        kept_invocations = list(dynamic_trace["kernel_invocations"])
        kept_invocation_ids = {
            row["source_kernel_invocation_id"]
            for row in kept_invocations
            if row.get("source_kernel_invocation_id")
        }
        scheduler_has_only_legacy_ids = _scheduler_metadata_has_only_legacy_ids(
            scheduler_metadata
        )
        if (
            invocation_limit is not None
            and invocation_ids is None
            and scheduler_has_only_legacy_ids
        ):
            scheduler_metadata = _limit_scheduler_metadata_invocations(
                scheduler_metadata,
                len(kept_invocations),
            )
        elif scheduler_has_only_legacy_ids:
            scheduler_metadata = _filter_legacy_scheduler_metadata_by_launch_orders(
                scheduler_metadata,
                {int(row["launch_order"]) for row in kept_invocations},
            )
        else:
            kept_invocation_ids.update(
                _legacy_scheduler_invocation_ids(kept_invocations)
            )
            scheduler_metadata = _filter_scheduler_metadata_by_invocation_ids(
                scheduler_metadata,
                kept_invocation_ids,
            )
    return ResNet50TraceSources(
        dynamic_trace=dynamic_trace,
        threadblocks=_load_threadblocks_from_scheduler(
            root,
            enhanced_execution_info,
            scheduler_metadata=scheduler_metadata,
            representative_sm_only=True,
        ),
        enhanced_execution_info=enhanced_execution_info,
        scheduler_metadata=scheduler_metadata,
        stats_rows=stats_rows,
        gate0_manifest=gate0_manifest,
    )


def build_resnet50_trace_adapter_bundle(
    root: Path,
    *,
    invocation_limit: int | None = None,
    invocation_ids: list[str] | None = None,
) -> dict[str, Any]:
    sources = load_resnet50_trace_sources(
        root,
        invocation_limit=invocation_limit,
        invocation_ids=invocation_ids,
    )
    input_scope = (
        "bounded_resnet50_invocation_slice"
        if invocation_limit is not None or invocation_ids is not None
        else "full_resnet50_inference_trace"
    )
    if sources.scheduler_metadata.get("scheduler_metadata_source") != "real_nvbit_smid":
        raise ValueError("scheduler_metadata_source must be real_nvbit_smid")
    kernel_invocation_table = _kernel_invocation_table(
        sources.dynamic_trace,
        prefer_source_invocation_id=True,
    )
    static_instruction_table = _static_instruction_table(sources.enhanced_execution_info)
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
        "artifact_status": "formal",
        "formal_input_eligible": True,
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": input_scope,
        "scheduler_metadata_source": "real_nvbit_smid",
        "source_gate0_manifest_hash": sources.gate0_manifest["gate0_manifest_hash"],
        "source_artifact_hashes": {
            "dynamic_trace.pb": sources.gate0_manifest["source_artifact_hashes"]["dynamic_trace.pb"],
            "threadblocks/": sources.gate0_manifest["source_artifact_hashes"]["threadblocks/"],
            "enhanced_execution_info.json": sources.gate0_manifest["source_artifact_hashes"][
                "enhanced_execution_info.json"
            ],
            "scheduler_metadata.json": sources.gate0_manifest["source_artifact_hashes"][
                "scheduler_metadata.json"
            ],
        },
        "kernel_invocation_table": kernel_invocation_table,
        "static_instruction_table": static_instruction_table,
        "cta_scheduler_records": cta_scheduler_records,
        "per_warp_trace_records": per_warp_trace_records,
        "adapter_validation_report": {
            "status": "passed",
            "scheduler_metadata_complete": True,
            "trace_materialization_scope": "representative_sm_all_ctas",
            "trace_count_reconciliation_policy": "scheduler_count_is_runtime_packet_count",
            **(
                {"formal_replay_invocation_limit": invocation_limit}
                if invocation_limit is not None
                else {}
            ),
            **(
                {"formal_replay_invocation_ids": invocation_ids}
                if invocation_ids is not None
                else {}
            ),
            "errors": [],
        },
    }
    bundle["adapter_bundle_hash"] = hash_without(bundle, "adapter_bundle_hash")
    validate_resnet50_trace_adapter_bundle(bundle)
    return bundle


def build_resnet50_debug_trace_adapter_bundle(root: Path) -> dict[str, Any]:
    stats_path = root / "stats.csv"
    with stats_path.open(newline="", encoding="utf-8") as handle:
        stats_rows = list(csv.DictReader(handle))
    sources = ResNet50TraceSources(
        dynamic_trace=read_json(root / "dynamic_trace.json"),
        threadblocks=read_json(root / "threadblocks.json"),
        enhanced_execution_info=read_json(root / "enhanced_execution_info.json"),
        scheduler_metadata=read_json(root / "scheduler_metadata.json"),
        stats_rows=stats_rows,
        gate0_manifest={},
    )
    kernel_invocation_table = _kernel_invocation_table(sources.dynamic_trace)
    invocation_lookup = _invocation_lookup(kernel_invocation_table)
    bundle = {
        "artifact_type": ADAPTER_ARTIFACT_TYPE,
        "artifact_version": ADAPTER_VERSION,
        "artifact_status": "debug_not_formal",
        "formal_input_eligible": False,
        "workload_id": "resnet50",
        "execution_mode": "debug_fixture",
        "trace_source": "fixture",
        "input_scope": "debug_resnet_like_fixture",
        "scheduler_metadata_source": sources.scheduler_metadata.get(
            "scheduler_metadata_source", "debug_fixture"
        ),
        "source_gate0_manifest_hash": None,
        "source_artifact_hashes": {
            "dynamic_trace.json": hash_without(sources.dynamic_trace, "_hash"),
            "threadblocks.json": hash_without(sources.threadblocks, "_hash"),
            "enhanced_execution_info.json": hash_without(sources.enhanced_execution_info, "_hash"),
            "scheduler_metadata.json": hash_without(sources.scheduler_metadata, "_hash"),
        },
        "kernel_invocation_table": kernel_invocation_table,
        "static_instruction_table": list(sources.enhanced_execution_info.get("instructions", [])),
        "cta_scheduler_records": _cta_scheduler_records(
            sources.scheduler_metadata, invocation_lookup
        ),
        "per_warp_trace_records": _per_warp_trace_records(
            sources.threadblocks, invocation_lookup
        ),
        "adapter_validation_report": {
            "status": "debug_not_formal",
            "scheduler_metadata_complete": True,
            "errors": [],
        },
    }
    bundle["adapter_bundle_hash"] = hash_without(bundle, "adapter_bundle_hash")
    _validate_invocation_provenance(bundle)
    return bundle


def build_resnet50_artifact_shape_trace_adapter_bundle(root: Path) -> dict[str, Any]:
    """Read protobuf artifact-shape fixtures without treating them as formal input."""

    root = Path(root)
    stats_path = root / "stats.csv"
    with stats_path.open(newline="", encoding="utf-8") as handle:
        stats_rows = list(csv.DictReader(handle))
    enhanced_execution_info = read_json(root / "enhanced_execution_info.json")
    scheduler_metadata = read_json(root / "scheduler_metadata.json")
    sources = ResNet50TraceSources(
        dynamic_trace=_load_dynamic_trace_pb(root / "dynamic_trace.pb"),
        threadblocks=_load_threadblocks_from_scheduler(
            root,
            enhanced_execution_info,
            scheduler_metadata=scheduler_metadata,
            representative_sm_only=False,
        ),
        enhanced_execution_info=enhanced_execution_info,
        scheduler_metadata=scheduler_metadata,
        stats_rows=stats_rows,
        gate0_manifest={},
    )
    kernel_invocation_table = _kernel_invocation_table(sources.dynamic_trace)
    invocation_lookup = _invocation_lookup(kernel_invocation_table)
    bundle = {
        "artifact_type": ADAPTER_ARTIFACT_TYPE,
        "artifact_version": ADAPTER_VERSION,
        "artifact_status": "debug_not_formal",
        "formal_input_eligible": False,
        "workload_id": "resnet50",
        "execution_mode": "artifact_shape_unit_test",
        "trace_source": "synthetic_protobuf_artifact_shape",
        "input_scope": "debug_resnet50_artifact_shape_fixture",
        "scheduler_metadata_source": sources.scheduler_metadata.get(
            "scheduler_metadata_source", "debug_fixture"
        ),
        "source_gate0_manifest_hash": None,
        "source_artifact_hashes": {
            "dynamic_trace.pb": hash_without(sources.dynamic_trace, "_hash"),
            "threadblocks/": hash_without(sources.threadblocks, "_hash"),
            "enhanced_execution_info.json": hash_without(sources.enhanced_execution_info, "_hash"),
            "scheduler_metadata.json": hash_without(sources.scheduler_metadata, "_hash"),
        },
        "kernel_invocation_table": kernel_invocation_table,
        "static_instruction_table": list(sources.enhanced_execution_info.get("instructions", [])),
        "cta_scheduler_records": _cta_scheduler_records(
            sources.scheduler_metadata, invocation_lookup
        ),
        "per_warp_trace_records": _per_warp_trace_records(
            sources.threadblocks, invocation_lookup
        ),
        "adapter_validation_report": {
            "status": "debug_not_formal",
            "scheduler_metadata_complete": True,
            "errors": [],
        },
    }
    bundle["adapter_bundle_hash"] = hash_without(bundle, "adapter_bundle_hash")
    _validate_invocation_provenance(bundle)
    return bundle


def _kernel_invocation_table(
    dynamic_trace: dict[str, Any],
    *,
    prefer_source_invocation_id: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for fallback_launch_order, row in enumerate(dynamic_trace.get("kernel_invocations", [])):
        launch_order = int(row.get("launch_order", fallback_launch_order))
        rows.append(
            {
                "kernel_invocation_id": row.get("source_kernel_invocation_id")
                if prefer_source_invocation_id and row.get("source_kernel_invocation_id")
                else _launch_order_invocation_id(launch_order),
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


def _load_dynamic_trace_pb(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("dynamic_trace.pb is required for formal Gate1")
    trace = trace_pb2.Trace()
    trace.ParseFromString(path.read_bytes())
    _reject_unordered_multi_stream_trace(trace)
    invocations = []
    launch_order = 0
    for device_id, device in sorted(trace.gpu_device.items()):
        for stream_id, stream in sorted(device.streams.items()):
            for kernel in stream.kernels:
                invocations.append(
                    {
                        "kernel_id": int(kernel.id),
                        "source_kernel_invocation_id": _launch_order_invocation_id(
                            launch_order
                        ),
                        "kernel_name": kernel.name,
                        "function_unique_id": int(kernel.function_unique_id),
                        "device_id": int(device_id),
                        "stream_id": int(stream_id),
                        "launch_order": launch_order,
                        "grid_dim": [
                            int(kernel.grid_dim.x),
                            int(kernel.grid_dim.y),
                            int(kernel.grid_dim.z),
                        ],
                        "block_dim": [
                            int(kernel.block_dim.x),
                            int(kernel.block_dim.y),
                            int(kernel.block_dim.z),
                        ],
                        "shared_memory_size": int(kernel.size_shared_memory),
                        "register_count": int(kernel.number_of_registers),
                    }
                )
                launch_order += 1
    if not invocations:
        raise ValueError("dynamic_trace.pb contains no kernel invocations")
    return {
        "artifact_type": "resnet50_dynamic_trace_pb",
        "kernel_invocations": invocations,
    }


def _reject_unordered_multi_stream_trace(trace) -> None:
    active_streams = []
    for device_id, device in trace.gpu_device.items():
        for stream_id, stream in device.streams.items():
            if stream.kernels:
                active_streams.append((int(device_id), int(stream_id)))
    if len(active_streams) > 1:
        raise ValueError(
            "multi-stream dynamic_trace.pb lacks global launch order; "
            "Gate1 requires scheduler-aligned invocation IDs before parsing protobuf kernels"
        )


def _load_threadblocks_from_scheduler(
    root: Path,
    enhanced_execution_info: dict[str, Any],
    *,
    scheduler_metadata: dict[str, Any] | None = None,
    representative_sm_only: bool,
) -> dict[str, Any]:
    if scheduler_metadata is None:
        scheduler_metadata = read_json(root / "scheduler_metadata.json")
    static_instruction_index = _static_instruction_index(enhanced_execution_info)
    selected_sm_by_invocation = (
        _selected_sm_by_invocation(scheduler_metadata) if representative_sm_only else {}
    )
    scheduler_invocations = scheduler_metadata.get("kernel_invocations", [])
    use_legacy_scheduler_order = (
        len(scheduler_invocations) > 1
        and all(
            "launch_order" not in invocation and not invocation.get("kernel_invocation_id")
            for invocation in scheduler_invocations
        )
    )
    records = []
    for scheduler_order, invocation in enumerate(scheduler_invocations):
        selected_sm = selected_sm_by_invocation.get(_scheduler_invocation_id(invocation))
        legacy_scheduler_order = scheduler_order if use_legacy_scheduler_order else None
        for cta in invocation.get("cta_records", []):
            if selected_sm is not None and int(cta["sm_id"]) != selected_sm:
                continue
            relative = _threadblock_relative_path(invocation, cta)
            pb_path = root / "threadblocks" / relative
            if not pb_path.is_file():
                raise FileNotFoundError(f"missing threadblock protobuf: {relative}")
            records.extend(
                _threadblock_records_from_pb(
                    pb_path=pb_path,
                    kernel_id=int(invocation["kernel_id"]),
                    kernel_invocation_id=_scheduler_canonical_invocation_id(invocation),
                    launch_order=invocation.get("launch_order"),
                    legacy_scheduler_order=legacy_scheduler_order,
                    cta_id=str(cta["cta_id"]),
                    static_instruction_index=static_instruction_index,
                )
            )
    if not records:
        raise ValueError("threadblocks/ contains no scheduler-referenced records")
    return {
        "artifact_type": "resnet50_threadblocks_pb",
        "threadblocks": records,
    }


def _limit_dynamic_trace_invocations(
    dynamic_trace: dict[str, Any],
    invocation_limit: int,
) -> dict[str, Any]:
    limited = dict(dynamic_trace)
    limited["kernel_invocations"] = list(
        dynamic_trace.get("kernel_invocations", [])[:invocation_limit]
    )
    if not limited["kernel_invocations"]:
        raise ValueError("invocation_limit selected no kernel invocations")
    return limited


def _filter_dynamic_trace_by_invocation_ids(
    dynamic_trace: dict[str, Any],
    invocation_ids: set[str],
) -> dict[str, Any]:
    if not invocation_ids:
        raise ValueError("invocation_ids must be non-empty")
    resolved_invocation_ids = _resolve_requested_dynamic_invocation_ids(
        dynamic_trace,
        invocation_ids,
    )
    filtered = dict(dynamic_trace)
    filtered["kernel_invocations"] = [
        invocation
        for invocation in dynamic_trace.get("kernel_invocations", [])
        if invocation.get("source_kernel_invocation_id") in resolved_invocation_ids
    ]
    found = {
        str(invocation.get("source_kernel_invocation_id"))
        for invocation in filtered["kernel_invocations"]
        if invocation.get("source_kernel_invocation_id")
    }
    missing = sorted(resolved_invocation_ids.difference(found))
    if missing:
        raise ValueError(f"invocation_ids not found in dynamic trace: {missing}")
    return filtered


def _resolve_requested_dynamic_invocation_ids(
    dynamic_trace: dict[str, Any],
    invocation_ids: set[str],
) -> set[str]:
    canonical_ids = {
        str(invocation.get("source_kernel_invocation_id"))
        for invocation in dynamic_trace.get("kernel_invocations", [])
        if invocation.get("source_kernel_invocation_id")
    }
    legacy_aliases: dict[str, list[str]] = {}
    for invocation in dynamic_trace.get("kernel_invocations", []):
        source_id = invocation.get("source_kernel_invocation_id")
        if not source_id:
            continue
        legacy_aliases.setdefault(_legacy_scheduler_invocation_id(invocation), []).append(
            str(source_id)
        )

    resolved = set()
    missing = []
    for requested_id in sorted(invocation_ids):
        if requested_id in canonical_ids:
            resolved.add(requested_id)
            continue
        legacy_matches = legacy_aliases.get(requested_id, [])
        if legacy_matches:
            if len(legacy_matches) > 1:
                raise ValueError(
                    "ambiguous legacy invocation_id "
                    f"{requested_id}; use canonical invocation ids: "
                    f"{sorted(legacy_matches)}"
                )
            resolved.add(legacy_matches[0])
            continue
        missing.append(requested_id)
    if missing:
        raise ValueError(f"invocation_ids not found in dynamic trace: {missing}")
    return resolved


def _filter_scheduler_metadata_by_invocation_ids(
    scheduler_metadata: dict[str, Any],
    kept_invocation_ids: set[str],
) -> dict[str, Any]:
    if not kept_invocation_ids:
        raise ValueError("invocation_limit selected no scheduler invocation ids")
    filtered = dict(scheduler_metadata)
    filtered["kernel_invocations"] = [
        invocation
        for invocation in scheduler_metadata.get("kernel_invocations", [])
        if _scheduler_invocation_id(invocation) in kept_invocation_ids
        or _scheduler_canonical_invocation_id(invocation) in kept_invocation_ids
    ]
    if not filtered["kernel_invocations"]:
        raise ValueError("invocation_limit selected no scheduler metadata")
    return filtered


def _scheduler_metadata_has_only_legacy_ids(scheduler_metadata: dict[str, Any]) -> bool:
    invocations = scheduler_metadata.get("kernel_invocations", [])
    return bool(invocations) and all(
        not invocation.get("kernel_invocation_id") and "launch_order" not in invocation
        for invocation in invocations
    )


def _limit_scheduler_metadata_invocations(
    scheduler_metadata: dict[str, Any],
    invocation_count: int,
) -> dict[str, Any]:
    selected_invocations = list(
        scheduler_metadata.get("kernel_invocations", [])[:invocation_count]
    )
    for index, invocation in enumerate(selected_invocations):
        _validate_legacy_scheduler_position_identity(invocation, index)
    filtered = dict(scheduler_metadata)
    filtered["kernel_invocations"] = selected_invocations
    if not filtered["kernel_invocations"]:
        raise ValueError("invocation_limit selected no scheduler metadata")
    return filtered


def _filter_legacy_scheduler_metadata_by_launch_orders(
    scheduler_metadata: dict[str, Any],
    launch_orders: set[int],
) -> dict[str, Any]:
    if not launch_orders:
        raise ValueError("invocation_ids selected no scheduler launch orders")
    filtered = dict(scheduler_metadata)
    scheduler_invocations = scheduler_metadata.get("kernel_invocations", [])
    filtered["kernel_invocations"] = [
        invocation
        for index, invocation in enumerate(scheduler_invocations)
        if _legacy_scheduler_position_matches_launch_order(invocation, index, launch_orders)
    ]
    if not filtered["kernel_invocations"]:
        raise ValueError("invocation_ids selected no scheduler metadata")
    return filtered


def _legacy_scheduler_position_matches_launch_order(
    invocation: dict[str, Any],
    index: int,
    launch_orders: set[int],
) -> bool:
    if index not in launch_orders:
        return False
    _validate_legacy_scheduler_position_identity(invocation, index)
    return True


def _validate_legacy_scheduler_position_identity(invocation: dict[str, Any], index: int) -> None:
    observed_launch_order = _infer_scheduler_threadblock_launch_order(invocation)
    if observed_launch_order is None:
        return
    if observed_launch_order != index:
        raise ValueError("legacy scheduler metadata lacks stable invocation identity")


def _infer_scheduler_threadblock_launch_order(invocation: dict[str, Any]) -> int | None:
    launch_orders = set()
    for cta in invocation.get("cta_records", []):
        relative = cta.get("threadblock_pb")
        if not relative:
            continue
        parts = Path(str(relative)).parts
        for part in parts:
            if part.startswith("kernel_"):
                suffix = part.removeprefix("kernel_")
                if suffix.isdigit():
                    launch_orders.add(int(suffix))
                break
    if len(launch_orders) > 1:
        raise ValueError("legacy scheduler metadata has inconsistent threadblock launch identity")
    return next(iter(launch_orders)) if launch_orders else None


def _launch_order_invocation_id(launch_order: int) -> str:
    return f"resnet50_k{int(launch_order):05d}"


def _legacy_scheduler_invocation_id(invocation: dict[str, Any]) -> str:
    return (
        f"d_{int(invocation.get('device_id', 0))}_"
        f"s_{int(invocation.get('stream_id', 0))}_"
        f"k_{int(invocation['kernel_id'])}"
    )


def _legacy_scheduler_invocation_ids(invocations: list[dict[str, Any]]) -> set[str]:
    return {_legacy_scheduler_invocation_id(invocation) for invocation in invocations}


def _enhanced_execution_info_path(root: Path) -> Path:
    root_level = root / "enhanced_execution_info.json"
    if root_level.is_file():
        return root_level
    nested = root / "extra_info" / "enhanced_execution_info.json"
    if nested.is_file():
        return nested
    raise FileNotFoundError("enhanced_execution_info.json is required for formal Gate1")


def _static_instruction_table(enhanced_execution_info: dict[str, Any]) -> list[dict[str, Any]]:
    if enhanced_execution_info.get("instructions"):
        return list(enhanced_execution_info["instructions"])
    rows = []
    for kernel in enhanced_execution_info.get("kernels", []):
        function_unique_id = int(kernel["unique_function_id"])
        for instruction in kernel.get("instructions", []):
            rows.append(_normalize_static_instruction(function_unique_id, instruction))
    return rows


def _static_instruction_index(enhanced_execution_info: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    index = {}
    for row in _static_instruction_table(enhanced_execution_info):
        index[(int(row["function_unique_id"]), int(row["pc"]))] = row
    if not index:
        raise ValueError("static instruction metadata must be non-empty")
    return index


def _normalize_static_instruction(
    function_unique_id: int,
    instruction: dict[str, Any],
) -> dict[str, Any]:
    operands = [
        operand.get("operand_string", str(operand))
        for operand in instruction.get("operands", [])
    ]
    return {
        "function_unique_id": function_unique_id,
        "pc": int(instruction["pc_num_dec"]),
        "opcode": instruction["op_code"],
        "operands": operands,
        "control_bits": instruction.get("control_bits", {}),
    }


def _scheduler_invocation_id(invocation: dict[str, Any]) -> str:
    explicit = invocation.get("kernel_invocation_id")
    if explicit:
        return str(explicit)
    if "launch_order" in invocation:
        return _launch_order_invocation_id(int(invocation["launch_order"]))
    return _legacy_scheduler_invocation_id(invocation)


def _scheduler_canonical_invocation_id(invocation: dict[str, Any]) -> str:
    if "launch_order" in invocation:
        return _launch_order_invocation_id(int(invocation["launch_order"]))
    return _scheduler_invocation_id(invocation)


def _threadblock_relative_path(invocation: dict[str, Any], cta: dict[str, Any]) -> str:
    if cta.get("threadblock_pb"):
        return str(cta["threadblock_pb"])
    device_id = int(invocation.get("device_id", 0))
    stream_id = int(invocation.get("stream_id", 0))
    kernel_directory_id = int(invocation.get("launch_order", invocation["kernel_id"]))
    cta_id = str(cta["cta_id"])
    return (
        f"device_{device_id}/stream_{stream_id}/kernel_{kernel_directory_id}/"
        f"d_{device_id}_s_{stream_id}_k_{kernel_directory_id}_{cta_id}.pb"
    )


def _selected_sm_by_invocation(scheduler_metadata: dict[str, Any]) -> dict[str, int]:
    selected = {}
    for invocation in scheduler_metadata.get("kernel_invocations", []):
        scheduler_by_sm: dict[str, dict[str, Any]] = {}
        for cta in invocation.get("cta_records", []):
            sm_id = str(cta["sm_id"])
            cta_id = str(cta["cta_id"])
            metadata = scheduler_by_sm.setdefault(
                sm_id,
                {
                    "sm_id": int(cta["sm_id"]),
                    "cta_ids": [],
                    "warp_ids_by_cta": {},
                    "trace_entry_count_by_cta": {},
                    "cta_start_order": {},
                    "cta_end_order": {},
                },
            )
            metadata["cta_ids"].append(cta_id)
            metadata["warp_ids_by_cta"][cta_id] = list(cta["warp_ids"])
            metadata["trace_entry_count_by_cta"][cta_id] = int(cta["trace_entry_count"])
            metadata["cta_start_order"][cta_id] = int(cta["first_seen_order"])
            metadata["cta_end_order"][cta_id] = int(cta["last_seen_order"])
        selection_input = {
            "kernel_invocation_id": _scheduler_invocation_id(invocation),
            "scheduler_metadata_by_sm": scheduler_by_sm,
        }
        selected[_scheduler_invocation_id(invocation)] = int(
            select_representative_sm(selection_input)["selected_sm"]
        )
    return selected


def _threadblock_records_from_pb(
    *,
    pb_path: Path,
    kernel_id: int,
    kernel_invocation_id: str,
    launch_order: int | None,
    legacy_scheduler_order: int | None,
    cta_id: str,
    static_instruction_index: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    block = threadblock_pb2.threadblock()
    block.ParseFromString(pb_path.read_bytes())
    block_cta_id = f"{int(block.block_id.x)},{int(block.block_id.y)},{int(block.block_id.z)}"
    if block_cta_id != cta_id:
        raise ValueError("threadblock protobuf CTA id does not match scheduler metadata")
    records = []
    for warp_id, warp in sorted(block.warps.items()):
        entries = []
        for trace_index, instruction in enumerate(warp.instructions):
            static = static_instruction_index.get(
                (int(instruction.function_unique_id), int(instruction.pc))
            )
            if static is None:
                raise ValueError("threadblock instruction missing static metadata")
            entries.append(
                _trace_entry_from_pb_instruction(
                    instruction=instruction,
                    static=static,
                    trace_index=trace_index,
                    source_entry_hash=hash_without(
                        {
                            "threadblock_pb": str(pb_path),
                            "warp_id": int(warp_id),
                            "trace_index": trace_index,
                            "pc": int(instruction.pc),
                            "function_unique_id": int(instruction.function_unique_id),
                        }
                    ),
                )
            )
        records.append(
            {
                "kernel_invocation_id": kernel_invocation_id,
                "kernel_id": kernel_id,
                **({"launch_order": int(launch_order)} if launch_order is not None else {}),
                **(
                    {"legacy_scheduler_order": int(legacy_scheduler_order)}
                    if legacy_scheduler_order is not None
                    else {}
                ),
                "cta_id": cta_id,
                "warp_id": int(warp_id),
                "entries": entries,
            }
        )
    return records


def _trace_entry_from_pb_instruction(
    *,
    instruction: Any,
    static: dict[str, Any],
    trace_index: int,
    source_entry_hash: str,
) -> dict[str, Any]:
    opcode = static["opcode"]
    operands = list(static.get("operands", []))
    destination_operands, source_operands = _split_operands(opcode, operands)
    memory_metadata = {}
    if instruction.addresses:
        memory_metadata = {
            "address_count": len(instruction.addresses),
            "space": "global" if opcode.startswith(("LDG", "STG")) else "unknown",
        }
    return {
        "trace_index": trace_index,
        "pc": int(instruction.pc),
        "opcode": opcode,
        "active_mask": _mask_string(instruction.active_mask),
        "predicate_mask": _mask_string(instruction.predicate_mask),
        "destination_operands": destination_operands,
        "source_operands": source_operands,
        "memory_address_metadata": memory_metadata,
        "observed_dynamic_values": [],
        "source_entry_hash": source_entry_hash,
    }


STORE_OPCODE_PREFIXES = ("STG", "STS", "STL")
TWO_DESTINATION_OPCODE_PREFIXES = ("LEA", "ISETP", "PSETP")
ZERO_DESTINATION_OPCODE_PREFIXES = (
    "BAR",
    "BRA",
    "BSSY",
    "BSYNC",
    "CALL",
    "DEPBAR",
    "EXIT",
    "JMP",
    "MEMBAR",
    "NOP",
    "RET",
    "WARPSYNC",
    "YIELD",
)


def _split_operands(opcode: str, operands: list[str]) -> tuple[list[str], list[str]]:
    if not operands:
        return [], []
    if opcode.startswith(STORE_OPCODE_PREFIXES):
        return [], operands
    if opcode.startswith(ZERO_DESTINATION_OPCODE_PREFIXES):
        return [], operands
    if opcode.startswith(TWO_DESTINATION_OPCODE_PREFIXES) and len(operands) >= 2:
        return operands[:2], operands[2:]
    return [operands[0]], operands[1:]


def _mask_string(value: int) -> str:
    return f"0x{int(value):08x}"


def _invocation_lookup(kernel_invocation_table: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["kernel_invocation_id"]: row for row in kernel_invocation_table}
    by_launch_order = {int(row["launch_order"]): row for row in kernel_invocation_table}
    by_kernel_id: dict[int, list[dict[str, Any]]] = {}
    legacy_candidates: dict[str, list[dict[str, Any]]] = {}
    for row in kernel_invocation_table:
        by_kernel_id.setdefault(int(row["kernel_id"]), []).append(row)
        legacy_candidates.setdefault(_legacy_scheduler_invocation_id(row), []).append(row)
    by_legacy_id = {
        legacy_id: rows[0] for legacy_id, rows in legacy_candidates.items() if len(rows) == 1
    }
    return {
        "by_id": by_id,
        "by_launch_order": by_launch_order,
        "by_kernel_id": by_kernel_id,
        "by_legacy_id": by_legacy_id,
        "legacy_candidates": legacy_candidates,
        "legacy_kernel_offsets": {},
    }


def _legacy_resolution_lookup(lookup: dict[str, Any]) -> dict[str, Any]:
    copied = dict(lookup)
    copied["legacy_kernel_offsets"] = {}
    return copied


def _resolve_kernel_invocation_id(record: dict[str, Any], lookup: dict[str, Any]) -> str:
    explicit = record.get("kernel_invocation_id")
    if explicit is not None:
        if explicit not in lookup["by_id"]:
            if "legacy_scheduler_order" in record:
                return _resolve_kernel_invocation_id_by_legacy_scheduler_order(
                    record,
                    lookup,
                )
            if "launch_order" in record:
                return _resolve_kernel_invocation_id_by_launch_order(record, lookup)
            if explicit in lookup["by_legacy_id"]:
                row = lookup["by_legacy_id"][explicit]
                _validate_record_identity_consistency(record, row)
                return row["kernel_invocation_id"]
            legacy_candidates = lookup["legacy_candidates"].get(str(explicit), [])
            if legacy_candidates:
                row = _consume_legacy_kernel_candidate(
                    int(record["kernel_id"]),
                    legacy_candidates,
                    lookup,
                )
                _validate_record_identity_consistency(record, row)
                return row["kernel_invocation_id"]
            raise ValueError("raw record references unknown kernel_invocation_id")
        _validate_record_identity_consistency(record, lookup["by_id"][explicit])
        return explicit
    if "launch_order" in record:
        return _resolve_kernel_invocation_id_by_launch_order(record, lookup)
    if "legacy_scheduler_order" in record:
        return _resolve_kernel_invocation_id_by_legacy_scheduler_order(record, lookup)
    kernel_id = int(record["kernel_id"])
    candidates = lookup["by_kernel_id"].get(kernel_id, [])
    if len(candidates) == 1:
        return candidates[0]["kernel_invocation_id"]
    if not candidates:
        raise ValueError(
            "raw record for repeated kernel_id requires kernel_invocation_id or launch_order"
        )
    return _consume_legacy_kernel_candidate(kernel_id, candidates, lookup)[
        "kernel_invocation_id"
    ]


def _consume_legacy_kernel_candidate(
    kernel_id: int,
    candidates: list[dict[str, Any]],
    lookup: dict[str, Any],
) -> dict[str, Any]:
    offset = lookup["legacy_kernel_offsets"].get(kernel_id, 0)
    if offset < len(candidates):
        lookup["legacy_kernel_offsets"][kernel_id] = offset + 1
        return candidates[offset]
    raise ValueError("legacy scheduler repeated kernel_id has more records than dynamic trace")


def _resolve_kernel_invocation_id_by_launch_order(
    record: dict[str, Any],
    lookup: dict[str, Any],
) -> str:
    launch_order = int(record["launch_order"])
    if launch_order not in lookup["by_launch_order"]:
        raise ValueError("raw record references unknown launch_order")
    row = lookup["by_launch_order"][launch_order]
    _validate_record_identity_consistency(record, row)
    return row["kernel_invocation_id"]


def _resolve_kernel_invocation_id_by_legacy_scheduler_order(
    record: dict[str, Any],
    lookup: dict[str, Any],
) -> str:
    scheduler_order = int(record["legacy_scheduler_order"])
    if scheduler_order not in lookup["by_launch_order"]:
        raise ValueError("legacy scheduler record order exceeds dynamic trace")
    row = lookup["by_launch_order"][scheduler_order]
    _validate_record_identity_consistency(record, row)
    return row["kernel_invocation_id"]


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
    lookup = _legacy_resolution_lookup(invocation_lookup)
    for invocation in scheduler_metadata.get("kernel_invocations", []):
        kernel_id = invocation["kernel_id"]
        kernel_invocation_id = _resolve_kernel_invocation_id(invocation, lookup)
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
    lookup = _legacy_resolution_lookup(invocation_lookup)
    for record in threadblocks.get("threadblocks", []):
        kernel_invocation_id = _resolve_kernel_invocation_id(record, lookup)
        records.append(
            {
                **record,
                "kernel_invocation_id": kernel_invocation_id,
            }
        )
    return records


def validate_resnet50_trace_adapter_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("artifact_type") != ADAPTER_ARTIFACT_TYPE:
        raise ValueError("unexpected adapter artifact_type")
    if bundle.get("artifact_version") != ADAPTER_VERSION:
        raise ValueError("unexpected adapter artifact_version")
    if bundle.get("artifact_status") != "formal":
        raise ValueError("adapter artifact_status must be formal")
    if bundle.get("formal_input_eligible") is not True:
        raise ValueError("adapter must be formal input eligible")
    if bundle.get("workload_id") != "resnet50":
        raise ValueError("workload_id must be resnet50")
    if bundle.get("execution_mode") != "real_trace":
        raise ValueError("execution_mode must be real_trace")
    if bundle.get("trace_source") != "nvbit":
        raise ValueError("trace_source must be nvbit")
    if bundle.get("input_scope") not in {
        "full_resnet50_inference_trace",
        "bounded_resnet50_invocation_slice",
    }:
        raise ValueError("input_scope must be full_resnet50_inference_trace or bounded_resnet50_invocation_slice")
    if bundle.get("scheduler_metadata_source") != "real_nvbit_smid":
        raise ValueError("scheduler_metadata_source must be real_nvbit_smid")
    if not bundle.get("source_gate0_manifest_hash"):
        raise ValueError("source_gate0_manifest_hash is required")
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
        raise ValueError("static instruction metadata must be non-empty")
    if not bundle.get("cta_scheduler_records"):
        raise ValueError("cta_scheduler_records must be non-empty")
    if not bundle.get("per_warp_trace_records"):
        raise ValueError("per_warp_trace_records must be non-empty")
    _validate_invocation_provenance(bundle)
    if bundle.get("adapter_bundle_hash") != hash_without(bundle, "adapter_bundle_hash"):
        raise ValueError("adapter_bundle_hash is not reproducible")


def _validate_invocation_provenance(bundle: dict[str, Any]) -> None:
    report = bundle.get("adapter_validation_report", {})
    representative_trace_only = (
        report.get("trace_materialization_scope") == "representative_sm_all_ctas"
        and report.get("trace_count_reconciliation_policy")
        == "scheduler_count_is_runtime_packet_count"
    )
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
    expected_scheduler_by_cta = scheduler_by_cta
    if representative_trace_only:
        expected_scheduler_by_cta = _selected_sm_scheduler_ctas(scheduler_by_cta)
    if set(expected_scheduler_by_cta) != set(trace_by_cta):
        raise ValueError("scheduler CTA set must match warp trace CTA set")
    for cta_key, trace_record in trace_by_cta.items():
        scheduler_record = expected_scheduler_by_cta[cta_key]
        if set(scheduler_record["warp_ids"]) != trace_record["warp_ids"]:
            raise ValueError("scheduler warp_ids must match traced warp IDs")
        if representative_trace_only:
            if trace_record["entry_count"] <= 0:
                raise ValueError("materialized warp trace entry count must be positive")
        elif int(scheduler_record["trace_entry_count"]) != trace_record["entry_count"]:
            raise ValueError("scheduler trace_entry_count must match traced entry count")
    for invocation_id in invocation_ids:
        if scheduled_by_invocation[invocation_id] <= 0:
            raise ValueError("each kernel_invocation_id must retain scheduler records")
        if trace_records_by_invocation[invocation_id] <= 0:
            raise ValueError("each kernel_invocation_id must retain warp trace records")


def _selected_sm_scheduler_ctas(
    scheduler_by_cta: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    by_invocation: dict[str, dict[str, dict[str, Any]]] = {}
    for (invocation_id, cta_id), record in scheduler_by_cta.items():
        sm_id = str(record["sm_id"])
        invocation_sms = by_invocation.setdefault(invocation_id, {})
        metadata = invocation_sms.setdefault(
            sm_id,
            {
                "sm_id": int(record["sm_id"]),
                "cta_ids": [],
                "warp_ids_by_cta": {},
                "trace_entry_count_by_cta": {},
                "cta_start_order": {},
                "cta_end_order": {},
            },
        )
        metadata["cta_ids"].append(cta_id)
        metadata["warp_ids_by_cta"][cta_id] = list(record["warp_ids"])
        metadata["trace_entry_count_by_cta"][cta_id] = int(record["trace_entry_count"])
        metadata["cta_start_order"][cta_id] = int(record["first_seen_order"])
        metadata["cta_end_order"][cta_id] = int(record["last_seen_order"])
    selected = {}
    for invocation_id, scheduler_metadata_by_sm in by_invocation.items():
        selected_sm = int(
            select_representative_sm(
                {
                    "kernel_invocation_id": invocation_id,
                    "scheduler_metadata_by_sm": scheduler_metadata_by_sm,
                }
            )["selected_sm"]
        )
        for cta_key, record in scheduler_by_cta.items():
            if cta_key[0] == invocation_id and int(record["sm_id"]) == selected_sm:
                selected[cta_key] = record
    return selected


def write_resnet50_trace_adapter_bundle(root: Path, out_path: Path) -> dict[str, Any]:
    bundle = build_resnet50_trace_adapter_bundle(root)
    write_json(out_path, bundle)
    return bundle


def mark_resnet50_fixture_debug_not_formal(root: Path) -> dict[str, Any]:
    sources = {
        "dynamic_trace.json": root / "dynamic_trace.json",
        "threadblocks.json": root / "threadblocks.json",
        "enhanced_execution_info.json": root / "enhanced_execution_info.json",
        "scheduler_metadata.json": root / "scheduler_metadata.json",
        "stats.csv": root / "stats.csv",
    }
    report = {
        "artifact_type": "gcl_resnet50_debug_fixture_report",
        "artifact_version": "debug_fixture_report_v1",
        "artifact_status": "debug_not_formal",
        "formal_input_eligible": False,
        "workload_id": "resnet50",
        "reason": "fixture path is allowed for unit, smoke, and debug only",
        "source_artifact_hashes": {
            name: _debug_source_hash(path) for name, path in sources.items() if path.exists()
        },
    }
    report["debug_fixture_report_hash"] = hash_without(report, "debug_fixture_report_hash")
    return report


def _debug_source_hash(path: Path) -> str:
    if path.suffix == ".json":
        return hash_without(read_json(path), "_hash")
    return hash_without({"content": path.read_text(encoding="utf-8")})
