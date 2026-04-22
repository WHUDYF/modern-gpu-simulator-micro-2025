"""Build v1 frontend invocation tables from repository-local inputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _select_exec_time(hardware_metrics: dict[str, Any]) -> tuple[float | None, str]:
    if "duration_ns" in hardware_metrics:
        return float(hardware_metrics["duration_ns"]), "duration_ns"
    if "elapsed_cycles" in hardware_metrics:
        return float(hardware_metrics["elapsed_cycles"]), "elapsed_cycles"
    return None, "missing"


def _feature_vector(item: dict[str, Any]) -> dict[str, Any]:
    dynamic = item.get("dynamic_stats", {})
    compression = item.get("compression_features", {})
    hardware = item.get("hardware_metrics", {})

    def _scalar(value: Any) -> Any:
        if isinstance(value, dict):
            if "mean" in value:
                return value["mean"]
            return None
        return value

    return {
        "total_dynamic_insts": dynamic.get("total_dynamic_insts"),
        "num_blocks": dynamic.get("num_blocks"),
        "threads_per_block": dynamic.get("threads_per_block"),
        "compute_throughput_pct": hardware.get("compute_throughput_pct"),
        "dram_throughput_pct": hardware.get("dram_throughput_pct"),
        "achieved_occupancy_pct": hardware.get("achieved_occupancy_pct"),
        "ipc_active": hardware.get("ipc_active"),
        "l1_hit_rate_pct": hardware.get("l1_hit_rate_pct"),
        "l2_hit_rate_pct": hardware.get("l2_hit_rate_pct"),
        "cross_tb_offset_coverage": _scalar(compression.get("cross_tb_offset_coverage")),
        "address_override_density": _scalar(compression.get("address_override_density")),
        "full_encoding_fallback_rate": _scalar(compression.get("full_encoding_fallback_rate")),
        "shared_pc_sequence_length": _scalar(compression.get("shared_pc_sequence_length")),
    }


def _build_kernel_level_squash_map(squash_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    kernel_level = squash_data.get("kernel_level", {})
    boundary_count = kernel_level.get("boundary_count")
    for segment in kernel_level.get("squash_segments", []):
        start, end = segment["kernel_range"]
        for zero_based_idx in range(start, end + 1):
            result[zero_based_idx] = {
                "kernel_squash_segment_id": segment["segment_id"],
                "kernel_squash_boundary_count": boundary_count,
                "kernel_squash_cohesion": segment.get("cohesion_score"),
                "kernel_squash_behavior_summary": segment.get("behavior_summary"),
            }
    return result


def _build_tb_level_squash_map(squash_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for kernel_id, payload in squash_data.get("tb_level", {}).items():
        segments = payload.get("squash_segments", [])
        result[int(kernel_id)] = {
            "tb_squash_segment_count": len(segments),
            "tb_squash_boundary_count": payload.get("boundary_count"),
        }
    return result


def _normalize_identity_source(identity_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    invocations = identity_data.get("invocations")
    if isinstance(invocations, list) and invocations:
        normalized: dict[int, dict[str, Any]] = {}
        for item in invocations:
            kernel_id = item.get("kernel_id")
            if kernel_id is None:
                raise ValueError("identity_json invocation missing kernel_id")
            normalized[int(kernel_id)] = {
                "source_invocation_key": item.get("source_invocation_key", f"kernel_{kernel_id}"),
                "kernel_name": item["kernel_name"],
                "kernel_id": int(kernel_id),
                "grid_dim": item.get("grid_dim"),
                "block_dim": item.get("block_dim"),
                "shape_hint": item.get("shape_hint"),
                "trace_order": item.get("trace_order"),
            }
        return normalized

    per_kernel = identity_data.get("per_kernel", {})
    if not isinstance(per_kernel, dict) or not per_kernel:
        raise ValueError("identity_json does not contain a non-empty per_kernel mapping or invocations list")
    normalized: dict[int, dict[str, Any]] = {}
    for source_invocation_key, item in per_kernel.items():
        kernel_id = item.get("kernel_id")
        if kernel_id is None:
            raise ValueError(f"identity_json entry {source_invocation_key} missing kernel_id")
        normalized[int(kernel_id)] = {
            "source_invocation_key": source_invocation_key,
            "kernel_name": item["kernel_name"],
            "kernel_id": int(kernel_id),
            "grid_dim": item.get("dynamic_stats", {}).get("grid_dim"),
            "block_dim": item.get("dynamic_stats", {}).get("block_dim"),
            "shape_hint": None,
            "trace_order": None,
        }
    return normalized


def _normalize_feature_source(features_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    feature_records = features_data.get("feature_records")
    if isinstance(feature_records, list) and feature_records:
        normalized: dict[int, dict[str, Any]] = {}
        for item in feature_records:
            kernel_id = item.get("kernel_id")
            if kernel_id is None:
                raise ValueError("features_json feature record missing kernel_id")
            normalized[int(kernel_id)] = {
                "source_invocation_key": item.get("source_invocation_key", f"kernel_{kernel_id}"),
                "kernel_name": item["kernel_name"],
                "kernel_id": int(kernel_id),
                "dynamic_inst_count": item.get("dynamic_inst_count"),
                "exec_time": item.get("exec_time"),
                "exec_time_source": item.get("exec_time_source", "unknown"),
                "feature_vector": item.get("feature_vector", {}),
                "feature_source_note": item.get("feature_source_note", "feature_records"),
            }
        return normalized

    per_kernel = features_data.get("per_kernel", {})
    if not isinstance(per_kernel, dict) or not per_kernel:
        raise ValueError("features_json does not contain a non-empty per_kernel mapping or feature_records list")
    normalized: dict[int, dict[str, Any]] = {}
    for source_invocation_key, item in per_kernel.items():
        kernel_id = item.get("kernel_id")
        if kernel_id is None:
            raise ValueError(f"features_json entry {source_invocation_key} missing kernel_id")
        hardware = item.get("hardware_metrics", {})
        dynamic = item.get("dynamic_stats", {})
        exec_time, exec_time_source = _select_exec_time(hardware)
        normalized[int(kernel_id)] = {
            "source_invocation_key": source_invocation_key,
            "kernel_name": item["kernel_name"],
            "kernel_id": int(kernel_id),
            "dynamic_inst_count": dynamic.get("total_dynamic_insts"),
            "exec_time": exec_time,
            "exec_time_source": exec_time_source,
            "feature_vector": _feature_vector(item),
            "feature_source_note": "explicit dual-source features_json",
        }
    return normalized


def build_records_from_dual_sources(
    identity_json_path: str | Path,
    features_json_path: str | Path,
    squash_json_path: str | Path | None = None,
) -> dict[str, Any]:
    identity_data = _load_json(identity_json_path)
    features_data = _load_json(features_json_path)
    identity_map = _normalize_identity_source(identity_data)
    feature_map = _normalize_feature_source(features_data)

    if set(identity_map.keys()) != set(feature_map.keys()):
        missing_in_features = sorted(set(identity_map.keys()) - set(feature_map.keys()))
        missing_in_identity = sorted(set(feature_map.keys()) - set(identity_map.keys()))
        raise ValueError(
            "dual-source alignment failed: "
            f"missing_in_features={missing_in_features}, "
            f"missing_in_identity={missing_in_identity}"
        )

    squash_kernel_map: dict[int, dict[str, Any]] = {}
    squash_tb_map: dict[int, dict[str, Any]] = {}
    if squash_json_path:
        squash_data = _load_json(squash_json_path)
        squash_kernel_map = _build_kernel_level_squash_map(squash_data)
        squash_tb_map = _build_tb_level_squash_map(squash_data)

    name_counts: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    ordered_identity = sorted(
        identity_map.values(),
        key=lambda item: (
            item["trace_order"] is None,
            item["trace_order"] if item["trace_order"] is not None else item["kernel_id"],
            item["kernel_id"],
        ),
    )
    for zero_based_idx, identity in enumerate(ordered_identity):
        kernel_id = identity["kernel_id"]
        features = feature_map[kernel_id]
        if identity["source_invocation_key"] != features["source_invocation_key"]:
            raise ValueError(
                "dual-source alignment failed: "
                f"kernel_id={kernel_id} has mismatched source_invocation_key "
                f"{identity['source_invocation_key']} != {features['source_invocation_key']}"
            )
        if identity["kernel_name"] != features["kernel_name"]:
            raise ValueError(
                "dual-source alignment failed: "
                f"kernel_id={kernel_id} has mismatched kernel_name "
                f"{identity['kernel_name']} != {features['kernel_name']}"
            )
        kernel_name = identity["kernel_name"]
        name_counts[kernel_name] += 1
        occurrence_index = name_counts[kernel_name]
        trace_order = identity["trace_order"] if identity["trace_order"] is not None else zero_based_idx + 1

        record = {
            "kernel_invocation_id": f"{kernel_name}#{trace_order}",
            "source_invocation_key": identity["source_invocation_key"],
            "kernel_name": kernel_name,
            "kernel_id": kernel_id,
            "trace_order": trace_order,
            "kernel_name_occurrence": occurrence_index,
            "grid_dim": identity["grid_dim"],
            "block_dim": identity["block_dim"],
            "shape_hint": identity["shape_hint"],
            "exec_time": features["exec_time"],
            "exec_time_source": features["exec_time_source"],
            "dynamic_inst_count": features["dynamic_inst_count"],
            "feature_vector": features["feature_vector"],
            "feature_source_note": features["feature_source_note"],
        }
        record.update(squash_kernel_map.get(zero_based_idx, {}))
        record.update(squash_tb_map.get(kernel_id, {}))
        records.append(record)

    return {
        "workload": identity_data.get("workload") or features_data.get("workload"),
        "hardware": identity_data.get("hardware") or features_data.get("hardware"),
        "source_mode": "explicit_dual_source",
        "identity_source": str(identity_json_path),
        "features_source": str(features_json_path),
        "records": records,
    }


def build_records_from_full_json(
    full_json_path: str | Path,
    squash_json_path: str | Path | None = None,
) -> dict[str, Any]:
    table = build_records_from_dual_sources(full_json_path, full_json_path, squash_json_path)
    table["source_mode"] = "full_json_shortcut"
    table["feature_source_note"] = "full_json shortcut path"
    return table
