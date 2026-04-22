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


def build_records_from_full_json(
    full_json_path: str | Path,
    squash_json_path: str | Path | None = None,
) -> dict[str, Any]:
    full_data = _load_json(full_json_path)
    per_kernel = full_data.get("per_kernel", {})
    if not isinstance(per_kernel, dict) or not per_kernel:
        raise ValueError("full_json does not contain a non-empty per_kernel mapping")

    squash_kernel_map: dict[int, dict[str, Any]] = {}
    squash_tb_map: dict[int, dict[str, Any]] = {}
    if squash_json_path:
        squash_data = _load_json(squash_json_path)
        squash_kernel_map = _build_kernel_level_squash_map(squash_data)
        squash_tb_map = _build_tb_level_squash_map(squash_data)

    ordered_items = sorted(
        per_kernel.items(),
        key=lambda kv: (int(kv[1].get("kernel_id", 10**9)), kv[0]),
    )

    name_counts: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    for zero_based_idx, (source_invocation_key, item) in enumerate(ordered_items):
        kernel_name = item["kernel_name"]
        name_counts[kernel_name] += 1
        occurrence_index = name_counts[kernel_name]
        trace_order = zero_based_idx + 1
        dynamic = item.get("dynamic_stats", {})
        hardware = item.get("hardware_metrics", {})
        exec_time, exec_time_source = _select_exec_time(hardware)

        record = {
            "kernel_invocation_id": f"{kernel_name}#{trace_order}",
            "source_invocation_key": source_invocation_key,
            "kernel_name": kernel_name,
            "kernel_id": item.get("kernel_id"),
            "trace_order": trace_order,
            "kernel_name_occurrence": occurrence_index,
            "grid_dim": dynamic.get("grid_dim"),
            "block_dim": dynamic.get("block_dim"),
            "shape_hint": None,
            "exec_time": exec_time,
            "exec_time_source": exec_time_source,
            "dynamic_inst_count": dynamic.get("total_dynamic_insts"),
            "feature_vector": _feature_vector(item),
            "feature_source_note": "mini_transformer_v4_full.json shortcut",
        }
        record.update(squash_kernel_map.get(zero_based_idx, {}))
        record.update(squash_tb_map.get(int(item.get("kernel_id", -1)), {}))
        records.append(record)

    return {
        "workload": full_data.get("workload"),
        "hardware": full_data.get("hardware"),
        "source_mode": "full_json_shortcut",
        "records": records,
    }
