"""Frontend selector modes for A-line anchor generation."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


def _percent_bucket(value: Any, step: int = 10) -> str:
    if value is None:
        return "na"
    return str(int(float(value) // step) * step)


def _magnitude_bucket(value: Any) -> str:
    if value is None:
        return "na"
    value = float(value)
    if value <= 0:
        return "0"
    thresholds = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    for idx, threshold in enumerate(thresholds):
        if value < threshold:
            return f"lt{int(threshold):d}"
    return "gte1000000000"


def _base_group_key(record: dict[str, Any]) -> tuple[str, ...]:
    return (record["kernel_name"],)


def _coarse_group_key(record: dict[str, Any]) -> tuple[str, ...]:
    feature = record.get("feature_vector", {})
    return (
        record["kernel_name"],
        str(record.get("grid_dim")),
        str(record.get("block_dim")),
        _percent_bucket(feature.get("compute_throughput_pct")),
        _percent_bucket(feature.get("dram_throughput_pct")),
    )


def _hybrid_subcluster_key(record: dict[str, Any]) -> tuple[str, ...]:
    feature = record.get("feature_vector", {})
    return (
        _magnitude_bucket(feature.get("total_dynamic_insts")),
        _percent_bucket(feature.get("achieved_occupancy_pct")),
        _percent_bucket(feature.get("compute_throughput_pct")),
        _percent_bucket(feature.get("dram_throughput_pct")),
        _percent_bucket(feature.get("ipc_active"), step=1),
        _percent_bucket(feature.get("cross_tb_offset_coverage"), step=1),
    )


def _group_records(records: list[dict[str, Any]], key_fn) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    return groups


def select_name_only(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = _group_records(records, _base_group_key)
    return _materialize_groups("name-only", groups)


def select_pka_like_coarse(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = _group_records(records, _coarse_group_key)
    return _materialize_groups("pka-like-coarse", groups)


def select_hybrid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coarse_groups = _group_records(records, _coarse_group_key)
    hybrid_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for coarse_key, members in coarse_groups.items():
        subgroups = _group_records(members, _hybrid_subcluster_key)
        for subkey, subgroup_members in subgroups.items():
            hybrid_groups[coarse_key + subkey] = subgroup_members
    return _materialize_groups("hybrid", hybrid_groups)


def _pick_anchor(members: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        members,
        key=lambda r: (
            r.get("exec_time") is not None,
            r.get("exec_time") or 0,
            -(r.get("trace_order") or 10**9),
        ),
    )


def _materialize_groups(method: str, groups: dict[tuple[str, ...], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for idx, (key, members) in enumerate(sorted(groups.items(), key=lambda kv: (kv[1][0]["trace_order"], kv[0]))):
        members = sorted(members, key=lambda r: r["trace_order"])
        anchor = _pick_anchor(members)
        exec_times = [m["exec_time"] for m in members if m.get("exec_time") is not None]
        inst_counts = [m["dynamic_inst_count"] for m in members if m.get("dynamic_inst_count") is not None]
        heterogeneity_flag = False
        if len(exec_times) >= 2:
            heterogeneity_flag = max(exec_times) != min(exec_times)
        result.append(
            {
                "method": method,
                "cluster_id": f"{method}-{idx+1}",
                "group_key": list(key),
                "anchor_record": anchor,
                "members": members,
                "member_count": len(members),
                "avg_exec_time": mean(exec_times) if exec_times else None,
                "avg_dynamic_inst_count": mean(inst_counts) if inst_counts else None,
                "heterogeneity_flag": heterogeneity_flag,
            }
        )
    return result


def run_selector(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "name-only":
        return select_name_only(records)
    if mode == "pka-like-coarse":
        return select_pka_like_coarse(records)
    if mode == "hybrid":
        return select_hybrid(records)
    raise ValueError(f"unknown selector mode: {mode}")

