"""Build backend decision-layer artifacts for mini_transformer_v4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


KERNEL_PREFIXES = {
    "_Z10gemm_tiled": "gemm_tiled",
    "_Z15attention_score": "attention_score",
    "_Z14softmax_kernel": "softmax_kernel",
    "_Z11context_mul": "context_mul",
    "_Z16layernorm_kernel": "layernorm_kernel",
    "_Z12residual_add": "residual_add",
}

ANCHOR_SPECS = {
    "gemm_tiled": ("A1", "C_dense_main", "Phase A", "projection-like dense M/N/K region", "Dense Projection/Transform", "Dense Tiled Compute"),
    "attention_score": ("A2", "C_dense_attention", "Phase B", "attention-score dense region", "Pairwise Score", "Dense Tiled Compute"),
    "softmax_kernel": ("A3", "C_reduce_attention", "Phase B", "softmax row-wise normalize region", "Reduction / Normalize", "Reduction Template"),
    "context_mul": ("A4", "C_streaming_attention", "Phase B", "attention aggregation region", "Weighted Aggregation", "Streaming Aggregation Template"),
    "layernorm_kernel": ("A5", "C_reduce_norm", "Phase C", "layernorm reduction region", "Reduction / Normalize", "Reduction Template"),
    "residual_add": ("A6", "C_elementwise_residual", "Phase C", "residual elementwise region", "Elementwise Fusion", "Elementwise Template"),
}

ANCHOR_CANON_STATUS = {
    "A1": "stable",
    "A2": "weak-share",
    "A3": "stable-with-context-split",
    "A4": "stable-singleton",
    "A5": "canonically-absorbed-review-needed",
    "A6": "canonically-absorbed-with-bottleneck-note",
}

FAMILY_SPECS = {
    "F1_dense_tiled": (["A1", "A2"], "weak-share", "Dense Projection/Transform + Pairwise Score", "Dense Tiled Compute", "weak_share", "High", "register-sensitive, occupancy-sensitive, tiled compute path"),
    "F2_reduction_normalize": (["A3", "A5"], "absorbed-with-review", "Reduction / Normalize", "Reduction Template", "strong_share_with_context_split", "High", "cache-capacity, reduction behavior, normalization path sensitivity"),
    "F3_streaming_aggregation": (["A4"], "stable-singleton", "Weighted Aggregation", "Streaming Aggregation Template", "stable_singleton", "Medium-High", "locality-sensitive, aggregation path validation"),
    "F4_elementwise_fusion": (["A6"], "absorbed-with-bottleneck-note", "Elementwise Fusion", "Elementwise Template", "stable_singleton", "Low-Medium", "constraint / regression checking with DRAM-side note retention"),
}

REGIME_SPECS = {
    "R1_projection_dense": ("F1_dense_tiled", "Phase A", ["A1"], "stable", "Dense Projection/Transform", "Dense Tiled Compute", "projection-like dense M/N/K region", "QKV / output projection / FFN projection context", "register-limited compute-heavy", "High", "L1_dense_projection"),
    "R2_attention_score_dense": ("F1_dense_tiled", "Phase B", ["A2"], "weak-share", "Pairwise Score", "Dense Tiled Compute", "attention-score dense region", "attention readout", "register + shmem coupled dense compute", "High", "L2_attention_score"),
    "R3_softmax_reduction": ("F2_reduction_normalize", "Phase B", ["A3"], "stable-with-context-split", "Reduction / Normalize", "Reduction Template", "softmax row-wise normalize region", "attention readout", "cache-capacity-sensitive / DRAM-pressure", "High", "L3_softmax"),
    "R4_layernorm_reduction": ("F2_reduction_normalize", "Phase C", ["A5"], "review-needed", "Reduction / Normalize", "Reduction Template", "layernorm reduction region", "normalization path", "reduction / normalization dominated", "Medium", "L4_layernorm"),
    "R5_context_streaming": ("F3_streaming_aggregation", "Phase B", ["A4"], "stable-singleton", "Weighted Aggregation", "Streaming Aggregation Template", "attention aggregation region", "attention readout", "L1-resident / locality-dominated streaming", "Medium-High", "L5_context_mul"),
    "R6_residual_elementwise": ("F4_elementwise_fusion", "Phase C", ["A6"], "absorbed-with-bottleneck-note", "Elementwise Fusion", "Elementwise Template", "residual fusion region", "residual path", "dram-dominated elementwise path", "Low", "L6_residual_add"),
}

LANE_SPECS = {
    "L1_dense_projection": ("compute", ["S1_register_pressure", "S2_occupancy_balance"], "dense projection responds to register and occupancy perturbations"),
    "L2_attention_score": ("compute-shmem", ["S1_register_pressure", "S5_shared_memory_coupling"], "attention score shows compute response with shmem-side divergence"),
    "L3_softmax": ("cache", ["S3_cache_capacity", "S4_reduction_path"], "softmax reacts to cache-capacity and reduction-side perturbations"),
    "L4_layernorm": ("reduction-review", ["S4_reduction_path"], "layernorm confirms whether reduction-path sharing is stable"),
    "L5_context_mul": ("locality", ["S6_locality_path"], "context_mul responds mainly to locality-sensitive perturbations"),
    "L6_residual_add": ("constraint-memory", ["S7_constraint_regression"], "residual_add acts as a low-priority constraint and memory-side check"),
}

REGIME_VALIDATION_ROLES = {
    "R1_projection_dense": "main-object",
    "R2_attention_score_dense": "main-object",
    "R3_softmax_reduction": "main-object",
    "R4_layernorm_reduction": "review-object",
    "R5_context_streaming": "main-object",
    "R6_residual_elementwise": "constraint-object",
}

REGIME_ORIGINAL_ORDER = [
    "R1_projection_dense",
    "R2_attention_score_dense",
    "R3_softmax_reduction",
    "R4_layernorm_reduction",
    "R5_context_streaming",
    "R6_residual_elementwise",
]

PARAMETER_SCENARIOS = {
    "S1_register_pressure": {"scenario_id": "S1_register_pressure", "focus": "register-sensitive"},
    "S2_occupancy_balance": {"scenario_id": "S2_occupancy_balance", "focus": "occupancy-sensitive"},
    "S3_cache_capacity": {"scenario_id": "S3_cache_capacity", "focus": "cache-sensitive"},
    "S4_reduction_path": {"scenario_id": "S4_reduction_path", "focus": "reduction-sensitive"},
    "S5_shared_memory_coupling": {"scenario_id": "S5_shared_memory_coupling", "focus": "shared-memory-coupled"},
    "S6_locality_path": {"scenario_id": "S6_locality_path", "focus": "locality-sensitive"},
    "S7_constraint_regression": {"scenario_id": "S7_constraint_regression", "focus": "constraint / regression"},
}

BASELINE_DEFS = [
    {"baseline_id": "B0_no_priority", "name": "No Priority"},
    {"baseline_id": "B1_time_only", "name": "Time-Only Priority"},
    {"baseline_id": "B2_name_based", "name": "Name-Based / Manual Priority"},
    {"baseline_id": "OURS_importance_guided", "name": "Importance-Guided Priority"},
]

DECISION_WEIGHT_VALUES = {"High": 1.0, "Medium-High": 0.75, "Medium": 0.6, "Low-Medium": 0.45, "Low": 0.3}


def load_full_features(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _short_name(invocation_name: str) -> str:
    for prefix, short in KERNEL_PREFIXES.items():
        if invocation_name.startswith(prefix):
            return short
    raise KeyError(invocation_name)


def _fmt(value: float) -> float:
    return round(value, 6)


def build_anchor_table(full_features: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    total_time = 0.0
    total_insts = 0
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trace_index, (invocation_name, kernel_data) in enumerate(full_features["per_kernel"].items(), start=1):
        hw = kernel_data.get("hardware_metrics", {}) or {}
        dyn = kernel_data.get("dynamic_stats", {}) or {}
        row = {
            "trace_index": trace_index,
            "invocation_name": invocation_name,
            "kernel_name": _short_name(invocation_name),
            "duration_ns": float(hw.get("duration_ns", 0.0) or 0.0),
            "dynamic_insts": int(dyn.get("total_dynamic_insts", 0) or 0),
            "grid_dim": dyn.get("grid_dim", ""),
            "block_dim": dyn.get("block_dim", ""),
        }
        rows.append(row)
        total_time += row["duration_ns"]
        total_insts += row["dynamic_insts"]
        grouped.setdefault(row["kernel_name"], []).append(row)

    anchors = []
    total_count = len(rows) or 1
    for kernel_name, (rep_id, cluster_id, phase_id, shape_hint, route_hint, template_hint) in ANCHOR_SPECS.items():
        members = grouped[kernel_name]
        coverage_count = len(members)
        duration_ns = sum(item["duration_ns"] for item in members)
        dynamic_insts = sum(item["dynamic_insts"] for item in members)
        anchors.append(
            {
                "rep_kernel_id": rep_id,
                "kernel_name": kernel_name,
                "cluster_id": cluster_id,
                "member_invocations": [item["invocation_name"] for item in members],
                "coverage_count": coverage_count,
                "coverage_weight": _fmt(coverage_count / total_count),
                "time_weight": _fmt(duration_ns / (total_time or 1.0)),
                "count_weight": _fmt(coverage_count / total_count),
                "inst_weight": _fmt(dynamic_insts / (total_insts or 1)),
                "trace_order_summary": ", ".join(str(item["trace_index"]) for item in members),
                "phase_id": phase_id,
                "grid_dim_summary": "; ".join(sorted({item["grid_dim"] for item in members})),
                "block_dim_summary": "; ".join(sorted({item["block_dim"] for item in members})),
                "shape_hint_summary": shape_hint,
                "route_hint": route_hint,
                "template_hint": template_hint,
                "canonical_status": ANCHOR_CANON_STATUS[rep_id],
            }
        )
    return sorted(anchors, key=lambda x: int(x["rep_kernel_id"][1:]))


def build_family_table(anchor_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor_by_id = {a["rep_kernel_id"]: a for a in anchor_table}
    families = []
    for family_id, (member_ids, canon_status, route, template, boundary, decision_label, tuning) in FAMILY_SPECS.items():
        members = [anchor_by_id[mid] for mid in member_ids]
        coverage = sum(m["coverage_weight"] for m in members)
        time_weight = sum(m["time_weight"] for m in members)
        decision = DECISION_WEIGHT_VALUES[decision_label]
        importance = 0.3 * coverage + 0.4 * time_weight + 0.3 * decision
        families.append(
            {
                "family_id": family_id,
                "phase_scope": sorted({m["phase_id"] for m in members}),
                "route_primitive": route,
                "hardware_template": template,
                "member_rep_kernels": member_ids,
                "member_count": len(member_ids),
                "canonical_status": canon_status,
                "boundary_status": boundary,
                "coverage_weight": _fmt(coverage),
                "time_weight": _fmt(time_weight),
                "decision_weight": decision,
                "decision_weight_label": decision_label,
                "importance_score": _fmt(importance),
                "priority_class": "High" if importance >= 0.7 else "Medium" if importance >= 0.4 else "Low",
                "recommended_tuning_target": tuning,
            }
        )
    return sorted(families, key=lambda x: x["importance_score"], reverse=True)


def build_regime_table(anchor_table: list[dict[str, Any]], family_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor_by_id = {a["rep_kernel_id"]: a for a in anchor_table}
    family_by_id = {f["family_id"]: f for f in family_table}
    regimes = []
    for regime_id, (family_id, phase_id, source_ids, canon_status, route, template, shape_regime, context_scope, resource_signature, local_label, lane_id) in REGIME_SPECS.items():
        members = [anchor_by_id[sid] for sid in source_ids]
        coverage = sum(m["coverage_weight"] for m in members)
        time_weight = sum(m["time_weight"] for m in members)
        local = DECISION_WEIGHT_VALUES[local_label]
        family_importance = family_by_id[family_id]["importance_score"]
        priority = 0.25 * coverage + 0.25 * time_weight + 0.25 * family_importance + 0.25 * local
        regimes.append(
            {
                "regime_id": regime_id,
                "family_id": family_id,
                "phase_id": phase_id,
                "route_primitive": route,
                "hardware_template": template,
                "source_rep_kernels": source_ids,
                "canonical_status": canon_status,
                "shape_regime": shape_regime,
                "context_scope": context_scope,
                "resource_signature": resource_signature,
                "coverage_weight": _fmt(coverage),
                "time_weight": _fmt(time_weight),
                "family_importance_score": family_importance,
                "local_decision_weight": local,
                "local_decision_weight_label": local_label,
                "regime_priority_score": _fmt(priority),
                "simulator_lane_id": lane_id,
                "validation_role": REGIME_VALIDATION_ROLES[regime_id],
                "original_order": REGIME_ORIGINAL_ORDER.index(regime_id),
                "validation_status": "pending-review" if canon_status == "review-needed" else "pending",
            }
        )
    return sorted(regimes, key=lambda x: x["regime_priority_score"], reverse=True)


def build_priority_lane_table(family_table: list[dict[str, Any]], regime_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    ordered_families = sorted(
        family_table,
        key=lambda x: (
            min(REGIME_ORIGINAL_ORDER.index(regime["regime_id"]) for regime in regime_table if regime["family_id"] == x["family_id"]),
            x["family_id"],
        ),
    )
    for source in ["importance-guided", "time-only", "name-based", "no-priority"]:
        if source == "importance-guided":
            fams = family_table[:]
        elif source == "time-only":
            fams = sorted(family_table, key=lambda x: x["time_weight"], reverse=True)
        elif source == "name-based":
            fams = sorted(family_table, key=lambda x: x["family_id"])
        else:
            fams = ordered_families
        for rank, family in enumerate(fams, start=1):
            rows.append({"priority_item_id": f"P_family_{family['family_id']}_{source.replace('-', '_')}", "object_level": "family", "object_id": family["family_id"], "family_id": family["family_id"], "regime_id": None, "priority_source": source, "priority_rank": rank, "canonical_status": family["canonical_status"], "simulator_lane_id": None, "lane_type": "family-level", "recommended_tuning_target": family["recommended_tuning_target"], "parameter_scenario_ids": [], "expected_signal": "family-level prioritization anchor", "score": family["importance_score"] if source == "importance-guided" else family["time_weight"] if source == "time-only" else 0.0, "status": "planned"})

    for source in ["importance-guided", "time-only", "name-based", "no-priority"]:
        if source == "importance-guided":
            regs = regime_table[:]
        elif source == "time-only":
            regs = sorted(regime_table, key=lambda x: x["time_weight"], reverse=True)
        elif source == "name-based":
            regs = sorted(regime_table, key=lambda x: x["regime_id"])
        else:
            regs = sorted(regime_table, key=lambda x: x["original_order"])
        for rank, regime in enumerate(regs, start=1):
            lane_type, scenarios, expected = LANE_SPECS[regime["simulator_lane_id"]]
            rows.append({"priority_item_id": f"P_regime_{regime['regime_id']}_{source.replace('-', '_')}", "object_level": "regime", "object_id": regime["regime_id"], "family_id": regime["family_id"], "regime_id": regime["regime_id"], "priority_source": source, "priority_rank": rank, "canonical_status": regime["canonical_status"], "simulator_lane_id": regime["simulator_lane_id"], "lane_type": lane_type, "recommended_tuning_target": next(f["recommended_tuning_target"] for f in family_table if f["family_id"] == regime["family_id"]), "parameter_scenario_ids": scenarios, "expected_signal": expected, "validation_role": regime["validation_role"], "original_order": regime["original_order"], "score": regime["regime_priority_score"] if source == "importance-guided" else regime["time_weight"] if source == "time-only" else 0.0, "status": "planned"})
    return rows


def build_validation_worksheet(priority_lane_table: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "validation_round_id": "mini_transformer_v4_backend_v1",
        "target_scope": "mini_transformer_v4",
        "baseline_defs": BASELINE_DEFS,
        "budget_definition": {
            "family_preselection_count": 3,
            "family_preselection": "Top-3 families (recommended target)",
            "comparison_strategies": ["importance-guided", "time-only", "name-based", "no-priority"],
            "main_object_max_scenarios": 2,
            "review_object_max_scenarios": 1,
            "constraint_object_max_scenarios": 1,
        },
        "parameter_scenarios": list(PARAMETER_SCENARIOS.values()),
        "notes": "First validation pass uses family -> regime expansion.",
    }


def build_writeback_map(regime_table: list[dict[str, Any]], anchor_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor_by_id = {a["rep_kernel_id"]: a for a in anchor_table}
    rows = []
    for regime in regime_table:
        _, scenarios, _ = LANE_SPECS[regime["simulator_lane_id"]]
        for scenario_id in scenarios:
            rep_kernel_ids = regime["source_rep_kernels"]
            member_invocations = []
            for rid in rep_kernel_ids:
                member_invocations.extend(anchor_by_id[rid]["member_invocations"])
            rows.append({"writeback_id": f"W_{regime['regime_id']}_{scenario_id}", "simulator_lane_id": regime["simulator_lane_id"], "regime_id": regime["regime_id"], "family_id": regime["family_id"], "rep_kernel_ids": rep_kernel_ids, "member_invocations": member_invocations, "parameter_scenario_id": scenario_id, "observed_response": None, "sensitivity_score": None, "decision_update": None, "importance_update": None, "validation_status_update": regime["validation_status"], "review_status_update": "keep-review" if regime["canonical_status"] == "review-needed" else "no-review", "workload_explanation_note": "Pending writeback from result summary."})
    return rows


def build_backend_outputs(full_features: dict[str, Any]) -> dict[str, Any]:
    anchors = build_anchor_table(full_features)
    families = build_family_table(anchors)
    regimes = build_regime_table(anchors, families)
    priority = build_priority_lane_table(families, regimes)
    worksheet = build_validation_worksheet(priority)
    writeback = build_writeback_map(regimes, anchors)
    return {"anchor_table": anchors, "family_table": families, "regime_table": regimes, "priority_lane_table": priority, "validation_worksheet": worksheet, "writeback_map": writeback}


def write_backend_outputs(outputs: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "backend_anchor_table_v1.json": outputs["anchor_table"],
        "backend_family_table_v1.json": outputs["family_table"],
        "backend_regime_table_v1.json": outputs["regime_table"],
        "backend_priority_lane_table_v1.json": outputs["priority_lane_table"],
        "backend_validation_worksheet_v1.json": outputs["validation_worksheet"],
        "backend_writeback_map_v1.json": outputs["writeback_map"],
    }
    for name, payload in mapping.items():
        (output_dir / name).write_text(json.dumps(payload, indent=2))
