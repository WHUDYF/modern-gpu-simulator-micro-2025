"""Build backend decision-layer artifacts from the middle-layer bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MIDDLE_LAYER_BUNDLE = REPO_ROOT / "artifacts" / "middle_layer" / "mini_transformer_v4" / "bundle.json"

ANCHOR_CANONICAL_STATUS = {
    "A1_qkv_projection_dense_48x32": "stable",
    "A2_attention_score_dense_32x32x12": "weak-share",
    "A3_softmax_reduce_24x1": "stable-with-context-split",
    "A4_context_stream_4x32x12": "stable-singleton",
    "A5_output_projection_dense_48x32": "stable",
    "A6_residual_elementwise_1536": "absorbed-with-bottleneck-note",
    "A7_layernorm_reduce_512": "review-needed",
    "A8_ffn_expand_dense_192x32": "stable",
    "A9_ffn_contract_dense_48x32": "stable",
}

FAMILY_CANONICAL_STATUS = {
    "F1_dense_tiled_backbone": "weak-share",
    "F2_reduction_normalize": "absorbed-with-review",
    "F3_streaming_aggregation": "stable-singleton",
    "F4_elementwise_residual": "absorbed-with-bottleneck-note",
}

REGIME_CANONICAL_STATUS = {
    "R1_qkv_projection_dense": "stable",
    "R2_attention_score_dense": "weak-share",
    "R3_output_projection_dense": "stable",
    "R4_ffn_expand_dense": "stable",
    "R5_ffn_contract_dense": "stable",
    "R6_softmax_reduction": "stable-with-context-split",
    "R7_layernorm_reduction": "review-needed",
    "R8_context_streaming": "stable-singleton",
    "R9_residual_elementwise": "absorbed-with-bottleneck-note",
}

REGIME_VALIDATION_ROLES = {
    "R1_qkv_projection_dense": "main-object",
    "R2_attention_score_dense": "main-object",
    "R3_output_projection_dense": "main-object",
    "R4_ffn_expand_dense": "main-object",
    "R5_ffn_contract_dense": "main-object",
    "R6_softmax_reduction": "main-object",
    "R7_layernorm_reduction": "review-object",
    "R8_context_streaming": "main-object",
    "R9_residual_elementwise": "constraint-object",
}

REGIME_ORIGINAL_ORDER = [
    "R1_qkv_projection_dense",
    "R2_attention_score_dense",
    "R3_output_projection_dense",
    "R4_ffn_expand_dense",
    "R5_ffn_contract_dense",
    "R6_softmax_reduction",
    "R7_layernorm_reduction",
    "R8_context_streaming",
    "R9_residual_elementwise",
]

LANE_SPECS = {
    "L1_dense_projection": ("compute", ["S1_register_pressure", "S2_occupancy_balance"], "cycles delta, occupancy response, top-k coverage gain"),
    "L2_attention_score": ("compute-shmem", ["S1_register_pressure", "S5_shared_memory_coupling"], "cycles delta, cache behavior shift, shmem-coupled response"),
    "L3_output_projection": ("projection-reuse", ["S1_register_pressure", "S2_occupancy_balance"], "cycles delta, reuse consistency, lane overlap"),
    "L4_ffn_expand": ("compute-large-shape", ["S1_register_pressure", "S2_occupancy_balance"], "cycles delta, priority rank gain, sensitivity concentration"),
    "L5_ffn_contract": ("dense-contraction", ["S2_occupancy_balance"], "cycles delta, marginal gain"),
    "L6_softmax": ("cache", ["S3_cache_capacity", "S4_reduction_path"], "cycles delta, dram throughput response, cache behavior response"),
    "L7_layernorm": ("reduction-review", ["S4_reduction_path"], "cycles delta, normalization consistency"),
    "L8_context_streaming": ("locality", ["S6_locality_path"], "cycles delta, l1 hit response, locality concentration"),
    "L9_residual_regression": ("constraint-memory", ["S7_constraint_regression"], "correctness-preserving delta, regression stability"),
}

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


def load_full_features(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_middle_bundle(path: Path = MIDDLE_LAYER_BUNDLE) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing middle-layer bundle: {path}")
    return json.loads(path.read_text())


def _per_kernel_by_id(full_features: dict[str, Any]) -> dict[int, tuple[str, dict[str, Any]]]:
    result: dict[int, tuple[str, dict[str, Any]]] = {}
    for invocation_name, kernel_data in full_features["per_kernel"].items():
        result[int(kernel_data["kernel_id"])] = (invocation_name, kernel_data)
    return result


def _fmt(value: float) -> float:
    return round(value, 6)


def build_anchor_table(full_features: dict[str, Any], bundle: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    bundle = bundle or load_middle_bundle()
    per_kernel = _per_kernel_by_id(full_features)
    anchors = []
    for anchor in bundle["anchors"]:
        member_invocation_names = [per_kernel[int(kernel_id)][0] for kernel_id in anchor["member_invocations"]]
        anchors.append(
            {
                "rep_kernel_id": anchor["anchor_id"],
                "kernel_name": anchor["kernel_name"],
                "cluster_id": anchor["cluster_id"],
                "member_invocations": member_invocation_names,
                "coverage_count": anchor["coverage_count"],
                "coverage_weight": _fmt(anchor["observed_coverage_ratio"]),
                "time_weight": _fmt(anchor["observed_time_ratio"]),
                "phase_id": anchor["phase_id"],
                "trace_order_summary": anchor["trace_order_summary"],
                "shape_hint_summary": anchor["shape_hint_summary"],
                "route_hint": anchor["route_hint"],
                "template_hint": anchor["template_hint"],
                "canonical_status": ANCHOR_CANONICAL_STATUS.get(anchor["anchor_id"], "stable"),
            }
        )
    return anchors


def build_family_table(anchor_table: list[dict[str, Any]], bundle: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    bundle = bundle or load_middle_bundle()
    families = []
    for family in bundle["families"]:
        families.append(
            {
                "family_id": family["family_id"],
                "phase_scope": family["phase_scope"],
                "route_primitive": family["route_primitive"],
                "hardware_template": family["hardware_template"],
                "member_rep_kernels": family["input_anchor_ids"],
                "member_count": len(family["input_anchor_ids"]),
                "canonical_status": FAMILY_CANONICAL_STATUS[family["family_id"]],
                "boundary_status": family["boundary_status"],
                "coverage_weight": _fmt(family["observed_coverage_ratio"]),
                "time_weight": _fmt(family["observed_time_ratio"]),
                "decision_weight": family["decision_weight_score"],
                "decision_weight_label": family["decision_weight"],
                "importance_score": family["importance_score"],
                "priority_class": family["priority_class"],
                "recommended_tuning_target": family["recommended_tuning_target"],
            }
        )
    return sorted(families, key=lambda row: row["importance_score"], reverse=True)


def build_regime_table(anchor_table: list[dict[str, Any]], family_table: list[dict[str, Any]], bundle: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    bundle = bundle or load_middle_bundle()
    regimes = []
    for regime in bundle["regimes"]:
        regimes.append(
            {
                "regime_id": regime["regime_id"],
                "family_id": regime["family_id"],
                "phase_id": regime["phase_id"],
                "route_primitive": regime["route_primitive"],
                "hardware_template": regime["hardware_template"],
                "source_rep_kernels": regime["source_anchor_ids"],
                "canonical_status": REGIME_CANONICAL_STATUS[regime["regime_id"]],
                "shape_regime": regime["shape_regime"],
                "context_scope": regime["context_scope"],
                "resource_signature": regime["resource_signature"],
                "coverage_weight": _fmt(regime["observed_coverage_ratio"]),
                "time_weight": _fmt(regime["observed_time_ratio"]),
                "family_importance_score": regime["family_importance_score"],
                "local_decision_weight": regime["local_decision_weight_score"],
                "local_decision_weight_label": regime["local_decision_weight"],
                "regime_priority_score": regime["regime_priority_score"],
                "simulator_lane_id": regime["simulator_lane_id"],
                "validation_role": REGIME_VALIDATION_ROLES[regime["regime_id"]],
                "original_order": REGIME_ORIGINAL_ORDER.index(regime["regime_id"]),
                "validation_status": "pending-review" if REGIME_CANONICAL_STATUS[regime["regime_id"]] == "review-needed" else "pending",
            }
        )
    return sorted(regimes, key=lambda row: row["regime_priority_score"], reverse=True)


def build_priority_lane_table(family_table: list[dict[str, Any]], regime_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered_families = [
        "F1_dense_tiled_backbone",
        "F2_reduction_normalize",
        "F3_streaming_aggregation",
        "F4_elementwise_residual",
    ]

    for source in ["importance-guided", "time-only", "name-based", "no-priority"]:
        if source == "importance-guided":
            families = family_table[:]
        elif source == "time-only":
            families = sorted(family_table, key=lambda row: row["time_weight"], reverse=True)
        elif source == "name-based":
            families = sorted(family_table, key=lambda row: row["family_id"])
        else:
            family_by_id = {row["family_id"]: row for row in family_table}
            families = [family_by_id[family_id] for family_id in ordered_families]
        for rank, family in enumerate(families, start=1):
            rows.append(
                {
                    "priority_item_id": f"P_family_{family['family_id']}_{source.replace('-', '_')}",
                    "object_level": "family",
                    "object_id": family["family_id"],
                    "family_id": family["family_id"],
                    "regime_id": None,
                    "priority_source": source,
                    "priority_rank": rank,
                    "canonical_status": family["canonical_status"],
                    "simulator_lane_id": None,
                    "lane_type": "family-level",
                    "recommended_tuning_target": family["recommended_tuning_target"],
                    "parameter_scenario_ids": [],
                    "expected_signal": "family-level prioritization anchor",
                    "score": family["importance_score"] if source == "importance-guided" else family["time_weight"] if source == "time-only" else 0.0,
                    "status": "planned",
                }
            )

    for source in ["importance-guided", "time-only", "name-based", "no-priority"]:
        if source == "importance-guided":
            regimes = regime_table[:]
        elif source == "time-only":
            regimes = sorted(regime_table, key=lambda row: row["time_weight"], reverse=True)
        elif source == "name-based":
            regimes = sorted(regime_table, key=lambda row: row["regime_id"])
        else:
            regimes = sorted(regime_table, key=lambda row: row["original_order"])
        for rank, regime in enumerate(regimes, start=1):
            lane_type, scenarios, expected_signal = LANE_SPECS[regime["simulator_lane_id"]]
            recommended_tuning_target = next(
                family["recommended_tuning_target"] for family in family_table if family["family_id"] == regime["family_id"]
            )
            rows.append(
                {
                    "priority_item_id": f"P_regime_{regime['regime_id']}_{source.replace('-', '_')}",
                    "object_level": "regime",
                    "object_id": regime["regime_id"],
                    "family_id": regime["family_id"],
                    "regime_id": regime["regime_id"],
                    "priority_source": source,
                    "priority_rank": rank,
                    "canonical_status": regime["canonical_status"],
                    "simulator_lane_id": regime["simulator_lane_id"],
                    "lane_type": lane_type,
                    "recommended_tuning_target": recommended_tuning_target,
                    "parameter_scenario_ids": scenarios,
                    "expected_signal": expected_signal,
                    "validation_role": regime["validation_role"],
                    "original_order": regime["original_order"],
                    "score": regime["regime_priority_score"] if source == "importance-guided" else regime["time_weight"] if source == "time-only" else 0.0,
                    "status": "planned",
                }
            )
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
    anchor_by_id = {anchor["rep_kernel_id"]: anchor for anchor in anchor_table}
    rows = []
    for regime in regime_table:
        _, scenarios, _ = LANE_SPECS[regime["simulator_lane_id"]]
        member_invocations = []
        for rep_kernel_id in regime["source_rep_kernels"]:
            member_invocations.extend(anchor_by_id[rep_kernel_id]["member_invocations"])
        for scenario_id in scenarios:
            rows.append(
                {
                    "writeback_id": f"W_{regime['regime_id']}_{scenario_id}",
                    "simulator_lane_id": regime["simulator_lane_id"],
                    "regime_id": regime["regime_id"],
                    "family_id": regime["family_id"],
                    "rep_kernel_ids": regime["source_rep_kernels"],
                    "member_invocations": member_invocations,
                    "parameter_scenario_id": scenario_id,
                    "observed_response": None,
                    "sensitivity_score": None,
                    "decision_update": None,
                    "importance_update": None,
                    "validation_status_update": regime["validation_status"],
                    "review_status_update": "keep-review" if regime["canonical_status"] == "review-needed" else "no-review",
                    "workload_explanation_note": "Pending writeback from result summary.",
                }
            )
    return rows


def build_backend_outputs(full_features: dict[str, Any]) -> dict[str, Any]:
    bundle = load_middle_bundle()
    anchors = build_anchor_table(full_features, bundle)
    families = build_family_table(anchors, bundle)
    regimes = build_regime_table(anchors, families, bundle)
    priority = build_priority_lane_table(families, regimes)
    worksheet = build_validation_worksheet(priority)
    writeback = build_writeback_map(regimes, anchors)
    return {
        "anchor_table": anchors,
        "family_table": families,
        "regime_table": regimes,
        "priority_lane_table": priority,
        "validation_worksheet": worksheet,
        "writeback_map": writeback,
    }


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
