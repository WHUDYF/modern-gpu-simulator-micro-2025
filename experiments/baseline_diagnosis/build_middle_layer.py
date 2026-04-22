from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "middle_layer" / "mini_transformer_v4"

LABEL_SCORES = {
    "Low": 0.30,
    "Low-Medium": 0.45,
    "Medium": 0.60,
    "Medium-High": 0.75,
    "High": 0.90,
}


ANCHOR_SPECS: list[dict[str, Any]] = [
    {
        "anchor_id": "A1_qkv_projection_dense_48x32",
        "canonical_kernel_name": "gemm_tiled",
        "kernel_ids": [1, 2, 3, 4],
        "phase_id": "Phase A",
        "context_scope": "Q/K/V projection path",
        "cluster_id": "C_dense_proj_qkv",
        "trace_order_summary": "trace front; kernels 1..4",
        "shape_hint_summary": "small dense projection region",
        "route_hint": "Dense Projection/Transform",
        "template_hint": "Dense Tiled Compute",
        "coverage_label": "High",
        "time_label": "High",
        "decision_label": "High",
        "notes": "Dense backbone front-end anchor",
    },
    {
        "anchor_id": "A2_attention_score_dense_32x32x12",
        "canonical_kernel_name": "attention_score",
        "kernel_ids": [5],
        "phase_id": "Phase B",
        "context_scope": "attention score path",
        "cluster_id": "C_dense_attention_score",
        "trace_order_summary": "attention score stage",
        "shape_hint_summary": "pairwise score dense region",
        "route_hint": "Pairwise Score",
        "template_hint": "Dense Tiled Compute",
        "coverage_label": "Medium",
        "time_label": "Medium-High",
        "decision_label": "High",
        "notes": "Dense family boundary anchor",
    },
    {
        "anchor_id": "A3_softmax_reduce_24x1",
        "canonical_kernel_name": "softmax_kernel",
        "kernel_ids": [6],
        "phase_id": "Phase B",
        "context_scope": "attention normalization path",
        "cluster_id": "C_reduce_attention_softmax",
        "trace_order_summary": "attention normalize stage",
        "shape_hint_summary": "row-wise reduction normalize region",
        "route_hint": "Reduction / Normalize",
        "template_hint": "Reduction Template",
        "coverage_label": "Medium",
        "time_label": "Medium-High",
        "decision_label": "High",
        "notes": "Reduction family attention-side anchor",
    },
    {
        "anchor_id": "A4_context_stream_4x32x12",
        "canonical_kernel_name": "context_mul",
        "kernel_ids": [7],
        "phase_id": "Phase B",
        "context_scope": "attention aggregation path",
        "cluster_id": "C_stream_attention_context",
        "trace_order_summary": "attention aggregation stage",
        "shape_hint_summary": "streaming weighted aggregation region",
        "route_hint": "Weighted Aggregation",
        "template_hint": "Streaming Aggregation Template",
        "coverage_label": "Medium",
        "time_label": "Medium",
        "decision_label": "Medium-High",
        "notes": "Streaming aggregation anchor",
    },
    {
        "anchor_id": "A5_output_projection_dense_48x32",
        "canonical_kernel_name": "gemm_tiled",
        "kernel_ids": [8],
        "phase_id": "Phase B_to_C",
        "context_scope": "attention output projection path",
        "cluster_id": "C_dense_output_proj",
        "trace_order_summary": "post-attention dense projection",
        "shape_hint_summary": "post-attention dense projection region",
        "route_hint": "Dense Projection/Transform",
        "template_hint": "Dense Tiled Compute",
        "coverage_label": "Medium",
        "time_label": "Medium",
        "decision_label": "Medium",
        "notes": "Post-attention dense projection anchor",
    },
    {
        "anchor_id": "A6_residual_elementwise_1536",
        "canonical_kernel_name": "residual_add",
        "kernel_ids": [9, 13],
        "phase_id": "Phase C",
        "context_scope": "residual path",
        "cluster_id": "C_elementwise_residual",
        "trace_order_summary": "repeated after attention and FFN",
        "shape_hint_summary": "elementwise residual region",
        "route_hint": "Elementwise Fusion",
        "template_hint": "Elementwise Template",
        "coverage_label": "High",
        "time_label": "Low",
        "decision_label": "Low-Medium",
        "notes": "High-frequency residual regression anchor",
    },
    {
        "anchor_id": "A7_layernorm_reduce_512",
        "canonical_kernel_name": "layernorm_kernel",
        "kernel_ids": [10, 14],
        "phase_id": "Phase C",
        "context_scope": "normalization path",
        "cluster_id": "C_reduce_layernorm",
        "trace_order_summary": "repeated after residuals",
        "shape_hint_summary": "layernorm reduction region",
        "route_hint": "Reduction / Normalize",
        "template_hint": "Reduction Template",
        "coverage_label": "Medium",
        "time_label": "Medium",
        "decision_label": "Medium",
        "notes": "Normalization-path reduction anchor",
    },
    {
        "anchor_id": "A8_ffn_expand_dense_192x32",
        "canonical_kernel_name": "gemm_tiled",
        "kernel_ids": [11],
        "phase_id": "Phase C",
        "context_scope": "FFN expansion path",
        "cluster_id": "C_dense_ffn_expand",
        "trace_order_summary": "FFN expansion stage",
        "shape_hint_summary": "large dense transform region",
        "route_hint": "Dense Projection/Transform",
        "template_hint": "Dense Tiled Compute",
        "coverage_label": "Medium",
        "time_label": "High",
        "decision_label": "High",
        "notes": "Dense family heavy back-end anchor",
    },
    {
        "anchor_id": "A9_ffn_contract_dense_48x32",
        "canonical_kernel_name": "gemm_tiled",
        "kernel_ids": [12],
        "phase_id": "Phase C",
        "context_scope": "FFN contraction path",
        "cluster_id": "C_dense_ffn_contract",
        "trace_order_summary": "FFN contraction stage",
        "shape_hint_summary": "dense contraction region",
        "route_hint": "Dense Projection/Transform",
        "template_hint": "Dense Tiled Compute",
        "coverage_label": "Medium",
        "time_label": "Medium",
        "decision_label": "Medium",
        "notes": "Dense family contraction anchor",
    },
]


FAMILY_SPECS: list[dict[str, Any]] = [
    {
        "family_id": "F1_dense_tiled_backbone",
        "input_anchor_ids": [
            "A1_qkv_projection_dense_48x32",
            "A2_attention_score_dense_32x32x12",
            "A5_output_projection_dense_48x32",
            "A8_ffn_expand_dense_192x32",
            "A9_ffn_contract_dense_48x32",
        ],
        "phase_scope": ["Phase A", "Phase B", "Phase B_to_C", "Phase C"],
        "route_primitive": "Dense Projection/Transform + Pairwise Score",
        "hardware_template": "Dense Tiled Compute",
        "boundary_status": "weak_share_but_keep_together",
        "boundary_notes": "attention_score differs in route and shmem signature, but shares the dense tiled execution template with projection and FFN anchors.",
        "shape_regime_summary": "48x32 projection/contract, 192x32 expansion, 32x32x12 pairwise score",
        "resource_signature_summary": "register / occupancy primary; attention_score carries stronger shmem coupling",
        "coverage_label": "High",
        "time_label": "High",
        "decision_label": "High",
        "priority_class": "High",
        "recommended_tuning_target": "register-sensitive, occupancy-sensitive, dense tiled path",
        "notes": "Most important compute backbone family.",
    },
    {
        "family_id": "F2_reduction_normalize",
        "input_anchor_ids": [
            "A3_softmax_reduce_24x1",
            "A7_layernorm_reduce_512",
        ],
        "phase_scope": ["Phase B", "Phase C"],
        "route_primitive": "Reduction / Normalize",
        "hardware_template": "Reduction Template",
        "boundary_status": "strong_share_with_context_split",
        "boundary_notes": "softmax and layernorm share reduction / normalize structure but should stay split at regime level because their context scopes differ.",
        "shape_regime_summary": "attention normalization vs residual normalization",
        "resource_signature_summary": "reduction / synchronization; softmax is more cache-capacity and DRAM-pressure sensitive",
        "coverage_label": "Medium",
        "time_label": "Medium-High",
        "decision_label": "High",
        "priority_class": "High",
        "recommended_tuning_target": "cache-capacity, reduction behavior, normalization path sensitivity",
        "notes": "Second-priority family because it captures critical reduction behavior.",
    },
    {
        "family_id": "F3_streaming_aggregation",
        "input_anchor_ids": ["A4_context_stream_4x32x12"],
        "phase_scope": ["Phase B"],
        "route_primitive": "Weighted Aggregation",
        "hardware_template": "Streaming Aggregation Template",
        "boundary_status": "stable_singleton",
        "boundary_notes": "context_mul currently remains a singleton because no other anchor shares its locality-dominated streaming behavior closely enough.",
        "shape_regime_summary": "attention aggregation region",
        "resource_signature_summary": "locality-dominated / L1-resident / streaming accumulation",
        "coverage_label": "Medium",
        "time_label": "Medium",
        "decision_label": "Medium-High",
        "priority_class": "Medium",
        "recommended_tuning_target": "locality-sensitive, aggregation path validation",
        "notes": "Independent attention aggregation family.",
    },
    {
        "family_id": "F4_elementwise_residual",
        "input_anchor_ids": ["A6_residual_elementwise_1536"],
        "phase_scope": ["Phase C"],
        "route_primitive": "Elementwise Fusion",
        "hardware_template": "Elementwise Template",
        "boundary_status": "stable_singleton",
        "boundary_notes": "residual_add is currently the stable elementwise constraint object.",
        "shape_regime_summary": "residual elementwise region",
        "resource_signature_summary": "lightweight memory-side fusion",
        "coverage_label": "High",
        "time_label": "Low",
        "decision_label": "Low-Medium",
        "priority_class": "Low",
        "recommended_tuning_target": "lightweight regression, constraint checking",
        "notes": "Constraint family rather than main tuning target.",
    },
]


REGIME_SPECS: list[dict[str, Any]] = [
    {
        "regime_id": "R1_qkv_projection_dense",
        "family_id": "F1_dense_tiled_backbone",
        "source_anchor_ids": ["A1_qkv_projection_dense_48x32"],
        "phase_id": "Phase A",
        "route_primitive": "Dense Projection/Transform",
        "hardware_template": "Dense Tiled Compute",
        "shape_regime": "48x32 projection-like dense region",
        "context_scope": "Q/K/V projection path",
        "resource_signature": "register-limited dense backbone",
        "coverage_label": "High",
        "time_label": "High",
        "local_decision_label": "High",
        "validation_status": "pending",
        "notes": "Primary dense regime.",
    },
    {
        "regime_id": "R2_attention_score_dense",
        "family_id": "F1_dense_tiled_backbone",
        "source_anchor_ids": ["A2_attention_score_dense_32x32x12"],
        "phase_id": "Phase B",
        "route_primitive": "Pairwise Score",
        "hardware_template": "Dense Tiled Compute",
        "shape_regime": "32x32x12 attention-score region",
        "context_scope": "attention score path",
        "resource_signature": "register + shmem coupled dense compute",
        "coverage_label": "Medium",
        "time_label": "Medium-High",
        "local_decision_label": "High",
        "validation_status": "pending",
        "notes": "Boundary regime that should not be merged into generic dense projection.",
    },
    {
        "regime_id": "R3_output_projection_dense",
        "family_id": "F1_dense_tiled_backbone",
        "source_anchor_ids": ["A5_output_projection_dense_48x32"],
        "phase_id": "Phase B_to_C",
        "route_primitive": "Dense Projection/Transform",
        "hardware_template": "Dense Tiled Compute",
        "shape_regime": "48x32 post-attention projection region",
        "context_scope": "attention output projection path",
        "resource_signature": "dense transform with post-attention context",
        "coverage_label": "Medium",
        "time_label": "Medium",
        "local_decision_label": "Medium",
        "validation_status": "pending",
        "notes": "Checks whether projection lane conclusions transfer after attention.",
    },
    {
        "regime_id": "R4_ffn_expand_dense",
        "family_id": "F1_dense_tiled_backbone",
        "source_anchor_ids": ["A8_ffn_expand_dense_192x32"],
        "phase_id": "Phase C",
        "route_primitive": "Dense Projection/Transform",
        "hardware_template": "Dense Tiled Compute",
        "shape_regime": "192x32 FFN expansion region",
        "context_scope": "FFN expansion path",
        "resource_signature": "large-shape dense compute, register-sensitive",
        "coverage_label": "Medium",
        "time_label": "High",
        "local_decision_label": "High",
        "validation_status": "pending",
        "notes": "Heavy dense regime in later pipeline stage.",
    },
    {
        "regime_id": "R5_ffn_contract_dense",
        "family_id": "F1_dense_tiled_backbone",
        "source_anchor_ids": ["A9_ffn_contract_dense_48x32"],
        "phase_id": "Phase C",
        "route_primitive": "Dense Projection/Transform",
        "hardware_template": "Dense Tiled Compute",
        "shape_regime": "48x32 FFN contraction region",
        "context_scope": "FFN contraction path",
        "resource_signature": "dense contraction with lower local leverage than expansion",
        "coverage_label": "Medium",
        "time_label": "Medium",
        "local_decision_label": "Medium",
        "validation_status": "pending",
        "notes": "Secondary dense regime.",
    },
    {
        "regime_id": "R6_softmax_reduction",
        "family_id": "F2_reduction_normalize",
        "source_anchor_ids": ["A3_softmax_reduce_24x1"],
        "phase_id": "Phase B",
        "route_primitive": "Reduction / Normalize",
        "hardware_template": "Reduction Template",
        "shape_regime": "24x1 row-wise normalization region",
        "context_scope": "attention normalization path",
        "resource_signature": "cache-capacity-sensitive, DRAM-pressure",
        "coverage_label": "Medium",
        "time_label": "Medium-High",
        "local_decision_label": "High",
        "validation_status": "pending",
        "notes": "Primary reduction regime.",
    },
    {
        "regime_id": "R7_layernorm_reduction",
        "family_id": "F2_reduction_normalize",
        "source_anchor_ids": ["A7_layernorm_reduce_512"],
        "phase_id": "Phase C",
        "route_primitive": "Reduction / Normalize",
        "hardware_template": "Reduction Template",
        "shape_regime": "512x1 layernorm reduction region",
        "context_scope": "residual normalization path",
        "resource_signature": "reduction / normalization dominated",
        "coverage_label": "Medium",
        "time_label": "Medium",
        "local_decision_label": "Medium",
        "validation_status": "pending",
        "notes": "Normalization-path constraint regime.",
    },
    {
        "regime_id": "R8_context_streaming",
        "family_id": "F3_streaming_aggregation",
        "source_anchor_ids": ["A4_context_stream_4x32x12"],
        "phase_id": "Phase B",
        "route_primitive": "Weighted Aggregation",
        "hardware_template": "Streaming Aggregation Template",
        "shape_regime": "4x32x12 weighted aggregation region",
        "context_scope": "attention aggregation path",
        "resource_signature": "locality-dominated, L1-resident streaming",
        "coverage_label": "Medium",
        "time_label": "Medium",
        "local_decision_label": "Medium-High",
        "validation_status": "pending",
        "notes": "Independent streaming regime.",
    },
    {
        "regime_id": "R9_residual_elementwise",
        "family_id": "F4_elementwise_residual",
        "source_anchor_ids": ["A6_residual_elementwise_1536"],
        "phase_id": "Phase C",
        "route_primitive": "Elementwise Fusion",
        "hardware_template": "Elementwise Template",
        "shape_regime": "1536-wide residual elementwise region",
        "context_scope": "residual path",
        "resource_signature": "lightweight elementwise memory-side",
        "coverage_label": "High",
        "time_label": "Low",
        "local_decision_label": "Low",
        "validation_status": "pending",
        "notes": "Regression / constraint regime.",
    },
]


LANE_SPECS: list[dict[str, Any]] = [
    {
        "lane_id": "L1_dense_projection",
        "target_regime_id": "R1_qkv_projection_dense",
        "lane_goal": "Validate whether front dense projection dominates register / occupancy sensitivity.",
        "parameter_direction": "register-sensitive, occupancy-sensitive",
        "baseline_type": "importance-guided vs time-only",
        "validation_metric": "cycles delta, occupancy response, top-k coverage gain",
        "writeback_target": "R1 -> F1 -> dense backbone summary",
        "notes": "Highest-priority lane.",
    },
    {
        "lane_id": "L2_attention_score",
        "target_regime_id": "R2_attention_score_dense",
        "lane_goal": "Check whether attention score should remain separate from generic dense projection in backend reasoning.",
        "parameter_direction": "shared-memory-coupled, register-sensitive",
        "baseline_type": "importance-guided vs manual",
        "validation_metric": "cycles delta, cache behavior shift, shmem-coupled response",
        "writeback_target": "R2 -> F1 boundary refinement",
        "notes": "Boundary-validation lane.",
    },
    {
        "lane_id": "L3_output_projection",
        "target_regime_id": "R3_output_projection_dense",
        "lane_goal": "Test whether projection conclusions transfer after attention output stage.",
        "parameter_direction": "register-sensitive, projection-path reuse",
        "baseline_type": "importance-guided vs family-shared baseline",
        "validation_metric": "cycles delta, reuse consistency, lane overlap",
        "writeback_target": "R3 -> F1 dense reuse note",
        "notes": "Dense reuse check.",
    },
    {
        "lane_id": "L4_ffn_expand",
        "target_regime_id": "R4_ffn_expand_dense",
        "lane_goal": "Check whether FFN expansion is the heaviest late dense regime.",
        "parameter_direction": "register-sensitive, large-shape dense compute",
        "baseline_type": "importance-guided vs time-only",
        "validation_metric": "cycles delta, priority rank gain, sensitivity concentration",
        "writeback_target": "R4 -> F1 FFN summary",
        "notes": "Second main dense lane.",
    },
    {
        "lane_id": "L5_ffn_contract",
        "target_regime_id": "R5_ffn_contract_dense",
        "lane_goal": "Measure marginal gain for FFN contraction after higher-value dense regimes are processed.",
        "parameter_direction": "dense contraction reuse, occupancy-sensitive",
        "baseline_type": "importance-guided vs no-priority",
        "validation_metric": "cycles delta, marginal gain",
        "writeback_target": "R5 -> F1 secondary regime note",
        "notes": "Secondary dense lane.",
    },
    {
        "lane_id": "L6_softmax",
        "target_regime_id": "R6_softmax_reduction",
        "lane_goal": "Validate softmax cache-capacity / DRAM-pressure explanation.",
        "parameter_direction": "cache-sensitive, reduction-sensitive",
        "baseline_type": "importance-guided vs time-only",
        "validation_metric": "cycles delta, dram throughput response, cache behavior response",
        "writeback_target": "R6 -> F2 reduction summary",
        "notes": "Primary reduction lane.",
    },
    {
        "lane_id": "L7_layernorm",
        "target_regime_id": "R7_layernorm_reduction",
        "lane_goal": "Keep layernorm as the normalization-path validation object.",
        "parameter_direction": "reduction-sensitive, normalization-path validation",
        "baseline_type": "importance-guided vs family-shared baseline",
        "validation_metric": "cycles delta, normalization consistency",
        "writeback_target": "R7 -> F2 normalization note",
        "notes": "Constraint lane for normalization path.",
    },
    {
        "lane_id": "L8_context_streaming",
        "target_regime_id": "R8_context_streaming",
        "lane_goal": "Validate locality-dominated streaming aggregation behavior.",
        "parameter_direction": "locality-sensitive, L1-sensitive",
        "baseline_type": "importance-guided vs manual",
        "validation_metric": "cycles delta, l1 hit response, locality concentration",
        "writeback_target": "R8 -> F3 streaming summary",
        "notes": "Independent streaming lane.",
    },
    {
        "lane_id": "L9_residual_regression",
        "target_regime_id": "R9_residual_elementwise",
        "lane_goal": "Keep residual elementwise as a lightweight regression / constraint object.",
        "parameter_direction": "lightweight memory-side, regression-check",
        "baseline_type": "no-priority baseline",
        "validation_metric": "correctness-preserving delta, regression stability",
        "writeback_target": "R9 -> F4 residual constraint note",
        "notes": "Not a primary tuning lane.",
    },
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _round4(value: float) -> float:
    return round(value, 4)


def _label_score(label: str) -> float:
    return LABEL_SCORES[label]


def load_middle_layer_sources(repo_root: Path) -> dict[str, Any]:
    experiment_dir = repo_root / "experiments" / "mini_transformer"
    result_dir = repo_root / "experiments" / "baseline_diagnosis" / "results" / "mini_transformer_v4"
    required = {
        "full": experiment_dir / "mini_transformer_v4_full.json",
        "squash": experiment_dir / "mechanisms" / "squash.json",
        "batch": experiment_dir / "mechanisms" / "batch.json",
        "baseline_ape": result_dir / "baseline_ape.json",
        "e5": result_dir / "E5_stageC_validation.md",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing middle-layer evidence files:\n{missing_list}")
    return {name: _load_json(path) if path.suffix == ".json" else path.read_text() for name, path in required.items()}


def _per_kernel_by_id(full_json: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for kernel in full_json["per_kernel"].values():
        result[int(kernel["kernel_id"])] = kernel
    return result


def _ape_key(short_name: str, grid_dim: str, block_dim: str) -> str:
    grid = f"({grid_dim.replace('x', ', ')})"
    block = f"({block_dim.replace('x', ', ')})"
    return f"{short_name}|{grid}|{block}"


def build_anchor_records(sources: dict[str, Any]) -> list[dict[str, Any]]:
    per_kernel = _per_kernel_by_id(sources["full"])
    ape_table = sources["baseline_ape"]["ape_table"]
    total_invocations = sum(len(spec["kernel_ids"]) for spec in ANCHOR_SPECS)
    total_cycles = sum(
        per_kernel[kernel_id]["hardware_metrics"]["elapsed_cycles"]
        for spec in ANCHOR_SPECS
        for kernel_id in spec["kernel_ids"]
    )

    anchors: list[dict[str, Any]] = []
    for spec in ANCHOR_SPECS:
        kernel_ids = spec["kernel_ids"]
        members = [per_kernel[kernel_id] for kernel_id in kernel_ids]
        primary = members[0]
        grid_dim = primary["dynamic_stats"]["grid_dim"]
        block_dim = primary["dynamic_stats"]["block_dim"]
        canonical_kernel_name = spec["canonical_kernel_name"]
        ape_lookup = _ape_key(canonical_kernel_name, grid_dim, block_dim)
        elapsed_cycles = sum(member["hardware_metrics"]["elapsed_cycles"] for member in members)
        coverage_ratio = len(kernel_ids) / total_invocations
        time_ratio = elapsed_cycles / total_cycles
        anchors.append(
            {
                "anchor_id": spec["anchor_id"],
                "kernel_name": canonical_kernel_name,
                "kernel_name_raw": primary["kernel_name"],
                "phase_id": spec["phase_id"],
                "context_scope": spec["context_scope"],
                "cluster_id": spec["cluster_id"],
                "member_invocations": kernel_ids,
                "coverage_count": len(kernel_ids),
                "observed_coverage_ratio": _round4(coverage_ratio),
                "observed_time_ratio": _round4(time_ratio),
                "coverage_label": spec["coverage_label"],
                "time_label": spec["time_label"],
                "decision_label": spec["decision_label"],
                "trace_order_summary": spec["trace_order_summary"],
                "grid_dim_summary": grid_dim,
                "block_dim_summary": block_dim,
                "shape_hint_summary": spec["shape_hint_summary"],
                "route_hint": spec["route_hint"],
                "template_hint": spec["template_hint"],
                "weighted_elapsed_cycles": _round4(elapsed_cycles),
                "ape_lookup_key": ape_lookup if ape_lookup in ape_table else None,
                "ape_elapsed_cycles_ape": (
                    ape_table[ape_lookup]["metrics"]["elapsed_cycles"]["ape"]
                    if ape_lookup in ape_table
                    else None
                ),
                "notes": spec["notes"],
            }
        )
    return anchors


def build_family_records(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor_by_id = {anchor["anchor_id"]: anchor for anchor in anchors}
    total_coverage_count = sum(anchor["coverage_count"] for anchor in anchors)
    total_weighted_cycles = sum(anchor["weighted_elapsed_cycles"] for anchor in anchors)
    families: list[dict[str, Any]] = []
    for spec in FAMILY_SPECS:
        selected = [anchor_by_id[anchor_id] for anchor_id in spec["input_anchor_ids"]]
        coverage_ratio = sum(anchor["coverage_count"] for anchor in selected) / total_coverage_count
        time_ratio = sum(anchor["weighted_elapsed_cycles"] for anchor in selected) / total_weighted_cycles
        importance = (
            0.3 * _label_score(spec["coverage_label"])
            + 0.4 * _label_score(spec["time_label"])
            + 0.3 * _label_score(spec["decision_label"])
        )
        families.append(
            {
                "family_id": spec["family_id"],
                "input_anchor_ids": spec["input_anchor_ids"],
                "phase_scope": spec["phase_scope"],
                "route_primitive": spec["route_primitive"],
                "hardware_template": spec["hardware_template"],
                "boundary_status": spec["boundary_status"],
                "boundary_notes": spec["boundary_notes"],
                "shape_regime_summary": spec["shape_regime_summary"],
                "resource_signature_summary": spec["resource_signature_summary"],
                "observed_coverage_ratio": _round4(coverage_ratio),
                "observed_time_ratio": _round4(time_ratio),
                "coverage_label": spec["coverage_label"],
                "time_label": spec["time_label"],
                "decision_label": spec["decision_label"],
                "importance_score": _round4(importance),
                "priority_class": spec["priority_class"],
                "recommended_tuning_target": spec["recommended_tuning_target"],
                "notes": spec["notes"],
            }
        )
    return families


def build_regime_records(
    anchors: list[dict[str, Any]],
    families: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anchor_by_id = {anchor["anchor_id"]: anchor for anchor in anchors}
    family_by_id = {family["family_id"]: family for family in families}
    total_coverage_count = sum(anchor["coverage_count"] for anchor in anchors)
    total_weighted_cycles = sum(anchor["weighted_elapsed_cycles"] for anchor in anchors)
    regimes: list[dict[str, Any]] = []
    for spec in REGIME_SPECS:
        selected = [anchor_by_id[anchor_id] for anchor_id in spec["source_anchor_ids"]]
        coverage_ratio = sum(anchor["coverage_count"] for anchor in selected) / total_coverage_count
        time_ratio = sum(anchor["weighted_elapsed_cycles"] for anchor in selected) / total_weighted_cycles
        family_importance = family_by_id[spec["family_id"]]["importance_score"]
        regime_priority = (
            0.35 * family_importance
            + 0.25 * _label_score(spec["coverage_label"])
            + 0.25 * _label_score(spec["time_label"])
            + 0.15 * _label_score(spec["local_decision_label"])
        )
        regimes.append(
            {
                "regime_id": spec["regime_id"],
                "family_id": spec["family_id"],
                "source_anchor_ids": spec["source_anchor_ids"],
                "phase_id": spec["phase_id"],
                "route_primitive": spec["route_primitive"],
                "hardware_template": spec["hardware_template"],
                "shape_regime": spec["shape_regime"],
                "context_scope": spec["context_scope"],
                "resource_signature": spec["resource_signature"],
                "observed_coverage_ratio": _round4(coverage_ratio),
                "observed_time_ratio": _round4(time_ratio),
                "coverage_label": spec["coverage_label"],
                "time_label": spec["time_label"],
                "family_importance_score": family_importance,
                "local_decision_label": spec["local_decision_label"],
                "regime_priority_score": _round4(regime_priority),
                "validation_status": spec["validation_status"],
                "notes": spec["notes"],
            }
        )
    return regimes


def build_lane_records(regimes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regime_by_id = {regime["regime_id"]: regime for regime in regimes}
    lanes: list[dict[str, Any]] = []
    for spec in LANE_SPECS:
        regime = regime_by_id[spec["target_regime_id"]]
        lanes.append(
            {
                "lane_id": spec["lane_id"],
                "target_regime_id": spec["target_regime_id"],
                "target_family_id": regime["family_id"],
                "lane_goal": spec["lane_goal"],
                "parameter_direction": spec["parameter_direction"],
                "baseline_type": spec["baseline_type"],
                "validation_metric": spec["validation_metric"],
                "writeback_target": spec["writeback_target"],
                "notes": spec["notes"],
            }
        )
    return lanes


def build_middle_layer_artifacts(repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root or REPO_ROOT
    sources = load_middle_layer_sources(repo_root)
    anchors = build_anchor_records(sources)
    families = build_family_records(anchors)
    regimes = build_regime_records(anchors, families)
    lanes = build_lane_records(regimes)
    return {
        "metadata": {
            "workload": "mini_transformer_v4",
            "builder": "experiments/baseline_diagnosis/build_middle_layer.py",
            "importance_formula": "0.3*coverage_label + 0.4*time_label + 0.3*decision_label",
            "regime_priority_formula": "0.35*family_importance + 0.25*coverage_label + 0.25*time_label + 0.15*local_decision_label",
            "notes": "Observed coverage/time ratios come from kernel invocation membership and weighted elapsed cycles; qualitative labels remain provisional.",
        },
        "anchors": anchors,
        "families": families,
        "regimes": regimes,
        "lanes": lanes,
    }


def _markdown_table(records: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|---" * len(columns) + "|"
    rows = []
    for record in records:
        rows.append("| " + " | ".join(str(record.get(column, "")) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def render_markdown_snapshots(bundle: dict[str, Any]) -> dict[str, str]:
    metadata = bundle["metadata"]
    header = [
        f"# mini_transformer_v4 Middle Layer Artifacts",
        "",
        f"- workload: `{metadata['workload']}`",
        f"- builder: `{metadata['builder']}`",
        f"- importance formula: `{metadata['importance_formula']}`",
        f"- regime priority formula: `{metadata['regime_priority_formula']}`",
        "",
    ]
    anchors_md = "\n".join(
        header
        + [
            "## Anchors",
            "",
            _markdown_table(
                bundle["anchors"],
                [
                    "anchor_id",
                    "phase_id",
                    "context_scope",
                    "member_invocations",
                    "observed_coverage_ratio",
                    "observed_time_ratio",
                    "coverage_label",
                    "time_label",
                    "route_hint",
                    "template_hint",
                ],
            ),
            "",
        ]
    )
    families_md = "\n".join(
        header
        + [
            "## Families",
            "",
            _markdown_table(
                bundle["families"],
                [
                    "family_id",
                    "input_anchor_ids",
                    "observed_coverage_ratio",
                    "observed_time_ratio",
                    "coverage_label",
                    "time_label",
                    "decision_label",
                    "importance_score",
                    "priority_class",
                ],
            ),
            "",
        ]
    )
    regimes_md = "\n".join(
        header
        + [
            "## Regimes",
            "",
            _markdown_table(
                bundle["regimes"],
                [
                    "regime_id",
                    "family_id",
                    "source_anchor_ids",
                    "observed_coverage_ratio",
                    "observed_time_ratio",
                    "local_decision_label",
                    "regime_priority_score",
                    "validation_status",
                ],
            ),
            "",
        ]
    )
    lanes_md = "\n".join(
        header
        + [
            "## Lanes",
            "",
            _markdown_table(
                bundle["lanes"],
                [
                    "lane_id",
                    "target_regime_id",
                    "target_family_id",
                    "parameter_direction",
                    "baseline_type",
                    "validation_metric",
                ],
            ),
            "",
        ]
    )
    return {
        "anchors.md": anchors_md,
        "families.md": families_md,
        "regimes.md": regimes_md,
        "lanes.md": lanes_md,
    }


def write_middle_layer_artifacts(bundle: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("anchors", "families", "regimes", "lanes"):
        (output_dir / f"{name}.json").write_text(json.dumps(bundle[name], indent=2, ensure_ascii=True) + "\n")
    (output_dir / "bundle.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=True) + "\n")
    for name, text in render_markdown_snapshots(bundle).items():
        (output_dir / name).write_text(text + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build middle-layer artifacts for mini_transformer_v4")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated middle-layer artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_middle_layer_artifacts(REPO_ROOT)
    write_middle_layer_artifacts(bundle, args.output_dir)
    print(f"wrote middle-layer artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
