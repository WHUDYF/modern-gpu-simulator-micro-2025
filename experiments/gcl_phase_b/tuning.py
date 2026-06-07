"""Gate 8 tuning vector proposal extension."""

from __future__ import annotations

from typing import Any

from .utils import stable_hash

EXTENSION_LABEL = "our_extension_not_original_gcl_sampler"


def generate_gate8_tuning_vectors(
    gate7_report: dict[str, Any],
    *,
    representative_anchors: list[dict[str, Any]],
    tunable_component_schema: dict[str, Any],
) -> dict[str, Any]:
    if gate7_report.get("artifact_type") != "gcl_resnet50_gate7_cluster_correctness_manifest":
        raise ValueError("Gate8 requires Gate7 correctness manifest")
    if gate7_report.get("claim_status") != "quantified_no_correctness_claim":
        raise ValueError("Gate8 requires quantified report-only Gate7 claim status")
    weighted_purity = gate7_report.get("family_alignment_metrics", {}).get("weighted_purity")
    if weighted_purity is not None and float(weighted_purity) < 0.8:
        raise ValueError("mixed-family cluster evidence cannot enter Gate8 tuning proposal")
    metric_error = gate7_report.get("metric_error_report", {}).get("global_weighted_mape")
    if metric_error is not None and float(metric_error) > 0.2:
        raise ValueError("high-error cluster evidence cannot enter Gate8 tuning proposal")
    quality = gate7_report.get("representative_quality_metrics", {})
    if int(quality.get("high_weight_outlier_count", 0)) > 0:
        raise ValueError("weak representative cluster cannot enter Gate8 tuning proposal")
    components = list(tunable_component_schema.get("components", []))
    if not components:
        raise ValueError("tunable component schema must contain components")
    proposals = [
        {
            "cluster_id": anchor.get("cluster_id"),
            "representative_record_id": anchor.get("representative_record_id"),
            "kernel_invocation_id": anchor.get("kernel_invocation_id"),
            "tuning_vector": {component: 1.0 for component in components},
            "proposal_status": "report_only_initial_vector",
        }
        for anchor in representative_anchors
    ]
    artifact = {
        "artifact_type": "gcl_resnet50_gate8_tuning_vector_proposal",
        "artifact_version": "gate8_tuning_vector_proposal_v1",
        "extension_label": EXTENSION_LABEL,
        "source_gate7_correctness_manifest_hash": gate7_report[
            "gate7_cluster_correctness_manifest_hash"
        ],
        "tunable_component_schema": tunable_component_schema,
        "proposals": proposals,
    }
    artifact["gate8_tuning_vector_proposal_hash"] = stable_hash(artifact)
    return artifact
