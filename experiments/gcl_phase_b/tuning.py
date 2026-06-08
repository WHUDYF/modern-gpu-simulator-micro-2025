"""Gate 8 tuning vector proposal extension."""

from __future__ import annotations

from typing import Any

from .utils import stable_hash

EXTENSION_LABEL = "our_extension_not_original_gcl_sampler"


def generate_gate8_tuning_vectors(
    gate7_report: dict[str, Any],
    *,
    representative_anchor_table: dict[str, Any],
    family_alignment_report: dict[str, Any],
    metric_error_report: dict[str, Any],
    tunable_component_schema: dict[str, Any],
) -> dict[str, Any]:
    if gate7_report.get("artifact_type") != "gcl_resnet50_gate7_cluster_correctness_manifest":
        raise ValueError("Gate8 requires Gate7 correctness manifest")
    if gate7_report.get("claim_status") != "quantified_no_correctness_claim":
        raise ValueError("Gate8 requires quantified report-only Gate7 claim status")
    if family_alignment_report.get("report_hash") != gate7_report.get(
        "family_alignment_report_hash"
    ):
        raise ValueError("family alignment report hash does not match Gate7 manifest")
    if metric_error_report.get("report_hash") != gate7_report.get("metric_error_report_hash"):
        raise ValueError("metric error report hash does not match Gate7 manifest")
    representative_anchors = list(representative_anchor_table.get("anchors", []))
    computed_anchor_hash = _representative_anchor_table_content_hash(representative_anchor_table)
    supplied_anchor_hash = representative_anchor_table.get("representative_anchor_table_hash")
    representative_anchor_table_hash = computed_anchor_hash
    expected_anchor_hash = gate7_report.get("source_representative_anchor_table_hash")
    if supplied_anchor_hash and supplied_anchor_hash != computed_anchor_hash:
        raise ValueError("representative anchor table content does not match supplied hash")
    if expected_anchor_hash and computed_anchor_hash != expected_anchor_hash:
        raise ValueError("representative anchor table hash does not match Gate7 manifest")
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
            "representative_anchor_hash": stable_hash(anchor),
            "representative_anchor_table_hash": representative_anchor_table_hash,
            "family_alignment_evidence_hash": family_alignment_report["report_hash"],
            "metric_error_evidence_hash": metric_error_report["report_hash"],
            "gate7_correctness_manifest_hash": gate7_report[
                "gate7_cluster_correctness_manifest_hash"
            ],
            "tuning_vector": {component: 1.0 for component in components},
            "proposal_status": "report_only_initial_vector",
        }
        for anchor in representative_anchors
    ]
    cluster_tuning_vector_table = {
        "artifact_type": "gcl_resnet50_cluster_tuning_vector_table",
        "artifact_version": "gate8_cluster_tuning_vector_table_v1",
        "source_gate7_correctness_manifest_hash": gate7_report[
            "gate7_cluster_correctness_manifest_hash"
        ],
        "tuning_vectors": proposals,
    }
    cluster_tuning_vector_table["cluster_tuning_vector_table_hash"] = stable_hash(
        cluster_tuning_vector_table
    )
    provenance_report = {
        "artifact_type": "gcl_resnet50_tuning_vector_provenance_report",
        "artifact_version": "gate8_tuning_vector_provenance_report_v1",
        "source_gate7_correctness_manifest_hash": gate7_report[
            "gate7_cluster_correctness_manifest_hash"
        ],
        "source_claim_status": gate7_report["claim_status"],
        "source_family_alignment_report_hash": family_alignment_report["report_hash"],
        "source_metric_error_report_hash": metric_error_report["report_hash"],
        "representative_anchor_table_hash": representative_anchor_table_hash,
        "representative_anchor_count": len(representative_anchors),
        "tunable_component_schema": tunable_component_schema,
    }
    provenance_report["tuning_vector_provenance_report_hash"] = stable_hash(provenance_report)
    safety_report = {
        "artifact_type": "gcl_resnet50_tuning_safety_report",
        "artifact_version": "gate8_tuning_safety_report_v1",
        "safety_status": "report_only_initial_vectors",
        "mixed_family_rejection_policy": "reject_weighted_purity_below_0_8",
        "metric_error_rejection_policy": "reject_global_weighted_mape_above_0_2",
        "accuracy_claim": "not_claimed",
    }
    safety_report["tuning_safety_report_hash"] = stable_hash(safety_report)
    gate8_manifest = {
        "artifact_type": "gcl_resnet50_gate8_tuning_manifest",
        "artifact_version": "gate8_tuning_manifest_v1",
        "extension_label": EXTENSION_LABEL,
        "source_gate7_correctness_manifest_hash": gate7_report[
            "gate7_cluster_correctness_manifest_hash"
        ],
        "source_family_alignment_report_hash": family_alignment_report["report_hash"],
        "source_metric_error_report_hash": metric_error_report["report_hash"],
        "representative_anchor_table_hash": representative_anchor_table_hash,
        "cluster_tuning_vector_table_hash": cluster_tuning_vector_table[
            "cluster_tuning_vector_table_hash"
        ],
        "tuning_vector_provenance_report_hash": provenance_report[
            "tuning_vector_provenance_report_hash"
        ],
        "tuning_safety_report_hash": safety_report["tuning_safety_report_hash"],
    }
    gate8_manifest["gate8_tuning_manifest_hash"] = stable_hash(gate8_manifest)
    artifact = {
        "artifact_type": "gcl_resnet50_gate8_tuning_vector_proposal",
        "artifact_version": "gate8_tuning_vector_proposal_v1",
        "extension_label": EXTENSION_LABEL,
        "source_gate7_correctness_manifest_hash": gate7_report[
            "gate7_cluster_correctness_manifest_hash"
        ],
        "tunable_component_schema": tunable_component_schema,
        "proposals": proposals,
        "cluster_tuning_vector_table": cluster_tuning_vector_table,
        "tuning_vector_provenance_report": provenance_report,
        "tuning_safety_report": safety_report,
        "gate8_tuning_manifest": gate8_manifest,
    }
    artifact["gate8_tuning_vector_proposal_hash"] = stable_hash(artifact)
    return artifact


def _representative_anchor_table_content_hash(table: dict[str, Any]) -> str:
    return stable_hash({
        key: value
        for key, value in table.items()
        if key != "representative_anchor_table_hash"
    })
