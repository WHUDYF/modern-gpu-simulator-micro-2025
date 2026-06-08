import pytest

from experiments.gcl_phase_b.tuning import generate_gate8_tuning_vectors
from experiments.gcl_phase_b.utils import stable_hash


def _gate7_report():
    anchor_hash = stable_hash(_anchor_table())
    return {
        "artifact_type": "gcl_resnet50_gate7_cluster_correctness_manifest",
        "artifact_version": "gate7_cluster_correctness_manifest_v1",
        "claim_status": "quantified_no_correctness_claim",
        "threshold_policy": "report_only_v1",
        "family_alignment_metrics": {"weighted_purity": 0.95},
        "representative_quality_metrics": {
            "representative_p95_distance": 0.2,
            "high_weight_outlier_count": 0,
        },
        "metric_error_report": {"global_weighted_mape": 0.08, "status": "reported"},
        "family_alignment_report_hash": "family-report-hash",
        "metric_error_report_hash": "metric-report-hash",
        "source_representative_anchor_table_hash": anchor_hash,
        "gate7_cluster_correctness_manifest_hash": "gate7-hash",
    }


def _family_report():
    return {
        "artifact_type": "gcl_resnet50_cluster_family_alignment_report",
        "report_hash": "family-report-hash",
        "report_payload": {"weighted_purity": 0.95},
    }


def _metric_report():
    return {
        "artifact_type": "gcl_resnet50_cluster_metric_error_report",
        "report_hash": "metric-report-hash",
        "report_payload": {"global_weighted_mape": 0.08},
    }


def _anchor_table():
    return {
        "artifact_type": "gcl_resnet50_representative_anchor_table",
        "anchors": [{"cluster_id": 0, "representative_record_id": "a", "kernel_invocation_id": "k0"}],
    }


def test_gate8_generates_tuning_vectors_from_trusted_clusters():
    report = generate_gate8_tuning_vectors(
        _gate7_report(),
        representative_anchor_table=_anchor_table(),
        family_alignment_report=_family_report(),
        metric_error_report=_metric_report(),
        tunable_component_schema={"components": ["memory_latency_scale", "compute_latency_scale"]},
    )

    assert report["artifact_type"] == "gcl_resnet50_gate8_tuning_vector_proposal"
    assert report["extension_label"] == "our_extension_not_original_gcl_sampler"
    assert report["source_gate7_correctness_manifest_hash"] == "gate7-hash"
    assert report["cluster_tuning_vector_table"]["artifact_type"] == (
        "gcl_resnet50_cluster_tuning_vector_table"
    )
    assert report["tuning_vector_provenance_report"]["source_gate7_correctness_manifest_hash"] == (
        "gate7-hash"
    )
    assert report["tuning_safety_report"]["safety_status"] == "report_only_initial_vectors"
    assert report["gate8_tuning_manifest"]["artifact_type"] == "gcl_resnet50_gate8_tuning_manifest"
    vector = report["proposals"][0]
    assert vector["representative_anchor_hash"]
    assert vector["representative_anchor_table_hash"] == stable_hash(_anchor_table())
    assert vector["family_alignment_evidence_hash"] == "family-report-hash"
    assert vector["metric_error_evidence_hash"] == "metric-report-hash"
    assert vector["gate7_correctness_manifest_hash"] == "gate7-hash"
    assert vector["tuning_vector"]["memory_latency_scale"] == 1.0
    assert vector["tuning_vector"]["compute_latency_scale"] == 1.0


def test_gate8_rejects_high_weight_mixed_or_high_error_clusters():
    gate7 = _gate7_report()
    gate7["family_alignment_metrics"]["weighted_purity"] = 0.4

    with pytest.raises(ValueError, match="mixed-family"):
        generate_gate8_tuning_vectors(
            gate7,
            representative_anchor_table=_anchor_table(),
            family_alignment_report=_family_report(),
            metric_error_report=_metric_report(),
            tunable_component_schema={"components": ["x"]},
        )

    gate7 = _gate7_report()
    gate7["metric_error_report"]["global_weighted_mape"] = 0.5
    with pytest.raises(ValueError, match="high-error"):
        generate_gate8_tuning_vectors(
            gate7,
            representative_anchor_table=_anchor_table(),
            family_alignment_report=_family_report(),
            metric_error_report=_metric_report(),
            tunable_component_schema={"components": ["x"]},
        )


def test_gate8_rejects_mismatched_persisted_gate7_reports():
    report = _family_report()
    report["report_hash"] = "wrong"

    with pytest.raises(ValueError, match="family alignment report"):
        generate_gate8_tuning_vectors(
            _gate7_report(),
            representative_anchor_table=_anchor_table(),
            family_alignment_report=report,
            metric_error_report=_metric_report(),
            tunable_component_schema={"components": ["x"]},
        )


def test_gate8_rejects_representative_anchor_hash_mismatch_against_gate7():
    anchors = _anchor_table()
    anchors["representative_anchor_table_hash"] = "wrong-anchor-hash"

    with pytest.raises(ValueError, match="representative anchor"):
        generate_gate8_tuning_vectors(
            _gate7_report(),
            representative_anchor_table=anchors,
            family_alignment_report=_family_report(),
            metric_error_report=_metric_report(),
            tunable_component_schema={"components": ["x"]},
        )


def test_gate8_rejects_anchor_content_mutation_with_stale_hash_string():
    anchors = _anchor_table()
    gate7 = _gate7_report()
    anchors["anchors"] = [
        {"cluster_id": 0, "representative_record_id": "tampered", "kernel_invocation_id": "k9"}
    ]
    anchors["representative_anchor_table_hash"] = gate7[
        "source_representative_anchor_table_hash"
    ]

    with pytest.raises(ValueError, match="representative anchor table content"):
        generate_gate8_tuning_vectors(
            gate7,
            representative_anchor_table=anchors,
            family_alignment_report=_family_report(),
            metric_error_report=_metric_report(),
            tunable_component_schema={"components": ["x"]},
        )
