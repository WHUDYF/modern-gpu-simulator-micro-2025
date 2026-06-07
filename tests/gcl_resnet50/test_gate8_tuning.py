import pytest

from experiments.gcl_phase_b.tuning import generate_gate8_tuning_vectors


def _gate7_report():
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
        "gate7_cluster_correctness_manifest_hash": "gate7-hash",
    }


def test_gate8_generates_tuning_vectors_from_trusted_clusters():
    report = generate_gate8_tuning_vectors(
        _gate7_report(),
        representative_anchors=[
            {"cluster_id": 0, "representative_record_id": "a", "kernel_invocation_id": "k0"}
        ],
        tunable_component_schema={"components": ["memory_latency_scale", "compute_latency_scale"]},
    )

    assert report["artifact_type"] == "gcl_resnet50_gate8_tuning_vector_proposal"
    assert report["extension_label"] == "our_extension_not_original_gcl_sampler"
    assert report["source_gate7_correctness_manifest_hash"] == "gate7-hash"
    assert report["proposals"][0]["tuning_vector"]["memory_latency_scale"] == 1.0
    assert report["proposals"][0]["tuning_vector"]["compute_latency_scale"] == 1.0


def test_gate8_rejects_high_weight_mixed_or_high_error_clusters():
    gate7 = _gate7_report()
    gate7["family_alignment_metrics"]["weighted_purity"] = 0.4

    with pytest.raises(ValueError, match="mixed-family"):
        generate_gate8_tuning_vectors(
            gate7,
            representative_anchors=[{"cluster_id": 0}],
            tunable_component_schema={"components": ["x"]},
        )

    gate7 = _gate7_report()
    gate7["metric_error_report"]["global_weighted_mape"] = 0.5
    with pytest.raises(ValueError, match="high-error"):
        generate_gate8_tuning_vectors(
            gate7,
            representative_anchors=[{"cluster_id": 0}],
            tunable_component_schema={"components": ["x"]},
        )
