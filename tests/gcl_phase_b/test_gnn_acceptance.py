import json

import pytest

from experiments.gcl_phase_b.trustworthiness import (
    render_gnn_acceptance_markdown,
    evaluate_gnn_acceptance,
    evaluate_gnn_acceptance_from_dir,
    validate_gnn_acceptance_manifest,
    write_gnn_acceptance_artifacts,
)


def _full_trace_manifest(**overrides):
    manifest = {
        "artifact_type": "gcl_resnet50_full_trace_reproduction_manifest",
        "artifact_version": "full_trace_reproduction_manifest_v1",
        "run_scope": "real_resnet50_full_trace",
        "formal_full_trace_run": True,
        "input_kernel_invocation_count": 265,
        "input_cta_record_count": 124876,
        "final_gate": "gate9_report_only",
        "full_trace_reproduction_manifest_hash": "full-trace-hash",
    }
    manifest.update(overrides)
    return manifest


def _training_manifest(**overrides):
    manifest = {
        "artifact_type": "gcl_resnet50_rgcn_training_run_manifest",
        "artifact_version": "gate5_rgcn_training_run_manifest_v1",
        "training_status": "formal_gate5_complete",
        "train_graph_count": 4,
        "export_graph_count": 265,
        "optimizer_config": {
            "optimizer": "Adam",
            "learning_rate": 0.005,
            "optimizer_step_count": 1,
        },
        "model_architecture": {
            "layers": 3,
            "input_dim": 64,
            "hidden_dim": 128,
            "kernel_embedding_dim": 256,
            "projection_hidden_dim": 128,
            "projection_output_dim": 64,
            "relation_count": 3,
        },
        "edge_relation_schema": {
            "control_flow": 0,
            "data_source": 1,
            "data_destination": 2,
        },
        "readout_hierarchy": "node_to_warp_to_cta_to_selected_sm_to_kernel",
        "representation_mode": "gcl_resnet50_mem_ref_only",
        "pseudo_node_mode": "mem_ref_only",
        "source_graph_tensor_bundle_hash": "tensor-hash",
        "training_run_manifest_hash": "training-hash",
    }
    manifest.update(overrides)
    return manifest


def _selector_artifacts(**overrides):
    artifact = {
        "artifact_type": "gcl_resnet50_gate6_selector_artifacts",
        "artifact_version": "gate6_selector_artifacts_v1",
        "source_embedding_table_hash": "embedding-hash",
        "selector_manifest_hash": "selector-hash",
        "k_selection_report": {
            "artifact_type": "gcl_resnet50_k_selection_report",
            "mode": "silhouette_k",
            "selected_k": 2,
            "selected_score": 0.53412531,
        },
        "kmeans_cluster_assignment_table": {
            "artifact_type": "gcl_resnet50_kmeans_cluster_assignment_table",
            "assignments": [
                {"record_id": "gcl_embedding:0000", "cluster_id": 0},
                {"record_id": "gcl_embedding:0001", "cluster_id": 0},
                {"record_id": "gcl_embedding:0002", "cluster_id": 1},
            ],
        },
        "representative_anchor_table": {
            "artifact_type": "gcl_resnet50_representative_anchor_table",
            "anchors": [
                {
                    "cluster_id": 0,
                    "representative_record_id": "gcl_embedding:0000",
                    "kernel_invocation_id": "d_0_s_0_k_333",
                },
                {
                    "cluster_id": 1,
                    "representative_record_id": "gcl_embedding:0002",
                    "kernel_invocation_id": "d_0_s_0_k_269",
                },
            ],
        },
    }
    artifact.update(overrides)
    return artifact


def _gate7_manifest(**overrides):
    manifest = {
        "artifact_type": "gcl_resnet50_gate7_cluster_correctness_manifest",
        "artifact_version": "gate7_cluster_correctness_manifest_v1",
        "assignment_count": 265,
        "claim_status": "quantified_no_correctness_claim",
        "gate7_cluster_correctness_manifest_hash": "gate7-hash",
        "embedding_geometry_metrics": {
            "silhouette": 0.48186617,
            "davies_bouldin": 0.78974237,
            "calinski_harabasz": 10.30193812,
            "intra_distance_mean": 0.9376443638426335,
            "inter_distance_mean": 1.8906091086825012,
            "inter_intra_ratio": 2.01633922,
        },
        "family_alignment_metrics": {
            "family_evidence_status": "available",
            "family_alignment_claim_status": "reported",
            "cluster_purity": 1.0,
            "weighted_purity": 1.0,
            "ari": 0.0,
            "nmi": 0.0,
            "homogeneity": 1.0,
            "completeness": 0.0,
            "v_measure": 0.0,
            "family_to_cluster_coverage": {
                "resnet50_real_trace": {
                    "cluster_count": 2,
                    "coverage_ratio": 0.91765889,
                    "primary_cluster_id": 0,
                }
            },
        },
        "metric_error_report": {
            "metric_claim_status": "unavailable",
            "status": "not_provided",
            "complete_row_count": 0,
        },
        "stability_report": {
            "stability_status": "single_run_not_evaluated",
            "assignment_stability_ari": None,
            "assignment_stability_nmi": None,
            "centroid_drift": None,
            "k_stability": None,
            "representative_stability_rate": None,
        },
    }
    manifest.update(overrides)
    return manifest


def _minimal_inputs(**overrides):
    values = {
        "full_trace_manifest": _full_trace_manifest(),
        "training_manifest": _training_manifest(),
        "selector_artifacts": _selector_artifacts(),
        "gate7_manifest": _gate7_manifest(),
    }
    values.update(overrides)
    return values


def _complete_baseline_report(**overrides):
    report = {
        "artifact_type": "gcl_gnn_baseline_ablation_report",
        "full_rgcn": {
            "silhouette": 0.8,
            "davies_bouldin": 0.4,
            "inter_intra_ratio": 3.0,
            "assignment_stability_ari": 0.9,
            "representative_metric_error": 0.1,
        },
        "random_embedding_baseline": {
            "silhouette": 0.1,
            "davies_bouldin": 2.0,
            "inter_intra_ratio": 1.0,
            "assignment_stability_ari": 0.2,
            "representative_metric_error": 0.8,
        },
        "opcode_histogram_baseline": {
            "silhouette": 0.15,
            "davies_bouldin": 1.8,
            "inter_intra_ratio": 1.2,
            "assignment_stability_ari": 0.3,
            "representative_metric_error": 0.7,
        },
        "node_feature_pooling_no_edge_baseline": {
            "silhouette": 0.2,
            "davies_bouldin": 1.5,
            "inter_intra_ratio": 1.1,
            "assignment_stability_ari": 0.35,
            "representative_metric_error": 0.6,
        },
        "control_flow_only_rgcn": {
            "silhouette": 0.3,
            "davies_bouldin": 1.2,
            "inter_intra_ratio": 1.3,
            "assignment_stability_ari": 0.45,
            "representative_metric_error": 0.5,
        },
        "data_flow_only_rgcn": {
            "silhouette": 0.35,
            "davies_bouldin": 1.1,
            "inter_intra_ratio": 1.4,
            "assignment_stability_ari": 0.5,
            "representative_metric_error": 0.45,
        },
    }
    report.update(overrides)
    return report


def _adequate_training_manifest():
    return _training_manifest(
        train_graph_count=128,
        epoch_count=3,
        positive_pair_count=128,
        negative_pair_count=512,
        loss_curve=[1.2, 0.8, 0.5],
        optimizer_config={
            "optimizer": "Adam",
            "learning_rate": 0.005,
            "optimizer_step_count": 100,
        },
    )


def _stable_gate7_manifest(**overrides):
    gate7 = _gate7_manifest(
        stability_report={
            "stability_status": "multi_seed_evaluated",
            "training_seed_count": 3,
            "kmeans_seed_count": 5,
            "assignment_stability_ari": 0.91,
            "assignment_stability_nmi": 0.93,
            "centroid_drift": 0.04,
            "k_stability": 1.0,
            "representative_stability_rate": 0.9,
        }
    )
    gate7.update(overrides)
    return gate7


def _semantic_gate7_manifest(**overrides):
    gate7 = _stable_gate7_manifest(
        family_alignment_metrics={
            "family_evidence_status": "available",
            "family_alignment_claim_status": "reported",
            "cluster_purity": 0.9,
            "weighted_purity": 0.9,
            "ari": 0.8,
            "nmi": 0.82,
            "homogeneity": 0.84,
            "completeness": 0.83,
            "v_measure": 0.835,
            "family_to_cluster_coverage": {"conv": {}, "activation": {}},
        }
    )
    gate7.update(overrides)
    return gate7


def _downstream_gate7_manifest(**overrides):
    gate7 = _semantic_gate7_manifest(
        metric_source_manifest_hash="metric-source-hash",
        metric_error_report={
            "metric_claim_status": "reported",
            "cluster_weighted_mape": {"0": 0.1},
            "global_weighted_mape": 0.1,
            "global_p95_relative_error": 0.2,
        },
        representative_quality_metrics={
            "representative_quality_status": "reported",
            "cluster_reports": [
                {
                    "cluster_id": 0,
                    "mean_distance_to_representative": 0.1,
                    "p95_distance_to_representative": 0.2,
                    "max_distance_to_representative": 0.3,
                    "representative_rank_to_centroid": 1,
                    "outlier_member_ratio": 0.0,
                }
            ],
        },
    )
    gate7.update(overrides)
    return gate7


def test_acceptance_evaluator_accepts_real_resnet50_full_trace_artifacts():
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    assert report["acceptance_items"]["input_provenance"]["status"] == "PASS"


def test_acceptance_evaluator_rejects_synthetic_or_debug_input():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            full_trace_manifest=_full_trace_manifest(
                run_scope="synthetic_fixture",
                formal_full_trace_run=False,
            )
        )
    )

    assert report["acceptance_items"]["input_provenance"]["status"] == "FAIL"
    assert "input provenance is not a formal real full-trace run" in report["blocking_gaps"]


def test_rgcn_structure_acceptance_passes_for_gate5_contract():
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    assert report["acceptance_items"]["rgcn_structure"]["status"] == "PASS"


def test_rgcn_structure_acceptance_rejects_pooling_only_or_projection_output_selector():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            training_manifest=_training_manifest(
                model_architecture={
                    "layers": 0,
                    "input_dim": 64,
                    "hidden_dim": 128,
                    "kernel_embedding_dim": 64,
                    "projection_output_dim": 64,
                    "relation_count": 0,
                },
                selector_embedding_source="projection_head_output",
            )
        )
    )

    assert report["acceptance_items"]["rgcn_structure"]["status"] == "FAIL"


def test_training_adequacy_fails_for_single_step_smoke_training():
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    assert report["acceptance_items"]["training_adequacy"]["status"] == "FAIL"
    assert "training is smoke-level or lacks training-curve evidence" in report["blocking_gaps"]


def test_training_adequacy_cannot_pass_without_loss_curve_and_multi_step_training():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            training_manifest=_training_manifest(
                train_graph_count=256,
                optimizer_config={
                    "optimizer": "Adam",
                    "learning_rate": 0.005,
                    "optimizer_step_count": 100,
                },
            )
        )
    )

    assert report["acceptance_items"]["training_adequacy"]["status"] == "FAIL"


def test_embedding_geometry_weak_pass_with_current_gate7_metrics():
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    assert report["acceptance_items"]["embedding_geometry_signal"]["status"] == "WEAK_PASS"


def test_embedding_geometry_cannot_be_final_pass_without_baseline_ablation():
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    assert report["acceptance_items"]["embedding_geometry_signal"]["status"] != "PASS"
    assert report["claim_status"] == "quantified_no_correctness_claim"


def test_missing_baseline_ablation_is_not_available():
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    assert report["acceptance_items"]["baseline_ablation"]["status"] == "NOT_AVAILABLE"


def test_baseline_ablation_accepts_spec_no_edge_key_when_full_rgcn_beats_all_metrics():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(baseline_ablation_report=_complete_baseline_report())
    )

    assert report["acceptance_items"]["baseline_ablation"]["status"] == "PASS"


def test_full_rgcn_must_not_pass_when_no_edge_baseline_matches():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            baseline_ablation_report=_complete_baseline_report(
                node_feature_pooling_no_edge_baseline={
                    "silhouette": 0.8,
                    "davies_bouldin": 0.4,
                    "inter_intra_ratio": 3.0,
                    "assignment_stability_ari": 0.9,
                    "representative_metric_error": 0.1,
                }
            )
        )
    )

    assert report["acceptance_items"]["baseline_ablation"]["status"] == "FAIL"
    assert report["gnn_acceptance_status"] == "rejected_no_graph_signal"


def test_incomplete_baseline_report_cannot_pass():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            baseline_ablation_report={
                "artifact_type": "gcl_gnn_baseline_ablation_report",
                "full_rgcn": {"silhouette": 0.8, "inter_intra_ratio": 3.0},
            }
        )
    )

    assert report["acceptance_items"]["baseline_ablation"]["status"] == "FAIL"
    assert report["gnn_acceptance_status"] == "rejected_no_graph_signal"


def test_baseline_ablation_rejects_when_full_rgcn_loses_on_davies_bouldin():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            baseline_ablation_report=_complete_baseline_report(
                node_feature_pooling_no_edge_baseline={
                    "silhouette": 0.2,
                    "davies_bouldin": 0.3,
                    "inter_intra_ratio": 1.1,
                    "assignment_stability_ari": 0.35,
                    "representative_metric_error": 0.6,
                }
            )
        )
    )

    assert report["acceptance_items"]["baseline_ablation"]["status"] == "FAIL"


def test_baseline_ablation_rejects_when_full_rgcn_loses_on_assignment_stability():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            baseline_ablation_report=_complete_baseline_report(
                data_flow_only_rgcn={
                    "silhouette": 0.35,
                    "davies_bouldin": 1.1,
                    "inter_intra_ratio": 1.4,
                    "assignment_stability_ari": 0.95,
                    "representative_metric_error": 0.45,
                }
            )
        )
    )

    assert report["acceptance_items"]["baseline_ablation"]["status"] == "FAIL"


def test_baseline_ablation_rejects_when_full_rgcn_loses_on_representative_error():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            baseline_ablation_report=_complete_baseline_report(
                control_flow_only_rgcn={
                    "silhouette": 0.3,
                    "davies_bouldin": 1.2,
                    "inter_intra_ratio": 1.3,
                    "assignment_stability_ari": 0.45,
                    "representative_metric_error": 0.05,
                }
            )
        )
    )

    assert report["acceptance_items"]["baseline_ablation"]["status"] == "FAIL"


def test_single_run_stability_is_not_available():
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    assert report["acceptance_items"]["multi_seed_stability"]["status"] == "NOT_AVAILABLE"


def test_multi_seed_stability_accepts_minimum_seed_counts():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            gate7_manifest=_gate7_manifest(
                stability_report={
                    "stability_status": "multi_seed_evaluated",
                    "training_seed_count": 3,
                    "kmeans_seed_count": 5,
                    "assignment_stability_ari": 0.91,
                    "assignment_stability_nmi": 0.93,
                    "centroid_drift": 0.04,
                    "k_stability": 1.0,
                    "representative_stability_rate": 0.9,
                }
            )
        )
    )

    assert report["acceptance_items"]["multi_seed_stability"]["status"] == "PASS"


def test_multi_seed_stability_rejects_under_seeded_reports():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            gate7_manifest=_gate7_manifest(
                stability_report={
                    "stability_status": "multi_seed_evaluated",
                    "training_seed_count": 2,
                    "kmeans_seed_count": 5,
                    "assignment_stability_ari": 0.91,
                    "assignment_stability_nmi": 0.93,
                    "centroid_drift": 0.04,
                    "k_stability": 1.0,
                    "representative_stability_rate": 0.9,
                }
            )
        )
    )

    assert report["acceptance_items"]["multi_seed_stability"]["status"] == "FAIL"


def test_multi_seed_stability_rejects_unstable_metric_values():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            gate7_manifest=_gate7_manifest(
                stability_report={
                    "stability_status": "multi_seed_evaluated",
                    "training_seed_count": 3,
                    "kmeans_seed_count": 5,
                    "assignment_stability_ari": 0.01,
                    "assignment_stability_nmi": 0.02,
                    "centroid_drift": 999.0,
                    "k_stability": 0.0,
                    "representative_stability_rate": 0.0,
                }
            )
        )
    )

    assert report["acceptance_items"]["multi_seed_stability"]["status"] == "FAIL"
    assert report["gnn_acceptance_status"] == "rejected_unstable_embedding"


def test_coarse_resnet50_label_keeps_semantic_correctness_unproven():
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    assert report["acceptance_items"]["semantic_cluster_correctness"]["status"] == "UNPROVEN"


def test_purity_alone_cannot_upgrade_semantic_claim():
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    assert report["acceptance_items"]["semantic_cluster_correctness"]["status"] != "PASS"
    assert report["claim_status"] == "quantified_no_correctness_claim"


def test_missing_metric_rows_keep_downstream_usefulness_not_available():
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    assert (
        report["acceptance_items"]["downstream_representative_usefulness"]["status"]
        == "NOT_AVAILABLE"
    )


def test_downstream_claim_requires_error_metrics():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(gate7_manifest=_gate7_manifest(metric_error_report={}))
    )

    assert (
        report["acceptance_items"]["downstream_representative_usefulness"]["status"]
        != "PASS"
    )


def test_downstream_metrics_without_provenance_cannot_pass():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            gate7_manifest=_gate7_manifest(
                metric_error_report={
                    "metric_claim_status": "reported",
                    "cluster_weighted_mape": {"0": 0.1},
                    "global_weighted_mape": 0.1,
                    "global_p95_relative_error": 0.2,
                },
                representative_quality_metrics={
                    "representative_quality_status": "reported",
                    "cluster_reports": [
                        {
                            "cluster_id": 0,
                            "mean_distance_to_representative": 0.1,
                            "p95_distance_to_representative": 0.2,
                            "max_distance_to_representative": 0.3,
                            "representative_rank_to_centroid": 1,
                            "outlier_member_ratio": 0.0,
                        }
                    ],
                },
            )
        )
    )

    assert (
        report["acceptance_items"]["downstream_representative_usefulness"]["status"]
        == "FAIL"
    )


def test_downstream_metrics_reject_unusable_error_values():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            gate7_manifest=_gate7_manifest(
                metric_source_manifest_hash="metric-source-hash",
                metric_error_report={
                    "metric_claim_status": "reported",
                    "cluster_weighted_mape": {"0": 9999.0},
                    "global_weighted_mape": 9999.0,
                    "global_p95_relative_error": 9999.0,
                },
                representative_quality_metrics={
                    "representative_quality_status": "reported",
                    "cluster_reports": [
                        {
                            "cluster_id": 0,
                            "mean_distance_to_representative": 0.1,
                            "p95_distance_to_representative": 0.2,
                            "max_distance_to_representative": 0.3,
                            "representative_rank_to_centroid": 1,
                            "outlier_member_ratio": 0.0,
                        }
                    ],
                },
            )
        )
    )

    assert (
        report["acceptance_items"]["downstream_representative_usefulness"]["status"]
        == "FAIL"
    )
    assert report["gnn_acceptance_status"] == "rejected_downstream_unproven"


def test_current_resnet50_run_keeps_quantified_no_correctness_claim():
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    assert (
        report["gnn_acceptance_status"]
        == "weak_acceptance_structure_valid_but_correctness_unproven"
    )
    assert report["claim_status"] == "quantified_no_correctness_claim"


def test_claim_status_cannot_upgrade_with_any_blocking_gap():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            baseline_ablation_report={
                "artifact_type": "gcl_gnn_baseline_ablation_report",
                "full_rgcn": {"silhouette": 0.8, "inter_intra_ratio": 3.0},
                "no_edge_baseline": {"silhouette": 0.2, "inter_intra_ratio": 1.1},
                "random_embedding_baseline": {"silhouette": 0.1},
                "opcode_histogram_baseline": {"silhouette": 0.15},
            }
        )
    )

    assert report["acceptance_items"]["training_adequacy"]["status"] == "FAIL"
    assert report["claim_status"] == "quantified_no_correctness_claim"
    assert report["gnn_acceptance_status"] != "accepted"


def test_claim_status_uses_intermediate_ladder_when_evidence_layers_pass():
    stable_report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            training_manifest=_adequate_training_manifest(),
            gate7_manifest=_stable_gate7_manifest(),
            baseline_ablation_report=_complete_baseline_report(),
        )
    )

    assert stable_report["claim_status"] == "cluster_stability_supported"

    semantic_report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            training_manifest=_adequate_training_manifest(),
            baseline_ablation_report=_complete_baseline_report(),
            gate7_manifest=_semantic_gate7_manifest(),
        )
    )

    assert semantic_report["claim_status"] == "semantic_cluster_supported"


def test_claim_status_does_not_upgrade_beyond_stability_without_semantic_pass():
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            training_manifest=_adequate_training_manifest(),
            baseline_ablation_report=_complete_baseline_report(),
            gate7_manifest=_stable_gate7_manifest(),
        )
    )

    assert report["acceptance_items"]["multi_seed_stability"]["status"] == "PASS"
    assert report["acceptance_items"]["semantic_cluster_correctness"]["status"] != "PASS"
    assert report["claim_status"] == "cluster_stability_supported"


def test_downstream_pass_cannot_skip_missing_semantic_tier():
    downstream_without_semantics = _downstream_gate7_manifest(
        family_alignment_metrics={
            "family_evidence_status": "available",
            "family_alignment_claim_status": "reported",
            "cluster_purity": 1.0,
            "weighted_purity": 1.0,
            "ari": 0.0,
            "nmi": 0.0,
            "homogeneity": 1.0,
            "completeness": 0.0,
            "v_measure": 0.0,
            "family_to_cluster_coverage": {"resnet50_real_trace": {}},
        }
    )
    report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            training_manifest=_adequate_training_manifest(),
            baseline_ablation_report=_complete_baseline_report(),
            gate7_manifest=downstream_without_semantics,
        )
    )

    assert (
        report["acceptance_items"]["downstream_representative_usefulness"]["status"]
        == "PASS"
    )
    assert report["acceptance_items"]["semantic_cluster_correctness"]["status"] != "PASS"
    assert report["claim_status"] == "cluster_stability_supported"


def test_claim_status_reaches_final_acceptance_only_after_all_tiers_pass():
    structure_report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            training_manifest=_adequate_training_manifest(),
            baseline_ablation_report=_complete_baseline_report(),
        )
    )
    stable_report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            training_manifest=_adequate_training_manifest(),
            baseline_ablation_report=_complete_baseline_report(),
            gate7_manifest=_stable_gate7_manifest(),
        )
    )
    semantic_report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            training_manifest=_adequate_training_manifest(),
            baseline_ablation_report=_complete_baseline_report(),
            gate7_manifest=_semantic_gate7_manifest(),
        )
    )
    downstream_report = evaluate_gnn_acceptance(
        **_minimal_inputs(
            training_manifest=_adequate_training_manifest(),
            baseline_ablation_report=_complete_baseline_report(),
            gate7_manifest=_downstream_gate7_manifest(),
        )
    )

    assert structure_report["claim_status"] == "structure_valid_embedding_signal_only"
    assert stable_report["claim_status"] == "cluster_stability_supported"
    assert semantic_report["claim_status"] == "semantic_cluster_supported"
    assert downstream_report["claim_status"] == "gnn_trustworthiness_accepted"
    assert downstream_report["gnn_acceptance_status"] == "accepted"


def test_acceptance_artifacts_written_with_hashes(tmp_path):
    report = evaluate_gnn_acceptance(**_minimal_inputs())

    written = write_gnn_acceptance_artifacts(tmp_path, report)

    manifest = json.loads((tmp_path / "gnn_acceptance_manifest.json").read_text())
    summary = json.loads((tmp_path / "gnn_acceptance_summary.json").read_text())
    markdown = (tmp_path / "gnn_acceptance_report.md").read_text()
    assert written["manifest_path"].name == "gnn_acceptance_manifest.json"
    assert manifest["report_hash"]
    assert manifest["input_artifact_hashes"]
    assert summary["gnn_acceptance_status"] == manifest["gnn_acceptance_status"]
    assert "weak_acceptance_structure_valid_but_correctness_unproven" in markdown
    validate_gnn_acceptance_manifest(manifest, summary=summary, markdown=markdown)


def test_acceptance_manifest_rejects_missing_hash_or_report_mismatch(tmp_path):
    report = evaluate_gnn_acceptance(**_minimal_inputs())
    write_gnn_acceptance_artifacts(tmp_path, report)
    manifest = json.loads((tmp_path / "gnn_acceptance_manifest.json").read_text())
    summary = json.loads((tmp_path / "gnn_acceptance_summary.json").read_text())

    manifest.pop("report_hash")
    with pytest.raises(ValueError, match="report_hash"):
        validate_gnn_acceptance_manifest(manifest, summary=summary, markdown="changed")


def test_missing_artifact_acceptance_manifest_is_validator_compatible(tmp_path):
    full_manifest = _full_trace_manifest()
    full_manifest["full_trace_reproduction_manifest_hash"] = "full-trace-hash"
    (tmp_path / "resnet50_full_trace_reproduction_manifest.json").write_text(
        json.dumps(full_manifest),
        encoding="utf-8",
    )

    report = evaluate_gnn_acceptance_from_dir(tmp_path)
    write_gnn_acceptance_artifacts(tmp_path, report)

    manifest = json.loads((tmp_path / "gnn_acceptance_manifest.json").read_text())
    summary = json.loads((tmp_path / "gnn_acceptance_summary.json").read_text())
    markdown = (tmp_path / "gnn_acceptance_report.md").read_text()
    assert manifest["gnn_acceptance_status"] == "not_evaluable_missing_artifacts"
    validate_gnn_acceptance_manifest(manifest, summary=summary, markdown=markdown)


def test_missing_artifact_acceptance_manifest_preserves_present_input_hashes(tmp_path):
    full_manifest = _full_trace_manifest()
    full_manifest["full_trace_reproduction_manifest_hash"] = "full-trace-hash"
    training_manifest = _training_manifest()
    training_manifest["training_run_manifest_hash"] = "training-present-hash"
    (tmp_path / "resnet50_full_trace_reproduction_manifest.json").write_text(
        json.dumps(full_manifest),
        encoding="utf-8",
    )
    (tmp_path / "rgcn_training_run_manifest.json").write_text(
        json.dumps(training_manifest),
        encoding="utf-8",
    )

    report = evaluate_gnn_acceptance_from_dir(tmp_path)

    assert report["input_artifact_hashes"]["full_trace_manifest_hash"] == "full-trace-hash"
    assert report["input_artifact_hashes"]["training_run_manifest_hash"] == (
        "training-present-hash"
    )
    assert report["input_artifact_hashes"]["selector_manifest_hash"] == (
        "missing:selector_artifacts.json"
    )


def test_acceptance_manifest_rejects_missing_individual_source_hash(tmp_path):
    report = evaluate_gnn_acceptance(**_minimal_inputs())
    write_gnn_acceptance_artifacts(tmp_path, report)
    manifest = json.loads((tmp_path / "gnn_acceptance_manifest.json").read_text())
    summary = json.loads((tmp_path / "gnn_acceptance_summary.json").read_text())
    markdown = (tmp_path / "gnn_acceptance_report.md").read_text()
    manifest["input_artifact_hashes"]["selector_manifest_hash"] = None

    with pytest.raises(ValueError, match="selector_manifest_hash"):
        validate_gnn_acceptance_manifest(manifest, summary=summary, markdown=markdown)


def test_acceptance_manifest_rejects_stale_summary_hash(tmp_path):
    report = evaluate_gnn_acceptance(**_minimal_inputs())
    write_gnn_acceptance_artifacts(tmp_path, report)
    manifest = json.loads((tmp_path / "gnn_acceptance_manifest.json").read_text())
    summary = json.loads((tmp_path / "gnn_acceptance_summary.json").read_text())
    markdown = (tmp_path / "gnn_acceptance_report.md").read_text()
    summary["manifest_hash"] = "stale"

    with pytest.raises(ValueError, match="manifest_hash"):
        validate_gnn_acceptance_manifest(manifest, summary=summary, markdown=markdown)


def test_acceptance_manifest_rejects_tampered_manifest_hash(tmp_path):
    report = evaluate_gnn_acceptance(**_minimal_inputs())
    write_gnn_acceptance_artifacts(tmp_path, report)
    manifest = json.loads((tmp_path / "gnn_acceptance_manifest.json").read_text())
    summary = json.loads((tmp_path / "gnn_acceptance_summary.json").read_text())
    markdown = (tmp_path / "gnn_acceptance_report.md").read_text()
    manifest["claim_status"] = "gnn_trustworthiness_accepted"

    with pytest.raises(ValueError, match="manifest_hash"):
        validate_gnn_acceptance_manifest(manifest, summary=summary, markdown=markdown)


def test_rendered_markdown_hash_changes_when_status_changes():
    report = evaluate_gnn_acceptance(**_minimal_inputs())
    original = render_gnn_acceptance_markdown(report)
    changed = dict(report)
    changed["claim_status"] = "gnn_trustworthiness_accepted"

    assert render_gnn_acceptance_markdown(changed) != original
