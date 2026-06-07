from pathlib import Path
import json

from experiments.gcl_phase_b.resnet50_gate0 import (
    record_resnet50_gate0_trace_acquisition,
    write_resnet50_gate0_blocker_report,
)
from experiments.gcl_phase_b.resnet50_gate_pipeline import (
    GATE1_7_PIPELINE_MANIFEST_FILENAME,
    _emit_gate8_gate9_extension_artifacts,
    run_resnet50_gate1_to_gate7,
)
from tests.gcl_resnet50.formal_fixture import write_minimal_artifact_shape_resnet50_root
from tests.gcl_resnet50.real_chain import FORMAL_ROOT, run_real_nondegenerate_gate1_to_gate7_artifacts


def _blocked_gate0_root(tmp_path):
    root = tmp_path / "blocked_gate0"
    root.mkdir()
    write_resnet50_gate0_blocker_report(
        root,
        reason="real ResNet-50 NVBit trace is not available",
        missing_requirements=["dynamic_trace.pb", "threadblocks/"],
    )
    return root


def test_resnet50_gate_pipeline_stops_at_gate0_blocker(tmp_path):
    out_dir = tmp_path / "gate0_blocked"

    manifest = run_resnet50_gate1_to_gate7(_blocked_gate0_root(tmp_path), out_dir, seed=20260606)

    assert manifest["final_gate"] == "gate0_blocked"
    assert manifest["artifact_status"] == "formal_blocked"
    assert manifest["formal_input_eligible"] is False
    assert (out_dir / "gate0_trace_acquisition_blocker_report.json").exists()
    assert not (out_dir / "resnet50_trace_adapter_bundle.json").exists()
    assert not (out_dir / "kernel_embedding_table.json").exists()


def test_resnet50_gate_pipeline_blocker_manifest_is_replayable(tmp_path):
    out_dir = tmp_path / "gate0_blocked_replay"

    manifest = run_resnet50_gate1_to_gate7(_blocked_gate0_root(tmp_path), out_dir, seed=20260606)

    stored = json.loads((out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME).read_text())
    assert stored["pipeline_manifest_hash"] == manifest["pipeline_manifest_hash"]
    assert not (out_dir / "gate1_5_pipeline_manifest.json").exists()


def test_resnet50_gate_pipeline_rejects_synthetic_artifact_shape_as_formal_root(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "artifact_shape_trace")

    try:
        record_resnet50_gate0_trace_acquisition(root)
    except ValueError as exc:
        assert "real NVBit runtime artifact origin" in str(exc)
    else:
        raise AssertionError("synthetic artifact-shape root must not produce formal Gate0")


def test_resnet50_gate_pipeline_keeps_baseline_artifacts_blocked_without_real_gate0(tmp_path):
    root = _blocked_gate0_root(tmp_path)
    baseline_path = tmp_path / "baseline_artifacts.json"
    baseline_path.write_text(
        json.dumps(
            {
                "metric_rows": [
                    {
                        "cluster_id": 0,
                        "measured": 100.0,
                        "predicted": 95.0,
                        "weight": 2.0,
                        "unit": "cycles",
                    }
                ],
                "sampled_metrics": {"cycles": 95.0},
                "full_baseline_metrics": {"cycles": 100.0},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "pipeline_out"

    manifest = run_resnet50_gate1_to_gate7(
        root,
        out_dir,
        seed=20260606,
        baseline_artifacts_path=baseline_path,
    )

    assert manifest["final_gate"] == "gate0_blocked"
    assert (out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME).exists()
    assert not (out_dir / "gate7_cluster_correctness_manifest.json").exists()
    assert not (out_dir / "gate9_sampled_vs_full_evaluation.json").exists()


def test_resnet50_gate_pipeline_runs_real_root_through_gate7(tmp_path):
    out_dir = tmp_path / "real_root_pipeline"

    manifest = run_resnet50_gate1_to_gate7(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        invocation_limit=1,
    )

    assert manifest["artifact_type"] == "gcl_resnet50_gate1_7_pipeline_manifest"
    assert manifest["final_gate"] == "gate9_report_only"
    for filename in [
        "resnet50_trace_adapter_bundle.json",
        "representative_sm_trace_manifest.json",
        "canonical_graph_bundle.json",
        "graph_tensor_bundle.json",
        "kernel_embedding_table.json",
        "selector_artifacts.json",
        "cluster_embedding_quality_report.json",
        "cluster_family_alignment_report.json",
        "representative_quality_report.json",
        "cluster_metric_error_report.json",
        "cluster_stability_report.json",
        "gate7_cluster_correctness_manifest.json",
        "cluster_tuning_vector_table.json",
        "tuning_vector_provenance_report.json",
        "tuning_safety_report.json",
        "gate8_tuning_manifest.json",
        "full_vs_sampled_simulation_report.json",
        "sampled_speedup_report.json",
        "sampled_error_report.json",
        "tuning_effect_report.json",
        "gate9_simulator_evaluation_manifest.json",
        "gate8_tuning_vector_proposal.json",
        "gate9_sampled_vs_full_evaluation.json",
        GATE1_7_PIPELINE_MANIFEST_FILENAME,
    ]:
        assert (out_dir / filename).exists()
    adapter = json.loads((out_dir / "resnet50_trace_adapter_bundle.json").read_text())
    assert adapter["artifact_status"] == "formal"
    assert adapter["trace_source"] == "nvbit"
    trace_manifest = json.loads((out_dir / "representative_sm_trace_manifest.json").read_text())
    assert trace_manifest["artifact_status"] == "formal"
    assert trace_manifest["collection_scope"] == "single_representative_sm_all_ctas"
    tensor_bundle = json.loads((out_dir / "graph_tensor_bundle.json").read_text())
    assert tensor_bundle["tensors"][0]["feature_width"] == 64
    embedding_table = json.loads((out_dir / "kernel_embedding_table.json").read_text())
    assert embedding_table["artifact_status"] == "formal"
    assert embedding_table["embedding_dim"] == 256
    selector = json.loads((out_dir / "selector_artifacts.json").read_text())
    assert selector["k_selection_report"]["mode"] == "silhouette_k"
    correctness = json.loads((out_dir / "gate7_cluster_correctness_manifest.json").read_text())
    assert correctness["threshold_policy"] == "report_only_v1"
    assert correctness["stability_report"]["stability_status"] == "single_run_not_evaluated"


def test_resnet50_gate_pipeline_real_root_records_gate6_and_gate7_contracts(tmp_path):
    out_dir = tmp_path / "real_root_gate6_gate7"

    run_resnet50_gate1_to_gate7(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        invocation_limit=1,
    )

    selector = json.loads((out_dir / "selector_artifacts.json").read_text())
    assert selector["artifact_type"] == "gcl_resnet50_gate6_selector_artifacts"
    assert selector["embedding_normalization_report"]["normalization_policy"] == (
        "engineering_default_z_score"
    )
    assert selector["embedding_normalization_report"]["input_fields"] == ["kernel_embedding"]
    assert selector["k_selection_report"]["mode"] == "silhouette_k"
    assert selector["kmeans_cluster_assignment_table"]["algorithm"] == "deterministic_kmeans"
    assert selector["representative_anchor_table"]["anchors"]
    assert selector["cluster_family_evidence_report"]["family_labels_used_for_clustering"] is False

    correctness = json.loads((out_dir / "gate7_cluster_correctness_manifest.json").read_text())
    assert correctness["artifact_type"] == "gcl_resnet50_gate7_cluster_correctness_manifest"
    assert correctness["artifact_version"] == "gate7_cluster_correctness_manifest_v1"
    assert correctness["threshold_policy"] == "report_only_v1"
    assert correctness["claim_status"] == "quantified_no_correctness_claim"
    assert correctness["threshold_claim_status"] == "not_set_until_real_resnet50_baseline"
    assert correctness["suggested_min_silhouette_score"] is None
    assert correctness["suggested_min_weighted_cluster_purity"] is None
    assert correctness["suggested_max_global_weighted_mape"] is None
    assert correctness["suggested_min_assignment_stability_ari"] is None
    for field in [
        "source_gate6_selector_manifest_hash",
        "source_cluster_assignment_table_hash",
        "source_representative_anchor_table_hash",
        "source_embedding_table_hash",
        "embedding_quality_report_hash",
        "family_alignment_report_hash",
        "representative_quality_report_hash",
        "metric_error_report_hash",
        "stability_report_hash",
        "gate7_cluster_correctness_manifest_hash",
    ]:
        assert correctness[field]
    assert correctness["embedding_geometry_metrics"]
    assert {
        "ari",
        "cluster_purity",
        "nmi",
        "weighted_purity",
    }.issubset(correctness["family_alignment_metrics"])
    assert correctness["family_alignment_metrics"]["ari"] is not None
    assert correctness["family_alignment_metrics"]["nmi"] is not None
    assert correctness["representative_quality_metrics"]["outlier_ratio"] == 0.0
    assert correctness["metric_error_report"]["status"] == "not_provided"
    assert correctness["stability_report"]["stability_status"] == "single_run_not_evaluated"
    report_files = {
        "embedding_quality_report_hash": "cluster_embedding_quality_report.json",
        "family_alignment_report_hash": "cluster_family_alignment_report.json",
        "representative_quality_report_hash": "representative_quality_report.json",
        "metric_error_report_hash": "cluster_metric_error_report.json",
        "stability_report_hash": "cluster_stability_report.json",
    }
    for hash_field, filename in report_files.items():
        stored_report = json.loads((out_dir / filename).read_text())
        assert stored_report["report_hash"] == correctness[hash_field]


def test_resnet50_gate_pipeline_real_root_reaches_gate9_with_baseline_artifacts(tmp_path):
    baseline_path = tmp_path / "baseline_artifacts.json"
    baseline_path.write_text(
        json.dumps(
            {
                "metric_rows": [
                    {
                        "cluster_id": 0,
                        "measured": 100.0,
                        "predicted": 95.0,
                        "weight": 1.0,
                        "unit": "cycles",
                    }
                ],
                "sampled_metrics": {"cycles": 95.0, "runtime_ms": 9.5},
                "full_baseline_metrics": {"cycles": 100.0, "runtime_ms": 10.0},
                "measured_baseline_metrics": {"cycles": 100.0, "runtime_ms": 10.0},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "real_root_gate9"

    manifest = run_resnet50_gate1_to_gate7(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        baseline_artifacts_path=baseline_path,
        invocation_limit=1,
    )

    assert manifest["final_gate"] == "gate9_evaluated"
    gate8 = json.loads((out_dir / "gate8_tuning_vector_proposal.json").read_text())
    assert gate8["artifact_type"] == "gcl_resnet50_gate8_tuning_vector_proposal"
    assert gate8["extension_label"] == "our_extension_not_original_gcl_sampler"
    assert gate8["proposals"]
    for vector in gate8["cluster_tuning_vector_table"]["tuning_vectors"]:
        assert vector["representative_anchor_hash"]
        assert vector["representative_anchor_table_hash"]
        assert vector["family_alignment_evidence_hash"]
        assert vector["metric_error_evidence_hash"]
        assert vector["gate7_correctness_manifest_hash"]
    gate8_manifest = json.loads((out_dir / "gate8_tuning_manifest.json").read_text())
    assert gate8_manifest["artifact_type"] == "gcl_resnet50_gate8_tuning_manifest"
    assert gate8_manifest["cluster_tuning_vector_table_hash"]
    assert gate8_manifest["tuning_vector_provenance_report_hash"]
    assert gate8_manifest["tuning_safety_report_hash"]
    gate9 = json.loads((out_dir / "gate9_sampled_vs_full_evaluation.json").read_text())
    assert gate9["artifact_type"] == "gcl_resnet50_gate9_sampled_vs_full_evaluation"
    assert gate9["extension_label"] == "our_extension_not_original_gcl_sampler"
    assert gate9["full_vs_sampled_simulation_report"]["cycles"]["relative_error"] == 0.05
    assert gate9["sampled_speedup_report"]["runtime_ms_speedup"] > 1.0
    assert gate9["sampled_error_report"]["cycles_relative_error"] == 0.05
    assert gate9["sampled_error_report"]["p95_relative_error"] == 0.05
    assert gate9["sampled_error_report"]["high_weight_bad_case_count"] == 0
    assert gate9["tuning_effect_report"]["status"] == "evaluated_from_gate8_proposal"
    assert gate9["tuning_effect_report"]["source_gate8_tuning_manifest_hash"] == (
        gate8_manifest["gate8_tuning_manifest_hash"]
    )
    gate9_manifest = json.loads((out_dir / "gate9_simulator_evaluation_manifest.json").read_text())
    assert gate9_manifest["artifact_type"] == "gcl_resnet50_gate9_simulator_evaluation_manifest"
    assert gate9_manifest["source_gate8_tuning_manifest_hash"] == (
        gate8_manifest["gate8_tuning_manifest_hash"]
    )
    assert gate9_manifest["representative_anchor_table_hash"]
    assert gate9_manifest["full_vs_sampled_simulation_report_hash"]
    assert gate9_manifest["sampled_speedup_report_hash"]
    assert gate9_manifest["sampled_error_report_hash"]
    assert gate9_manifest["tuning_effect_report_hash"]
    correctness = json.loads((out_dir / "gate7_cluster_correctness_manifest.json").read_text())
    assert correctness["metric_error_report"]["status"] == "reported"


def test_gate8_gate9_extension_stage_rejects_anchor_hash_mismatch_without_outputs(tmp_path):
    chain = run_real_nondegenerate_gate1_to_gate7_artifacts(tmp_path / "real_chain")
    out_dir = tmp_path / "mismatch_extension"
    out_dir.mkdir()
    selector = json.loads(json.dumps(chain["selector_artifacts"]))
    selector["representative_anchor_table"]["representative_anchor_table_hash"] = (
        "wrong-anchor-hash"
    )

    try:
        _emit_gate8_gate9_extension_artifacts(
            correctness_manifest=chain["correctness_manifest"],
            selector_artifacts=selector,
            baseline_artifacts={
                "metric_rows": [
                    {
                        "cluster_id": 0,
                        "measured": 100.0,
                        "predicted": 95.0,
                        "weight": 1.0,
                        "unit": "cycles",
                    }
                ],
                "sampled_metrics": {"cycles": 95.0},
                "full_baseline_metrics": {"cycles": 100.0},
            },
            out_dir=out_dir,
        )
    except ValueError as exc:
        assert "representative anchor" in str(exc)
    else:
        raise AssertionError("mismatched representative anchor hash must fail extension stage")

    for filename in [
        "gate8_tuning_vector_proposal.json",
        "gate8_tuning_manifest.json",
        "gate9_sampled_vs_full_evaluation.json",
        "gate9_simulator_evaluation_manifest.json",
    ]:
        assert not (out_dir / filename).exists()
