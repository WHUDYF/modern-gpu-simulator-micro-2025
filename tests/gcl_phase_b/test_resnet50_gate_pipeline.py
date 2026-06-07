from pathlib import Path
import json

from experiments.gcl_phase_b.resnet50_gate0 import (
    record_resnet50_gate0_trace_acquisition,
    write_resnet50_gate0_blocker_report,
)
from experiments.gcl_phase_b.resnet50_gate_pipeline import (
    GATE1_7_PIPELINE_MANIFEST_FILENAME,
    run_resnet50_gate1_to_gate7,
)
from tests.gcl_resnet50.formal_fixture import write_minimal_artifact_shape_resnet50_root
from tests.gcl_resnet50.real_chain import FORMAL_ROOT


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
    assert not (out_dir / "gate7_correctness_manifest.json").exists()
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
        "gate7_correctness_manifest.json",
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
    correctness = json.loads((out_dir / "gate7_correctness_manifest.json").read_text())
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

    correctness = json.loads((out_dir / "gate7_correctness_manifest.json").read_text())
    assert correctness["artifact_type"] == "gcl_resnet50_gate7_correctness_manifest"
    assert correctness["threshold_policy"] == "report_only_v1"
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
    gate9 = json.loads((out_dir / "gate9_sampled_vs_full_evaluation.json").read_text())
    assert gate9["artifact_type"] == "gcl_resnet50_gate9_sampled_vs_full_evaluation"
    assert gate9["extension_label"] == "our_extension_not_original_gcl_sampler"
    assert gate9["full_vs_sampled_simulation_report"]["cycles"]["relative_error"] == 0.05
    assert gate9["sampled_speedup_report"]["runtime_ms_speedup"] > 1.0
    assert gate9["sampled_error_report"]["cycles_relative_error"] == 0.05
    correctness = json.loads((out_dir / "gate7_correctness_manifest.json").read_text())
    assert correctness["metric_error_report"]["status"] == "reported"
