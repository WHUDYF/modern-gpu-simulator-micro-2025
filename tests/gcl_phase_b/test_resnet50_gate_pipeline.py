from pathlib import Path
import json

from experiments.gcl_phase_b.resnet50_gate0 import (
    record_resnet50_gate0_trace_acquisition,
    write_resnet50_gate0_blocker_report,
)
from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.utils import hash_without, write_json
from experiments.gcl_phase_b.resnet50_gate_pipeline import (
    GATE1_7_PIPELINE_MANIFEST_FILENAME,
    _emit_gate8_gate9_extension_artifacts,
    resume_resnet50_gate5_to_gate9_from_disk,
    run_resnet50_gate1_to_gate5,
    run_resnet50_gate1_to_gate7,
)
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records
from gcl_resnet50.formal_fixture import write_minimal_artifact_shape_resnet50_root
from gcl_resnet50.real_chain import FORMAL_ROOT, run_real_nondegenerate_gate1_to_gate7_artifacts


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


def test_resnet50_gate_pipeline_clears_stale_downstream_artifacts_on_gate0_blocked_rerun(
    tmp_path,
):
    root = _blocked_gate0_root(tmp_path)
    out_dir = tmp_path / "gate0_blocked_stale_cleanup"
    out_dir.mkdir()
    for filename in [
        "resnet50_trace_adapter_bundle.json",
        "kernel_embedding_table.json",
        "selector_artifacts.json",
        "gate7_cluster_correctness_manifest.json",
        "gate8_tuning_vector_proposal.json",
        "gate9_sampled_vs_full_evaluation.json",
    ]:
        write_json(out_dir / filename, {"stale": filename})

    manifest = run_resnet50_gate1_to_gate7(root, out_dir, seed=20260606)

    assert manifest["final_gate"] == "gate0_blocked"
    for filename in [
        "resnet50_trace_adapter_bundle.json",
        "kernel_embedding_table.json",
        "selector_artifacts.json",
        "gate7_cluster_correctness_manifest.json",
        "gate8_tuning_vector_proposal.json",
        "gate9_sampled_vs_full_evaluation.json",
    ]:
        assert not (out_dir / filename).exists()


def test_resnet50_gate_pipeline_rejects_synthetic_artifact_shape_as_formal_root(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "artifact_shape_trace")

    try:
        from experiments.gcl_phase_b import resnet50_gate0

        resnet50_gate0._ACTIVE_COLLECTOR_SESSION_IDS.add("test-session")
        try:
            record_resnet50_gate0_trace_acquisition(
                root,
                active_collector_session_id="test-session",
            )
        finally:
            resnet50_gate0._ACTIVE_COLLECTOR_SESSION_IDS.discard("test-session")
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


def test_resnet50_gate_pipeline_ignores_stale_gate0_blocker_when_formal_manifest_exists(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.resnet50_gate_pipeline as pipeline_module

    root = _blocked_gate0_root(tmp_path)
    manifest = {
        "artifact_type": "gcl_resnet50_gate0_trace_acquisition_manifest",
        "artifact_version": "gate0_trace_acquisition_manifest_v1",
        "artifact_status": "formal",
        "formal_input_eligible": True,
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "nvbit_collection_evidence_hash": "formal-evidence",
        "source_artifact_hashes": {
            "dynamic_trace.pb": "dynamic",
            "threadblocks/": "threadblocks",
            "enhanced_execution_info.json": "enhanced",
            "scheduler_metadata.json": "scheduler",
            "stats.csv": "stats",
        },
    }
    manifest["gate0_manifest_hash"] = hash_without(manifest, "gate0_manifest_hash")
    write_json(root / "gate0_trace_acquisition_manifest.json", manifest)

    def prove_not_short_circuited(*args, **kwargs):
        raise RuntimeError("adapter reached after stale blocker was ignored")

    monkeypatch.setattr(
        pipeline_module,
        "build_resnet50_trace_adapter_bundle",
        prove_not_short_circuited,
    )

    try:
        run_resnet50_gate1_to_gate7(root, tmp_path / "out", seed=20260606)
    except RuntimeError as exc:
        assert "adapter reached" in str(exc)
    else:
        raise AssertionError("pipeline should continue past stale Gate0 blocker")


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
    assert manifest["run_scope"] == "bounded_resnet50_trace_replay"
    assert manifest["invocation_limit"] == 1
    assert manifest["invocation_ids"] is None
    assert manifest["input_kernel_invocation_count"] == 1
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


def test_resnet50_gate1_to_gate5_entrypoint_stops_before_selector_and_reports(tmp_path):
    out_dir = tmp_path / "real_root_gate5_only"

    manifest = run_resnet50_gate1_to_gate5(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        invocation_limit=1,
    )

    assert manifest["artifact_type"] == "gcl_resnet50_gate1_7_pipeline_manifest"
    assert manifest["final_gate"] == "gate5_embedding_exported"
    assert manifest["run_scope"] == "bounded_resnet50_trace_replay"
    assert manifest["invocation_limit"] == 1
    assert manifest["invocation_ids"] is None
    assert manifest["input_kernel_invocation_count"] == 1
    assert manifest["hashes"]["embedding_table_hash"]
    assert manifest["hashes"]["selector_manifest_hash"] is None
    assert manifest["hashes"]["gate7_correctness_manifest_hash"] is None
    assert manifest["hashes"]["gate8_tuning_vector_proposal_hash"] is None
    assert manifest["hashes"]["gate9_sampled_vs_full_evaluation_hash"] is None
    for filename in [
        "resnet50_trace_adapter_bundle.json",
        "representative_sm_trace_manifest.json",
        "canonical_graph_bundle.json",
        "graph_tensor_bundle.json",
        "kernel_embedding_table.json",
        "embedding_export_report.json",
        GATE1_7_PIPELINE_MANIFEST_FILENAME,
    ]:
        assert (out_dir / filename).exists()
    for filename in [
        "selector_artifacts.json",
        "gate7_cluster_correctness_manifest.json",
        "cluster_embedding_quality_report.json",
        "gate8_tuning_vector_proposal.json",
        "gate8_tuning_manifest.json",
        "gate9_sampled_vs_full_evaluation.json",
        "gate9_simulator_evaluation_manifest.json",
    ]:
        assert not (out_dir / filename).exists()


def test_resnet50_gate1_to_gate5_rerun_removes_stale_gate6_gate9_artifacts(tmp_path):
    out_dir = tmp_path / "gate5_only_stale_cleanup"
    run_resnet50_gate1_to_gate7(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        invocation_limit=1,
    )
    stale_outputs = [
        "selector_artifacts.json",
        "gate7_cluster_correctness_manifest.json",
        "cluster_embedding_quality_report.json",
        "cluster_family_alignment_report.json",
        "representative_quality_report.json",
        "cluster_metric_error_report.json",
        "cluster_stability_report.json",
        "gate8_tuning_vector_proposal.json",
        "gate8_tuning_manifest.json",
        "gate9_sampled_vs_full_evaluation.json",
        "gate9_simulator_evaluation_manifest.json",
    ]
    for filename in stale_outputs:
        assert (out_dir / filename).exists()

    manifest = run_resnet50_gate1_to_gate5(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        invocation_limit=1,
    )

    assert manifest["final_gate"] == "gate5_embedding_exported"
    assert (out_dir / "kernel_embedding_table.json").exists()
    for filename in stale_outputs:
        assert not (out_dir / filename).exists()


def test_resnet50_gate_pipeline_resumes_gate5_to_gate9_from_persisted_gate4(tmp_path):
    out_dir = tmp_path / "resume_from_gate4"
    gate5_manifest = run_resnet50_gate1_to_gate5(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        invocation_limit=1,
    )
    assert gate5_manifest["final_gate"] == "gate5_embedding_exported"
    assert (out_dir / "graph_tensor_bundle.json").exists()

    manifest = resume_resnet50_gate5_to_gate9_from_disk(
        out_dir,
        seed=20260607,
    )

    assert manifest["final_gate"] == "gate9_report_only"
    assert manifest["resumed_from_persisted_gate4"] is True
    assert manifest["run_scope"] == "bounded_resnet50_trace_replay"
    assert manifest["invocation_limit"] == 1
    assert manifest["invocation_ids"] is None
    assert (out_dir / "kernel_embedding_table.json").exists()
    assert (out_dir / "selector_artifacts.json").exists()
    assert (out_dir / "gate7_cluster_correctness_manifest.json").exists()
    stored = json.loads((out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME).read_text())
    assert stored["pipeline_manifest_hash"] == manifest["pipeline_manifest_hash"]


def test_resnet50_gate_pipeline_resume_uses_persisted_seed_by_default(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.resnet50_gate_pipeline as pipeline_module

    out_dir = tmp_path / "resume_uses_persisted_seed"
    run_resnet50_gate1_to_gate5(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        invocation_limit=1,
    )
    captured = {}

    def fake_training_and_export(*, tensors, graph_tensor_bundle, augmentation_bundle, out_dir, seed):
        captured["seed"] = seed
        return {
            "kernel_embedding_table_hash": "embedding-hash",
            "embeddings": [
                {
                    "record_id": "gcl_embedding:0000",
                    "kernel_invocation_id": tensors[0]["kernel_invocation_id"],
                    "kernel_embedding": [0.0] * 256,
                }
            ],
        }

    monkeypatch.setattr(
        pipeline_module,
        "_run_gate5_training_and_export",
        fake_training_and_export,
    )
    monkeypatch.setattr(
        pipeline_module,
        "select_phase_b_representatives",
        lambda *args, **kwargs: {
            "selector_manifest_hash": "selector-hash",
            "representative_anchor_table": {
                "artifact_type": "gcl_resnet50_representative_anchor_table",
                "anchors": [],
            },
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "evaluate_gate7_correctness_from_artifacts",
        lambda **kwargs: {
            "gate7_cluster_correctness_manifest_hash": "gate7-hash",
            "source_representative_anchor_table_hash": "anchor-hash",
            "gate7_report_artifacts": {
                key: {"report_hash": f"{key}-hash"}
                for key in pipeline_module.GATE7_REPORT_FILENAMES
            },
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "_emit_gate8_gate9_extension_artifacts",
        lambda **kwargs: (
            {"gate8_tuning_vector_proposal_hash": "gate8-hash"},
            {"gate9_sampled_vs_full_evaluation_hash": "gate9-hash"},
            "gate9_report_only",
        ),
    )

    manifest = resume_resnet50_gate5_to_gate9_from_disk(out_dir)

    assert captured["seed"] == 20260607
    assert manifest["seed"] == 20260607


def test_resnet50_gate_pipeline_resume_preserves_adapter_scope_when_previous_manifest_missing(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.resnet50_gate_pipeline as pipeline_module

    out_dir = tmp_path / "resume_missing_manifest"
    run_resnet50_gate1_to_gate5(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        invocation_limit=1,
    )
    (out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME).unlink()

    def fake_training_and_export(*, tensors, graph_tensor_bundle, augmentation_bundle, out_dir, seed):
        return {
            "kernel_embedding_table_hash": "embedding-hash",
            "embeddings": [
                {
                    "record_id": "gcl_embedding:0000",
                    "kernel_invocation_id": tensors[0]["kernel_invocation_id"],
                    "kernel_embedding": [0.0] * 256,
                }
            ],
        }

    monkeypatch.setattr(
        pipeline_module,
        "_run_gate5_training_and_export",
        fake_training_and_export,
    )
    monkeypatch.setattr(
        pipeline_module,
        "select_phase_b_representatives",
        lambda *args, **kwargs: {
            "selector_manifest_hash": "selector-hash",
            "representative_anchor_table": {
                "artifact_type": "gcl_resnet50_representative_anchor_table",
                "anchors": [],
            },
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "evaluate_gate7_correctness_from_artifacts",
        lambda **kwargs: {
            "gate7_cluster_correctness_manifest_hash": "gate7-hash",
            "source_representative_anchor_table_hash": "anchor-hash",
            "gate7_report_artifacts": {
                key: {"report_hash": f"{key}-hash"}
                for key in pipeline_module.GATE7_REPORT_FILENAMES
            },
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "_emit_gate8_gate9_extension_artifacts",
        lambda **kwargs: (
            {"gate8_tuning_vector_proposal_hash": "gate8-hash"},
            {"gate9_sampled_vs_full_evaluation_hash": "gate9-hash"},
            "gate9_report_only",
        ),
    )

    manifest = resume_resnet50_gate5_to_gate9_from_disk(out_dir, seed=20260607)

    assert manifest["resumed_from_persisted_gate4"] is True
    assert manifest["run_scope"] == "bounded_resnet50_trace_replay"
    assert manifest["invocation_limit"] == 1
    assert manifest["invocation_ids"] is None


def test_resnet50_gate5_trains_on_bounded_subset_but_exports_all_embeddings(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.resnet50_gate_pipeline as pipeline_module
    from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest

    original_train = pipeline_module.train_minimal_contrastive
    captured = {}

    def spy_train(tensors, out_dir, seed=20260602, **kwargs):
        captured["training_count"] = len(tensors)
        captured["kwargs"] = kwargs
        return original_train(tensors, out_dir, seed=seed, **kwargs)

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", spy_train)
    manifest = build_representative_sm_trace_manifest()
    source_invocation = manifest["kernel_invocations"][0]
    manifest["kernel_invocations"] = []
    for index in range(5):
        invocation = {**source_invocation, "kernel_invocation_id": f"debug_train_subset_{index}"}
        invocation["trace_hash"] = hash_without(invocation, "trace_hash")
        manifest["kernel_invocations"].append(invocation)
    manifest["trace_manifest_hash"] = hash_without(manifest, "trace_manifest_hash")
    records = build_phase_b_trace_records(manifest)
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)
    graph_tensor_bundle = {
        "artifact_type": "gcl_resnet50_graph_tensor_bundle",
        "artifact_version": "gate4_graph_tensor_bundle_v1",
        "source_canonical_graph_bundle_hash": "graph-bundle-hash",
        "graph_tensor_bundle_hash": "tensor-bundle-hash",
        "tensors": [],
    }
    augmentation_bundle = pipeline_module.create_augmentation_manifest_bundle(
        tensors,
        seed=20260607,
    )

    embedding_table = pipeline_module._run_gate5_training_and_export(
        tensors=tensors,
        graph_tensor_bundle=graph_tensor_bundle,
        augmentation_bundle=augmentation_bundle,
        out_dir=tmp_path,
        seed=20260607,
    )

    training_manifest = json.loads((tmp_path / "rgcn_training_run_manifest.json").read_text())
    assert len(embedding_table["embeddings"]) == len(tensors)
    assert captured["training_count"] < len(tensors)
    assert training_manifest["train_graph_count"] == captured["training_count"]
    assert training_manifest["export_graph_count"] == len(tensors)
    assert training_manifest["training_subset_policy"] == "deterministic_prefix_for_full_trace_scalability"


def test_resnet50_gate5_reuses_existing_checkpoint_for_export_resume(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.resnet50_gate_pipeline as pipeline_module
    from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest

    manifest = build_representative_sm_trace_manifest()
    records = build_phase_b_trace_records(manifest)
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)
    graph_tensor_bundle = {
        "artifact_type": "gcl_resnet50_graph_tensor_bundle",
        "artifact_version": "gate4_graph_tensor_bundle_v1",
        "source_canonical_graph_bundle_hash": "graph-bundle-hash",
        "graph_tensor_bundle_hash": "tensor-bundle-hash",
        "tensors": [],
    }
    augmentation_bundle = pipeline_module.create_augmentation_manifest_bundle(
        tensors,
        seed=20260607,
    )
    pipeline_module._run_gate5_training_and_export(
        tensors=tensors,
        graph_tensor_bundle=graph_tensor_bundle,
        augmentation_bundle=augmentation_bundle,
        out_dir=tmp_path,
        seed=20260607,
    )

    def fail_if_retrained(*args, **kwargs):
        raise AssertionError("Gate5 should reuse existing checkpoint")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_if_retrained)

    embedding_table = pipeline_module._run_gate5_training_and_export(
        tensors=tensors,
        graph_tensor_bundle=graph_tensor_bundle,
        augmentation_bundle=augmentation_bundle,
        out_dir=tmp_path,
        seed=20260607,
    )

    assert len(embedding_table["embeddings"]) == len(tensors)
    assert (tmp_path / "kernel_embedding_table.json").exists()


def test_resnet50_gate5_reexports_when_cached_table_lacks_side_artifacts(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.resnet50_gate_pipeline as pipeline_module
    from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest

    manifest = build_representative_sm_trace_manifest()
    records = build_phase_b_trace_records(manifest)
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)
    graph_tensor_bundle = {
        "artifact_type": "gcl_resnet50_graph_tensor_bundle",
        "artifact_version": "gate4_graph_tensor_bundle_v1",
        "source_canonical_graph_bundle_hash": "graph-bundle-hash",
        "graph_tensor_bundle_hash": "tensor-bundle-hash",
        "tensors": [],
    }
    augmentation_bundle = pipeline_module.create_augmentation_manifest_bundle(
        tensors,
        seed=20260607,
    )
    pipeline_module._run_gate5_training_and_export(
        tensors=tensors,
        graph_tensor_bundle=graph_tensor_bundle,
        augmentation_bundle=augmentation_bundle,
        out_dir=tmp_path,
        seed=20260607,
    )
    (tmp_path / "readout_manifest.json").unlink()

    def fail_if_retrained(*args, **kwargs):
        raise AssertionError("Gate5 should reuse training checkpoint for re-export")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_if_retrained)

    embedding_table = pipeline_module._run_gate5_training_and_export(
        tensors=tensors,
        graph_tensor_bundle=graph_tensor_bundle,
        augmentation_bundle=augmentation_bundle,
        out_dir=tmp_path,
        seed=20260607,
    )

    assert len(embedding_table["embeddings"]) == len(tensors)
    assert (tmp_path / "readout_manifest.json").exists()
    assert (tmp_path / "gate5_lineage_bundle.json").exists()
    assert (tmp_path / "embedding_export_report.json").exists()


def test_resnet50_gate5_retrains_when_checkpoint_training_manifest_missing(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.resnet50_gate_pipeline as pipeline_module
    from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest

    manifest = build_representative_sm_trace_manifest()
    records = build_phase_b_trace_records(manifest)
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)
    graph_tensor_bundle = {
        "artifact_type": "gcl_resnet50_graph_tensor_bundle",
        "artifact_version": "gate4_graph_tensor_bundle_v1",
        "source_canonical_graph_bundle_hash": "graph-bundle-hash",
        "graph_tensor_bundle_hash": "tensor-bundle-hash",
        "tensors": [],
    }
    augmentation_bundle = pipeline_module.create_augmentation_manifest_bundle(
        tensors,
        seed=20260607,
    )
    pipeline_module._run_gate5_training_and_export(
        tensors=tensors,
        graph_tensor_bundle=graph_tensor_bundle,
        augmentation_bundle=augmentation_bundle,
        out_dir=tmp_path,
        seed=20260607,
    )
    (tmp_path / "kernel_embedding_table.json").unlink()
    (tmp_path / "embedding_export_report.json").unlink()
    (tmp_path / "gate5_lineage_bundle.json").unlink()
    (tmp_path / "readout_manifest.json").unlink()
    (tmp_path / "rgcn_training_run_manifest.json").unlink()
    retrain_calls = {"count": 0}
    original_train = pipeline_module.train_minimal_contrastive

    def spy_train(*args, **kwargs):
        retrain_calls["count"] += 1
        return original_train(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", spy_train)

    embedding_table = pipeline_module._run_gate5_training_and_export(
        tensors=tensors,
        graph_tensor_bundle=graph_tensor_bundle,
        augmentation_bundle=augmentation_bundle,
        out_dir=tmp_path,
        seed=20260607,
    )

    assert retrain_calls["count"] == 1
    assert len(embedding_table["embeddings"]) == len(tensors)
    assert (tmp_path / "rgcn_training_run_manifest.json").exists()


def test_resnet50_gate5_retrains_when_requested_seed_changes(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.resnet50_gate_pipeline as pipeline_module
    from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest

    manifest = build_representative_sm_trace_manifest()
    records = build_phase_b_trace_records(manifest)
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)
    graph_tensor_bundle = {
        "artifact_type": "gcl_resnet50_graph_tensor_bundle",
        "artifact_version": "gate4_graph_tensor_bundle_v1",
        "source_canonical_graph_bundle_hash": "graph-bundle-hash",
        "graph_tensor_bundle_hash": "tensor-bundle-hash",
        "tensors": [],
    }
    augmentation_bundle = pipeline_module.create_augmentation_manifest_bundle(
        tensors,
        seed=20260607,
    )
    pipeline_module._run_gate5_training_and_export(
        tensors=tensors,
        graph_tensor_bundle=graph_tensor_bundle,
        augmentation_bundle=augmentation_bundle,
        out_dir=tmp_path,
        seed=20260607,
    )
    retrain_calls = {"count": 0}
    original_train = pipeline_module.train_minimal_contrastive

    def spy_train(*args, **kwargs):
        retrain_calls["count"] += 1
        return original_train(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", spy_train)

    pipeline_module._run_gate5_training_and_export(
        tensors=tensors,
        graph_tensor_bundle=graph_tensor_bundle,
        augmentation_bundle=augmentation_bundle,
        out_dir=tmp_path,
        seed=20260608,
    )

    training_manifest = json.loads((tmp_path / "rgcn_training_run_manifest.json").read_text())
    assert retrain_calls["count"] == 1
    assert training_manifest["random_seed"] == 20260608
    assert training_manifest["checkpoint_reuse"] is False


def test_resnet50_gate8_report_only_handles_weak_representatives_without_blocking(tmp_path):
    import experiments.gcl_phase_b.resnet50_gate_pipeline as pipeline_module

    correctness_manifest = {
        "artifact_type": "gcl_resnet50_gate7_cluster_correctness_manifest",
        "claim_status": "quantified_no_correctness_claim",
        "gate7_cluster_correctness_manifest_hash": "gate7-hash",
        "source_representative_anchor_table_hash": "anchor-hash",
        "family_alignment_report_hash": "family-hash",
        "metric_error_report_hash": "metric-hash",
        "family_alignment_metrics": {"weighted_purity": 1.0},
        "metric_error_report": {"global_weighted_mape": None},
        "representative_quality_metrics": {"high_weight_outlier_count": 1},
        "gate7_report_artifacts": {
            "family_alignment_report": {"report_hash": "family-hash"},
            "metric_error_report": {"report_hash": "metric-hash"},
        },
    }
    selector_artifacts = {
        "representative_anchor_table": {
            "artifact_type": "gcl_resnet50_representative_anchor_table",
            "anchors": [
                {
                    "cluster_id": 0,
                    "representative_record_id": "record-0",
                    "kernel_invocation_id": "kernel-0",
                    "distance_to_centroid": 0.0,
                }
            ],
        }
    }
    monkeypatch_hash = pipeline_module.stable_hash(
        {
            key: value
            for key, value in selector_artifacts["representative_anchor_table"].items()
            if key != "representative_anchor_table_hash"
        }
    )
    correctness_manifest["source_representative_anchor_table_hash"] = monkeypatch_hash
    selector_artifacts["representative_anchor_table"][
        "representative_anchor_table_hash"
    ] = monkeypatch_hash

    gate8_proposal, gate9_report, final_gate = pipeline_module._emit_gate8_gate9_extension_artifacts(
        correctness_manifest=correctness_manifest,
        selector_artifacts=selector_artifacts,
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
        out_dir=tmp_path,
    )

    assert final_gate == "gate9_report_only"
    assert gate8_proposal["gate8_tuning_manifest"]["tuning_safety_status"] == "blocked_report_only"
    assert gate9_report["claim_status"] == (
        "baseline_missing_no_speedup_or_accuracy_claim"
    )
    assert gate9_report["tuning_effect_report"]["status"] == (
        "not_evaluated_without_baseline"
    )
    assert gate9_report["gate9_simulator_evaluation_manifest"]["artifact_type"] == (
        "gcl_resnet50_gate9_simulator_evaluation_manifest"
    )
    assert (tmp_path / "gate8_tuning_vector_proposal.json").exists()
    assert (tmp_path / "gate9_sampled_vs_full_evaluation.json").exists()


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


def test_resnet50_gate_pipeline_manifest_records_full_trace_scope(tmp_path, monkeypatch):
    import experiments.gcl_phase_b.resnet50_gate_pipeline as pipeline_module

    out_dir = tmp_path / "full_scope"
    fake_adapter_bundle = {
        "adapter_bundle_hash": "adapter-hash",
        "kernel_invocation_table": [
            {"kernel_invocation_id": f"d_0_s_0_k_{index}"}
            for index in range(265)
        ],
    }

    def fake_manifest_from_bundle(bundle):
        assert bundle is fake_adapter_bundle
        manifest = build_representative_sm_trace_manifest(invocation_count=2)
        return (
            manifest,
            {
                "artifact_type": "gcl_resnet50_selected_sm_policy_report_bundle",
                "selected_sm_policy_report_bundle_hash": "policy-hash",
                "reports": [],
            },
            {
                "artifact_type": "gcl_resnet50_scope_preview_report",
                "scope_preview_report_hash": "preview-hash",
                "kernel_invocation_count": 2,
                "invocations": [],
            },
        )

    monkeypatch.setattr(
        pipeline_module,
        "build_resnet50_trace_adapter_bundle",
        lambda *args, **kwargs: fake_adapter_bundle,
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_representative_sm_manifest_from_bundle",
        fake_manifest_from_bundle,
    )

    manifest = run_resnet50_gate1_to_gate7(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        invocation_limit=None,
        invocation_ids=None,
    )

    assert manifest["run_scope"] == "real_resnet50_full_trace"
    assert manifest["invocation_limit"] is None
    assert manifest["invocation_ids"] is None
    assert manifest["input_kernel_invocation_count"] == 265


def test_resnet50_gate_pipeline_manifest_records_full_trace_scope_at_gate5(tmp_path, monkeypatch):
    import experiments.gcl_phase_b.resnet50_gate_pipeline as pipeline_module

    out_dir = tmp_path / "full_scope_gate5"
    fake_adapter_bundle = {
        "adapter_bundle_hash": "adapter-hash",
        "kernel_invocation_table": [
            {"kernel_invocation_id": f"d_0_s_0_k_{index}"}
            for index in range(265)
        ],
    }

    monkeypatch.setattr(
        pipeline_module,
        "build_resnet50_trace_adapter_bundle",
        lambda *args, **kwargs: fake_adapter_bundle,
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_representative_sm_manifest_from_bundle",
        lambda bundle: (
            build_representative_sm_trace_manifest(invocation_count=2),
            {
                "artifact_type": "gcl_resnet50_selected_sm_policy_report_bundle",
                "selected_sm_policy_report_bundle_hash": "policy-hash",
                "reports": [],
            },
            {
                "artifact_type": "gcl_resnet50_scope_preview_report",
                "scope_preview_report_hash": "preview-hash",
                "kernel_invocation_count": 2,
                "invocations": [],
            },
        ),
    )

    manifest = run_resnet50_gate1_to_gate5(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        invocation_limit=None,
        invocation_ids=None,
    )

    assert manifest["final_gate"] == "gate5_embedding_exported"
    assert manifest["run_scope"] == "real_resnet50_full_trace"
    assert manifest["invocation_limit"] is None
    assert manifest["invocation_ids"] is None
    assert manifest["input_kernel_invocation_count"] == 265


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
