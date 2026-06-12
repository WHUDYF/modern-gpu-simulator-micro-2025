import json
import subprocess
import sys

import pytest

from experiments.gcl_phase_b.resnet50_gate0 import GATE0_ARTIFACT_TYPE, GATE0_ARTIFACT_VERSION
from experiments.gcl_phase_b.resnet50_gate0 import write_resnet50_gate0_blocker_report
from experiments.gcl_phase_b.utils import hash_without
from scripts import run_resnet50_full_trace_gcl


def _write_gate0_manifest(
    root,
    *,
    artifact_status="formal",
    eligible=True,
    scope="full_resnet50_inference_trace",
    bad_hash=False,
):
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_type": GATE0_ARTIFACT_TYPE,
        "artifact_version": GATE0_ARTIFACT_VERSION,
        "artifact_status": artifact_status,
        "formal_input_eligible": eligible,
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": scope,
        "scheduler_metadata_source": "real_nvbit_smid",
        "nvbit_collection_evidence_hash": "evidence-hash",
        "source_artifact_hashes": {
            "dynamic_trace.pb": "dynamic",
            "threadblocks/": "threadblocks",
            "enhanced_execution_info.json": "enhanced",
            "scheduler_metadata.json": "scheduler",
            "stats.csv": "stats",
        },
    }
    manifest["gate0_manifest_hash"] = (
        "bad-hash" if bad_hash else hash_without(manifest, "gate0_manifest_hash")
    )
    (root / "gate0_trace_acquisition_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (root / "scheduler_metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": "gcl_real_trace_scheduler_metadata",
                "artifact_version": "resnet50_scheduler_metadata_v1",
                "scheduler_metadata_source": "real_nvbit_smid",
                "kernel_invocations": [
                    {
                        "kernel_invocation_id": "d_0_s_0_k_1",
                        "cta_records": [{"cta_id": "cta_0"}, {"cta_id": "cta_1"}],
                    },
                    {
                        "kernel_invocation_id": "d_0_s_0_k_2",
                        "cta_records": [{"cta_id": "cta_2"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _fake_success_pipeline(calls):
    def fake_pipeline(
        root,
        out_dir,
        seed,
        baseline_artifacts_path=None,
        invocation_limit=None,
        invocation_ids=None,
    ):
        calls["root"] = root
        calls["out_dir"] = out_dir
        calls["seed"] = seed
        calls["baseline_artifacts_path"] = baseline_artifacts_path
        calls["invocation_limit"] = invocation_limit
        calls["invocation_ids"] = invocation_ids
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "resnet50_trace_adapter_bundle.json").write_text(
            json.dumps({"adapter_validation_report": {"status": "passed"}}),
            encoding="utf-8",
        )
        (out_dir / "kernel_embedding_table.json").write_text(
            json.dumps({"embeddings": [1, 2]}),
            encoding="utf-8",
        )
        (out_dir / "selector_artifacts.json").write_text(
            json.dumps({"k_selection_report": {"selected_k": 2}}),
            encoding="utf-8",
        )
        (out_dir / "gate7_cluster_correctness_manifest.json").write_text(
            json.dumps({"gate7_cluster_correctness_manifest_hash": "gate7-hash"}),
            encoding="utf-8",
        )
        return {
            "artifact_type": "gcl_resnet50_gate1_7_pipeline_manifest",
            "final_gate": "gate9_report_only",
            "run_scope": "real_resnet50_full_trace",
            "invocation_limit": None,
            "invocation_ids": None,
            "input_kernel_invocation_count": 265,
            "hashes": {
                "embedding_table_hash": "embedding-hash",
                "selector_manifest_hash": "selector-hash",
                "gate7_correctness_manifest_hash": "gate7-hash",
                "gate8_tuning_vector_proposal_hash": "gate8-hash",
                "gate9_sampled_vs_full_evaluation_hash": "gate9-hash",
            },
            "pipeline_manifest_hash": "pipeline-hash",
        }

    return fake_pipeline


def _fake_success_resume(calls):
    def fake_resume(out_dir, seed, baseline_artifacts_path=None):
        calls["resume_out_dir"] = out_dir
        calls["resume_seed"] = seed
        calls["resume_baseline_artifacts_path"] = baseline_artifacts_path
        (out_dir / "resnet50_trace_adapter_bundle.json").write_text(
            json.dumps({"adapter_validation_report": {"status": "passed"}}),
            encoding="utf-8",
        )
        (out_dir / "kernel_embedding_table.json").write_text(
            json.dumps({"embeddings": [1, 2]}),
            encoding="utf-8",
        )
        (out_dir / "selector_artifacts.json").write_text(
            json.dumps({"k_selection_report": {"selected_k": 2}}),
            encoding="utf-8",
        )
        (out_dir / "gate7_cluster_correctness_manifest.json").write_text(
            json.dumps({"gate7_cluster_correctness_manifest_hash": "gate7-hash"}),
            encoding="utf-8",
        )
        return {
            "artifact_type": "gcl_resnet50_gate1_7_pipeline_manifest",
            "final_gate": "gate9_report_only",
            "run_scope": "real_resnet50_full_trace",
            "invocation_limit": None,
            "invocation_ids": None,
            "input_kernel_invocation_count": 265,
            "hashes": {
                "embedding_table_hash": "embedding-hash",
                "selector_manifest_hash": "selector-hash",
                "gate7_correctness_manifest_hash": "gate7-hash",
                "gate8_tuning_vector_proposal_hash": "gate8-hash",
                "gate9_sampled_vs_full_evaluation_hash": "gate9-hash",
            },
            "pipeline_manifest_hash": "pipeline-hash",
        }

    return fake_resume


def _write_complete_gate4_artifact_set(out_dir, gate0_manifest):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph_tensor_bundle.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (out_dir / "resnet50_trace_adapter_bundle.json").write_text(
        json.dumps(
            {
                "source_gate0_manifest_hash": gate0_manifest["gate0_manifest_hash"],
                "adapter_validation_report": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "representative_sm_trace_manifest.json").write_text(
        json.dumps({"artifact_type": "gcl_representative_sm_trace_manifest"}),
        encoding="utf-8",
    )
    (out_dir / "canonical_graph_bundle.json").write_text(
        json.dumps({"artifact_type": "gcl_resnet50_canonical_graph_bundle"}),
        encoding="utf-8",
    )


def _write_success_manifest(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resnet50_full_trace_reproduction_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "gcl_resnet50_full_trace_reproduction_manifest",
                "resource_status": "completed",
                "formal_full_trace_run": True,
            }
        ),
        encoding="utf-8",
    )


def _write_gnn_acceptance_required_inputs(out_dir):
    (out_dir / "rgcn_training_run_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "gcl_resnet50_rgcn_training_run_manifest",
                "artifact_version": "gate5_rgcn_training_run_manifest_v1",
                "training_status": "formal_gate5_complete",
                "train_graph_count": 4,
                "export_graph_count": 265,
                "optimizer_config": {"optimizer_step_count": 1},
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
                "training_run_manifest_hash": "training-hash",
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "selector_artifacts.json").write_text(
        json.dumps(
            {
                "artifact_type": "gcl_resnet50_gate6_selector_artifacts",
                "artifact_version": "gate6_selector_artifacts_v1",
                "selector_manifest_hash": "selector-hash",
                "source_embedding_table_hash": "embedding-hash",
                "k_selection_report": {"mode": "silhouette_k", "selected_k": 2},
                "kmeans_cluster_assignment_table": {
                    "assignments": [
                        {"record_id": "gcl_embedding:0000", "cluster_id": 0},
                        {"record_id": "gcl_embedding:0001", "cluster_id": 1},
                    ]
                },
                "representative_anchor_table": {"anchors": []},
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "gate7_cluster_correctness_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "gcl_resnet50_gate7_cluster_correctness_manifest",
                "artifact_version": "gate7_cluster_correctness_manifest_v1",
                "claim_status": "quantified_no_correctness_claim",
                "gate7_cluster_correctness_manifest_hash": "gate7-hash",
                "embedding_geometry_metrics": {
                    "silhouette": 0.48186617,
                    "davies_bouldin": 0.78974237,
                    "inter_intra_ratio": 2.01633922,
                },
                "family_alignment_metrics": {
                    "cluster_purity": 1.0,
                    "weighted_purity": 1.0,
                    "ari": 0.0,
                    "nmi": 0.0,
                    "completeness": 0.0,
                    "family_to_cluster_coverage": {"resnet50_real_trace": {}},
                },
                "metric_error_report": {
                    "metric_claim_status": "unavailable",
                    "status": "not_provided",
                },
                "stability_report": {"stability_status": "single_run_not_evaluated"},
            }
        ),
        encoding="utf-8",
    )


def test_full_trace_runner_preserves_gate0_blocker_details(tmp_path):
    root = tmp_path / "gate0_blocked_root"
    write_resnet50_gate0_blocker_report(
        root,
        reason="real ResNet-50 NVBit trace is not available",
        missing_requirements=["dynamic_trace.pb", "threadblocks/"],
    )
    out_dir = tmp_path / "out"

    result = run_resnet50_full_trace_gcl.run_full_trace_reproduction(
        input_root=root,
        out_dir=out_dir,
        seed=20260608,
        baseline_artifacts=None,
    )

    copied_gate0_blocker = json.loads(
        (out_dir / "gate0_trace_acquisition_blocker_report.json").read_text()
    )
    assert result["final_gate"] == "gate0_blocked"
    assert result["formal_full_trace_run"] is False
    assert result["resource_status"] == "blocked"
    assert result["pipeline_hashes"]["gate0_blocker_report_hash"] == copied_gate0_blocker[
        "gate0_blocker_report_hash"
    ]
    assert not (out_dir / "resnet50_full_trace_reproduction_blocker_report.json").exists()


def test_full_trace_runner_calls_pipeline_without_invocation_slicing(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "run_resnet50_gate1_to_gate7",
        _fake_success_pipeline(calls),
    )
    root = tmp_path / "formal_root"
    _write_gate0_manifest(root)

    result = run_resnet50_full_trace_gcl.run_full_trace_reproduction(
        input_root=root,
        out_dir=tmp_path / "out",
        seed=20260608,
        baseline_artifacts=None,
    )

    assert calls["invocation_limit"] is None
    assert calls["invocation_ids"] is None
    assert result["run_scope"] == "real_resnet50_full_trace"
    assert result["formal_full_trace_run"] is True
    assert result["source_gate0_manifest_hash"]
    assert result["input_cta_record_count"] == 3
    assert result["invocation_limit"] is None
    assert result["invocation_ids"] is None


def test_full_trace_runner_resumes_from_persisted_gate4_when_available(tmp_path, monkeypatch):
    calls = {}

    def fail_if_rebuilt(*args, **kwargs):
        raise AssertionError("full runner should resume from persisted Gate4 artifacts")

    monkeypatch.setattr(run_resnet50_full_trace_gcl, "run_resnet50_gate1_to_gate7", fail_if_rebuilt)
    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "resume_resnet50_gate5_to_gate9_from_disk",
        _fake_success_resume(calls),
    )
    root = tmp_path / "formal_root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"
    gate0_manifest = json.loads((root / "gate0_trace_acquisition_manifest.json").read_text())
    _write_complete_gate4_artifact_set(out_dir, gate0_manifest)

    result = run_resnet50_full_trace_gcl.run_full_trace_reproduction(
        input_root=root,
        out_dir=out_dir,
        seed=20260608,
        baseline_artifacts=None,
    )

    assert calls["resume_out_dir"] == out_dir
    assert calls["resume_seed"] == 20260608
    assert calls["resume_baseline_artifacts_path"] is None
    assert result["formal_full_trace_run"] is True
    assert result["input_kernel_invocation_count"] == 265


def test_full_trace_runner_rebuilds_incomplete_gate4_artifact_set(tmp_path, monkeypatch):
    calls = {}

    def fail_if_resumed(*args, **kwargs):
        raise AssertionError("incomplete Gate4 artifacts must be rebuilt")

    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "resume_resnet50_gate5_to_gate9_from_disk",
        fail_if_resumed,
    )
    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "run_resnet50_gate1_to_gate7",
        _fake_success_pipeline(calls),
    )
    root = tmp_path / "formal_root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "graph_tensor_bundle.json").write_text(json.dumps({"ready": True}), encoding="utf-8")

    result = run_resnet50_full_trace_gcl.run_full_trace_reproduction(
        input_root=root,
        out_dir=out_dir,
        seed=20260608,
        baseline_artifacts=None,
    )

    assert calls["invocation_limit"] is None
    assert calls["invocation_ids"] is None
    assert result["formal_full_trace_run"] is True


def test_full_trace_runner_rebuilds_bounded_gate4_from_same_gate0_root(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fail_if_resumed(*args, **kwargs):
        raise AssertionError("bounded Gate4 artifacts must not be resumed as full trace")

    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "resume_resnet50_gate5_to_gate9_from_disk",
        fail_if_resumed,
    )
    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "run_resnet50_gate1_to_gate7",
        _fake_success_pipeline(calls),
    )
    root = tmp_path / "formal_root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "graph_tensor_bundle.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    gate0_manifest = json.loads((root / "gate0_trace_acquisition_manifest.json").read_text())
    (out_dir / "resnet50_trace_adapter_bundle.json").write_text(
        json.dumps(
            {
                "source_gate0_manifest_hash": gate0_manifest["gate0_manifest_hash"],
                "adapter_validation_report": {
                    "status": "passed",
                    "formal_replay_invocation_limit": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_resnet50_full_trace_gcl.run_full_trace_reproduction(
        input_root=root,
        out_dir=out_dir,
        seed=20260608,
        baseline_artifacts=None,
    )

    assert calls["invocation_limit"] is None
    assert calls["invocation_ids"] is None
    assert result["formal_full_trace_run"] is True
    rebuilt_adapter = json.loads((out_dir / "resnet50_trace_adapter_bundle.json").read_text())
    assert "formal_replay_invocation_limit" not in rebuilt_adapter["adapter_validation_report"]


def test_full_trace_runner_clears_bounded_sidecars_before_failed_full_rebuild(
    tmp_path,
    monkeypatch,
):
    def fail_rebuild(*args, **kwargs):
        raise RuntimeError("full rebuild failed before regenerating sidecars")

    monkeypatch.setattr(run_resnet50_full_trace_gcl, "run_resnet50_gate1_to_gate7", fail_rebuild)
    root = tmp_path / "formal_root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    gate0_manifest = json.loads((root / "gate0_trace_acquisition_manifest.json").read_text())
    (out_dir / "graph_tensor_bundle.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (out_dir / "resnet50_trace_adapter_bundle.json").write_text(
        json.dumps(
            {
                "source_gate0_manifest_hash": gate0_manifest["gate0_manifest_hash"],
                "adapter_validation_report": {
                    "status": "passed",
                    "formal_replay_invocation_limit": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    stale_sidecars = [
        "selected_sm_policy_report.json",
        "scope_preview_report.json",
        "cluster_embedding_quality_report.json",
        "cluster_family_alignment_report.json",
        "representative_quality_report.json",
        "cluster_metric_error_report.json",
        "cluster_stability_report.json",
        "cluster_tuning_vector_table.json",
        "tuning_vector_provenance_report.json",
        "tuning_safety_report.json",
        "gate8_tuning_manifest.json",
        "full_vs_sampled_simulation_report.json",
        "sampled_speedup_report.json",
        "sampled_error_report.json",
        "tuning_effect_report.json",
        "gate9_simulator_evaluation_manifest.json",
    ]
    for filename in stale_sidecars:
        (out_dir / filename).write_text(json.dumps({"stale": filename}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="full rebuild failed"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=out_dir,
            seed=20260608,
            baseline_artifacts=None,
        )

    assert (out_dir / "resnet50_full_trace_reproduction_blocker_report.json").exists()
    for filename in stale_sidecars:
        assert not (out_dir / filename).exists()


def test_full_trace_runner_rejects_non_full_gate0_scope(tmp_path):
    root = tmp_path / "root"
    _write_gate0_manifest(root, scope="bounded_invocation_slice")

    with pytest.raises(ValueError, match="input_scope"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=tmp_path / "out",
            seed=20260608,
            baseline_artifacts=None,
        )


def test_full_trace_runner_rejects_debug_gate0_manifest(tmp_path):
    root = tmp_path / "root"
    _write_gate0_manifest(root, artifact_status="debug_not_formal", eligible=False)

    with pytest.raises(ValueError, match="must be formal"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=tmp_path / "out",
            seed=20260608,
            baseline_artifacts=None,
        )


def test_full_trace_runner_rejects_malformed_gate0_manifest_hash(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "run_resnet50_gate1_to_gate7",
        _fake_success_pipeline(calls),
    )
    root = tmp_path / "root"
    _write_gate0_manifest(root, bad_hash=True)

    with pytest.raises(ValueError, match="gate0_manifest_hash"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=tmp_path / "out",
            seed=20260608,
            baseline_artifacts=None,
        )

    assert calls == {}


def test_full_trace_runner_preflight_failure_writes_blocker_and_removes_stale_success(
    tmp_path,
    monkeypatch,
):
    calls = {}
    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "run_resnet50_gate1_to_gate7",
        _fake_success_pipeline(calls),
    )
    root = tmp_path / "root"
    _write_gate0_manifest(root, bad_hash=True)
    out_dir = tmp_path / "out"
    _write_success_manifest(out_dir)

    with pytest.raises(ValueError, match="gate0_manifest_hash"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=out_dir,
            seed=20260608,
            baseline_artifacts=None,
        )

    blocker = json.loads(
        (out_dir / "resnet50_full_trace_reproduction_blocker_report.json").read_text()
    )
    assert blocker["blocker_reason"] == "ValueError"
    assert blocker["resource_status"] == "blocked"
    assert not (out_dir / "resnet50_full_trace_reproduction_manifest.json").exists()
    assert calls == {}


def test_full_trace_runner_cli_writes_manifest(tmp_path, monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "run_resnet50_gate1_to_gate7",
        _fake_success_pipeline(calls),
    )
    root = tmp_path / "root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"

    run_resnet50_full_trace_gcl.main_args(
        [
            "--input-root",
            str(root),
            "--out",
            str(out_dir),
            "--seed",
            "20260608",
        ]
    )

    manifest = json.loads(
        (out_dir / "resnet50_full_trace_reproduction_manifest.json").read_text()
    )
    assert manifest["formal_full_trace_run"] is True
    assert manifest["artifact_presence"]["kernel_embedding_table.json"] is True
    assert manifest["input_cta_record_count"] == 3
    assert "real_resnet50_full_trace" in capsys.readouterr().out


def test_full_trace_runner_removes_stale_blocker_report_on_success(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "run_resnet50_gate1_to_gate7",
        _fake_success_pipeline(calls),
    )
    root = tmp_path / "root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    blocker_path = out_dir / "resnet50_full_trace_reproduction_blocker_report.json"
    blocker_path.write_text(
        json.dumps(
            {
                "artifact_type": "gcl_resnet50_full_trace_reproduction_blocker_report",
                "resource_status": "blocked",
                "blocker_reason": "TimeoutError",
            }
        ),
        encoding="utf-8",
    )

    run_resnet50_full_trace_gcl.run_full_trace_reproduction(
        input_root=root,
        out_dir=out_dir,
        seed=20260608,
        baseline_artifacts=None,
    )

    assert (out_dir / "resnet50_full_trace_reproduction_manifest.json").exists()
    assert not blocker_path.exists()


def test_full_trace_runner_can_append_gnn_acceptance_report(tmp_path, monkeypatch):
    calls = {}

    def fake_pipeline(
        root,
        out_dir,
        seed,
        baseline_artifacts_path=None,
        invocation_limit=None,
        invocation_ids=None,
    ):
        result = _fake_success_pipeline(calls)(
            root,
            out_dir,
            seed,
            baseline_artifacts_path=baseline_artifacts_path,
            invocation_limit=invocation_limit,
            invocation_ids=invocation_ids,
        )
        _write_gnn_acceptance_required_inputs(out_dir)
        return result

    monkeypatch.setattr(run_resnet50_full_trace_gcl, "run_resnet50_gate1_to_gate7", fake_pipeline)
    root = tmp_path / "root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"

    result = run_resnet50_full_trace_gcl.run_full_trace_reproduction(
        input_root=root,
        out_dir=out_dir,
        seed=20260608,
        baseline_artifacts=None,
    )

    assert result["gnn_acceptance_status"] == (
        "weak_acceptance_structure_valid_but_correctness_unproven"
    )
    manifest = json.loads((out_dir / "gnn_acceptance_manifest.json").read_text())
    full_manifest = json.loads(
        (out_dir / "resnet50_full_trace_reproduction_manifest.json").read_text()
    )
    assert manifest["claim_status"] == "structure_valid_embedding_signal_only"
    assert result["gnn_acceptance_manifest_hash"] == manifest[
        "gnn_acceptance_manifest_hash"
    ]
    assert manifest["input_artifact_hashes"]["full_trace_manifest_hash"] == full_manifest[
        "pre_gnn_acceptance_manifest_hash"
    ]
    assert (out_dir / "gnn_acceptance_summary.json").exists()
    assert (out_dir / "gnn_acceptance_report.md").exists()
    assert json.loads((out_dir / "selector_artifacts.json").read_text())[
        "selector_manifest_hash"
    ] == "selector-hash"


def test_acceptance_stage_rejects_missing_gate5_gate6_gate7_inputs(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "run_resnet50_gate1_to_gate7",
        _fake_success_pipeline(calls),
    )
    root = tmp_path / "root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"

    result = run_resnet50_full_trace_gcl.run_full_trace_reproduction(
        input_root=root,
        out_dir=out_dir,
        seed=20260608,
        baseline_artifacts=None,
    )

    manifest = json.loads((out_dir / "gnn_acceptance_manifest.json").read_text())
    assert result["gnn_acceptance_status"] == "not_evaluable_missing_artifacts"
    assert manifest["artifact_type"] == "gcl_gnn_acceptance_blocker_report"
    assert "rgcn_training_run_manifest.json" in manifest["missing_artifacts"]


def test_acceptance_only_stage_preserves_existing_gate_artifacts(tmp_path):
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    full_manifest = {
        "artifact_type": "gcl_resnet50_full_trace_reproduction_manifest",
        "artifact_version": "full_trace_reproduction_manifest_v1",
        "run_scope": "real_resnet50_full_trace",
        "formal_full_trace_run": True,
        "input_kernel_invocation_count": 265,
        "input_cta_record_count": 124876,
        "final_gate": "gate9_report_only",
    }
    full_manifest["full_trace_reproduction_manifest_hash"] = hash_without(
        full_manifest,
        "full_trace_reproduction_manifest_hash",
    )
    (out_dir / "resnet50_full_trace_reproduction_manifest.json").write_text(
        json.dumps(full_manifest),
        encoding="utf-8",
    )
    _write_gnn_acceptance_required_inputs(out_dir)
    (out_dir / "kernel_embedding_table.json").write_text(
        json.dumps({"kernel_embedding_table_hash": "embedding-hash"}),
        encoding="utf-8",
    )
    protected_files = [
        "kernel_embedding_table.json",
        "selector_artifacts.json",
        "gate7_cluster_correctness_manifest.json",
    ]
    before = {
        name: (
            (out_dir / name).read_bytes(),
            (out_dir / name).stat().st_mtime_ns,
        )
        for name in protected_files
    }

    result = run_resnet50_full_trace_gcl.append_gnn_acceptance_report(out_dir)

    assert result["gnn_acceptance_status"] == (
        "weak_acceptance_structure_valid_but_correctness_unproven"
    )
    for name, (content, mtime_ns) in before.items():
        assert (out_dir / name).read_bytes() == content
        assert (out_dir / name).stat().st_mtime_ns == mtime_ns
    assert (out_dir / "gnn_acceptance_manifest.json").exists()
    assert (out_dir / "gnn_acceptance_summary.json").exists()
    assert (out_dir / "gnn_acceptance_report.md").exists()


def test_full_trace_runner_resume_rejects_gate4_from_different_gate0_root(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def fail_if_resumed(*args, **kwargs):
        raise AssertionError("resume should be rejected before Gate5-to-Gate9 execution")

    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "resume_resnet50_gate5_to_gate9_from_disk",
        fail_if_resumed,
    )
    root = tmp_path / "root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_success_manifest(out_dir)
    (out_dir / "graph_tensor_bundle.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    (out_dir / "resnet50_trace_adapter_bundle.json").write_text(
        json.dumps(
            {
                "source_gate0_manifest_hash": "different-gate0-hash",
                "adapter_validation_report": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "representative_sm_trace_manifest.json").write_text(
        json.dumps({"artifact_type": "gcl_representative_sm_trace_manifest"}),
        encoding="utf-8",
    )
    (out_dir / "canonical_graph_bundle.json").write_text(
        json.dumps({"artifact_type": "gcl_resnet50_canonical_graph_bundle"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Gate4 resume artifacts"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=out_dir,
            seed=20260608,
            baseline_artifacts=None,
        )

    blocker = json.loads(
        (out_dir / "resnet50_full_trace_reproduction_blocker_report.json").read_text()
    )
    assert blocker["blocker_reason"] == "ValueError"
    assert not (out_dir / "resnet50_full_trace_reproduction_manifest.json").exists()
    assert calls == {}


def test_full_trace_runner_writes_blocker_report_on_resource_failure(tmp_path, monkeypatch):
    def fake_pipeline(*args, **kwargs):
        raise RuntimeError("out of memory while tensorizing full trace")

    monkeypatch.setattr(run_resnet50_full_trace_gcl, "run_resnet50_gate1_to_gate7", fake_pipeline)
    root = tmp_path / "root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match="out of memory"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=out_dir,
            seed=20260608,
            baseline_artifacts=None,
        )

    blocker = json.loads(
        (out_dir / "resnet50_full_trace_reproduction_blocker_report.json").read_text()
    )
    assert blocker["run_scope"] == "real_resnet50_full_trace"
    assert blocker["formal_full_trace_run"] is False
    assert blocker["blocker_reason"] == "RuntimeError"
    assert not (out_dir / "resnet50_full_trace_reproduction_manifest.json").exists()


@pytest.mark.parametrize(
    "bounded_report",
    [
        {"formal_replay_invocation_limit": 1},
        {"formal_replay_invocation_ids": ["d_0_s_0_k_1"]},
    ],
)
def test_full_trace_runner_rejects_bounded_adapter_bundle(
    tmp_path,
    monkeypatch,
    bounded_report,
):
    def fake_pipeline(
        root,
        out_dir,
        seed,
        baseline_artifacts_path=None,
        invocation_limit=None,
        invocation_ids=None,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "resnet50_trace_adapter_bundle.json").write_text(
            json.dumps({"adapter_validation_report": {"status": "passed", **bounded_report}}),
            encoding="utf-8",
        )
        return {
            "artifact_type": "gcl_resnet50_gate1_7_pipeline_manifest",
            "final_gate": "gate9_report_only",
            "input_kernel_invocation_count": 265,
            "hashes": {},
            "pipeline_manifest_hash": "pipeline-hash",
        }

    monkeypatch.setattr(run_resnet50_full_trace_gcl, "run_resnet50_gate1_to_gate7", fake_pipeline)
    root = tmp_path / "root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="bounded replay adapter"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=out_dir,
            seed=20260608,
            baseline_artifacts=None,
        )

    blocker = json.loads(
        (out_dir / "resnet50_full_trace_reproduction_blocker_report.json").read_text()
    )
    assert blocker["blocker_reason"] == "ValueError"
    assert not (out_dir / "resnet50_full_trace_reproduction_manifest.json").exists()


def test_full_trace_runner_writes_blocker_report_on_deadline(tmp_path, monkeypatch):
    def fake_pipeline(*args, **kwargs):
        raise TimeoutError("full trace reproduction exceeded 1 seconds")

    monkeypatch.setattr(run_resnet50_full_trace_gcl, "run_resnet50_gate1_to_gate7", fake_pipeline)
    root = tmp_path / "root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"

    with pytest.raises(TimeoutError, match="exceeded"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=out_dir,
            seed=20260608,
            baseline_artifacts=None,
            deadline_seconds=1,
        )

    blocker = json.loads(
        (out_dir / "resnet50_full_trace_reproduction_blocker_report.json").read_text()
    )
    assert blocker["blocker_reason"] == "TimeoutError"
    assert blocker["resource_status"] == "blocked"
    assert blocker["deadline_seconds"] == 1
    assert not (out_dir / "resnet50_full_trace_reproduction_manifest.json").exists()


def test_full_trace_runner_script_entrypoint_is_importable_from_repo_root():
    result = subprocess.run(
        [sys.executable, "scripts/run_resnet50_full_trace_gcl.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--input-root" in result.stdout
