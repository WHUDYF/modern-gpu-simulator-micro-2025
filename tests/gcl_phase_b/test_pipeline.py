import json
import shutil
import subprocess
import sys

import pytest

from experiments.gcl_phase_b.embedding_export import _kernel_embedding_hash
from experiments.gcl_phase_b.pipeline import (
    ARTIFACT_FILENAMES,
    run_embedding_export_stage_from_disk,
    run_graph_construction_stage_from_disk,
    run_pipeline,
    run_selector_stage_from_disk,
    run_tensorization_stage_from_disk,
    validate_phase_b_replay_from_disk,
)
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.utils import hash_without, stable_hash, write_json


def test_phase_b_pipeline_e2e_on_eligible_trace_batch(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "phase_b"
    write_json(manifest_path, build_representative_sm_trace_manifest())

    manifest = run_pipeline(manifest_path, out_dir)

    assert manifest["artifact_type"] == "gcl_phase_b_pipeline_manifest"
    for filename in ARTIFACT_FILENAMES.values():
        assert (out_dir / filename).exists()
    assert manifest["hashes"]["selection_hashes"]
    assert manifest["hashes"]["trace_scope_hashes"]
    assert manifest["hashes"]["graph_hashes"]
    assert manifest["hashes"]["augmentation_manifest_hashes"]
    assert manifest["hashes"]["readout_manifest_hashes"]
    assert manifest["hashes"]["selector_manifest_hash"]

    expected_artifacts = {
        "trace_manifest.json",
        "selected_sm_policy_report.json",
        "scope_audits.json",
        "graph_bundle.json",
        "graph_size_audits.json",
        "tensor_bundle.json",
        "augmentation_manifests.json",
        "training_report.json",
        "checkpoint_manifest.json",
        "rgcn_training_run_manifest.json",
        "rgcn_checkpoint_manifest.json",
        "readout_manifest.json",
        "gate5_lineage_bundle.json",
        "embedding_export_report.json",
        "embedding_table.json",
        "selector_artifacts.json",
        "pipeline_manifest.json",
    }
    assert expected_artifacts.issubset({path.name for path in out_dir.iterdir()})


def test_phase_b_pipeline_cli(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "phase_b_cli"
    write_json(manifest_path, build_representative_sm_trace_manifest())

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.gcl_phase_b.pipeline",
            "--input",
            str(manifest_path),
            "--out",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip()
    assert (out_dir / ARTIFACT_FILENAMES["selector_artifacts"]).exists()


def test_phase_b_pipeline_cli_accepts_plan_fixture_path(tmp_path):
    fixture_path = "tests/fixtures/gcl_phase_b/representative_sm_trace_manifest.json"
    out_dir = tmp_path / "phase_b_fixture_cli"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.gcl_phase_b.pipeline",
            "--input",
            fixture_path,
            "--out",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip()
    assert (out_dir / ARTIFACT_FILENAMES["selected_sm_policy_report"]).exists()
    assert (out_dir / ARTIFACT_FILENAMES["augmentation_manifests"]).exists()
    assert (out_dir / ARTIFACT_FILENAMES["readout_manifest"]).exists()


def test_training_resource_failure_writes_resource_blocked_artifact(tmp_path, monkeypatch):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "resource_blocked"
    write_json(manifest_path, build_representative_sm_trace_manifest())

    def fail_training(*args, **kwargs):
        raise pipeline_module.PhaseBResourceError("simulated CUDA memory exhaustion")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)

    manifest = run_pipeline(manifest_path, out_dir)

    blocked_path = out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"]
    assert blocked_path.exists()
    blocked = json.loads(blocked_path.read_text())
    assert blocked["failed_stage"] == "training"
    assert "simulated CUDA memory exhaustion" in blocked["resource_failure_reason"]
    assert manifest["resource_blocked"] is True
    assert not (out_dir / ARTIFACT_FILENAMES["embedding_table"]).exists()


def test_resource_blocked_rerun_removes_stale_success_artifacts(tmp_path, monkeypatch):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "blocked_rerun"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)
    stale_pipeline_hash = json.loads(
        (out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text()
    )["pipeline_manifest_hash"]
    assert (out_dir / ARTIFACT_FILENAMES["embedding_table"]).exists()
    assert (out_dir / ARTIFACT_FILENAMES["tensor_bundle"]).exists()
    assert (out_dir / ARTIFACT_FILENAMES["augmentation_manifests"]).exists()

    def fail_training(*args, **kwargs):
        raise pipeline_module.PhaseBResourceError("simulated CUDA memory exhaustion")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)
    manifest = run_pipeline(manifest_path, out_dir, seed=43)

    assert manifest["resource_blocked"] is True
    tensor_bundle = json.loads((out_dir / ARTIFACT_FILENAMES["tensor_bundle"]).read_text())
    tensors = tensor_bundle["tensors"]
    augmentation_bundle = json.loads(
        (out_dir / ARTIFACT_FILENAMES["augmentation_manifests"]).read_text()
    )
    assert manifest["hashes"]["tensor_hashes"] == [tensor["tensor_hash"] for tensor in tensors]
    assert manifest["hashes"]["augmentation_manifest_hashes"] == [
        item["augmentation_manifest_hash"] for item in augmentation_bundle["manifests"]
    ]
    assert manifest["hashes"]["augmentation_manifest_bundle_hash"] == augmentation_bundle[
        "augmentation_manifest_bundle_hash"
    ]
    assert [item["random_seed"] for item in augmentation_bundle["manifests"]] == [43, 44]
    stale_downstream_artifacts = {
        "training_report",
        "checkpoint_manifest",
        "readout_manifest",
        "embedding_table",
        "selector_artifacts",
    }
    for key in stale_downstream_artifacts:
        assert not (out_dir / ARTIFACT_FILENAMES[key]).exists()
    assert not (out_dir / "rgcn_checkpoint.pt").exists()
    assert json.loads(
        (out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text()
    )["pipeline_manifest_hash"] != stale_pipeline_hash
    for stale_hash in {
        "encoder_manifest_hash",
        "readout_manifest_hashes",
        "readout_manifest_bundle_hash",
        "embedding_table_hash",
        "selector_manifest_hash",
        "resource_blocked_hash",
    }:
        if stale_hash == "resource_blocked_hash":
            assert manifest["hashes"][stale_hash]
        else:
            assert manifest["hashes"][stale_hash] is None
    assert manifest["hashes"]["tensor_hashes"]


def test_cuda_oom_runtime_error_writes_resource_blocked_artifact(tmp_path, monkeypatch):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "cuda_oom_resource_blocked"
    write_json(manifest_path, build_representative_sm_trace_manifest())

    def fail_training(*args, **kwargs):
        raise RuntimeError("CUDA out of memory while allocating tensor")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)

    manifest = run_pipeline(manifest_path, out_dir)

    blocked = json.loads((out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"]).read_text())
    assert blocked["failed_stage"] == "training"
    assert "CUDA out of memory" in blocked["resource_failure_reason"]
    assert manifest["resource_blocked"] is True
    assert not (out_dir / ARTIFACT_FILENAMES["embedding_table"]).exists()


def test_selector_resource_failure_writes_resource_blocked_artifact(tmp_path, monkeypatch):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "selector_resource_blocked"
    write_json(manifest_path, build_representative_sm_trace_manifest())

    def fail_selector(*args, **kwargs):
        raise pipeline_module.PhaseBResourceError("simulated selector memory exhaustion")

    monkeypatch.setattr(pipeline_module, "select_phase_b_representatives", fail_selector)

    manifest = run_pipeline(manifest_path, out_dir, seed=42)

    blocked = json.loads((out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"]).read_text())
    assert blocked["failed_stage"] == "selector"
    assert "simulated selector memory exhaustion" in blocked["resource_failure_reason"]
    assert manifest["resource_blocked"] is True
    for key in {"training_report", "checkpoint_manifest", "readout_manifest", "embedding_table", "selector_artifacts"}:
        assert not (out_dir / ARTIFACT_FILENAMES[key]).exists()


def test_non_resource_runtime_error_is_not_marked_resource_blocked(tmp_path, monkeypatch):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "runtime_error"
    write_json(manifest_path, build_representative_sm_trace_manifest())

    def fail_training(*args, **kwargs):
        raise RuntimeError("simulated programming error")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)

    with pytest.raises(RuntimeError, match="simulated programming error"):
        run_pipeline(manifest_path, out_dir)

    assert not (out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"]).exists()


def test_pipeline_requires_selected_sm_policy_report(tmp_path):
    manifest = build_representative_sm_trace_manifest()
    del manifest["kernel_invocations"][0]["selected_sm_policy_report_hash"]
    manifest["trace_manifest_hash"] = hash_without(manifest, "trace_manifest_hash")
    manifest_path = tmp_path / "bad_manifest.json"
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="selected_sm_policy_report_hash"):
        run_pipeline(manifest_path, tmp_path / "bad")


def test_pipeline_requires_inline_selected_sm_policy_report(tmp_path):
    manifest = build_representative_sm_trace_manifest()
    del manifest["kernel_invocations"][0]["selected_sm_policy_report"]
    manifest["kernel_invocations"][0]["trace_hash"] = hash_without(
        manifest["kernel_invocations"][0], "trace_hash"
    )
    manifest["trace_manifest_hash"] = hash_without(manifest, "trace_manifest_hash")
    manifest_path = tmp_path / "bad_inline_manifest.json"
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="selected_sm_policy_report"):
        run_pipeline(manifest_path, tmp_path / "bad_inline")


def test_pipeline_rejects_selected_sm_policy_report_scope_mismatch(tmp_path):
    manifest = build_representative_sm_trace_manifest()
    mismatched_report = build_representative_sm_trace_manifest(selected_sm=0)["kernel_invocations"][0][
        "selected_sm_policy_report"
    ]
    invocation = manifest["kernel_invocations"][0]
    invocation["selected_sm_policy_report"] = mismatched_report
    invocation["selected_sm_policy_report_hash"] = mismatched_report["selection_hash"]
    invocation["trace_hash"] = hash_without(invocation, "trace_hash")
    manifest["trace_manifest_hash"] = hash_without(manifest, "trace_manifest_hash")
    manifest_path = tmp_path / "bad_scope_report_manifest.json"
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="selected_sm_policy_report"):
        run_pipeline(manifest_path, tmp_path / "bad_scope_report")


def test_from_disk_graph_stage_requires_selected_sm_policy_report(tmp_path):
    out_dir = tmp_path / "stage_missing_selection"
    out_dir.mkdir()
    manifest = build_representative_sm_trace_manifest()
    del manifest["kernel_invocations"][0]["selected_sm_policy_report"]
    write_json(out_dir / ARTIFACT_FILENAMES["trace_manifest"], manifest)

    with pytest.raises(FileNotFoundError, match="selected SM policy report"):
        run_graph_construction_stage_from_disk(out_dir)


def test_from_disk_tensorization_stage_requires_graph_size_audit(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_missing_audit"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir)
    (out_dir / ARTIFACT_FILENAMES["graph_size_audits"]).unlink()

    with pytest.raises(FileNotFoundError, match="graph size audit"):
        run_tensorization_stage_from_disk(out_dir)


def test_from_disk_embedding_export_stage_requires_tensor_bundle(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_missing_tensor"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir)
    (out_dir / ARTIFACT_FILENAMES["tensor_bundle"]).unlink()

    with pytest.raises(FileNotFoundError, match="tensor bundle"):
        run_embedding_export_stage_from_disk(out_dir)


def test_from_disk_embedding_export_stage_runs_without_pipeline_manifest(tmp_path):
    out_dir = tmp_path / "stage_embedding_without_manifest"
    out_dir.mkdir()
    manifest = build_representative_sm_trace_manifest()
    write_json(out_dir / ARTIFACT_FILENAMES["trace_manifest"], manifest)
    write_json(
        out_dir / ARTIFACT_FILENAMES["selected_sm_policy_report"],
        {
            "artifact_type": "gcl_phase_b_selected_sm_policy_report_bundle",
            "reports": [
                invocation["selected_sm_policy_report"]
                for invocation in manifest["kernel_invocations"]
            ],
        },
    )
    run_graph_construction_stage_from_disk(out_dir)
    run_tensorization_stage_from_disk(out_dir)

    table = run_embedding_export_stage_from_disk(out_dir, seed=42)
    artifacts = run_selector_stage_from_disk(out_dir, seed=42)

    assert table["kernel_embedding_table_hash"]
    assert (out_dir / ARTIFACT_FILENAMES["embedding_table"]).exists()
    assert artifacts["selector_manifest_hash"]
    assert (out_dir / ARTIFACT_FILENAMES["selector_artifacts"]).exists()
    assert not (out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).exists()


def test_from_disk_embedding_and_selector_stages_reuse_pipeline_seed(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_seed"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    manifest = run_pipeline(manifest_path, out_dir, seed=42)

    table = run_embedding_export_stage_from_disk(out_dir)
    artifacts = run_selector_stage_from_disk(out_dir)

    training_report = json.loads((out_dir / ARTIFACT_FILENAMES["training_report"]).read_text())
    assert training_report["checkpoint_manifest"]["seed"] == 42
    assert table["kernel_embedding_table_hash"] == manifest["hashes"]["embedding_table_hash"]
    assert artifacts["structural_evaluation_artifacts"]["seed"] == 42


def test_from_disk_embedding_and_selector_stages_respect_explicit_seed_override(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_seed_override"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)
    captured = {}
    original_export = pipeline_module.run_embedding_export
    original_select = pipeline_module._select_phase_b_representatives_for_artifact_status

    def spy_export(tensors, out_dir, seed):
        captured["embedding_seed"] = seed
        return original_export(tensors, out_dir, seed=seed)

    def spy_select(table, seed, out_dir):
        captured.setdefault("selector_seeds", []).append(seed)
        return original_select(table, seed=seed, out_dir=out_dir)

    monkeypatch.setattr(pipeline_module, "run_embedding_export", spy_export)
    monkeypatch.setattr(
        pipeline_module,
        "_select_phase_b_representatives_for_artifact_status",
        spy_select,
    )

    run_embedding_export_stage_from_disk(out_dir, seed=7)
    run_selector_stage_from_disk(out_dir, seed=9)

    assert captured["embedding_seed"] == 7
    assert captured["selector_seeds"] == [7, 9]


def test_from_disk_embedding_stage_refreshes_downstream_manifest_hashes(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_embedding_refresh"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)

    pipeline_manifest_path = out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]
    pipeline_manifest = json.loads(pipeline_manifest_path.read_text())
    pipeline_manifest["seed"] = 43
    pipeline_manifest["pipeline_manifest_hash"] = stable_hash(
        {
            key: value
            for key, value in pipeline_manifest.items()
            if key != "pipeline_manifest_hash"
        }
    )
    pipeline_manifest_path.write_text(json.dumps(pipeline_manifest, sort_keys=True))

    table = run_embedding_export_stage_from_disk(out_dir)
    validation = validate_phase_b_replay_from_disk(out_dir)
    refreshed_manifest = json.loads(pipeline_manifest_path.read_text())
    augmentation_bundle = json.loads(
        (out_dir / ARTIFACT_FILENAMES["augmentation_manifests"]).read_text()
    )
    selector_artifacts = json.loads(
        (out_dir / ARTIFACT_FILENAMES["selector_artifacts"]).read_text()
    )

    assert refreshed_manifest["seed"] == 43
    assert refreshed_manifest["hashes"]["augmentation_manifest_hashes"] == [
        manifest["augmentation_manifest_hash"]
        for manifest in augmentation_bundle["manifests"]
    ]
    assert refreshed_manifest["hashes"]["augmentation_manifest_bundle_hash"] == augmentation_bundle[
        "augmentation_manifest_bundle_hash"
    ]
    assert [manifest["random_seed"] for manifest in augmentation_bundle["manifests"]] == [43, 44]
    assert refreshed_manifest["hashes"]["embedding_table_hash"] == table[
        "kernel_embedding_table_hash"
    ]
    assert refreshed_manifest["hashes"]["selector_manifest_hash"] == selector_artifacts[
        "selector_manifest_hash"
    ]
    assert validation["embedding_table_hash"] == table["kernel_embedding_table_hash"]


def test_from_disk_embedding_stage_clears_resource_blocked_after_success(tmp_path, monkeypatch):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_blocked_recovery"
    write_json(manifest_path, build_representative_sm_trace_manifest())

    def fail_training(*args, **kwargs):
        raise pipeline_module.PhaseBResourceError("simulated CUDA memory exhaustion")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)
    blocked_manifest = run_pipeline(manifest_path, out_dir, seed=42)
    assert blocked_manifest["resource_blocked"] is True

    monkeypatch.undo()
    table = run_embedding_export_stage_from_disk(out_dir)
    validation = validate_phase_b_replay_from_disk(out_dir)
    refreshed_manifest = json.loads(
        (out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text()
    )

    assert refreshed_manifest["resource_blocked"] is False
    assert refreshed_manifest["hashes"]["resource_blocked_hash"] is not None
    assert validation["embedding_table_hash"] == table["kernel_embedding_table_hash"]


def test_from_disk_embedding_stage_failure_marks_resource_blocked_and_clears_stale_outputs(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_embedding_failure"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)
    assert (out_dir / ARTIFACT_FILENAMES["embedding_table"]).exists()

    def fail_training(*args, **kwargs):
        raise pipeline_module.PhaseBResourceError("simulated CUDA memory exhaustion")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)
    with pytest.raises(pipeline_module.PhaseBResourceError):
        run_embedding_export_stage_from_disk(out_dir)

    blocked = json.loads((out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text())
    assert blocked["resource_blocked"] is True
    for key in {"training_report", "checkpoint_manifest", "readout_manifest", "embedding_table", "selector_artifacts"}:
        assert not (out_dir / ARTIFACT_FILENAMES[key]).exists()
    assert not (out_dir / "rgcn_checkpoint.pt").exists()
    for stale_hash in {
        "encoder_manifest_hash",
        "readout_manifest_hashes",
        "readout_manifest_bundle_hash",
        "embedding_table_hash",
        "selector_manifest_hash",
    }:
        assert blocked["hashes"][stale_hash] is None


def test_from_disk_embedding_stage_reuses_exported_readout_bundle(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_reuse_exported_readout"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)

    def duplicate_readout_call(*args, **kwargs):
        raise AssertionError("embedding stage must reuse the readout bundle returned by export")

    monkeypatch.setattr(pipeline_module, "build_readout_manifest_bundle", duplicate_readout_call)
    table = run_embedding_export_stage_from_disk(out_dir)

    assert table["kernel_embedding_table_hash"]


def test_from_disk_embedding_stage_rejects_stale_tensor_bundle_against_graphs(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_embedding_stale_tensor"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)

    graph_bundle_path = out_dir / ARTIFACT_FILENAMES["graph_bundle"]
    graph_bundle = json.loads(graph_bundle_path.read_text())
    graph_bundle["graphs"][0]["graph_hash"] = "new_graph_hash_after_rebuild"
    graph_bundle_path.write_text(json.dumps(graph_bundle, sort_keys=True))

    with pytest.raises(ValueError, match="tensor bundle input_graph_hash"):
        run_embedding_export_stage_from_disk(out_dir)


def test_from_disk_embedding_stage_rejects_stale_graph_size_audit(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_embedding_stale_audit"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)

    audit_path = out_dir / ARTIFACT_FILENAMES["graph_size_audits"]
    audit_bundle = json.loads(audit_path.read_text())
    audit_bundle["audits"][0]["node_count"] += 1
    audit_path.write_text(json.dumps(audit_bundle, sort_keys=True))

    with pytest.raises(ValueError, match="node_count mismatch"):
        run_embedding_export_stage_from_disk(out_dir)


def test_from_disk_embedding_stage_failure_writes_refreshed_augmentation_bundle(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_embedding_failure_aug"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)
    pipeline_manifest_path = out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]
    pipeline_manifest = json.loads(pipeline_manifest_path.read_text())
    pipeline_manifest["seed"] = 43
    pipeline_manifest["pipeline_manifest_hash"] = stable_hash(
        {
            key: value
            for key, value in pipeline_manifest.items()
            if key != "pipeline_manifest_hash"
        }
    )
    pipeline_manifest_path.write_text(json.dumps(pipeline_manifest, sort_keys=True))

    def fail_training(*args, **kwargs):
        raise pipeline_module.PhaseBResourceError("simulated CUDA memory exhaustion")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)
    with pytest.raises(pipeline_module.PhaseBResourceError):
        run_embedding_export_stage_from_disk(out_dir)

    blocked_manifest = json.loads(pipeline_manifest_path.read_text())
    augmentation_bundle = json.loads(
        (out_dir / ARTIFACT_FILENAMES["augmentation_manifests"]).read_text()
    )

    assert [manifest["random_seed"] for manifest in augmentation_bundle["manifests"]] == [43, 44]
    assert blocked_manifest["hashes"]["augmentation_manifest_hashes"] == [
        manifest["augmentation_manifest_hash"]
        for manifest in augmentation_bundle["manifests"]
    ]
    assert blocked_manifest["hashes"]["augmentation_manifest_bundle_hash"] == augmentation_bundle[
        "augmentation_manifest_bundle_hash"
    ]


def test_from_disk_repair_refreshes_pipeline_manifest_paths_after_copy(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    original_out = tmp_path / "stage_original_paths"
    copied_out = tmp_path / "stage_copied_paths"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, original_out, seed=42)
    shutil.copytree(original_out, copied_out)

    run_selector_stage_from_disk(copied_out)
    refreshed_manifest = json.loads(
        (copied_out / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text()
    )

    assert refreshed_manifest["paths"]["embedding_table"] == str(
        copied_out / ARTIFACT_FILENAMES["embedding_table"]
    )
    assert all(str(copied_out) in path for path in refreshed_manifest["paths"].values())


def test_from_disk_selector_stage_refreshes_pipeline_manifest_hashes(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_selector_refresh"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)

    table_path = out_dir / ARTIFACT_FILENAMES["embedding_table"]
    table = json.loads(table_path.read_text())
    table["embeddings"][0]["kernel_embedding"][0] = round(
        table["embeddings"][0]["kernel_embedding"][0] + 0.125, 8
    )
    table["embeddings"][0]["kernel_embedding_hash"] = _kernel_embedding_hash(
        table["embeddings"][0]["kernel_embedding"]
    )
    table["embeddings"][0]["embedding_hash"] = hash_without(
        table["embeddings"][0], "embedding_hash"
    )
    table["kernel_embedding_table_hash"] = hash_without(table, "kernel_embedding_table_hash")
    table_path.write_text(json.dumps(table, sort_keys=True))

    artifacts = run_selector_stage_from_disk(out_dir)
    refreshed_manifest = json.loads(
        (out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text()
    )

    assert refreshed_manifest["hashes"]["embedding_table_hash"] == table[
        "kernel_embedding_table_hash"
    ]
    assert refreshed_manifest["hashes"]["selector_manifest_hash"] == artifacts[
        "selector_manifest_hash"
    ]


def test_from_disk_selector_stage_failure_marks_resource_blocked_and_clears_stale_selector(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_selector_resource_failure"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)
    assert (out_dir / ARTIFACT_FILENAMES["selector_artifacts"]).exists()

    def fail_selector(*args, **kwargs):
        raise pipeline_module.PhaseBResourceError("simulated selector memory exhaustion")

    monkeypatch.setattr(
        pipeline_module,
        "_select_phase_b_representatives_for_artifact_status",
        fail_selector,
    )
    with pytest.raises(pipeline_module.PhaseBResourceError):
        run_selector_stage_from_disk(out_dir)

    blocked = json.loads((out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text())
    resource = json.loads((out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"]).read_text())
    assert blocked["resource_blocked"] is True
    assert blocked["hashes"]["selector_manifest_hash"] is None
    embedding_table = json.loads((out_dir / ARTIFACT_FILENAMES["embedding_table"]).read_text())
    readout_manifest = json.loads((out_dir / ARTIFACT_FILENAMES["readout_manifest"]).read_text())
    assert blocked["hashes"]["embedding_table_hash"] == embedding_table[
        "kernel_embedding_table_hash"
    ]
    assert blocked["hashes"]["readout_manifest_bundle_hash"] == readout_manifest[
        "readout_manifest_bundle_hash"
    ]
    assert blocked["hashes"]["readout_manifest_hashes"] == [
        manifest["readout_manifest_hash"] for manifest in readout_manifest["manifests"]
    ]
    assert blocked["hashes"]["encoder_manifest_hash"]
    assert resource["failed_stage"] == "selector"
    assert not (out_dir / ARTIFACT_FILENAMES["selector_artifacts"]).exists()
    assert (out_dir / ARTIFACT_FILENAMES["training_report"]).exists()
    assert (out_dir / ARTIFACT_FILENAMES["checkpoint_manifest"]).exists()
    assert (out_dir / "rgcn_checkpoint.pt").exists()
    validate_phase_b_replay_from_disk(out_dir)


def test_from_disk_embedding_stage_selector_failure_preserves_gate5_outputs(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "embedding_stage_selector_failure"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)
    embedding_path = out_dir / ARTIFACT_FILENAMES["embedding_table"]
    checkpoint_path = out_dir / "rgcn_checkpoint.pt"
    assert embedding_path.exists()
    assert checkpoint_path.exists()

    def fail_selector(*args, **kwargs):
        raise pipeline_module.PhaseBResourceError("simulated selector memory exhaustion")

    monkeypatch.setattr(
        pipeline_module,
        "_select_phase_b_representatives_for_artifact_status",
        fail_selector,
    )

    with pytest.raises(pipeline_module.PhaseBResourceError):
        run_embedding_export_stage_from_disk(out_dir)

    blocked = json.loads((out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text())
    resource = json.loads((out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"]).read_text())
    assert blocked["resource_blocked"] is True
    assert resource["failed_stage"] == "selector"
    table = json.loads(embedding_path.read_text())
    readout_manifest = json.loads(
        (out_dir / ARTIFACT_FILENAMES["readout_manifest"]).read_text()
    )
    assert blocked["hashes"]["embedding_table_hash"] == table[
        "kernel_embedding_table_hash"
    ]
    assert blocked["hashes"]["readout_manifest_bundle_hash"] == readout_manifest[
        "readout_manifest_bundle_hash"
    ]
    assert blocked["hashes"]["readout_manifest_hashes"] == [
        manifest["readout_manifest_hash"] for manifest in readout_manifest["manifests"]
    ]
    assert blocked["hashes"]["encoder_manifest_hash"]
    assert checkpoint_path.exists()
    assert not (out_dir / ARTIFACT_FILENAMES["selector_artifacts"]).exists()
    validate_phase_b_replay_from_disk(out_dir)


def test_from_disk_selector_stage_clears_resource_blocked_after_successful_retry(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_selector_recovery"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)

    def fail_selector(*args, **kwargs):
        raise pipeline_module.PhaseBResourceError("simulated selector memory exhaustion")

    monkeypatch.setattr(
        pipeline_module,
        "_select_phase_b_representatives_for_artifact_status",
        fail_selector,
    )
    with pytest.raises(pipeline_module.PhaseBResourceError):
        run_selector_stage_from_disk(out_dir)

    monkeypatch.undo()
    run_embedding_export_stage_from_disk(out_dir)
    artifacts = run_selector_stage_from_disk(out_dir)
    validation = validate_phase_b_replay_from_disk(out_dir)
    refreshed_manifest = json.loads(
        (out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text()
    )

    assert refreshed_manifest["resource_blocked"] is False
    assert refreshed_manifest["hashes"]["selector_manifest_hash"] == artifacts[
        "selector_manifest_hash"
    ]
    assert validation["selector_manifest_hash"] == artifacts["selector_manifest_hash"]


def test_from_disk_selector_stage_preserves_formal_gate5_lineage_validation(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_selector_formal_lineage"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)

    for key in {
        "gate5_lineage_bundle",
        "rgcn_training_run_manifest",
        "rgcn_checkpoint_manifest",
        "embedding_export_report",
    }:
        assert (out_dir / ARTIFACT_FILENAMES[key]).exists()

    export_report_path = out_dir / ARTIFACT_FILENAMES["embedding_export_report"]
    export_report = json.loads(export_report_path.read_text())
    export_report["source_graph_tensor_bundle_hash"] = "tampered-source-bundle"
    export_report["embedding_export_report_hash"] = hash_without(
        export_report, "embedding_export_report_hash"
    )
    export_report_path.write_text(json.dumps(export_report, sort_keys=True))

    with pytest.raises(ValueError, match="persisted Gate5 manifest objects"):
        run_selector_stage_from_disk(out_dir)


def test_from_disk_graph_stage_refreshes_pipeline_manifest_scope_and_graph_hashes(tmp_path):
    out_dir = tmp_path / "stage_graph_hash_refresh"
    out_dir.mkdir()
    manifest = build_representative_sm_trace_manifest()
    write_json(out_dir / ARTIFACT_FILENAMES["trace_manifest"], manifest)
    write_json(
        out_dir / ARTIFACT_FILENAMES["selected_sm_policy_report"],
        {
            "artifact_type": "gcl_phase_b_selected_sm_policy_report_bundle",
            "reports": [
                invocation["selected_sm_policy_report"]
                for invocation in manifest["kernel_invocations"]
            ],
        },
    )
    write_json(
        out_dir / ARTIFACT_FILENAMES["pipeline_manifest"],
        {
            "artifact_type": "gcl_phase_b_pipeline_manifest",
            "seed": 42,
            "resource_blocked": False,
            "paths": {},
            "hashes": {
                "selection_hashes": ["stale"],
                "trace_scope_hashes": ["stale"],
                "graph_hashes": ["stale"],
                "graph_size_audit_hashes": ["stale"],
            },
            "pipeline_manifest_hash": "stale",
        },
    )

    graphs = run_graph_construction_stage_from_disk(out_dir)
    refreshed_manifest = json.loads(
        (out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text()
    )
    audit_bundle = json.loads((out_dir / ARTIFACT_FILENAMES["graph_size_audits"]).read_text())
    scope_bundle = json.loads((out_dir / ARTIFACT_FILENAMES["scope_audits"]).read_text())

    assert refreshed_manifest["hashes"]["selection_hashes"] == [
        invocation["selected_sm_policy_report_hash"]
        for invocation in manifest["kernel_invocations"]
    ]
    assert refreshed_manifest["hashes"]["trace_scope_hashes"] == [
        audit["trace_scope_hash"] for audit in scope_bundle["audits"]
    ]
    assert refreshed_manifest["hashes"]["graph_hashes"] == [
        graph["graph_hash"] for graph in graphs
    ]
    assert refreshed_manifest["hashes"]["graph_size_audit_hashes"] == [
        audit["graph_size_audit_hash"] for audit in audit_bundle["audits"]
    ]


def test_from_disk_graph_stage_invalidates_stale_tensor_and_downstream_artifacts(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_graph_invalidates_downstream"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)

    run_graph_construction_stage_from_disk(out_dir)

    invalidated_keys = {
        "tensor_bundle",
        "augmentation_manifests",
        "training_report",
        "checkpoint_manifest",
        "readout_manifest",
        "embedding_table",
        "selector_artifacts",
        "resource_blocked_artifact",
    }
    for key in invalidated_keys:
        assert not (out_dir / ARTIFACT_FILENAMES[key]).exists()
    assert not (out_dir / "rgcn_checkpoint.pt").exists()
    refreshed_manifest = json.loads(
        (out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text()
    )
    for stale_hash in {
        "tensor_hashes",
        "augmentation_manifest_hashes",
        "augmentation_manifest_bundle_hash",
        "encoder_manifest_hash",
        "readout_manifest_hashes",
        "readout_manifest_bundle_hash",
        "embedding_table_hash",
        "selector_manifest_hash",
        "resource_blocked_hash",
    }:
        assert refreshed_manifest["hashes"][stale_hash] is None


def test_from_disk_graph_stage_rejects_selected_sm_policy_report_scope_mismatch(tmp_path):
    out_dir = tmp_path / "stage_graph_scope_mismatch"
    out_dir.mkdir()
    manifest = build_representative_sm_trace_manifest()
    mismatched_report = build_representative_sm_trace_manifest(selected_sm=0)["kernel_invocations"][0][
        "selected_sm_policy_report"
    ]
    invocation = manifest["kernel_invocations"][0]
    invocation["selected_sm_policy_report"] = mismatched_report
    invocation["selected_sm_policy_report_hash"] = mismatched_report["selection_hash"]
    invocation["trace_hash"] = hash_without(invocation, "trace_hash")
    manifest["trace_manifest_hash"] = hash_without(manifest, "trace_manifest_hash")
    write_json(out_dir / ARTIFACT_FILENAMES["trace_manifest"], manifest)
    write_json(
        out_dir / ARTIFACT_FILENAMES["selected_sm_policy_report"],
        {
            "artifact_type": "gcl_phase_b_selected_sm_policy_report_bundle",
            "reports": [mismatched_report],
        },
    )

    with pytest.raises(ValueError, match="selected_sm_policy_report"):
        run_graph_construction_stage_from_disk(out_dir)


def test_from_disk_tensorization_stage_refreshes_pipeline_manifest_tensor_hashes(tmp_path):
    out_dir = tmp_path / "stage_tensor_hash_refresh"
    out_dir.mkdir()
    manifest = build_representative_sm_trace_manifest()
    write_json(out_dir / ARTIFACT_FILENAMES["trace_manifest"], manifest)
    write_json(
        out_dir / ARTIFACT_FILENAMES["selected_sm_policy_report"],
        {
            "artifact_type": "gcl_phase_b_selected_sm_policy_report_bundle",
            "reports": [
                invocation["selected_sm_policy_report"]
                for invocation in manifest["kernel_invocations"]
            ],
        },
    )
    run_graph_construction_stage_from_disk(out_dir)
    write_json(
        out_dir / ARTIFACT_FILENAMES["pipeline_manifest"],
        {
            "artifact_type": "gcl_phase_b_pipeline_manifest",
            "seed": 42,
            "resource_blocked": False,
            "paths": {},
            "hashes": {"tensor_hashes": ["stale"]},
            "pipeline_manifest_hash": "stale",
        },
    )

    tensors = run_tensorization_stage_from_disk(out_dir)
    refreshed_manifest = json.loads(
        (out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text()
    )

    assert refreshed_manifest["hashes"]["tensor_hashes"] == [
        tensor["tensor_hash"] for tensor in tensors
    ]


def test_from_disk_tensorization_stage_invalidates_stale_embedding_and_selector_artifacts(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_tensor_invalidates_downstream"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)

    run_tensorization_stage_from_disk(out_dir)

    invalidated_keys = {
        "augmentation_manifests",
        "training_report",
        "checkpoint_manifest",
        "readout_manifest",
        "embedding_table",
        "selector_artifacts",
        "resource_blocked_artifact",
    }
    for key in invalidated_keys:
        assert not (out_dir / ARTIFACT_FILENAMES[key]).exists()
    assert not (out_dir / "rgcn_checkpoint.pt").exists()
    refreshed_manifest = json.loads(
        (out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text()
    )
    assert refreshed_manifest["hashes"]["tensor_hashes"]
    for stale_hash in {
        "augmentation_manifest_hashes",
        "augmentation_manifest_bundle_hash",
        "encoder_manifest_hash",
        "readout_manifest_hashes",
        "readout_manifest_bundle_hash",
        "embedding_table_hash",
        "selector_manifest_hash",
        "resource_blocked_hash",
    }:
        assert refreshed_manifest["hashes"][stale_hash] is None


def test_from_disk_graph_stage_rebuilds_matching_graph_size_audits(tmp_path):
    out_dir = tmp_path / "stage_graph_rebuild"
    out_dir.mkdir()
    manifest = build_representative_sm_trace_manifest()
    write_json(out_dir / ARTIFACT_FILENAMES["trace_manifest"], manifest)
    write_json(
        out_dir / ARTIFACT_FILENAMES["selected_sm_policy_report"],
        {
            "artifact_type": "gcl_phase_b_selected_sm_policy_report_bundle",
            "reports": [
                invocation["selected_sm_policy_report"]
                for invocation in manifest["kernel_invocations"]
            ],
        },
    )

    graphs = run_graph_construction_stage_from_disk(out_dir)
    tensors = run_tensorization_stage_from_disk(out_dir)

    audit_bundle = json.loads((out_dir / ARTIFACT_FILENAMES["graph_size_audits"]).read_text())
    assert [audit["graph_hash"] for audit in audit_bundle["audits"]] == [
        graph["graph_hash"] for graph in graphs
    ]
    assert tensors


def test_from_disk_selector_stage_requires_embedding_table(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_missing_embedding"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir)
    (out_dir / ARTIFACT_FILENAMES["embedding_table"]).unlink()

    with pytest.raises(FileNotFoundError, match="embedding table"):
        run_selector_stage_from_disk(out_dir)
