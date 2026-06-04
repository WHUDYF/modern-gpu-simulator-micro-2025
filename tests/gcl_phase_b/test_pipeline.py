import json
import subprocess
import sys

import pytest

from experiments.gcl_phase_b.pipeline import (
    ARTIFACT_FILENAMES,
    run_embedding_export_stage_from_disk,
    run_graph_construction_stage_from_disk,
    run_pipeline,
    run_selector_stage_from_disk,
    run_tensorization_stage_from_disk,
)
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.utils import write_json


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
        "readout_manifest.json",
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


def test_pipeline_requires_selected_sm_policy_report(tmp_path):
    manifest = build_representative_sm_trace_manifest()
    del manifest["kernel_invocations"][0]["selected_sm_policy_report_hash"]
    manifest_path = tmp_path / "bad_manifest.json"
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="selected_sm_policy_report_hash"):
        run_pipeline(manifest_path, tmp_path / "bad")


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


def test_from_disk_selector_stage_requires_embedding_table(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "stage_missing_embedding"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir)
    (out_dir / ARTIFACT_FILENAMES["embedding_table"]).unlink()

    with pytest.raises(FileNotFoundError, match="embedding table"):
        run_selector_stage_from_disk(out_dir)
