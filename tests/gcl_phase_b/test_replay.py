import json

import pytest

from experiments.gcl_phase_b.pipeline import (
    ARTIFACT_FILENAMES,
    PhaseBResourceError,
    run_pipeline,
    validate_phase_b_replay_from_disk,
)
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.utils import write_json


def test_phase_b_artifacts_are_replayable(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"

    first = run_pipeline(manifest_path, first_out, seed=42)
    second = run_pipeline(manifest_path, second_out, seed=42)

    assert first["hashes"]["selection_hashes"] == second["hashes"]["selection_hashes"]
    assert first["hashes"]["trace_scope_hashes"] == second["hashes"]["trace_scope_hashes"]
    assert first["hashes"]["graph_hashes"] == second["hashes"]["graph_hashes"]
    assert first["hashes"]["graph_size_audit_hashes"] == second["hashes"]["graph_size_audit_hashes"]
    assert first["hashes"]["tensor_hashes"] == second["hashes"]["tensor_hashes"]
    assert first["hashes"]["augmentation_manifest_hashes"] == second["hashes"]["augmentation_manifest_hashes"]
    assert first["hashes"]["readout_manifest_hashes"] == second["hashes"]["readout_manifest_hashes"]
    assert first["hashes"]["embedding_table_hash"] == second["hashes"]["embedding_table_hash"]
    assert first["hashes"]["selector_manifest_hash"] == second["hashes"]["selector_manifest_hash"]


def test_phase_b_replay_hash_changes_when_selected_sm_changes(tmp_path):
    manifest = build_representative_sm_trace_manifest()
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, manifest)
    first = run_pipeline(manifest_path, tmp_path / "first", seed=42)

    mutated = build_representative_sm_trace_manifest(selected_sm=0)
    mutated_path = tmp_path / "mutated_manifest.json"
    write_json(mutated_path, mutated)
    second = run_pipeline(mutated_path, tmp_path / "second", seed=42)

    assert first["hashes"]["selection_hashes"] != second["hashes"]["selection_hashes"]
    assert first["hashes"]["graph_hashes"] != second["hashes"]["graph_hashes"]
    assert first["hashes"]["augmentation_manifest_hashes"] != second["hashes"]["augmentation_manifest_hashes"]
    assert first["hashes"]["readout_manifest_hashes"] != second["hashes"]["readout_manifest_hashes"]

    first_manifest = json.loads((tmp_path / "first" / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text())
    assert first_manifest["pipeline_manifest_hash"] == first["pipeline_manifest_hash"]


def test_phase_b_disk_replay_validator_accepts_clean_artifacts(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    out_dir = tmp_path / "clean_replay"
    manifest = run_pipeline(manifest_path, out_dir, seed=42)

    validation = validate_phase_b_replay_from_disk(out_dir)

    assert validation["encoder_manifest_hash"] == manifest["hashes"]["encoder_manifest_hash"]
    assert validation["embedding_table_hash"] == manifest["hashes"]["embedding_table_hash"]
    assert validation["selector_manifest_hash"] == manifest["hashes"]["selector_manifest_hash"]


def test_phase_b_replay_rejects_stale_checkpoint_backed_artifacts(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    out_dir = tmp_path / "checkpoint_stale"
    run_pipeline(manifest_path, out_dir, seed=42)

    checkpoint_path = out_dir / "rgcn_checkpoint.pt"
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"stale")

    with pytest.raises(ValueError, match="checkpoint_hash"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_stale_upstream_graph_artifact(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    out_dir = tmp_path / "stale_graph"
    run_pipeline(manifest_path, out_dir, seed=42)

    graph_bundle_path = out_dir / ARTIFACT_FILENAMES["graph_bundle"]
    graph_bundle = json.loads(graph_bundle_path.read_text())
    graph_bundle["graphs"][0]["graph_hash"] = "stale"
    graph_bundle_path.write_text(json.dumps(graph_bundle, sort_keys=True))

    with pytest.raises(ValueError, match="graph"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_stale_selected_sm_policy_report_bundle(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    out_dir = tmp_path / "stale_selection"
    run_pipeline(manifest_path, out_dir, seed=42)

    report_path = out_dir / ARTIFACT_FILENAMES["selected_sm_policy_report"]
    report_bundle = json.loads(report_path.read_text())
    report_bundle["reports"][0]["selection_hash"] = "stale"
    report_path.write_text(json.dumps(report_bundle, sort_keys=True))

    with pytest.raises(ValueError, match="selected_sm_policy_report"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_stale_augmentation_manifest_bundle(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    out_dir = tmp_path / "stale_augmentation"
    run_pipeline(manifest_path, out_dir, seed=42)

    augmentation_path = out_dir / ARTIFACT_FILENAMES["augmentation_manifests"]
    augmentation_bundle = json.loads(augmentation_path.read_text())
    augmentation_bundle["manifests"][0]["augmentation_manifest_hash"] = "stale"
    augmentation_path.write_text(json.dumps(augmentation_bundle, sort_keys=True))

    with pytest.raises(ValueError, match="augmentation_manifest"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_accepts_resource_blocked_artifacts(tmp_path, monkeypatch):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "blocked_replay"
    write_json(manifest_path, build_representative_sm_trace_manifest())

    def fail_training(*args, **kwargs):
        raise PhaseBResourceError("simulated CUDA memory exhaustion")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)
    manifest = run_pipeline(manifest_path, out_dir, seed=42)

    validation = validate_phase_b_replay_from_disk(out_dir)

    assert manifest["resource_blocked"] is True
    assert validation["resource_blocked"] is True


def test_phase_b_replay_rejects_stale_resource_blocked_pipeline_manifest(tmp_path, monkeypatch):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "blocked_stale_manifest"
    write_json(manifest_path, build_representative_sm_trace_manifest())

    def fail_training(*args, **kwargs):
        raise PhaseBResourceError("simulated CUDA memory exhaustion")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)
    run_pipeline(manifest_path, out_dir, seed=42)

    pipeline_manifest_path = out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]
    pipeline_manifest = json.loads(pipeline_manifest_path.read_text())
    pipeline_manifest["seed"] = 43
    pipeline_manifest_path.write_text(json.dumps(pipeline_manifest, sort_keys=True))

    with pytest.raises(ValueError, match="pipeline_manifest_hash"):
        validate_phase_b_replay_from_disk(out_dir)
