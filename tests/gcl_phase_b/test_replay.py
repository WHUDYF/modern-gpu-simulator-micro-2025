import json

import pytest

from experiments.gcl_phase_b.pipeline import (
    ARTIFACT_FILENAMES,
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
