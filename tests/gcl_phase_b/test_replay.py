import json

import pytest

from experiments.gcl_phase_b.pipeline import (
    ARTIFACT_FILENAMES,
    PhaseBResourceError,
    run_pipeline,
    validate_phase_b_replay_from_disk,
)
from experiments.gcl_phase_b.embedding_export import _kernel_embedding_hash
from experiments.gcl_phase_b.selector import select_phase_b_representatives
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.utils import hash_without, stable_hash, write_json


def _refresh_pipeline_manifest_hashes(out_dir, hash_updates):
    manifest_path = out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["hashes"] = {**manifest["hashes"], **hash_updates}
    manifest["pipeline_manifest_hash"] = stable_hash(
        {
            key: value
            for key, value in manifest.items()
            if key != "pipeline_manifest_hash"
        }
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))


def _graph_tensor_bundle_reference_hash(tensors):
    return stable_hash(
        {
            "artifact_type": "gcl_resnet50_graph_tensor_bundle_reference",
            "tensor_hashes": [tensor["tensor_hash"] for tensor in tensors],
        }
    )


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


def test_phase_b_replay_rejects_stale_pipeline_manifest_paths_after_hash_refresh(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    out_dir = tmp_path / "stale_manifest_paths"
    run_pipeline(manifest_path, out_dir, seed=42)

    pipeline_manifest_path = out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]
    pipeline_manifest = json.loads(pipeline_manifest_path.read_text())
    pipeline_manifest["paths"]["trace_manifest"] = str(tmp_path / "wrong" / "trace_manifest.json")
    pipeline_manifest["pipeline_manifest_hash"] = stable_hash(
        {
            key: value
            for key, value in pipeline_manifest.items()
            if key != "pipeline_manifest_hash"
        }
    )
    pipeline_manifest_path.write_text(json.dumps(pipeline_manifest, sort_keys=True))

    with pytest.raises(ValueError, match="pipeline_manifest paths"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_tampered_selector_artifacts(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    out_dir = tmp_path / "tampered_selector"
    run_pipeline(manifest_path, out_dir, seed=42)

    selector_path = out_dir / ARTIFACT_FILENAMES["selector_artifacts"]
    selector_artifacts = json.loads(selector_path.read_text())
    selector_artifacts["representative_anchor_table"]["anchors"][0][
        "representative_record_id"
    ] = "tampered"
    selector_path.write_text(json.dumps(selector_artifacts, sort_keys=True))

    with pytest.raises(ValueError, match="selector_manifest_hash"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_selector_semantic_tamper_after_hash_refresh(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest(invocation_count=2))
    out_dir = tmp_path / "selector_semantic_tamper"
    run_pipeline(manifest_path, out_dir, seed=42)

    selector_path = out_dir / ARTIFACT_FILENAMES["selector_artifacts"]
    selector_artifacts = json.loads(selector_path.read_text())
    assignments = selector_artifacts["kmeans_cluster_assignment_table"]["assignments"]
    selector_artifacts["kmeans_cluster_assignment_table"]["assignments"] = assignments[:-1]
    selector_artifacts["selector_manifest_hash"] = hash_without(
        selector_artifacts, "selector_manifest_hash"
    )
    selector_path.write_text(json.dumps(selector_artifacts, sort_keys=True))
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {"selector_manifest_hash": selector_artifacts["selector_manifest_hash"]},
    )

    with pytest.raises(ValueError, match="selector cluster_assignments"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_embedding_payload_tamper_after_hash_refresh(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest(invocation_count=2))
    out_dir = tmp_path / "embedding_payload_tamper"
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

    selector_artifacts = select_phase_b_representatives(table, seed=42, allow_debug=True)
    write_json(out_dir / ARTIFACT_FILENAMES["selector_artifacts"], selector_artifacts)
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {
            "embedding_table_hash": table["kernel_embedding_table_hash"],
            "selector_manifest_hash": selector_artifacts["selector_manifest_hash"],
        },
    )

    with pytest.raises(ValueError, match="kernel_embedding_hash coverage"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_embedding_readout_hash_tamper_after_hash_refresh(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest(invocation_count=2))
    out_dir = tmp_path / "embedding_readout_hash_tamper"
    run_pipeline(manifest_path, out_dir, seed=42)

    readout_bundle = json.loads(
        (out_dir / ARTIFACT_FILENAMES["readout_manifest"]).read_text()
    )
    table_path = out_dir / ARTIFACT_FILENAMES["embedding_table"]
    table = json.loads(table_path.read_text())
    table["embeddings"][0]["readout_manifest_hash"] = readout_bundle["manifests"][1][
        "readout_manifest_hash"
    ]
    table["embeddings"][0]["embedding_hash"] = hash_without(
        table["embeddings"][0], "embedding_hash"
    )
    table["kernel_embedding_table_hash"] = hash_without(table, "kernel_embedding_table_hash")
    table_path.write_text(json.dumps(table, sort_keys=True))

    selector_artifacts = select_phase_b_representatives(table, seed=42, allow_debug=True)
    write_json(out_dir / ARTIFACT_FILENAMES["selector_artifacts"], selector_artifacts)
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {
            "embedding_table_hash": table["kernel_embedding_table_hash"],
            "selector_manifest_hash": selector_artifacts["selector_manifest_hash"],
        },
    )

    with pytest.raises(ValueError, match="readout_manifest_hash coverage"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_embedding_table_lineage_retarget_after_hash_refresh(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest(invocation_count=2))
    out_dir = tmp_path / "embedding_lineage_retarget"
    run_pipeline(manifest_path, out_dir, seed=42)

    tensor_bundle = json.loads(
        (out_dir / ARTIFACT_FILENAMES["tensor_bundle"]).read_text()
    )
    tensors = tensor_bundle["tensors"]
    table_expected_bundle_hash = _graph_tensor_bundle_reference_hash(tensors)
    assert table_expected_bundle_hash
    wrong_bundle_hash = stable_hash(
        {
            "artifact_type": "gcl_resnet50_graph_tensor_bundle_reference",
            "tensor_hashes": ["not-the-real-tensor-bundle"],
        }
    )
    assert wrong_bundle_hash != table_expected_bundle_hash
    table_path = out_dir / ARTIFACT_FILENAMES["embedding_table"]
    table = json.loads(table_path.read_text())
    table["source_graph_tensor_bundle_hash"] = wrong_bundle_hash
    table["gate5_lineage"]["source_graph_tensor_bundle_hash"] = wrong_bundle_hash
    table["gate5_lineage_hash"] = hash_without(table["gate5_lineage"])
    for row in table["embeddings"]:
        row["gate5_lineage_hash"] = table["gate5_lineage_hash"]
        row["embedding_hash"] = hash_without(row, "embedding_hash")
    table["kernel_embedding_table_hash"] = hash_without(table, "kernel_embedding_table_hash")
    table_path.write_text(json.dumps(table, sort_keys=True))

    selector_artifacts = select_phase_b_representatives(table, seed=42, allow_debug=True)
    write_json(out_dir / ARTIFACT_FILENAMES["selector_artifacts"], selector_artifacts)
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {
            "embedding_table_hash": table["kernel_embedding_table_hash"],
            "selector_manifest_hash": selector_artifacts["selector_manifest_hash"],
        },
    )

    with pytest.raises(ValueError, match="source_graph_tensor_bundle_hash"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_truncated_embedding_table_even_after_hash_refresh(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest(invocation_count=2))
    out_dir = tmp_path / "truncated_embedding"
    run_pipeline(manifest_path, out_dir, seed=42)

    table_path = out_dir / ARTIFACT_FILENAMES["embedding_table"]
    table = json.loads(table_path.read_text())
    table["embeddings"] = table["embeddings"][:-1]
    table["kernel_embedding_table_hash"] = hash_without(table, "kernel_embedding_table_hash")
    table_path.write_text(json.dumps(table, sort_keys=True))

    selector_artifacts = select_phase_b_representatives(table, seed=42, allow_debug=True)
    write_json(out_dir / ARTIFACT_FILENAMES["selector_artifacts"], selector_artifacts)
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {
            "embedding_table_hash": table["kernel_embedding_table_hash"],
            "selector_manifest_hash": selector_artifacts["selector_manifest_hash"],
        },
    )

    with pytest.raises(ValueError, match="embedding table"):
        validate_phase_b_replay_from_disk(out_dir)


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

    with pytest.raises(ValueError, match="selected_sm_policy_report|scope audit"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_trace_manifest_selected_sm_report_mismatch(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    out_dir = tmp_path / "trace_selection_mismatch"
    run_pipeline(manifest_path, out_dir, seed=42)

    trace_path = out_dir / ARTIFACT_FILENAMES["trace_manifest"]
    trace_manifest = json.loads(trace_path.read_text())
    invocation = trace_manifest["kernel_invocations"][0]
    invocation["selected_sm_reason"] = "tampered_reason"
    invocation["trace_hash"] = hash_without(invocation, "trace_hash")
    trace_manifest["trace_manifest_hash"] = hash_without(trace_manifest, "trace_manifest_hash")
    trace_path.write_text(json.dumps(trace_manifest, sort_keys=True))

    with pytest.raises(ValueError, match="selected_sm_policy_report|scope audit"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_scope_audit_selection_mismatch(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    out_dir = tmp_path / "scope_audit_mismatch"
    run_pipeline(manifest_path, out_dir, seed=42)

    scope_path = out_dir / ARTIFACT_FILENAMES["scope_audits"]
    scope_bundle = json.loads(scope_path.read_text())
    audit = scope_bundle["audits"][0]
    audit["selected_sm"] = 99
    audit["trace_scope_hash"] = hash_without(audit, "trace_scope_hash")
    scope_path.write_text(json.dumps(scope_bundle, sort_keys=True))
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {"trace_scope_hashes": [audit["trace_scope_hash"]]},
    )

    with pytest.raises(ValueError, match="scope audit"):
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


def test_phase_b_replay_rejects_truncated_augmentation_bundle_after_hash_refresh(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest(invocation_count=2))
    out_dir = tmp_path / "truncated_augmentation"
    run_pipeline(manifest_path, out_dir, seed=42)

    augmentation_path = out_dir / ARTIFACT_FILENAMES["augmentation_manifests"]
    augmentation_bundle = json.loads(augmentation_path.read_text())
    augmentation_bundle["manifests"] = augmentation_bundle["manifests"][:-1]
    augmentation_bundle["augmentation_manifest_bundle_hash"] = hash_without(
        augmentation_bundle, "augmentation_manifest_bundle_hash"
    )
    augmentation_path.write_text(json.dumps(augmentation_bundle, sort_keys=True))
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {
            "augmentation_manifest_hashes": [
                manifest["augmentation_manifest_hash"]
                for manifest in augmentation_bundle["manifests"]
            ],
            "augmentation_manifest_bundle_hash": augmentation_bundle[
                "augmentation_manifest_bundle_hash"
            ],
        },
    )

    with pytest.raises(ValueError, match="augmentation manifest count"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_swapped_augmentation_manifests_after_hash_refresh(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest(invocation_count=2))
    out_dir = tmp_path / "swapped_augmentation"
    run_pipeline(manifest_path, out_dir, seed=42)

    augmentation_path = out_dir / ARTIFACT_FILENAMES["augmentation_manifests"]
    augmentation_bundle = json.loads(augmentation_path.read_text())
    augmentation_bundle["manifests"][2] = dict(augmentation_bundle["manifests"][0])
    augmentation_bundle["manifests"][3] = dict(augmentation_bundle["manifests"][1])
    augmentation_bundle["augmentation_manifest_bundle_hash"] = hash_without(
        augmentation_bundle, "augmentation_manifest_bundle_hash"
    )
    augmentation_path.write_text(json.dumps(augmentation_bundle, sort_keys=True))
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {
            "augmentation_manifest_hashes": [
                manifest["augmentation_manifest_hash"]
                for manifest in augmentation_bundle["manifests"]
            ],
            "augmentation_manifest_bundle_hash": augmentation_bundle[
                "augmentation_manifest_bundle_hash"
            ],
        },
    )

    with pytest.raises(ValueError, match="augmentation manifest source tensor"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_scope_audit_before_count_drift_after_hash_refresh(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    out_dir = tmp_path / "scope_audit_before_count_drift"
    run_pipeline(manifest_path, out_dir, seed=42)

    scope_path = out_dir / ARTIFACT_FILENAMES["scope_audits"]
    scope_bundle = json.loads(scope_path.read_text())
    audit = scope_bundle["audits"][0]
    audit["instruction_count_before_scope"] = audit["instruction_count_before_scope"] + 1
    audit["trace_scope_hash"] = hash_without(audit, "trace_scope_hash")
    scope_path.write_text(json.dumps(scope_bundle, sort_keys=True))
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {"trace_scope_hashes": [audit["trace_scope_hash"]]},
    )

    with pytest.raises(ValueError, match="instruction_count_before_scope"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_missing_non_blocked_resource_status(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    out_dir = tmp_path / "missing_non_blocked_resource_status"
    run_pipeline(manifest_path, out_dir, seed=42)

    (out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"]).unlink()

    with pytest.raises(FileNotFoundError, match="resource blocked artifact"):
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


def test_phase_b_replay_rejects_resource_blocked_output_with_stale_success_artifacts(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "blocked_with_stale_success"
    write_json(manifest_path, build_representative_sm_trace_manifest())
    run_pipeline(manifest_path, out_dir, seed=42)

    def fail_training(*args, **kwargs):
        raise PhaseBResourceError("simulated CUDA memory exhaustion")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)
    with pytest.raises(PhaseBResourceError):
        pipeline_module.run_embedding_export_stage_from_disk(out_dir)

    write_json(out_dir / ARTIFACT_FILENAMES["embedding_table"], {"stale": "success"})

    with pytest.raises(ValueError, match="resource-blocked replay contains stale success artifact"):
        validate_phase_b_replay_from_disk(out_dir)


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


def test_phase_b_replay_rejects_resource_blocked_manifest_with_stale_success_hashes(
    tmp_path,
    monkeypatch,
):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "blocked_stale_success_hashes"
    write_json(manifest_path, build_representative_sm_trace_manifest())

    def fail_training(*args, **kwargs):
        raise PhaseBResourceError("simulated CUDA memory exhaustion")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)
    run_pipeline(manifest_path, out_dir, seed=42)

    pipeline_manifest_path = out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]
    pipeline_manifest = json.loads(pipeline_manifest_path.read_text())
    pipeline_manifest["hashes"]["embedding_table_hash"] = "stale-success-hash"
    pipeline_manifest["pipeline_manifest_hash"] = stable_hash(
        {
            key: value
            for key, value in pipeline_manifest.items()
            if key != "pipeline_manifest_hash"
        }
    )
    pipeline_manifest_path.write_text(json.dumps(pipeline_manifest, sort_keys=True))

    with pytest.raises(ValueError, match="resource-blocked replay contains stale success hash"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_tampered_resource_blocked_artifact(tmp_path, monkeypatch):
    import experiments.gcl_phase_b.pipeline as pipeline_module

    manifest_path = tmp_path / "trace_manifest.json"
    out_dir = tmp_path / "blocked_tampered_resource_status"
    write_json(manifest_path, build_representative_sm_trace_manifest())

    def fail_training(*args, **kwargs):
        raise PhaseBResourceError("simulated CUDA memory exhaustion")

    monkeypatch.setattr(pipeline_module, "train_minimal_contrastive", fail_training)
    run_pipeline(manifest_path, out_dir, seed=42)

    resource_path = out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"]
    resource_status = json.loads(resource_path.read_text())
    resource_status["resource_failure_reason"] = "tampered"
    resource_path.write_text(json.dumps(resource_status, sort_keys=True))

    with pytest.raises(ValueError, match="resource_blocked_hash"):
        validate_phase_b_replay_from_disk(out_dir)


def test_phase_b_replay_rejects_truncated_readout_bundle_after_hash_refresh(tmp_path):
    manifest_path = tmp_path / "trace_manifest.json"
    write_json(manifest_path, build_representative_sm_trace_manifest(invocation_count=2))
    out_dir = tmp_path / "truncated_readout"
    run_pipeline(manifest_path, out_dir, seed=42)

    readout_path = out_dir / ARTIFACT_FILENAMES["readout_manifest"]
    readout_bundle = json.loads(readout_path.read_text())
    readout_bundle["manifests"] = readout_bundle["manifests"][:-1]
    readout_bundle["readout_manifest_bundle_hash"] = hash_without(
        readout_bundle, "readout_manifest_bundle_hash"
    )
    readout_path.write_text(json.dumps(readout_bundle, sort_keys=True))
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {
            "readout_manifest_hashes": [
                manifest["readout_manifest_hash"]
                for manifest in readout_bundle["manifests"]
            ],
            "readout_manifest_bundle_hash": readout_bundle["readout_manifest_bundle_hash"],
        },
    )

    with pytest.raises(ValueError, match="readout manifest count"):
        validate_phase_b_replay_from_disk(out_dir)
