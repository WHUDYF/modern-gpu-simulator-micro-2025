import subprocess
import sys
from shutil import copytree
import json

import pytest

from experiments.gcl_phase_a.embedding_export import EMBEDDING_DIM
from experiments.gcl_phase_a.pipeline import (
    ARTIFACT_FILENAMES,
    run_embedding_export_stage_from_disk,
    run_pipeline,
    run_selector_stage_from_disk,
)
from experiments.gcl_phase_a.embedding_export import validate_embedding_table
from experiments.gcl_phase_a.utils import hash_without, stable_hash


@pytest.fixture(scope="module")
def pipeline_out_dir(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("pipeline") / "gcl_phase_a"
    run_pipeline(out_dir)
    return out_dir


def test_phase_a_pipeline_e2e(pipeline_out_dir):
    manifest_path = pipeline_out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]
    import json

    manifest = json.loads(manifest_path.read_text())

    assert manifest["fixture_summary"]["kernel_invocation_count"] == 12
    assert len(manifest["hashes"]["graph_hashes"]) == 12
    assert len(manifest["hashes"]["tensor_hashes"]) == 12
    for filename in ARTIFACT_FILENAMES.values():
        assert (pipeline_out_dir / filename).exists()


def test_phase_a_pipeline_cli(tmp_path):
    out_dir = tmp_path / "cli_gcl_phase_a"

    completed = subprocess.run(
        [sys.executable, "-m", "experiments.gcl_phase_a.pipeline", "--out", str(out_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip()
    assert (out_dir / ARTIFACT_FILENAMES["selector_artifacts"]).exists()


def test_pipeline_reports_missing_graph_bundle_for_embedding_export(tmp_path, pipeline_out_dir):
    out_dir = copytree(pipeline_out_dir, tmp_path / "missing_graph")
    (out_dir / ARTIFACT_FILENAMES["graph_bundle"]).unlink()

    with pytest.raises(FileNotFoundError, match="graph bundle"):
        run_embedding_export_stage_from_disk(out_dir)


def test_embedding_export_stage_from_disk_recreates_embedding_table(tmp_path, pipeline_out_dir):
    out_dir = copytree(pipeline_out_dir, tmp_path / "recreate_embedding")
    original_table = json.loads((out_dir / ARTIFACT_FILENAMES["embedding_table"]).read_text())
    (out_dir / ARTIFACT_FILENAMES["embedding_table"]).unlink()

    table = run_embedding_export_stage_from_disk(out_dir)

    validate_embedding_table(table)
    assert (out_dir / ARTIFACT_FILENAMES["embedding_table"]).exists()
    assert table["row_count"] == 12
    assert table["embedding_table_hash"] == original_table["embedding_table_hash"]


def test_embedding_export_stage_from_disk_invalidates_existing_selector_artifacts(tmp_path, pipeline_out_dir):
    out_dir = copytree(pipeline_out_dir, tmp_path / "invalidate_selector")
    selector_path = out_dir / ARTIFACT_FILENAMES["selector_artifacts"]
    assert selector_path.exists()

    run_embedding_export_stage_from_disk(out_dir)

    assert not selector_path.exists()


def test_embedding_export_stage_from_disk_rejects_checkpoint_manifest_mismatch(tmp_path, pipeline_out_dir):
    out_dir = copytree(pipeline_out_dir, tmp_path / "mutated_checkpoint")
    checkpoint_path = out_dir / "rgcn_checkpoint.pt"
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"\nmutated")

    with pytest.raises(ValueError, match="checkpoint_hash"):
        run_embedding_export_stage_from_disk(out_dir)


def test_embedding_export_stage_from_disk_rejects_tensor_checkpoint_mismatch(tmp_path, pipeline_out_dir):
    out_dir = copytree(pipeline_out_dir, tmp_path / "mutated_tensor_bundle")
    graph_path = out_dir / ARTIFACT_FILENAMES["graph_bundle"]
    tensor_path = out_dir / ARTIFACT_FILENAMES["tensor_bundle"]
    graph_bundle = json.loads(graph_path.read_text())
    tensor_bundle = json.loads(tensor_path.read_text())
    graph_bundle["graphs"] = graph_bundle["graphs"][1:]
    tensor_bundle["tensors"] = tensor_bundle["tensors"][1:]
    graph_path.write_text(json.dumps(graph_bundle, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    tensor_path.write_text(json.dumps(tensor_bundle, ensure_ascii=False, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError, match="source_tensor_hashes"):
        run_embedding_export_stage_from_disk(out_dir)


def test_pipeline_reports_missing_embedding_table_for_selector(tmp_path, pipeline_out_dir):
    out_dir = copytree(pipeline_out_dir, tmp_path / "missing_embedding")
    (out_dir / ARTIFACT_FILENAMES["embedding_table"]).unlink()

    with pytest.raises(FileNotFoundError, match="embedding table"):
        run_selector_stage_from_disk(out_dir)


def test_selector_stage_from_disk_recreates_selector_artifacts(tmp_path, pipeline_out_dir):
    out_dir = copytree(pipeline_out_dir, tmp_path / "recreate_selector")
    selector_path = out_dir / ARTIFACT_FILENAMES["selector_artifacts"]
    selector_path.unlink()

    artifacts = run_selector_stage_from_disk(out_dir)

    assert selector_path.exists()
    assert artifacts["artifact_type"] == "gcl_m0_selector_artifacts"
    assert artifacts["selector_manifest_hash"]


def test_selector_stage_from_disk_preserves_non_default_seed(tmp_path):
    out_dir = tmp_path / "non_default_seed"
    run_pipeline(out_dir, seed=123)
    selector_path = out_dir / ARTIFACT_FILENAMES["selector_artifacts"]
    original_artifacts = json.loads(selector_path.read_text())
    selector_path.unlink()

    repaired_artifacts = run_selector_stage_from_disk(out_dir)

    assert selector_path.exists()
    assert repaired_artifacts["structural_evaluation_artifacts"]["seed"] == 123
    assert repaired_artifacts["selector_manifest_hash"] == original_artifacts["selector_manifest_hash"]


def test_selector_stage_from_disk_refreshes_pipeline_manifest(tmp_path, pipeline_out_dir):
    out_dir = copytree(pipeline_out_dir, tmp_path / "refresh_manifest")
    embedding_path = out_dir / ARTIFACT_FILENAMES["embedding_table"]
    manifest_path = out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]
    embedding_table = json.loads(embedding_path.read_text())
    for row in embedding_table["rows"]:
        row["embedding"] = [0.0] * EMBEDDING_DIM
        row["embedding_hash"] = hash_without(row, "embedding_hash")
    embedding_table["embedding_table_hash"] = hash_without(embedding_table, "embedding_table_hash")
    embedding_path.write_text(json.dumps(embedding_table, ensure_ascii=False, sort_keys=True, indent=2) + "\n")

    repaired_artifacts = run_selector_stage_from_disk(out_dir)
    repaired_manifest = json.loads(manifest_path.read_text())
    expected_manifest_hash = stable_hash(
        {key: value for key, value in repaired_manifest.items() if key != "pipeline_manifest_hash"}
    )

    assert repaired_manifest["hashes"]["embedding_table_hash"] == embedding_table["embedding_table_hash"]
    assert repaired_manifest["hashes"]["selector_manifest_hash"] == repaired_artifacts["selector_manifest_hash"]
    assert repaired_manifest["pipeline_manifest_hash"] == expected_manifest_hash


def test_repair_stages_refresh_manifest_paths_after_copy(tmp_path, pipeline_out_dir):
    out_dir = copytree(pipeline_out_dir, tmp_path / "copied_pipeline")

    run_embedding_export_stage_from_disk(out_dir)
    manifest_after_export = json.loads((out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text())

    for artifact_key, filename in ARTIFACT_FILENAMES.items():
        if artifact_key == "pipeline_manifest":
            continue
        assert manifest_after_export["paths"][artifact_key] == str(out_dir / filename)

    run_selector_stage_from_disk(out_dir)
    manifest_after_selector = json.loads((out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).read_text())

    for artifact_key, filename in ARTIFACT_FILENAMES.items():
        if artifact_key == "pipeline_manifest":
            continue
        assert manifest_after_selector["paths"][artifact_key] == str(out_dir / filename)
