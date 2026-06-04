import pytest

from experiments.gcl_phase_a.embedding_export import export_embedding_table, validate_embedding_table
from experiments.gcl_phase_a.graph_builder import build_canonical_graphs
from experiments.gcl_phase_a.tensorizer import tensorize_graphs
from experiments.gcl_phase_a.trace_fixture import build_controlled_trace_fixture
from experiments.gcl_phase_a.train import train_minimal_contrastive


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    fixture = build_controlled_trace_fixture()
    tensors = tensorize_graphs(build_canonical_graphs(fixture))
    report = train_minimal_contrastive(tensors, tmp_path_factory.mktemp("embedding_export"))
    return tensors, report


def test_exports_m0_compatible_embedding_table(trained_artifacts):
    tensors, report = trained_artifacts

    table = export_embedding_table(tensors, report["encoder"], report["checkpoint_manifest"])

    validate_embedding_table(table)
    assert table["row_count"] == 12
    assert table["embedding_dim"] == 256
    for row in table["rows"]:
        assert {
            "record_id",
            "kernel_invocation_id",
            "representation_mode",
            "embedding_dim",
            "embedding",
            "source_graph_hash",
            "encoder_manifest_hash",
            "embedding_hash",
            "weight_input",
        }.issubset(row)
        assert row["embedding_dim"] == 256
        assert len(row["embedding"]) == 256
        assert row["source_graph_hash"] in {tensor["input_graph_hash"] for tensor in tensors}
        assert row["encoder_manifest_hash"] == report["checkpoint_manifest"]["encoder_manifest_hash"]


def test_embedding_export_rejects_projection_output_dimension(trained_artifacts):
    tensors, report = trained_artifacts
    table = export_embedding_table(tensors, report["encoder"], report["checkpoint_manifest"])
    table["rows"][0]["embedding_dim"] = 64

    with pytest.raises(ValueError, match="projection output"):
        validate_embedding_table(table)


def test_embedding_export_rejects_missing_hashes(trained_artifacts):
    tensors, report = trained_artifacts
    table = export_embedding_table(tensors, report["encoder"], report["checkpoint_manifest"])
    del table["rows"][0]["source_graph_hash"]

    with pytest.raises(ValueError, match="missing required fields"):
        validate_embedding_table(table)


def test_embedding_export_rejects_augmented_tensor(trained_artifacts):
    tensors, report = trained_artifacts
    tensors = list(tensors)
    tensors[0] = dict(tensors[0])
    tensors[0]["augmentation_manifest"] = {"seed": 1}

    with pytest.raises(ValueError, match="canonical non-augmented"):
        export_embedding_table(tensors, report["encoder"], report["checkpoint_manifest"])
