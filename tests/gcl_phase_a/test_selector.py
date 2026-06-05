import pytest

from experiments.gcl_phase_a.embedding_export import EMBEDDING_DIM, REPRESENTATION_MODE
from experiments.gcl_phase_a.selector import select_representatives
from experiments.gcl_phase_a.utils import hash_without


def _synthetic_table(row_count=12, embedding_dim=EMBEDDING_DIM):
    rows = []
    for index in range(row_count):
        base = -1.0 if index < row_count // 2 else 1.0
        embedding = [0.0] * embedding_dim
        for dim in range(embedding_dim):
            embedding[dim] = round(base + index * 0.001 + dim * 0.00001, 8)
        row = {
            "record_id": f"gcl_embedding:{index:04d}",
            "kernel_invocation_id": f"kernel_{index:02d}",
            "representation_mode": REPRESENTATION_MODE,
            "embedding_dim": embedding_dim,
            "embedding": embedding,
            "source_graph_hash": f"graph_hash_{index}",
            "encoder_manifest_hash": "encoder_manifest_hash",
            "weight_input": {"graph_id": f"graph:{index}", "node_count": 10, "edge_count": 20},
        }
        row["embedding_hash"] = hash_without(row, "embedding_hash")
        rows.append(row)
    table = {
        "artifact_type": "gcl_kernel_embedding_table",
        "representation_mode": REPRESENTATION_MODE,
        "embedding_dim": embedding_dim,
        "row_count": row_count,
        "rows": rows,
        "encoder_manifest_hash": "encoder_manifest_hash",
    }
    table["embedding_table_hash"] = hash_without(table, "embedding_table_hash")
    return table


def test_selector_outputs_clusters_and_anchors():
    artifacts = select_representatives(_synthetic_table())

    assert artifacts["normalization"]["mode"] == "z_score"
    assert artifacts["silhouette_report"]["mode"] == "silhouette_k"
    assert artifacts["silhouette_report"]["selected_k"] >= 2
    assert len(artifacts["cluster_assignments"]) == 12
    assert artifacts["representative_anchor_table"]
    assert artifacts["structural_evaluation_artifacts"]["anchor_count"] == len(
        artifacts["representative_anchor_table"]
    )
    assert artifacts["selector_manifest_hash"]


def test_selector_reduces_k_when_embeddings_collapse_to_one_unique_point():
    table = _synthetic_table()
    for row in table["rows"]:
        row["embedding"] = [0.0] * EMBEDDING_DIM
        row["embedding_hash"] = hash_without(row, "embedding_hash")
    table["embedding_table_hash"] = hash_without(table, "embedding_table_hash")

    artifacts = select_representatives(table)

    assert artifacts["silhouette_report"]["selected_k"] == 1
    assert {assignment["cluster_id"] for assignment in artifacts["cluster_assignments"]} == {0}
    assert artifacts["structural_evaluation_artifacts"]["cluster_count"] == 1
    assert artifacts["structural_evaluation_artifacts"]["anchor_count"] == 1
    assert len(artifacts["representative_anchor_table"]) == 1


def test_selector_handles_single_embedding_table_as_single_cluster():
    artifacts = select_representatives(_synthetic_table(row_count=1), seed=7)

    assert artifacts["silhouette_report"]["selected_k"] == 1
    assert artifacts["silhouette_report"]["fallback_reason"] == "single_embedding_batch"
    assert artifacts["cluster_assignments"] == [
        {
            "record_id": "gcl_embedding:0000",
            "kernel_invocation_id": "kernel_00",
            "cluster_id": 0,
        }
    ]
    assert artifacts["representative_anchor_table"][0]["representative_record_id"] == (
        "gcl_embedding:0000"
    )
    assert artifacts["structural_evaluation_artifacts"]["row_count"] == 1
    assert artifacts["structural_evaluation_artifacts"]["cluster_count"] == 1
    assert artifacts["structural_evaluation_artifacts"]["anchor_count"] == 1
    assert artifacts["structural_evaluation_artifacts"]["seed"] == 7


def test_selector_rejects_empty_embedding_table():
    table = _synthetic_table(row_count=2)
    table["rows"] = []
    table["row_count"] = 0
    table["embedding_table_hash"] = hash_without(table, "embedding_table_hash")

    with pytest.raises(ValueError, match="row count"):
        select_representatives(table)


def test_selector_rejects_embedding_dim_mismatch():
    table = _synthetic_table()
    table["embedding_dim"] = 128

    with pytest.raises(ValueError, match="embedding_dim"):
        select_representatives(table)
