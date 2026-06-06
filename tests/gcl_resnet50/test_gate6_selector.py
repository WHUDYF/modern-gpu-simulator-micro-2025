import copy

import pytest

from experiments.gcl_phase_b.embedding_export import _kernel_embedding_hash
from experiments.gcl_phase_b.selector import (
    select_phase_b_representatives,
    validate_gate6_selector_artifacts,
)
from experiments.gcl_phase_b.utils import hash_without


def _embedding_row(index, vector):
    row = {
        "record_id": f"row-{index}",
        "kernel_invocation_id": f"resnet50_k{index:05d}",
        "graph_id": f"graph-{index}",
        "source_tensor_hash": f"tensor-{index}",
        "source_graph_hash": f"graph-hash-{index}",
        "representation_mode": "gcl_resnet50_rgcn_selected_sm_kernel_embedding",
        "input_representation_mode": "gcl_resnet50_mem_ref_only",
        "pseudo_node_mode": "mem_ref_only",
        "paper_reproduction_mode": "strict_gcl_sampler_reproduction",
        "collection_scope": "single_representative_sm_all_ctas",
        "selected_sm": index,
        "embedding_dim": 256,
        "kernel_embedding": vector,
        "kernel_embedding_hash": _kernel_embedding_hash(vector),
        "encoder_manifest_hash": "encoder",
        "readout_manifest_hash": "readout",
        "weight_input": {
            "graph_id": f"graph-{index}",
            "node_count": 10 + index,
            "edge_count": 20 + index,
            "readout_hierarchy": "node_to_warp_to_cta_to_selected_sm_to_kernel",
        },
    }
    row["embedding_hash"] = hash_without(row, "embedding_hash")
    return row


def _formal_embedding_table():
    rows = [
        _embedding_row(0, [0.0] * 256),
        _embedding_row(1, [0.1] * 256),
        _embedding_row(2, [10.0] * 256),
        _embedding_row(3, [10.1] * 256),
    ]
    table = {
        "artifact_type": "gcl_resnet50_kernel_embedding_table",
        "artifact_version": "gate5_kernel_embedding_table_v1",
        "source_graph_tensor_bundle_hash": "tensor-bundle",
        "representation_mode": "gcl_resnet50_rgcn_selected_sm_kernel_embedding",
        "encoder_manifest_hash": "encoder",
        "checkpoint_hash": "checkpoint",
        "embedding_dim": 256,
        "readout_hierarchy": "node_to_warp_to_cta_to_selected_sm_to_kernel",
        "embeddings": rows,
    }
    table["kernel_embedding_table_hash"] = hash_without(table, "kernel_embedding_table_hash")
    return table


def test_gate6_accepts_real_resnet50_gate5_embedding_table():
    artifacts = select_phase_b_representatives(_formal_embedding_table(), seed=7)

    validate_gate6_selector_artifacts(artifacts)
    assert artifacts["artifact_type"] == "gcl_resnet50_gate6_selector_artifacts"
    assert artifacts["source_embedding_table_hash"] == _formal_embedding_table()[
        "kernel_embedding_table_hash"
    ]
    assert artifacts["embedding_normalization_report"]["normalization_policy"] == (
        "engineering_default_z_score"
    )
    assert artifacts["embedding_normalization_report"]["paper_defined"] is False
    assert artifacts["k_selection_report"]["mode"] == "silhouette_k"
    assert artifacts["kmeans_cluster_assignment_table"]["assignments"]
    assert artifacts["representative_anchor_table"]["anchors"]
    assert artifacts["cluster_family_evidence_report"]["family_labels_used_for_clustering"] is False


def test_gate6_rejects_fixture_projection_or_augmented_embeddings():
    table = _formal_embedding_table()
    table["embeddings"][0]["embedding_dim"] = 64
    table["embeddings"][0]["kernel_embedding"] = [0.0] * 64

    with pytest.raises(ValueError, match="256"):
        select_phase_b_representatives(table)

    table = _formal_embedding_table()
    table["embeddings"][0]["source_view"] = "augmented"
    with pytest.raises(ValueError, match="canonical non-augmented"):
        select_phase_b_representatives(table)

    table = _formal_embedding_table()
    table["artifact_status"] = "debug_not_formal"
    with pytest.raises(ValueError, match="formal"):
        select_phase_b_representatives(table)


def test_gate6_rejects_forbidden_fields_in_clustering_path():
    table = _formal_embedding_table()
    table["clustering_input_fields"] = ["kernel_embedding", "kernel_name"]

    with pytest.raises(ValueError, match="forbidden clustering field"):
        select_phase_b_representatives(table)


def test_gate6_rejects_family_label_guided_clustering():
    table = _formal_embedding_table()
    table["family_labels_used_for_clustering"] = True

    with pytest.raises(ValueError, match="family labels"):
        select_phase_b_representatives(table)
