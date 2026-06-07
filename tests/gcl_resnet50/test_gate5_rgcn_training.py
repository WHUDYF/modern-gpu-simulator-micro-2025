from pathlib import Path

import pytest

from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.pipeline import create_augmentation_manifest_bundle, run_embedding_export
from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_debug_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records
from experiments.gcl_phase_b.selector import select_phase_b_representatives
from gcl_resnet50.formal_chain import build_artifact_shape_tensors
from gcl_resnet50.real_chain import build_real_tensors

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def _debug_tensors():
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)
    manifest, _reports, _preview = build_representative_sm_manifest_from_bundle(bundle)
    graphs = build_phase_b_graphs(build_phase_b_trace_records(manifest))
    return tensorize_phase_b_graphs(graphs)


def test_gate5_augmentation_does_not_overwrite_canonical_tensor():
    tensors = _debug_tensors()
    canonical_hashes = [tensor["tensor_hash"] for tensor in tensors]

    augmentation = create_augmentation_manifest_bundle(tensors, seed=20260606)

    assert [tensor["tensor_hash"] for tensor in tensors] == canonical_hashes
    assert augmentation["manifests"]
    canonical_graph_hashes = {tensor["input_graph_hash"] for tensor in tensors}
    assert all(
        manifest["input_graph_hash"] in canonical_graph_hashes
        for manifest in augmentation["manifests"]
    )


def test_gate5_exports_256d_canonical_kernel_embeddings_for_debug_smoke(tmp_path):
    table, _training = run_embedding_export(_debug_tensors(), tmp_path)

    assert table["embedding_dim"] == 256
    assert table["readout_hierarchy"] == "node_to_warp_to_cta_to_selected_sm_to_kernel"
    assert all(len(row["kernel_embedding"]) == 256 for row in table["embeddings"])


def test_gate5_rejects_projection_head_output_for_selector(tmp_path):
    table, _training = run_embedding_export(_debug_tensors(), tmp_path)
    table["embeddings"][0]["embedding_dim"] = 64
    table["embeddings"][0]["kernel_embedding"] = table["embeddings"][0]["kernel_embedding"][:64]

    with pytest.raises(ValueError, match="256"):
        select_phase_b_representatives(table, allow_debug=True)


def test_gate5_artifact_shape_embedding_table_carries_auditable_training_lineage(tmp_path):
    table, _training = run_embedding_export(build_artifact_shape_tensors(tmp_path), tmp_path)

    assert table["artifact_status"] == "debug_not_formal"
    assert table["formal_input_eligible"] is False
    lineage = table["gate5_lineage"]
    assert lineage["source_graph_tensor_bundle_hash"]
    assert lineage["training_run_manifest_hash"]
    assert lineage["checkpoint_manifest_hash"]
    assert lineage["readout_manifest_bundle_hash"]
    assert lineage["embedding_export_report_hash"]
    for row in table["embeddings"]:
        assert row["gate5_lineage_hash"] == table["gate5_lineage_hash"]


def test_gate5_exports_256d_canonical_kernel_embeddings_from_real_resnet50_root(tmp_path):
    _manifest, _reports, _preview, _graphs, tensors = build_real_tensors()

    table, _training = run_embedding_export(tensors, tmp_path, seed=20260607)

    assert table["artifact_status"] == "formal"
    assert table["formal_input_eligible"] is True
    assert table["trace_source"] == "nvbit"
    assert table["scheduler_metadata_source"] == "real_nvbit_smid"
    assert table["embedding_dim"] == 256
    assert table["readout_hierarchy"] == "node_to_warp_to_cta_to_selected_sm_to_kernel"
    assert all(len(row["kernel_embedding"]) == 256 for row in table["embeddings"])
