import numpy as np
from pathlib import Path

from experiments.gcl_phase_b.embedding_export import (
    export_phase_b_embedding_table,
    validate_phase_b_embedding_table,
)
from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.pipeline import run_embedding_export
from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.tensorizer import _tensor_hash, tensorize_phase_b_graphs
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records


def test_phase_b_exports_m0_compatible_256_dim_embeddings(tmp_path):
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)

    table, _training_report = run_embedding_export(tensors, tmp_path)

    assert table["artifact_type"] == "gcl_kernel_embedding_table"
    assert table["embedding_dim"] == 256
    assert table["row_count"] == 1
    row = table["rows"][0]
    assert row["representation_mode"] == table["representation_mode"]
    assert len(row["embedding"]) == 256
    assert row["source_graph_hash"] == graphs[0]["graph_hash"]
    assert row["weight_input"]["readout_hierarchy"] == "node_to_warp_to_cta_to_selected_sm_to_kernel"
    assert "readout_manifest_hash" in row


def test_phase_b_embedding_export_uses_cta_aware_readout(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(Path("tests/fixtures/gcl_resnet50_gate1"))
    manifest, _reports, _preview = build_representative_sm_manifest_from_bundle(bundle)
    assert all(len(invocation["included_cta_ids"]) >= 2 for invocation in manifest["kernel_invocations"])
    records = build_phase_b_trace_records(manifest)
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)

    table, training_report = run_embedding_export(tensors, tmp_path)
    regrouped_tensors = [dict(tensor) for tensor in tensors]
    regrouped_tensors[0]["warp_partition_tensors"] = {
        key: dict(value) for key, value in tensors[0]["warp_partition_tensors"].items()
    }
    partition_ids = sorted(regrouped_tensors[0]["warp_partition_tensors"])
    first_cta = regrouped_tensors[0]["warp_partition_tensors"][partition_ids[0]]["cta_id"]
    regrouped_tensors[0]["warp_partition_tensors"][partition_ids[-1]]["cta_id"] = first_cta
    regrouped_tensors[0]["tensor_hash"] = _tensor_hash(regrouped_tensors[0])
    regrouped_table, _readout_bundle = export_phase_b_embedding_table(
        regrouped_tensors,
        training_report["encoder"],
        training_report["checkpoint_manifest"],
    )

    exported = np.asarray(table["rows"][0]["embedding"], dtype=np.float32)
    regrouped = np.asarray(regrouped_table["rows"][0]["embedding"], dtype=np.float32)
    assert not np.allclose(exported, regrouped)


def test_phase_b_export_function_returns_readout_manifest_bundle(tmp_path):
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)
    _table, training_report = run_embedding_export(tensors, tmp_path)

    table, readout_bundle = export_phase_b_embedding_table(
        tensors,
        training_report["encoder"],
        training_report["checkpoint_manifest"],
    )

    validate_phase_b_embedding_table(table)
    assert readout_bundle["artifact_type"] == "gcl_phase_b_readout_manifest_bundle"
    assert readout_bundle["manifests"][0]["readout_hierarchy"] == (
        "node_to_warp_to_cta_to_selected_sm_to_kernel"
    )
