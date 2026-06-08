import numpy as np
import pytest
from pathlib import Path

from experiments.gcl_phase_b.embedding_export import (
    READOUT_HIERARCHY,
    GATE5_EXPORT_PROGRESS_FILENAME,
    export_phase_b_embedding_table,
    validate_phase_b_embedding_table,
)
from experiments.gcl_phase_a.rgcn import require_torch
from experiments.gcl_phase_b.utils import hash_without
from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.pipeline import run_embedding_export
from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_debug_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.tensorizer import _tensor_hash, tensorize_phase_b_graphs
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records


FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def test_phase_b_exports_m0_compatible_256_dim_embeddings(tmp_path):
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)

    table, _training_report, _ = run_embedding_export(tensors, tmp_path)

    assert table["artifact_type"] == "gcl_resnet50_kernel_embedding_table"
    assert table["artifact_version"] == "gate5_kernel_embedding_table_v1"
    assert table["source_graph_tensor_bundle_hash"]
    assert table["checkpoint_hash"] == _training_report["checkpoint_manifest"]["checkpoint_hash"]
    assert table["kernel_embedding_table_hash"] == hash_without(
        table, "kernel_embedding_table_hash"
    )
    assert table["readout_hierarchy"] == READOUT_HIERARCHY
    assert table["embedding_dim"] == 256
    assert len(table["embeddings"]) == 1
    row = table["embeddings"][0]
    assert row["representation_mode"] == table["representation_mode"]
    assert row["graph_id"] == tensors[0]["graph_id"]
    assert row["source_tensor_hash"] == tensors[0]["tensor_hash"]
    assert row["collection_scope"] == "single_representative_sm_all_ctas"
    assert row["selected_sm"] == tensors[0]["graph_batch_metadata"]["selected_sm"]
    assert len(row["kernel_embedding"]) == 256
    assert row["source_graph_hash"] == graphs[0]["graph_hash"]
    assert row["weight_input"]["readout_hierarchy"] == READOUT_HIERARCHY
    assert "readout_manifest_hash" in row


def test_phase_b_embedding_export_uses_cta_aware_readout(tmp_path):
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)
    manifest, _reports, _preview = build_representative_sm_manifest_from_bundle(bundle)
    assert all(len(invocation["included_cta_ids"]) >= 2 for invocation in manifest["kernel_invocations"])
    records = build_phase_b_trace_records(manifest)
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)

    table, training_report, _ = run_embedding_export(tensors, tmp_path)
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

    exported = np.asarray(table["embeddings"][0]["kernel_embedding"], dtype=np.float32)
    regrouped = np.asarray(regrouped_table["embeddings"][0]["kernel_embedding"], dtype=np.float32)
    assert not np.allclose(exported, regrouped)


def test_phase_b_export_function_returns_readout_manifest_bundle(tmp_path):
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)
    _table, training_report, _ = run_embedding_export(tensors, tmp_path)

    table, readout_bundle = export_phase_b_embedding_table(
        tensors,
        training_report["encoder"],
        training_report["checkpoint_manifest"],
    )

    validate_phase_b_embedding_table(table)
    assert readout_bundle["artifact_type"] == "gcl_phase_b_readout_manifest_bundle"
    assert readout_bundle["manifests"][0]["readout_hierarchy"] == READOUT_HIERARCHY


def test_phase_b_embedding_export_resumes_partial_progress(tmp_path):
    manifest = build_representative_sm_trace_manifest()
    source_invocation = manifest["kernel_invocations"][0]
    manifest["kernel_invocations"] = []
    for index in range(3):
        invocation = {**source_invocation, "kernel_invocation_id": f"debug_invocation_{index}"}
        invocation["trace_hash"] = hash_without(invocation, "trace_hash")
        manifest["kernel_invocations"].append(invocation)
    manifest["trace_manifest_hash"] = hash_without(manifest, "trace_manifest_hash")
    records = build_phase_b_trace_records(manifest)
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)
    torch = require_torch()
    calls = []

    class FailingEncoder:
        def eval(self):
            return self

        def encode_kernel_partitioned(self, tensor):
            calls.append(tensor["tensor_hash"])
            if len(calls) == 2:
                raise RuntimeError("simulated export interruption")
            return torch.full((256,), float(len(calls)), dtype=torch.float32)

    encoder_manifest = {
        "encoder_manifest_hash": "encoder-hash",
        "checkpoint_hash": "checkpoint-hash",
    }

    with pytest.raises(RuntimeError, match="simulated export interruption"):
        export_phase_b_embedding_table(
            tensors,
            FailingEncoder(),
            encoder_manifest,
            progress_dir=tmp_path,
        )

    progress_path = tmp_path / GATE5_EXPORT_PROGRESS_FILENAME
    assert progress_path.exists()
    assert calls == [tensors[0]["tensor_hash"], tensors[1]["tensor_hash"]]

    class ResumeEncoder:
        def eval(self):
            return self

        def encode_kernel_partitioned(self, tensor):
            calls.append(tensor["tensor_hash"])
            return torch.full((256,), float(len(calls)), dtype=torch.float32)

    table, readout_bundle = export_phase_b_embedding_table(
        tensors,
        ResumeEncoder(),
        encoder_manifest,
        progress_dir=tmp_path,
    )

    assert calls == [
        tensors[0]["tensor_hash"],
        tensors[1]["tensor_hash"],
        tensors[1]["tensor_hash"],
        tensors[2]["tensor_hash"],
    ]
    assert len(table["embeddings"]) == 3
    assert len(readout_bundle["manifests"]) == 3
    assert not progress_path.exists()
    validate_phase_b_embedding_table(table)


def test_phase_b_embedding_table_validator_rejects_missing_formal_top_level_field(tmp_path):
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)
    table, _training_report, _ = run_embedding_export(tensors, tmp_path)

    del table["source_graph_tensor_bundle_hash"]

    with pytest.raises(ValueError, match="source_graph_tensor_bundle_hash"):
        validate_phase_b_embedding_table(table)
