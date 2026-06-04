from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.pipeline import run_embedding_export
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs
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
