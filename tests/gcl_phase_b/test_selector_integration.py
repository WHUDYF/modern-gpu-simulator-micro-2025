import pytest

from experiments.gcl_phase_b.pipeline import run_embedding_export
from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.selector import select_phase_b_representatives
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records


def _embedding_table(tmp_path):
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest(invocation_count=2))
    graphs = build_phase_b_graphs(records)
    tensors = tensorize_phase_b_graphs(graphs)
    table, _ = run_embedding_export(tensors, tmp_path)
    return table


def test_m0_selector_consumes_phase_b_embedding_table(tmp_path):
    table = _embedding_table(tmp_path)

    artifacts = select_phase_b_representatives(table, seed=9)

    assert artifacts["artifact_type"] == "gcl_m0_selector_artifacts"
    assert artifacts["cluster_assignments"]
    assert artifacts["representative_anchor_table"]
    assert artifacts["silhouette_report"]["mode"] == "silhouette_k"


def test_selector_rejects_resource_blocked_rows(tmp_path):
    table = _embedding_table(tmp_path)
    table["embeddings"][0]["resource_blocked"] = True

    with pytest.raises(ValueError, match="resource-blocked"):
        select_phase_b_representatives(table)
