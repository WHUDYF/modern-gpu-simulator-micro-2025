import copy

import pytest

from experiments.gcl_phase_b.graph_audit import build_graph_size_audit, validate_graph_size_audit
from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records


def _graph():
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    return build_phase_b_graphs(records)[0]


def test_graph_size_audit_records_size_class_without_auto_blocking():
    graph = _graph()

    audit = build_graph_size_audit(graph)

    validate_graph_size_audit(audit, graph)
    assert audit["size_policy_version"] == "phase_b_audit_guardrail_v1"
    assert audit["training_resource_status"] == "not_checked"
    assert audit["graph_size_class"] in {"small", "medium", "large", "oversized"}
    assert "training_eligibility" not in audit


def test_graph_size_audit_rejects_count_mismatch():
    graph = _graph()
    audit = build_graph_size_audit(graph)
    audit["node_count"] -= 1

    with pytest.raises(ValueError, match="node_count"):
        validate_graph_size_audit(audit, graph)


def test_graph_size_audit_rejects_truncated_large_marked_complete():
    graph = _graph()
    audit = build_graph_size_audit(graph)
    mutated = copy.deepcopy(audit)
    mutated["graph_size_class"] = "large"
    mutated["trace_scope_modified_after_audit"] = True
    mutated["phase_b_completion_status"] = "phase_b_complete"

    with pytest.raises(ValueError, match="truncated|scope"):
        validate_graph_size_audit(mutated, graph)
