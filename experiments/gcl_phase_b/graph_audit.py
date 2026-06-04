"""Audit-only graph size reports for GCL Phase B."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .graph_builder import validate_phase_b_graph_artifact
from .utils import hash_without

SIZE_POLICY_VERSION = "phase_b_audit_guardrail_v1"


def _size_class(node_count: int, edge_count: int) -> str:
    total = node_count + edge_count
    if total <= 500:
        return "small"
    if total <= 5_000:
        return "medium"
    if total <= 50_000:
        return "large"
    return "oversized"


def build_graph_size_audit(graph: dict[str, Any]) -> dict[str, Any]:
    validate_phase_b_graph_artifact(graph)
    node_counts = Counter(node["node_type"] for node in graph["nodes"])
    edge_counts = Counter(edge["relation"] for edge in graph["edges"])
    partitions = graph["warp_partitions"].values()
    audit = {
        "artifact_type": "gcl_phase_b_graph_size_audit",
        "graph_id": graph["graph_id"],
        "kernel_invocation_id": graph["kernel_invocation_id"],
        "graph_hash": graph["graph_hash"],
        "instruction_count": graph["graph_summary"]["instruction_node_count"],
        "warp_count": len(graph["warp_partitions"]),
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "node_type_counts": dict(sorted(node_counts.items())),
        "edge_type_counts": dict(sorted(edge_counts.items())),
        "max_warp_instruction_count": max(partition["instruction_count"] for partition in partitions),
        "max_warp_node_count": max(partition["node_count"] for partition in graph["warp_partitions"].values()),
        "max_warp_edge_count": max(partition["edge_count"] for partition in graph["warp_partitions"].values()),
        "graph_size_class": _size_class(len(graph["nodes"]), len(graph["edges"])),
        "size_policy_version": SIZE_POLICY_VERSION,
        "training_resource_status": "not_checked",
        "trace_scope_modified_after_audit": False,
        "phase_b_completion_status": "phase_b_complete",
    }
    audit["graph_size_audit_hash"] = hash_without(audit, "graph_size_audit_hash")
    return audit


def validate_graph_size_audit(audit: dict[str, Any], graph: dict[str, Any]) -> None:
    validate_phase_b_graph_artifact(graph)
    required = {
        "instruction_count",
        "warp_count",
        "node_count",
        "edge_count",
        "node_type_counts",
        "edge_type_counts",
        "max_warp_instruction_count",
        "max_warp_node_count",
        "max_warp_edge_count",
        "graph_size_class",
        "size_policy_version",
        "training_resource_status",
        "graph_size_audit_hash",
    }
    missing = required.difference(audit)
    if missing:
        raise ValueError(f"graph size audit missing required fields: {sorted(missing)}")
    if "training_eligibility" in audit:
        raise ValueError("graph size audit must not emit training_eligibility in Phase B")
    if audit["size_policy_version"] != SIZE_POLICY_VERSION:
        raise ValueError("unexpected graph size audit policy version")
    if audit["node_count"] != len(graph["nodes"]):
        raise ValueError("node_count mismatch")
    if audit["edge_count"] != len(graph["edges"]):
        raise ValueError("edge_count mismatch")
    if audit["instruction_count"] != graph["graph_summary"]["instruction_node_count"]:
        raise ValueError("instruction_count mismatch")
    if audit["warp_count"] != len(graph["warp_partitions"]):
        raise ValueError("warp_count mismatch")
    if (
        audit.get("graph_size_class") in {"large", "oversized"}
        and audit.get("trace_scope_modified_after_audit")
        and audit.get("phase_b_completion_status") == "phase_b_complete"
    ):
        raise ValueError("large graph scope cannot be truncated and marked phase_b_complete")
    if audit["graph_size_audit_hash"] != hash_without(audit, "graph_size_audit_hash"):
        raise ValueError("graph_size_audit_hash is not reproducible")
