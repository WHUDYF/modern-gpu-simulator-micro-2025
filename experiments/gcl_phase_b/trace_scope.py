"""Trace scope validation and extraction for GCL Phase B."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .trace_fixture import COLLECTION_SCOPE
from .utils import hash_without


def validate_phase_b_trace_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("collection_scope") != COLLECTION_SCOPE:
        raise ValueError("manifest collection_scope must be single_representative_sm_all_ctas")
    for invocation in manifest.get("kernel_invocations", []):
        required = {
            "collection_scope",
            "selected_sm",
            "selected_sm_policy",
            "selected_sm_reason",
            "candidate_sm_count",
            "included_cta_ids",
            "instruction_count",
            "warp_count",
            "selected_sm_policy_report_hash",
            "trace_hash",
        }
        missing = required.difference(invocation)
        if missing:
            raise ValueError(f"trace invocation missing required fields: {sorted(missing)}")
        if invocation["collection_scope"] != COLLECTION_SCOPE:
            raise ValueError("invocation collection_scope must be single_representative_sm_all_ctas")
        selected_sm = invocation["selected_sm"]
        for cta_id in invocation["included_cta_ids"]:
            if invocation["cta_to_sm"][cta_id] != selected_sm:
                raise ValueError("included_cta_ids must only come from selected SM")
        expected_ctas = set(invocation["scheduler_metadata_by_sm"][str(selected_sm)]["cta_ids"])
        if set(invocation["included_cta_ids"]) != expected_ctas:
            raise ValueError("included_cta_ids must cover all selected SM CTAs")


def build_scope_audit(invocation: dict[str, Any]) -> dict[str, Any]:
    audit = {
        "artifact_type": "gcl_phase_b_scope_audit",
        "scope_policy": COLLECTION_SCOPE,
        "scope_reason": invocation["selected_sm_reason"],
        "selected_sm": invocation["selected_sm"],
        "included_cta_ids": invocation["included_cta_ids"],
        "before_scope_counts_available": "instruction_count_before_scope" in invocation,
        "instruction_count_before_scope": invocation.get("instruction_count_before_scope"),
        "instruction_count_after_scope": invocation["instruction_count"],
        "warp_count_before_scope": invocation.get("warp_count_before_scope"),
        "warp_count_after_scope": invocation["warp_count"],
        "missing_before_scope_reason": invocation.get("missing_before_scope_reason"),
    }
    audit["trace_scope_hash"] = hash_without(audit, "trace_scope_hash")
    return audit


def validate_scope_audit(audit: dict[str, Any], invocation: dict[str, Any]) -> None:
    if audit.get("scope_policy") != COLLECTION_SCOPE:
        raise ValueError("scope_policy must be single_representative_sm_all_ctas")
    if not audit.get("before_scope_counts_available"):
        if audit.get("instruction_count_before_scope") in (0, None) and not audit.get("missing_before_scope_reason"):
            raise ValueError("missing_before_scope_reason is required when before counts are unavailable")
    if audit["instruction_count_after_scope"] != invocation["instruction_count"]:
        raise ValueError("instruction_count_after_scope mismatch")
    if audit["warp_count_after_scope"] != invocation["warp_count"]:
        raise ValueError("warp_count_after_scope mismatch")
    if audit["trace_scope_hash"] != hash_without(audit, "trace_scope_hash"):
        raise ValueError("trace_scope_hash is not reproducible")


def build_phase_b_trace_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    validate_phase_b_trace_manifest(manifest)
    records = []
    for invocation in manifest["kernel_invocations"]:
        scoped_entries = [
            entry
            for entry in invocation["all_trace_entries"]
            if entry["cta_id"] in invocation["included_cta_ids"]
        ]
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for entry in scoped_entries:
            grouped[(entry["cta_id"], entry["warp_id"])].append(entry)
        cta_ordinals = {
            cta_id: ordinal
            for ordinal, cta_id in enumerate(sorted(invocation["included_cta_ids"]), start=1)
        }
        warps = []
        for (cta_id, warp_id), entries in sorted(grouped.items()):
            ordered = sorted(entries, key=lambda entry: entry["trace_index"])
            warps.append(
                {
                    "cta_id": cta_id,
                    "warp_id": warp_id,
                    "warp_partition_id": f"{cta_ordinals[cta_id]}:{warp_id}",
                    "entries": ordered,
                }
            )
        records.append(
            {
                "kernel_invocation_id": invocation["kernel_invocation_id"],
                "trace_family": invocation["trace_family"],
                "collection_scope": COLLECTION_SCOPE,
                "selected_sm": invocation["selected_sm"],
                "included_cta_ids": invocation["included_cta_ids"],
                "selected_sm_policy_report_hash": invocation["selected_sm_policy_report_hash"],
                "warps": warps,
                "source_trace_hash": invocation["trace_hash"],
            }
        )
    return records
