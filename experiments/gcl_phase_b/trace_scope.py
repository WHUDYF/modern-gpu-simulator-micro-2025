"""Trace scope validation and extraction for GCL Phase B."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .trace_fixture import COLLECTION_SCOPE
from .utils import hash_without


def validate_phase_b_trace_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("collection_scope") != COLLECTION_SCOPE:
        raise ValueError("manifest collection_scope must be single_representative_sm_all_ctas")
    if manifest.get("trace_manifest_hash") != hash_without(manifest, "trace_manifest_hash"):
        raise ValueError("trace_manifest_hash is not reproducible")
    for invocation in manifest.get("kernel_invocations", []):
        required = {
            "collection_scope",
            "kernel_invocation_id",
            "trace_family",
            "selected_sm",
            "selected_sm_policy",
            "selected_sm_reason",
            "candidate_sm_count",
            "included_cta_ids",
            "instruction_count",
            "warp_count",
            "selected_sm_policy_report_hash",
            "trace_hash",
            "cta_to_sm",
            "scheduler_metadata_by_sm",
        }
        missing = required.difference(invocation)
        if missing:
            raise ValueError(f"trace invocation missing required fields: {sorted(missing)}")
        if invocation["collection_scope"] != COLLECTION_SCOPE:
            raise ValueError("invocation collection_scope must be single_representative_sm_all_ctas")
        selected_sm = invocation["selected_sm"]
        selected_sm_key = str(selected_sm)
        if selected_sm_key not in invocation["scheduler_metadata_by_sm"]:
            raise ValueError("scheduler_metadata_by_sm must include selected SM metadata")
        for cta_id in invocation["included_cta_ids"]:
            if cta_id not in invocation["cta_to_sm"]:
                raise ValueError("cta_to_sm must include every included CTA")
            if invocation["cta_to_sm"][cta_id] != selected_sm:
                raise ValueError("included_cta_ids must only come from selected SM")
        expected_ctas = set(invocation["scheduler_metadata_by_sm"][selected_sm_key]["cta_ids"])
        if set(invocation["included_cta_ids"]) != expected_ctas:
            raise ValueError("included_cta_ids must cover all selected SM CTAs")
        scoped_entries = _scoped_entries(invocation)
        _validate_selected_sm_trace_entry_coverage(invocation, scoped_entries)
        expected_instruction_count = len(scoped_entries)
        expected_warp_count = len({(entry["cta_id"], entry["warp_id"]) for entry in scoped_entries})
        if invocation["instruction_count"] != expected_instruction_count:
            raise ValueError("instruction_count does not match selected SM trace entries")
        if invocation["warp_count"] != expected_warp_count:
            raise ValueError("warp_count does not match selected SM trace entries")
        if "instruction_count_before_scope" in invocation:
            if invocation["instruction_count_before_scope"] != len(invocation.get("all_trace_entries", [])):
                raise ValueError("instruction_count_before_scope mismatch")
        if "warp_count_before_scope" in invocation:
            before_warps = {
                (entry["cta_id"], entry["warp_id"])
                for entry in invocation.get("all_trace_entries", [])
            }
            if invocation["warp_count_before_scope"] != len(before_warps):
                raise ValueError("warp_count_before_scope mismatch")
        if invocation["trace_hash"] != hash_without(invocation, "trace_hash"):
            raise ValueError("trace_hash is not reproducible")


def _scoped_entries(invocation: dict[str, Any]) -> list[dict[str, Any]]:
    included = set(invocation["included_cta_ids"])
    return [entry for entry in invocation.get("all_trace_entries", []) if entry["cta_id"] in included]


def _validate_selected_sm_trace_entry_coverage(
    invocation: dict[str, Any],
    scoped_entries: list[dict[str, Any]],
) -> None:
    included_ctas = set(invocation["included_cta_ids"])
    entries_by_cta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_warps_by_cta: dict[str, set[int]] = defaultdict(set)
    for entry in scoped_entries:
        cta_id = entry["cta_id"]
        entries_by_cta[cta_id].append(entry)
        observed_warps_by_cta[cta_id].add(int(entry["warp_id"]))

    missing_ctas = sorted(included_ctas.difference(entries_by_cta))
    if missing_ctas:
        raise ValueError(f"CTA trace entries missing for selected SM CTAs: {missing_ctas}")

    selected_sm_metadata = invocation["scheduler_metadata_by_sm"][str(invocation["selected_sm"])]
    warp_ids_by_cta = selected_sm_metadata.get("warp_ids_by_cta")
    if not isinstance(warp_ids_by_cta, dict):
        raise ValueError("selected SM scheduler metadata must define warp_ids_by_cta")
    for cta_id in sorted(included_ctas):
        expected_warps = {int(warp_id) for warp_id in warp_ids_by_cta.get(cta_id, [])}
        missing_warps = sorted(expected_warps.difference(observed_warps_by_cta.get(cta_id, set())))
        if missing_warps:
            raise ValueError(
                f"warp trace entries missing for selected SM CTA {cta_id}: {missing_warps}"
            )


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
    if audit.get("scope_reason") != invocation["selected_sm_reason"]:
        raise ValueError("scope audit scope_reason mismatch")
    if audit.get("selected_sm") != invocation["selected_sm"]:
        raise ValueError("scope audit selected_sm mismatch")
    if audit.get("included_cta_ids") != invocation["included_cta_ids"]:
        raise ValueError("scope audit included_cta_ids mismatch")
    if audit.get("instruction_count_before_scope") != invocation.get("instruction_count_before_scope"):
        raise ValueError("instruction_count_before_scope mismatch")
    if audit.get("warp_count_before_scope") != invocation.get("warp_count_before_scope"):
        raise ValueError("warp_count_before_scope mismatch")
    if audit.get("missing_before_scope_reason") != invocation.get("missing_before_scope_reason"):
        raise ValueError("missing_before_scope_reason mismatch")
    if not audit.get("before_scope_counts_available"):
        if audit.get("instruction_count_before_scope") in (0, None) and not audit.get("missing_before_scope_reason"):
            raise ValueError("missing_before_scope_reason is required when before counts are unavailable")
    scoped_entries = _scoped_entries(invocation)
    actual_instruction_count = len(scoped_entries)
    actual_warp_count = len({(entry["cta_id"], entry["warp_id"]) for entry in scoped_entries})
    if audit["instruction_count_after_scope"] != actual_instruction_count:
        raise ValueError("instruction_count_after_scope mismatch")
    if audit["warp_count_after_scope"] != actual_warp_count:
        raise ValueError("warp_count_after_scope mismatch")
    if audit["trace_scope_hash"] != hash_without(audit, "trace_scope_hash"):
        raise ValueError("trace_scope_hash is not reproducible")


def _cta_ordinals_by_scheduler_order(invocation: dict[str, Any]) -> dict[str, int]:
    selected_sm = str(invocation["selected_sm"])
    scheduler_metadata = invocation["scheduler_metadata_by_sm"][selected_sm]
    cta_start_order = scheduler_metadata.get("cta_start_order")
    if not isinstance(cta_start_order, dict):
        raise ValueError("selected SM scheduler metadata must define cta_start_order")
    ordered_ctas = sorted(
        invocation["included_cta_ids"],
        key=lambda cta_id: (int(cta_start_order[cta_id]), cta_id),
    )
    return {cta_id: ordinal for ordinal, cta_id in enumerate(ordered_ctas, start=1)}


def build_phase_b_trace_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    validate_phase_b_trace_manifest(manifest)
    records = []
    for invocation in manifest["kernel_invocations"]:
        scoped_entries = _scoped_entries(invocation)
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for entry in scoped_entries:
            grouped[(entry["cta_id"], entry["warp_id"])].append(entry)
        cta_ordinals = _cta_ordinals_by_scheduler_order(invocation)
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
                "artifact_status": invocation.get("artifact_status", "formal"),
                "formal_input_eligible": invocation.get("formal_input_eligible", True),
                "workload_id": invocation.get("workload_id"),
                "execution_mode": invocation.get("execution_mode"),
                "trace_source": invocation.get("trace_source"),
                "input_scope": invocation.get("input_scope"),
                "scheduler_metadata_source": invocation.get("scheduler_metadata_source"),
                "collection_scope": COLLECTION_SCOPE,
                "selected_sm": invocation["selected_sm"],
                "included_cta_ids": invocation["included_cta_ids"],
                "selected_sm_policy_report_hash": invocation["selected_sm_policy_report_hash"],
                "warps": warps,
                "source_trace_hash": invocation["trace_hash"],
            }
        )
    return records
