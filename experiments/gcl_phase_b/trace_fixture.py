"""Small representative-SM trace manifest fixture for GCL Phase B."""

from __future__ import annotations

from typing import Any

from .sm_selection import select_representative_sm
from .utils import hash_without

COLLECTION_SCOPE = "single_representative_sm_all_ctas"


def _entry(
    kernel_invocation_id: str,
    cta_id: str,
    warp_id: int,
    trace_index: int,
    opcode: str,
    dests: list[str],
    srcs: list[str],
) -> dict[str, Any]:
    entry = {
        "kernel_invocation_id": kernel_invocation_id,
        "trace_family": "phase_b_representative_sm_fixture",
        "collection_scope": COLLECTION_SCOPE,
        "cta_id": cta_id,
        "warp_id": warp_id,
        "trace_index": trace_index,
        "pc": 0x2000 + trace_index * 8 + warp_id * 0x40,
        "opcode": opcode,
        "active_mask": "0xffffffff",
        "destination_operands": dests,
        "source_operands": srcs,
        "observed_dynamic_values": [float(trace_index + 1), float(trace_index + 2), 3.0, 4.0],
    }
    entry["source_entry_hash"] = hash_without(entry, "source_entry_hash")
    return entry


def _cta_entries(kernel_invocation_id: str, cta_id: str) -> list[dict[str, Any]]:
    entries = []
    for warp_id in (0, 1):
        entries.extend(
            [
                _entry(kernel_invocation_id, cta_id, warp_id, 0, "MOV", ["R4"], ["input:base"]),
                _entry(kernel_invocation_id, cta_id, warp_id, 1, "LDG.E.64.SYS", ["R8"], ["R4"]),
                _entry(kernel_invocation_id, cta_id, warp_id, 2, "FADD", ["R9"], ["R8", "input:bias"]),
                _entry(kernel_invocation_id, cta_id, warp_id, 3, "STG.E.64.SYS", [], ["R4", "R9"]),
            ]
        )
    return entries


def _base_invocation(index: int, selected_sm: int | None = None) -> dict[str, Any]:
    kernel_invocation_id = f"gcl_pb_k{index:03d}"
    cta_to_sm = {"cta_0": 0, "cta_1": 0, "cta_2": 1, "cta_3": 1, "cta_4": 2}
    trace_entries = []
    for cta_id in cta_to_sm:
        trace_entries.extend(_cta_entries(kernel_invocation_id, cta_id))
    scheduler_metadata_by_sm = {
        "0": {
            "sm_id": 0,
            "cta_ids": ["cta_0", "cta_1"],
            "warp_ids_by_cta": {"cta_0": [0, 1], "cta_1": [0, 1]},
            "trace_entry_count_by_cta": {"cta_0": 8, "cta_1": 8},
            "cta_start_order": {"cta_0": 0, "cta_1": 1},
            "cta_end_order": {"cta_0": 1, "cta_1": 2},
        },
        "1": {
            "sm_id": 1,
            "cta_ids": ["cta_2", "cta_3"],
            "warp_ids_by_cta": {"cta_2": [0, 1], "cta_3": [0, 1]},
            "trace_entry_count_by_cta": {"cta_2": 8, "cta_3": 8},
            "cta_start_order": {"cta_2": 1, "cta_3": 3},
            "cta_end_order": {"cta_2": 3, "cta_3": 4},
        },
        "2": {
            "sm_id": 2,
            "cta_ids": ["cta_4"],
            "warp_ids_by_cta": {"cta_4": [0, 1]},
            "trace_entry_count_by_cta": {"cta_4": 8},
            "cta_start_order": {"cta_4": 4},
            "cta_end_order": {"cta_4": 4},
        },
    }
    invocation = {
        "kernel_invocation_id": kernel_invocation_id,
        "trace_family": "phase_b_representative_sm_fixture",
        "collection_scope": COLLECTION_SCOPE,
        "selected_sm_policy": "scheduler_signature_medoid_sm",
        "scheduler_metadata_by_sm": scheduler_metadata_by_sm,
        "cta_to_sm": cta_to_sm,
        "all_trace_entries": trace_entries,
        "instruction_count_before_scope": len(trace_entries),
        "warp_count_before_scope": len({(entry["cta_id"], entry["warp_id"]) for entry in trace_entries}),
    }
    if selected_sm is not None:
        invocation["selected_sm"] = selected_sm
        report = select_representative_sm(invocation, policy="explicit_sm_id")
    else:
        report = select_representative_sm(invocation)
    invocation["selected_sm_policy_report"] = report
    invocation["selected_sm_policy_report_hash"] = report["selection_hash"]
    invocation["selected_sm_policy"] = report["selected_sm_policy"]
    invocation["selected_sm"] = report["selected_sm"]
    invocation["selected_sm_reason"] = report["selected_sm_reason"]
    invocation["candidate_sm_count"] = report["candidate_sm_count"]
    invocation["included_cta_ids"] = scheduler_metadata_by_sm[str(report["selected_sm"])]["cta_ids"]
    scoped_entries = [entry for entry in trace_entries if entry["cta_id"] in invocation["included_cta_ids"]]
    invocation["instruction_count"] = len(scoped_entries)
    invocation["warp_count"] = len({(entry["cta_id"], entry["warp_id"]) for entry in scoped_entries})
    invocation["trace_hash"] = hash_without(invocation, "trace_hash")
    return invocation


def build_representative_sm_trace_manifest(
    invocation_count: int = 1,
    selected_sm: int | None = None,
) -> dict[str, Any]:
    invocations = [
        _base_invocation(index, selected_sm=selected_sm if index == 0 else None)
        for index in range(invocation_count)
    ]
    manifest = {
        "artifact_type": "gcl_phase_b_trace_manifest",
        "manifest_version": "gcl_phase_b_trace_manifest_v1",
        "collection_scope": COLLECTION_SCOPE,
        "kernel_invocations": invocations,
    }
    manifest["trace_manifest_hash"] = hash_without(manifest, "trace_manifest_hash")
    return manifest
