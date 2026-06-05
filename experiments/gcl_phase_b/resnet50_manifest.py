"""Gate 2 representative-SM manifest construction from ResNet-50 adapter bundle."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .resnet50_adapter import validate_resnet50_trace_adapter_bundle
from .sm_selection import select_representative_sm
from .trace_fixture import COLLECTION_SCOPE
from .trace_scope import validate_phase_b_trace_manifest
from .utils import hash_without


def build_representative_sm_manifest_from_bundle(
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_resnet50_trace_adapter_bundle(bundle)
    kernel_by_invocation = {
        row["kernel_invocation_id"]: row for row in bundle["kernel_invocation_table"]
    }
    scheduler_by_invocation = _scheduler_records_by_invocation(bundle["cta_scheduler_records"])
    trace_records_by_invocation = _trace_records_by_invocation(bundle["per_warp_trace_records"])
    invocations = []
    reports = []
    for invocation_id, kernel_row in sorted(kernel_by_invocation.items()):
        selection_input = _selection_input(
            kernel_row,
            scheduler_by_invocation[invocation_id],
            trace_records_by_invocation[invocation_id],
        )
        report = select_representative_sm(selection_input)
        invocation = _manifest_invocation(selection_input, report)
        invocations.append(invocation)
        reports.append(report)
    manifest = {
        "artifact_type": "gcl_phase_b_trace_manifest",
        "manifest_version": "gcl_phase_b_trace_manifest_v1",
        "collection_scope": COLLECTION_SCOPE,
        "trace_family": "resnet50_real_trace",
        "kernel_invocations": invocations,
    }
    manifest["trace_manifest_hash"] = hash_without(manifest, "trace_manifest_hash")
    validate_phase_b_trace_manifest(manifest)
    report_bundle = {
        "artifact_type": "gcl_resnet50_selected_sm_policy_report_bundle",
        "artifact_version": "gate2_selected_sm_policy_report_bundle_v1",
        "source_adapter_bundle_hash": bundle["adapter_bundle_hash"],
        "reports": reports,
    }
    report_bundle["selected_sm_policy_report_bundle_hash"] = hash_without(
        report_bundle, "selected_sm_policy_report_bundle_hash"
    )
    preview = {
        "artifact_type": "gcl_resnet50_scope_preview_report",
        "artifact_version": "gate2_scope_preview_report_v1",
        "source_adapter_bundle_hash": bundle["adapter_bundle_hash"],
        "kernel_invocation_count": len(invocations),
        "invocations": [
            {
                "kernel_invocation_id": invocation["kernel_invocation_id"],
                "selected_sm": invocation["selected_sm"],
                "included_cta_ids": invocation["included_cta_ids"],
                "instruction_count_before_scope": invocation["instruction_count_before_scope"],
                "instruction_count_after_scope": invocation["instruction_count"],
                "warp_count_before_scope": invocation["warp_count_before_scope"],
                "warp_count_after_scope": invocation["warp_count"],
            }
            for invocation in invocations
        ],
    }
    preview["scope_preview_report_hash"] = hash_without(preview, "scope_preview_report_hash")
    return manifest, report_bundle, preview


def _scheduler_records_by_invocation(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["kernel_invocation_id"]].append(record)
    return grouped


def _trace_records_by_invocation(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["kernel_invocation_id"]].append(record)
    return grouped


def _selection_input(
    kernel_row: dict[str, Any],
    scheduler_records: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
) -> dict[str, Any]:
    scheduler_metadata_by_sm: dict[str, dict[str, Any]] = {}
    cta_to_sm = {}
    for record in scheduler_records:
        sm_id = str(record["sm_id"])
        metadata = scheduler_metadata_by_sm.setdefault(
            sm_id,
            {
                "sm_id": int(record["sm_id"]),
                "cta_ids": [],
                "warp_ids_by_cta": {},
                "trace_entry_count_by_cta": {},
                "cta_start_order": {},
                "cta_end_order": {},
            },
        )
        cta_id = record["cta_id"]
        metadata["cta_ids"].append(cta_id)
        metadata["warp_ids_by_cta"][cta_id] = list(record["warp_ids"])
        metadata["trace_entry_count_by_cta"][cta_id] = int(record["trace_entry_count"])
        metadata["cta_start_order"][cta_id] = int(record["first_seen_order"])
        metadata["cta_end_order"][cta_id] = int(record["last_seen_order"])
        cta_to_sm[cta_id] = int(record["sm_id"])
    for metadata in scheduler_metadata_by_sm.values():
        metadata["cta_ids"] = sorted(
            metadata["cta_ids"],
            key=lambda cta_id: (int(metadata["cta_start_order"][cta_id]), cta_id),
        )
    all_trace_entries = _flatten_trace_entries(kernel_row, trace_records)
    return {
        "kernel_invocation_id": kernel_row["kernel_invocation_id"],
        "trace_family": "resnet50_real_trace",
        "selected_sm_policy": "scheduler_signature_medoid_sm",
        "scheduler_metadata_by_sm": scheduler_metadata_by_sm,
        "cta_to_sm": cta_to_sm,
        "all_trace_entries": all_trace_entries,
        "instruction_count_before_scope": len(all_trace_entries),
        "warp_count_before_scope": len(
            {(entry["cta_id"], entry["warp_id"]) for entry in all_trace_entries}
        ),
    }


def _flatten_trace_entries(
    kernel_row: dict[str, Any],
    trace_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries = []
    for record in trace_records:
        for entry in record["entries"]:
            entries.append(
                {
                    "kernel_invocation_id": kernel_row["kernel_invocation_id"],
                    "trace_family": "resnet50_real_trace",
                    "collection_scope": COLLECTION_SCOPE,
                    "cta_id": record["cta_id"],
                    "warp_id": int(record["warp_id"]),
                    "trace_index": int(entry["trace_index"]),
                    "pc": int(entry["pc"]),
                    "opcode": entry["opcode"],
                    "active_mask": entry.get("active_mask", "0xffffffff"),
                    "predicate_mask": entry.get("predicate_mask", "0xffffffff"),
                    "destination_operands": list(entry.get("destination_operands", [])),
                    "source_operands": list(entry.get("source_operands", [])),
                    "memory_address_metadata": dict(entry.get("memory_address_metadata", {})),
                    "observed_dynamic_values": list(entry.get("observed_dynamic_values", [])),
                    "source_entry_hash": entry["source_entry_hash"],
                }
            )
    return sorted(entries, key=lambda item: (item["cta_id"], item["warp_id"], item["trace_index"]))


def _manifest_invocation(selection_input: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    selected_sm = int(report["selected_sm"])
    included_cta_ids = list(selection_input["scheduler_metadata_by_sm"][str(selected_sm)]["cta_ids"])
    included_cta_set = set(included_cta_ids)
    scoped_entries = [
        entry for entry in selection_input["all_trace_entries"] if entry["cta_id"] in included_cta_set
    ]
    invocation = {
        **selection_input,
        "collection_scope": COLLECTION_SCOPE,
        "selected_sm": selected_sm,
        "selected_sm_policy": report["selected_sm_policy"],
        "selected_sm_reason": report["selected_sm_reason"],
        "candidate_sm_count": report["candidate_sm_count"],
        "included_cta_ids": included_cta_ids,
        "instruction_count": len(scoped_entries),
        "warp_count": len({(entry["cta_id"], entry["warp_id"]) for entry in scoped_entries}),
        "selected_sm_policy_report": report,
        "selected_sm_policy_report_hash": report["selection_hash"],
    }
    invocation["trace_hash"] = hash_without(invocation, "trace_hash")
    return invocation
