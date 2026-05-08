"""Gate 4 selector eligibility and backward repair for PKA-M1."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from shared_acquisition import ARTIFACT_DIR, FEATURE_ORDER, artifact_ref, file_hash, read_json, write_json

MANIFEST_PATH = ARTIFACT_DIR / "kernel_validation_manifest_l1.json"
RESOLUTION_GAP_PATH = ARTIFACT_DIR / "m1_workload_resolution_gap_l1.json"
CAPTURE_GAP_PATH = ARTIFACT_DIR / "m1_ncu_capture_gap_l1.json"
FEATURE_TABLE_PATH = ARTIFACT_DIR / "pka_feature_table_l1.json"
ACQ_GAP_PATH = ARTIFACT_DIR / "pka_acquisition_gap_l1.json"
ELIGIBILITY_PATH = ARTIFACT_DIR / "m1_selector_eligibility_l1.json"
SELECTOR_INPUT_PATH = ARTIFACT_DIR / "m1_selector_input_l1.json"
REPAIR_JSON_PATH = ARTIFACT_DIR / "m1_backward_repair_report_l1.json"
REPAIR_MD_PATH = ARTIFACT_DIR / "m1_backward_repair_report_l1.md"

FORBIDDEN_SELECTOR_FIELDS = {
    "kernel_name",
    "source_path",
    "expected_behavior_axis",
    "family",
    "regime",
    "shape_hint",
    "trace_order",
}


def _p0_entries() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return [row for row in manifest.get("entries", []) if row.get("priority") == "P0"]


def _valid_feature_row(row: dict[str, Any]) -> list[str]:
    errors = []
    if row.get("feature_mode") != "pka_m1_measured":
        errors.append("feature_mode_not_pka_m1_measured")
    features = row.get("features", {})
    for name in FEATURE_ORDER:
        feature = features.get(name)
        if not isinstance(feature, dict):
            errors.append(f"missing_{name}")
            continue
        if feature.get("status") != "measured":
            errors.append(f"{name}_not_measured")
        if not isinstance(feature.get("value"), (int, float)):
            errors.append(f"{name}_not_numeric")
        for key in ("canonical_metric", "actual_source_metric", "provenance"):
            if key not in feature or feature.get(key) in (None, ""):
                errors.append(f"{name}_missing_{key}")
    return errors


def _earliest_gap(entry_id: str, resolution_gaps: list[dict[str, Any]], capture_gaps: list[dict[str, Any]], acq_gaps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for gap in resolution_gaps:
        if gap.get("manifest_entry_id") == entry_id:
            return {"earliest_failed_gate": "Gate1", "gap_reason": gap.get("gap_reason"), "source": gap}
    for gap in capture_gaps:
        if entry_id in gap.get("consuming_manifest_entry_ids", []):
            return {"earliest_failed_gate": "Gate2", "gap_reason": gap.get("gap_reason"), "source": gap}
    for gap in acq_gaps:
        if gap.get("manifest_entry_id") == entry_id:
            return {"earliest_failed_gate": "Gate3", "gap_reason": gap.get("gap_reason"), "source": gap}
    return None


def _repair_action(gate: str, reason: str | None) -> dict[str, str | None]:
    if gate == "Gate1":
        return {"action_type": "manual_action", "description": f"Fix workload registry/binary/smoke run for reason: {reason}", "command": None}
    if gate == "Gate2":
        return {"action_type": "environment_action", "description": f"Fix NCU capture environment or metric selection for reason: {reason}", "command": None}
    if gate == "Gate3":
        return {"action_type": "code_fix_required", "description": f"Fix capture CSV parsing, kernel join, or missing metrics for reason: {reason}", "command": None}
    return {"action_type": "code_fix_required", "description": f"Fix selector preflight for reason: {reason}", "command": None}


def evaluate() -> dict[str, Any]:
    entries = _p0_entries()
    features = read_json(FEATURE_TABLE_PATH, [])
    acq_gaps = read_json(ACQ_GAP_PATH, [])
    resolution_gaps = read_json(RESOLUTION_GAP_PATH, [])
    capture_gaps = read_json(CAPTURE_GAP_PATH, [])
    measured_ids = {row.get("manifest_id") for row in features}

    row_errors = []
    for row in features:
        errors = _valid_feature_row(row)
        if errors:
            row_errors.append({"record_id": row.get("record_id"), "errors": errors})

    timing_units = {row.get("timing_basis") for row in features if row.get("timing_basis")}
    if row_errors:
        state = "selector_blocked_invalid_feature_table"
    elif len(timing_units) > 1:
        state = "selector_blocked_mixed_timing_unit"
    elif len(features) < 3:
        state = "selector_blocked_insufficient_measured_records"
    elif len(measured_ids) < len(entries):
        state = "selector_ready_with_remaining_gaps"
    else:
        state = "selector_ready"

    gate5_allowed = state in {"selector_ready", "selector_ready_with_remaining_gaps"}
    weight_mode = "member_count_fallback" if not timing_units else "timing_weight"
    selector_records = []
    if not row_errors:
        for row in features:
            selector_records.append({
                "record_id": row.get("record_id"),
                "kernel_invocation_id": row.get("kernel_invocation_id"),
                "features": row.get("features"),
                "feature_mode": row.get("feature_mode"),
                "weight_input": {"weight_mode": weight_mode, "value": 1.0},
            })
    write_json(SELECTOR_INPUT_PATH, selector_records)

    repair_rows = []
    for entry in entries:
        entry_id = entry["id"]
        if entry_id in measured_ids:
            repair_rows.append({
                "manifest_entry_id": entry_id,
                "status": "measured",
                "earliest_failed_gate": None,
                "repair_action": None,
            })
            continue
        gap = _earliest_gap(entry_id, resolution_gaps, capture_gaps, acq_gaps)
        gate = gap["earliest_failed_gate"] if gap else "Gate4"
        reason = gap["gap_reason"] if gap else "not_measured_without_prior_gap"
        repair_rows.append({
            "manifest_entry_id": entry_id,
            "status": "gap",
            "earliest_failed_gate": gate,
            "gap_reason": reason,
            "repair_action": _repair_action(gate, reason),
        })

    eligibility = {
        "selector_eligibility_state": state,
        "gate5_allowed": gate5_allowed,
        "measured_rows": len(features),
        "total_p0_entries": len(entries),
        "remaining_gap_count": len([row for row in repair_rows if row["status"] == "gap"]),
        "feature_table_errors": row_errors,
        "timing_units": sorted(timing_units),
        "weight_mode": weight_mode,
        "selector_input_path": artifact_ref(SELECTOR_INPUT_PATH),
        "selector_input_hash": file_hash(SELECTOR_INPUT_PATH),
        "repair_report_path": artifact_ref(REPAIR_JSON_PATH),
    }
    repair = {"summary": eligibility, "entries": repair_rows}
    write_json(ELIGIBILITY_PATH, eligibility)
    write_json(REPAIR_JSON_PATH, repair)
    _write_repair_md(repair)
    return eligibility


def _write_repair_md(repair: dict[str, Any]) -> None:
    lines = [
        "# M1 Backward Repair Report",
        "",
        f"State: `{repair['summary']['selector_eligibility_state']}`",
        f"Measured rows: {repair['summary']['measured_rows']}",
        "",
        "| Entry | Status | Earliest Failed Gate | Reason | Action |",
        "|---|---|---|---|---|",
    ]
    for row in repair["entries"]:
        action = row.get("repair_action") or {}
        lines.append(
            f"| {row['manifest_entry_id']} | {row['status']} | {row.get('earliest_failed_gate') or ''} | "
            f"{row.get('gap_reason') or ''} | {action.get('description') or ''} |"
        )
    REPAIR_MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    eligibility = evaluate()
    print(f"Gate4 selector eligibility: {eligibility['selector_eligibility_state']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
