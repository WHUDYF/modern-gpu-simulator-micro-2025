"""Gate 4 selector eligibility and backward repair for PKA-M1."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
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


def _unwrap_rows(doc: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(doc, dict):
        return doc.get(key, [])
    return doc or []


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


def _repair_action(gate: str, reason: str | None, source: dict[str, Any] | None = None) -> dict[str, Any]:
    source = source or {}
    if gate == "Gate1":
        if source.get("build_command"):
            return {
                "action_type": "executable_command",
                "description": f"Run the registry build command for missing binary: {reason}",
                "command": source.get("build_command"),
                "working_directory": source.get("working_directory"),
                "source_of_command": "workload_registry_l1.json",
            }
        return {
            "action_type": "manual_action",
            "description": f"Provide the registry binary or add an allowlisted build command for reason: {reason}",
            "command": None,
            "working_directory": source.get("working_directory"),
            "source_of_command": "workload_registry_l1.json",
        }
    if gate == "Gate2":
        return {"action_type": "environment_action", "description": f"Fix NCU capture environment or metric selection for reason: {reason}", "command": None}
    if gate == "Gate3":
        return {"action_type": "code_fix_required", "description": f"Fix capture CSV parsing, kernel join, or missing metrics for reason: {reason}", "command": None}
    return {"action_type": "code_fix_required", "description": f"Fix selector preflight for reason: {reason}", "command": None}


def _gate_status(entry_id: str, measured_ids: set[str], resolution_gaps: list[dict[str, Any]], capture_gaps: list[dict[str, Any]], acq_gaps: list[dict[str, Any]]) -> tuple[str, str, str]:
    if entry_id in measured_ids:
        return "passed", "passed", "passed"
    if any(gap.get("manifest_entry_id") == entry_id for gap in resolution_gaps):
        return "blocked", "not_attempted", "not_attempted"
    if any(entry_id in gap.get("consuming_manifest_entry_ids", []) for gap in capture_gaps):
        return "passed", "blocked", "not_attempted"
    if any(gap.get("manifest_entry_id") == entry_id for gap in acq_gaps):
        return "passed", "passed", "blocked"
    return "unknown", "unknown", "unknown"


def evaluate() -> dict[str, Any]:
    entries = _p0_entries()
    feature_table_exists = FEATURE_TABLE_PATH.exists()
    features = read_json(FEATURE_TABLE_PATH, []) if feature_table_exists else []
    acq_gaps = read_json(ACQ_GAP_PATH, [])
    resolution_gaps = read_json(RESOLUTION_GAP_PATH, [])
    capture_gaps = _unwrap_rows(read_json(CAPTURE_GAP_PATH, []), "gaps")
    measured_ids = {row.get("manifest_id") for row in features}

    row_errors = []
    forbidden_violations = []
    feature_mode_violations = []
    for row in features:
        errors = _valid_feature_row(row)
        if errors:
            row_errors.append({"record_id": row.get("record_id"), "errors": errors})
        if row.get("feature_mode") != "pka_m1_measured":
            feature_mode_violations.append(row.get("record_id"))
        forbidden = sorted(FORBIDDEN_SELECTOR_FIELDS & set(row))
        if forbidden:
            forbidden_violations.append({"record_id": row.get("record_id"), "fields": forbidden})

    timing_units = set()
    for row in features:
        if row.get("duration_ns") is not None:
            timing_units.add("duration_ns")
        elif row.get("elapsed_cycles") is not None:
            timing_units.add("elapsed_cycles")
        elif row.get("timing_basis"):
            timing_units.add(row.get("timing_basis"))
    blocking_reasons = []
    if not feature_table_exists:
        state = "selector_blocked_invalid_feature_table"
        blocking_reasons.append("feature_table_missing")
    elif row_errors:
        state = "selector_blocked_invalid_feature_table"
        blocking_reasons.append("invalid_feature_table")
    elif len(timing_units) > 1:
        state = "selector_blocked_mixed_timing_unit"
        blocking_reasons.append("mixed_timing_unit")
    elif len(features) < 3:
        state = "selector_blocked_insufficient_measured_records"
        blocking_reasons.append("insufficient_measured_records")
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
                "weight_input": {
                    "weight_mode": weight_mode,
                    "timing_unit": next(iter(timing_units), None) if weight_mode == "timing_weight" else None,
                    "value": float(row.get("duration_ns") or row.get("elapsed_cycles") or row.get("duration") or row.get("elapsed_time") or 1.0),
                },
            })
    write_json(SELECTOR_INPUT_PATH, selector_records)

    repair_rows = []
    for entry in entries:
        entry_id = entry["id"]
        gate1_status, gate2_status, gate3_status = _gate_status(entry_id, measured_ids, resolution_gaps, capture_gaps, acq_gaps)
        if entry_id in measured_ids:
            repair_rows.append({
                "manifest_entry_id": entry_id,
                "kernel_or_case": entry.get("kernel_or_case"),
                "entry_status": "measured",
                "gate1_status": gate1_status,
                "gate2_status": gate2_status,
                "gate3_status": gate3_status,
                "earliest_failed_gate": None,
                "blocking_reason": None,
                "repair_action_type": None,
                "suggested_repair_action": None,
                "executable_command": None,
                "allowed_to_auto_run": False,
            })
            continue
        gap = _earliest_gap(entry_id, resolution_gaps, capture_gaps, acq_gaps)
        gate = gap["earliest_failed_gate"] if gap else "Gate4"
        reason = gap["gap_reason"] if gap else "not_measured_without_prior_gap"
        action = _repair_action(gate, reason, gap.get("source") if gap else None)
        repair_rows.append({
            "manifest_entry_id": entry_id,
            "kernel_or_case": entry.get("kernel_or_case"),
            "entry_status": "blocked",
            "gate1_status": gate1_status,
            "gate2_status": gate2_status,
            "gate3_status": gate3_status,
            "earliest_failed_gate": gate,
            "blocking_reason": reason,
            "repair_action_type": action["action_type"],
            "suggested_repair_action": action["description"],
            "executable_command": action["command"],
            "working_directory": action.get("working_directory"),
            "source_of_command": action.get("source_of_command"),
            "allowed_to_auto_run": False,
        })

    generated_at = datetime.now(timezone.utc).isoformat()
    preflight = {
        "status": "failed" if (not feature_table_exists) or row_errors or forbidden_violations or feature_mode_violations else "passed",
        "checked_rows": len(features),
        "complete_12d_rows": len(features) - len(row_errors),
        "invalid_rows": row_errors,
        "forbidden_field_violations": forbidden_violations,
        "feature_mode_violations": feature_mode_violations,
    }
    timing_check = {
        "status": "failed" if len(timing_units) > 1 else "passed",
        "weight_mode": weight_mode,
        "timing_unit": next(iter(timing_units), None) if len(timing_units) == 1 else None,
        "conflicting_units": sorted(timing_units) if len(timing_units) > 1 else [],
        "conflict_records": [
            {
                "record_id": row.get("record_id"),
                "timing_basis": row.get("timing_basis"),
                "duration_ns": row.get("duration_ns"),
                "elapsed_cycles": row.get("elapsed_cycles"),
            }
            for row in features
            if len(timing_units) > 1
        ],
    }
    eligibility = {
        "artifact_name": "m1_selector_eligibility_l1",
        "generated_at": generated_at,
        "selector_eligibility_state": state,
        "gate5_allowed": gate5_allowed,
        "measured_rows": len(features),
        "gap_rows": len(acq_gaps) + len(resolution_gaps) + len(capture_gaps),
        "total_p0_entries": len(entries),
        "feature_table_path": artifact_ref(FEATURE_TABLE_PATH),
        "acquisition_gap_path": artifact_ref(ACQ_GAP_PATH),
        "feature_table_preflight": preflight,
        "timing_check": timing_check,
        "remaining_gap_count": len([row for row in repair_rows if row["entry_status"] == "blocked"]),
        "feature_table_errors": row_errors,
        "timing_units": sorted(timing_units),
        "weight_mode": weight_mode,
        "timing_unit": timing_check["timing_unit"],
        "selector_input_projection_path": artifact_ref(SELECTOR_INPUT_PATH),
        "selector_input_path": artifact_ref(SELECTOR_INPUT_PATH),
        "selector_input_hash": file_hash(SELECTOR_INPUT_PATH),
        "backward_repair_report_path": artifact_ref(REPAIR_JSON_PATH),
        "repair_report_path": artifact_ref(REPAIR_JSON_PATH),
        "blocking_reasons": blocking_reasons,
    }
    repair_summary = {
        "total_p0_entries": len(entries),
        "measured_entries": sum(1 for row in repair_rows if row["entry_status"] == "measured"),
        "blocked_entries": sum(1 for row in repair_rows if row["entry_status"] == "blocked"),
        "not_attempted_entries": sum(1 for row in repair_rows if "not_attempted" in {row["gate1_status"], row["gate2_status"], row["gate3_status"]}),
        "gate1_blocked_count": sum(1 for row in repair_rows if row.get("earliest_failed_gate") == "Gate1"),
        "gate2_blocked_count": sum(1 for row in repair_rows if row.get("earliest_failed_gate") == "Gate2"),
        "gate3_blocked_count": sum(1 for row in repair_rows if row.get("earliest_failed_gate") == "Gate3"),
        "gate4_blocked_count": sum(1 for row in repair_rows if row.get("earliest_failed_gate") == "Gate4"),
        "auto_runnable_repairs": sum(1 for row in repair_rows if row.get("allowed_to_auto_run")),
        "manual_repairs": sum(1 for row in repair_rows if row.get("repair_action_type") == "manual_action"),
        "environment_actions": sum(1 for row in repair_rows if row.get("repair_action_type") == "environment_action"),
        "code_fixes_required": sum(1 for row in repair_rows if row.get("repair_action_type") == "code_fix_required"),
    }
    eligibility["remaining_gap_count"] = repair_summary["blocked_entries"]
    repair = {
        "artifact_name": "m1_backward_repair_report_l1",
        "generated_at": generated_at,
        "selector_eligibility_state": state,
        "gate5_allowed": gate5_allowed,
        "entries": repair_rows,
        "summary": repair_summary,
    }
    write_json(ELIGIBILITY_PATH, eligibility)
    write_json(REPAIR_JSON_PATH, repair)
    _write_repair_md(repair)
    return eligibility


def _write_repair_md(repair: dict[str, Any]) -> None:
    lines = [
        "# M1 Backward Repair Report",
        "",
        f"State: `{repair['selector_eligibility_state']}`",
        f"Measured entries: {repair['summary']['measured_entries']}",
        "",
        "## Selector readiness summary",
        "",
        json.dumps(repair["summary"], sort_keys=True),
        "",
    ]
    for gate in ("Gate1", "Gate2", "Gate3", "Gate4"):
        rows = [row for row in repair["entries"] if row.get("earliest_failed_gate") == gate]
        lines.extend([
            f"## {gate} blockers",
            "",
            "| Entry | Kernel/Case | Reason | Action Type | Suggested Action |",
            "|---|---|---|---|---|",
        ])
        if not rows:
            lines.append("|  |  |  |  |  |")
        for row in rows:
            lines.append(
                f"| {row['manifest_entry_id']} | {row.get('kernel_or_case') or ''} | "
                f"{row.get('blocking_reason') or ''} | {row.get('repair_action_type') or ''} | "
                f"{row.get('suggested_repair_action') or ''} |"
            )
        lines.append("")
    lines.extend([
        "| Entry | Status | Earliest Failed Gate | Reason | Action |",
        "|---|---|---|---|---|",
    ])
    for row in repair["entries"]:
        lines.append(
            f"| {row['manifest_entry_id']} | {row['entry_status']} | {row.get('earliest_failed_gate') or ''} | "
            f"{row.get('blocking_reason') or ''} | {row.get('suggested_repair_action') or ''} |"
        )
    REPAIR_MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    eligibility = evaluate()
    print(f"Gate4 selector eligibility: {eligibility['selector_eligibility_state']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
