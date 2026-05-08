"""Gate 3 strict measured feature extractor for PKA-M1."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from shared_acquisition import (
    ARTIFACT_DIR,
    FEATURE_ORDER,
    FEATURE_SPECS,
    REPO_ROOT,
    feature_record,
    kernel_name_matches,
    missing_feature_record,
    parse_ncu_csv,
    read_json,
    repo_path,
    write_json,
)

ATTEMPTS_PATH = ARTIFACT_DIR / "m1_ncu_capture_attempts_l1.json"
FEATURE_TABLE_PATH = ARTIFACT_DIR / "pka_feature_table_l1.json"
GAP_PATH = ARTIFACT_DIR / "pka_acquisition_gap_l1.json"
FEATURE_AUDIT_PATH = ARTIFACT_DIR / "pka_feature_audit_l1.json"
JOIN_AUDIT_PATH = ARTIFACT_DIR / "pka_join_audit_l1.json"


def _allowed_sources(attempt: dict[str, Any]) -> dict[str, str]:
    selected_path = repo_path(Path(attempt["capture_csv_path"]).parent / "selected_metrics.json")
    rows = read_json(selected_path, [])
    return {
        row["feature_name"]: row.get("actual_source_metric")
        for row in rows
        if row.get("resolution_status") in {"available", "rollup_resolved", "launch_metadata"}
    }


def _join(invocations: list[dict[str, Any]], kernel_or_case: str) -> tuple[dict[str, Any] | None, str, str | None]:
    matches = [inv for inv in invocations if kernel_name_matches(inv.get("kernel_name", ""), kernel_or_case)]
    if not matches:
        return None, "missing_kernel", "no matching kernel name in capture CSV"
    if len(matches) > 1:
        return None, "ambiguous_kernel", f"{len(matches)} matching invocations"
    return matches[0], "matched", None


def _extract_features(inv: dict[str, Any], attempt: dict[str, Any], allowed: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    source_path = attempt["capture_csv_path"]
    features: dict[str, Any] = {}
    missing: list[str] = []
    for feature_name in FEATURE_ORDER:
        if feature_name == "num_thread_blocks":
            grid_value = inv.get("grid_size_normalized")
            if grid_value is None:
                features[feature_name] = missing_feature_record(feature_name, source_path, "grid_size_missing")
                missing.append(feature_name)
            else:
                features[feature_name] = feature_record(
                    feature_name,
                    float(grid_value),
                    "Grid Size",
                    source_path,
                    inv.get("grid_size_provenance", {}),
                )
            continue
        actual_metric = allowed.get(feature_name)
        value = inv.get("metric_map", {}).get(actual_metric)
        if actual_metric is None or value is None:
            features[feature_name] = missing_feature_record(feature_name, source_path, "selected_metric_absent_in_csv")
            missing.append(feature_name)
        else:
            features[feature_name] = feature_record(
                feature_name,
                float(value),
                actual_metric,
                source_path,
                {
                    "capture_job_id": attempt["capture_job_id"],
                    "csv_invocation_id": inv["csv_invocation_id"],
                    "kernel_name": inv.get("kernel_name"),
                },
            )
    return features, missing


def extract() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempts = read_json(ATTEMPTS_PATH, [])
    features: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    join_audit: list[dict[str, Any]] = []
    for attempt in [row for row in attempts if row.get("gate3_eligible") is True]:
        csv_path = repo_path(attempt["capture_csv_path"])
        try:
            invocations = parse_ncu_csv(csv_path)
        except Exception as exc:  # noqa: BLE001 - gate artifact should capture parser failure.
            for entry_id, kernel_or_case in zip(attempt["consuming_manifest_entry_ids"], attempt["consuming_kernel_or_cases"]):
                gaps.append({
                    "manifest_entry_id": entry_id,
                    "capture_job_id": attempt["capture_job_id"],
                    "kernel_or_case": kernel_or_case,
                    "gate": "Gate3",
                    "gap_reason": "csv_parse_failed",
                    "detail": str(exc),
                })
            continue
        allowed = _allowed_sources(attempt)
        for entry_id, kernel_or_case in zip(attempt["consuming_manifest_entry_ids"], attempt["consuming_kernel_or_cases"]):
            inv, join_status, reason = _join(invocations, kernel_or_case)
            join_audit.append({
                "capture_job_id": attempt["capture_job_id"],
                "manifest_entry_id": entry_id,
                "kernel_or_case": kernel_or_case,
                "csv_invocation_id": inv.get("csv_invocation_id") if inv else None,
                "kernel_name": inv.get("kernel_name") if inv else None,
                "occurrence_index": inv.get("occurrence_index") if inv else None,
                "join_status": join_status,
                "auxiliary_grid_size_evidence": inv.get("grid_size_provenance") if inv else None,
                "reason": reason,
            })
            if inv is None:
                gaps.append({
                    "manifest_entry_id": entry_id,
                    "capture_job_id": attempt["capture_job_id"],
                    "kernel_or_case": kernel_or_case,
                    "gate": "Gate3",
                    "gap_reason": join_status,
                    "detail": reason,
                })
                continue
            record_features, missing = _extract_features(inv, attempt, allowed)
            if missing:
                gaps.append({
                    "manifest_entry_id": entry_id,
                    "capture_job_id": attempt["capture_job_id"],
                    "kernel_or_case": kernel_or_case,
                    "gate": "Gate3",
                    "gap_reason": "missing_required_metrics",
                    "missing_features": missing,
                })
                continue
            features.append({
                "record_id": f"{entry_id}:{inv['csv_invocation_id']}",
                "manifest_id": entry_id,
                "kernel_invocation_id": f"{kernel_or_case}#{inv.get('occurrence_index', 0) + 1}",
                "feature_mode": "pka_m1_measured",
                "capture_job_id": attempt["capture_job_id"],
                "capture_warning": "non_zero_exit" if attempt.get("capture_status") == "capture_non_zero_exit_with_partial_csv" else None,
                "features": record_features,
            })
    audit = {
        "measured_records": len(features),
        "gap_records": len(gaps),
        "missing_feature_counts": {
            name: sum(1 for gap in gaps if name in gap.get("missing_features", []))
            for name in FEATURE_ORDER
        },
    }
    write_json(FEATURE_TABLE_PATH, features)
    write_json(GAP_PATH, gaps)
    write_json(FEATURE_AUDIT_PATH, audit)
    write_json(JOIN_AUDIT_PATH, join_audit)
    return features, gaps


def main() -> int:
    features, gaps = extract()
    print(f"Gate3 measured extractor: {len(features)} measured, {len(gaps)} gaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())

