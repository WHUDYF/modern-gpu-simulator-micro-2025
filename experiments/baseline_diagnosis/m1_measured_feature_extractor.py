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
    artifact_ref,
    feature_record,
    kernel_name_matches,
    missing_feature_record,
    parse_ncu_csv,
    read_json,
    repo_path,
    valid_environment_manifest,
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


def _gap_row(
    entry_id: str,
    attempt: dict[str, Any],
    kernel_or_case: str,
    reason: str,
    missing_features: list[str] | None = None,
    join_status: str | None = None,
    detail: str | None = None,
    meta: dict[str, Any] | None = None,
    inv: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = meta or {}
    return {
        "record_id": f"{entry_id}:{reason}",
        "manifest_entry_id": entry_id,
        "capture_job_id": attempt.get("capture_job_id"),
        "dataset_level": "L1",
        "source_type": meta.get("source_type"),
        "benchmark_name": meta.get("benchmark_name"),
        "kernel_or_case": kernel_or_case,
        "kernel_invocation_id": f"{kernel_or_case}#{(inv or {}).get('occurrence_index', 0) + 1}" if kernel_or_case else None,
        "failed_gate": "Gate3",
        "gate": "Gate3",
        "gap_reason": reason,
        "missing_features": missing_features or [],
        "join_status": join_status,
        "capture_status": attempt.get("capture_status"),
        "source_artifact_path": attempt.get("capture_csv_path"),
        "selected_metrics_path": artifact_ref(Path(attempt.get("capture_csv_path", ".")).parent / "selected_metrics.json"),
        "environment_manifest_path": attempt.get("environment_manifest_path"),
        "suggested_repair_action": f"Repair Gate3 acquisition gap {reason}",
        "detail": detail,
    }


def _entry_metadata(attempt: dict[str, Any], index: int, entry_id: str, kernel_or_case: str) -> dict[str, Any]:
    entries = attempt.get("consuming_manifest_entries") or []
    if index < len(entries):
        return entries[index]
    return {
        "manifest_entry_id": entry_id,
        "kernel_or_case": kernel_or_case,
        "source_type": None,
        "benchmark_name": None,
        "workload_id": None,
    }


def _join(
    invocations: list[dict[str, Any]],
    kernel_or_case: str,
    requested_occurrence_index: int,
    total_requested_for_kernel: int,
) -> tuple[dict[str, Any] | None, str, str | None]:
    if not kernel_or_case:
        return None, "empty_kernel_name", "manifest kernel_or_case is empty"
    if any(not inv.get("kernel_name") for inv in invocations):
        return None, "empty_kernel_name", "capture CSV contains empty kernel name"
    matches = [inv for inv in invocations if kernel_name_matches(inv.get("kernel_name", ""), kernel_or_case)]
    if not matches:
        return None, "missing_kernel", "no matching kernel name in capture CSV"
    if len(matches) > 1 and total_requested_for_kernel <= 1:
        return None, "ambiguous_kernel", f"{len(matches)} matching invocations"
    if requested_occurrence_index >= len(matches):
        return None, "occurrence_mismatch", f"requested occurrence {requested_occurrence_index}, only {len(matches)} matches"
    return matches[requested_occurrence_index], "matched", None


def _extract_features(inv: dict[str, Any], attempt: dict[str, Any], allowed: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    source_path = attempt["capture_csv_path"]
    features: dict[str, Any] = {}
    missing: list[str] = []
    for feature_name in FEATURE_ORDER:
        if feature_name == "num_thread_blocks":
            grid_value = inv.get("grid_size_normalized")
            if grid_value is None:
                grid_reason = "grid_size_parse_failed" if inv.get("grid_size_provenance", {}).get("normalization_rule") == "parse_failed" else "grid_size_missing"
                features[feature_name] = missing_feature_record(feature_name, source_path, grid_reason)
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
    attempts_doc = read_json(ATTEMPTS_PATH, [])
    attempts = attempts_doc.get("attempts", []) if isinstance(attempts_doc, dict) else attempts_doc
    features: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    join_audit: list[dict[str, Any]] = []
    for attempt in [row for row in attempts if row.get("gate3_eligible") is True]:
        env_ref = attempt.get("environment_manifest_path")
        env_path = repo_path(env_ref) if env_ref else None
        env_manifest = read_json(env_path, {}) if env_path and env_path.is_file() else {}
        if not env_ref or not valid_environment_manifest(env_manifest):
            for index, (entry_id, kernel_or_case) in enumerate(zip(attempt["consuming_manifest_entry_ids"], attempt["consuming_kernel_or_cases"])):
                gaps.append(_gap_row(
                    entry_id,
                    attempt,
                    kernel_or_case,
                    "env_manifest_missing" if not env_path or not env_path.exists() else "env_manifest_invalid",
                    meta=_entry_metadata(attempt, index, entry_id, kernel_or_case),
                    detail="missing or invalid Gate2 environment manifest",
                ))
            continue
        csv_path = repo_path(attempt["capture_csv_path"])
        try:
            invocations = parse_ncu_csv(csv_path)
        except Exception as exc:  # noqa: BLE001 - gate artifact should capture parser failure.
            for index, (entry_id, kernel_or_case) in enumerate(zip(attempt["consuming_manifest_entry_ids"], attempt["consuming_kernel_or_cases"])):
                gaps.append(_gap_row(
                    entry_id,
                    attempt,
                    kernel_or_case,
                    "incomplete_12d_feature_vector",
                    join_status="csv_parse_failed",
                    meta=_entry_metadata(attempt, index, entry_id, kernel_or_case),
                    detail=str(exc),
                ))
            continue
        allowed = _allowed_sources(attempt)
        kernel_counts = {
            kernel: attempt.get("consuming_kernel_or_cases", []).count(kernel)
            for kernel in attempt.get("consuming_kernel_or_cases", [])
        }
        seen_kernel_counts: dict[str, int] = {}
        for index, (entry_id, kernel_or_case) in enumerate(zip(attempt["consuming_manifest_entry_ids"], attempt["consuming_kernel_or_cases"])):
            requested_occurrence_index = seen_kernel_counts.get(kernel_or_case, 0)
            seen_kernel_counts[kernel_or_case] = requested_occurrence_index + 1
            meta = _entry_metadata(attempt, index, entry_id, kernel_or_case)
            inv, join_status, reason = _join(
                invocations,
                kernel_or_case,
                requested_occurrence_index,
                kernel_counts.get(kernel_or_case, 1),
            )
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
                reason_map = {
                    "missing_kernel": "missing_kernel_in_csv",
                    "ambiguous_kernel": "ambiguous_kernel_match",
                    "occurrence_mismatch": "occurrence_mismatch",
                    "empty_kernel_name": "empty_kernel_name",
                }
                gaps.append(_gap_row(
                    entry_id,
                    attempt,
                    kernel_or_case,
                    reason_map.get(join_status, "incomplete_12d_feature_vector"),
                    join_status=join_status,
                    meta=meta,
                    detail=reason,
                ))
                continue
            record_features, missing = _extract_features(inv, attempt, allowed)
            if missing:
                missing_reasons = {
                    name: record_features[name]["provenance"].get("missing_reason")
                    for name in missing
                }
                if "num_thread_blocks" in missing:
                    gap_reason = missing_reasons["num_thread_blocks"]
                elif any(reason == "selected_metric_absent_in_csv" for reason in missing_reasons.values()):
                    gap_reason = "missing_canonical_metric"
                else:
                    gap_reason = "incomplete_12d_feature_vector"
                gaps.append(_gap_row(
                    entry_id,
                    attempt,
                    kernel_or_case,
                    gap_reason,
                    missing_features=missing,
                    join_status=join_status,
                    meta=meta,
                    inv=inv,
                ))
                continue
            duration_ns = inv.get("duration")
            elapsed_cycles = inv.get("elapsed_cycles")
            features.append({
                "record_id": f"{entry_id}:{inv['csv_invocation_id']}",
                "dataset_level": "L1",
                "source_type": meta.get("source_type"),
                "benchmark_name": meta.get("benchmark_name"),
                "kernel_or_case": kernel_or_case,
                "manifest_id": entry_id,
                "manifest_entry_id": entry_id,
                "kernel_invocation_id": f"{kernel_or_case}#{inv.get('occurrence_index', 0) + 1}",
                "feature_mode": "pka_m1_measured",
                "feature_status": "complete_measured",
                "source_path": attempt["capture_csv_path"],
                "capture_job_id": attempt["capture_job_id"],
                "capture_status": attempt.get("capture_status"),
                "capture_exit_code": attempt.get("capture_exit_code"),
                "capture_stderr_path": attempt.get("capture_stderr_path"),
                "capture_warning": "non_zero_exit" if attempt.get("capture_status") == "capture_non_zero_exit_with_partial_csv" else None,
                "duration_ns": duration_ns,
                "elapsed_cycles": elapsed_cycles,
                "timing_basis": "duration_ns" if duration_ns is not None else ("elapsed_cycles" if elapsed_cycles is not None else None),
                "feature_provenance": {
                    "selected_metrics_path": str(Path(attempt["capture_csv_path"]).parent / "selected_metrics.json"),
                    "environment_manifest_path": attempt.get("environment_manifest_path"),
                    "join_status": join_status,
                    "csv_invocation_id": inv["csv_invocation_id"],
                    "capture_status": attempt.get("capture_status"),
                },
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
