"""End-to-end orchestrator for the PKA-M1 measured loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from shared_acquisition import ARTIFACT_DIR, read_json, write_json

import m1_measured_feature_extractor
import m1_ncu_capture_dispatcher
import m1_selector_eligibility
import m1_workload_resolver
import pka_m1_selector

STATUS_PATH = ARTIFACT_DIR / "m1_measured_loop_status_l1.json"
def _gate5_artifacts() -> list[Path]:
    return [
        pka_m1_selector.ARTIFACT_DIR / "pka_pca_projection_l1.json",
        pka_m1_selector.ARTIFACT_DIR / "pka_kmeans_clusters_l1.json",
        pka_m1_selector.ARTIFACT_DIR / "representative_anchor_table_l1.json",
        pka_m1_selector.ARTIFACT_DIR / "pka_compression_evaluation_l1.json",
    ]


def _repair_report_complete(path: Path, total_p0_entries: int) -> bool:
    report = read_json(path, {})
    entries = report.get("entries", [])
    if len(entries) != total_p0_entries:
        return False
    required = {
        "manifest_entry_id",
        "entry_status",
        "gate1_status",
        "gate2_status",
        "gate3_status",
        "earliest_failed_gate",
        "blocking_reason",
        "repair_action_type",
        "suggested_repair_action",
        "allowed_to_auto_run",
    }
    for row in entries:
        if row.get("entry_status") == "measured":
            continue
        if not required.issubset(row):
            return False
        if not row.get("earliest_failed_gate"):
            return False
    return True


def classify_completion(summary: dict[str, Any]) -> str:
    state = summary.get("selector_eligibility_state")
    if state == "selector_blocked_invalid_feature_table":
        return "stop_fail_invalid_feature_table"
    if state == "selector_blocked_mixed_timing_unit":
        return "stop_fail_mixed_timing_unit"
    if summary.get("gate5_allowed"):
        if all(summary.get("gate5_artifacts", {}).values()):
            return "completed_gate5_formal_selector"
        return "stop_fail_missing_gate5_artifacts"
    if not summary.get("backward_repair_report_exists") or not summary.get("per_entry_earliest_gate_complete"):
        return "stop_fail_missing_backward_repair_report"
    return "blocked_on_acquisition_with_repair_report"


def run_all(dry_run_capture: bool = False, dry_run_smoke: bool = False) -> dict:
    resolutions, resolution_gaps = m1_workload_resolver.resolve(dry_run_smoke=dry_run_smoke)
    write_json(m1_workload_resolver.RESOLUTION_PATH, resolutions)
    write_json(m1_workload_resolver.GAP_PATH, resolution_gaps)

    attempts, capture_gaps = m1_ncu_capture_dispatcher.dispatch(dry_run=dry_run_capture)
    features, acq_gaps = m1_measured_feature_extractor.extract()
    eligibility = m1_selector_eligibility.evaluate()
    if eligibility.get("gate5_allowed"):
        pka_m1_selector.run()
    repair_path = m1_selector_eligibility.REPAIR_JSON_PATH
    gate5_artifacts = {path.name: path.exists() for path in _gate5_artifacts()}
    classification_input = {
        "selector_eligibility_state": eligibility.get("selector_eligibility_state"),
        "gate5_allowed": eligibility.get("gate5_allowed"),
        "measured_rows": eligibility.get("measured_rows"),
        "backward_repair_report_exists": repair_path.exists(),
        "per_entry_earliest_gate_complete": _repair_report_complete(repair_path, eligibility.get("total_p0_entries", 0)),
        "gate5_artifacts": gate5_artifacts,
    }
    status = classify_completion(classification_input)
    report = {
        "status": status,
        "gate1": {"resolved": len(resolutions), "gaps": len(resolution_gaps)},
        "gate2": {"attempts": len(attempts), "gaps": len(capture_gaps)},
        "gate3": {"measured": len(features), "gaps": len(acq_gaps)},
        "gate4": eligibility,
        "completion_classifier": classification_input,
    }
    write_json(STATUS_PATH, report)
    return report


def run(dry_run_capture: bool = False, dry_run_smoke: bool = False) -> dict:
    return run_all(dry_run_capture=dry_run_capture, dry_run_smoke=dry_run_smoke)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run-capture", action="store_true")
    parser.add_argument("--dry-run-smoke", action="store_true")
    args = parser.parse_args(argv)
    report = run(dry_run_capture=args.dry_run_capture, dry_run_smoke=args.dry_run_smoke)
    print(report["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
