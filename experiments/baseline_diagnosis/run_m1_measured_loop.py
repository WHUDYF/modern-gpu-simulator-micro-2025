"""End-to-end orchestrator for the PKA-M1 measured loop."""

from __future__ import annotations

import argparse
import sys

from shared_acquisition import ARTIFACT_DIR, write_json

import m1_measured_feature_extractor
import m1_ncu_capture_dispatcher
import m1_selector_eligibility
import m1_workload_resolver
import pka_m1_selector

STATUS_PATH = ARTIFACT_DIR / "m1_measured_loop_status_l1.json"


def run(dry_run_capture: bool = False, dry_run_smoke: bool = False) -> dict:
    resolutions, resolution_gaps = m1_workload_resolver.resolve(dry_run_smoke=dry_run_smoke)
    from shared_acquisition import write_json
    write_json(ARTIFACT_DIR / "m1_workload_resolution_l1.json", resolutions)
    write_json(ARTIFACT_DIR / "m1_workload_resolution_gap_l1.json", resolution_gaps)

    attempts, capture_gaps = m1_ncu_capture_dispatcher.dispatch(dry_run=dry_run_capture)
    features, acq_gaps = m1_measured_feature_extractor.extract()
    eligibility = m1_selector_eligibility.evaluate()
    if eligibility.get("gate5_allowed"):
        pka_m1_selector.run()
        status = "completed_gate5_formal_selector"
    else:
        status = "blocked_on_acquisition_with_repair_report"
    report = {
        "status": status,
        "gate1": {"resolved": len(resolutions), "gaps": len(resolution_gaps)},
        "gate2": {"attempts": len(attempts), "gaps": len(capture_gaps)},
        "gate3": {"measured": len(features), "gaps": len(acq_gaps)},
        "gate4": eligibility,
    }
    write_json(STATUS_PATH, report)
    return report


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

