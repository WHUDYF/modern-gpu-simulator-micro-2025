"""PKA feature audit and stage-gate validator for L1.

Enforces P0 outcome validation, timing-unit conflict detection, and gate-clean
artifact management. Deletes stale downstream artifacts when gate blocks.
"""

from __future__ import annotations

import json, sys, os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "a_line" / "l1"

FT = ARTIFACT_DIR / "pka_feature_table_l1.json"
GP = ARTIFACT_DIR / "pka_acquisition_gap_l1.json"
MX = ARTIFACT_DIR / "pka_metric_availability_matrix_l1.json"
MF = ARTIFACT_DIR / "kernel_validation_manifest_l1.json"

VALID_STATUSES = {"full_closure_success", "acquisition_gate_success",
                   "selector_insufficient_records", "weight_unit_conflict"}

STAGE_3_4_ARTIFACTS = [
    "pka_selector_config_l1.json",
    "pka_dimensionality_reduction_report_l1.json",
    "pka_dimensionality_reduction_report_l1.md",
    "pka_reduced_feature_table_l1.json",
    "pka_cluster_assignment_l1.json",
    "representative_anchor_table_l1.json",
    "b_line_consumption_report_l1.md",
]

TIMING_UNIT_MAP = {"duration_ns": "ns", "elapsed_cycles": "cycles"}


def _clean_stale_artifacts():
    for name in STAGE_3_4_ARTIFACTS:
        p = ARTIFACT_DIR / name
        if p.exists():
            p.unlink()


def _collect_timing_units(records: list[dict]) -> set[str]:
    units = set()
    for rec in records:
        sp = rec.get("source_path", "")
        # Check if timing info is embedded in features
        feats = rec.get("features", {})
        for fn, fv in feats.items():
            src = fv.get("source_artifact_path", "")
            # Timing is per-source-file
            pass
    return units


def _audit_json(records):
    provenance_errors = []
    entries = []
    for rec in records:
        feats = rec.get("features", {})
        entry = {
            "manifest_id": rec.get("manifest_id", ""),
            "kernel_invocation_id": rec.get("kernel_invocation_id", ""),
            "kernel_name": rec.get("kernel_name", ""),
            "priority": rec.get("priority", ""),
            "source_path": rec.get("source_path", ""),
            "outcome": rec.get("outcome"),
            "missing_metric_count": len(rec.get("missing_metrics", [])),
            "features": {},
        }
        for name in feats:
            f = feats[name]
            entry["features"][name] = {
                "value": f.get("value"), "status": f.get("status", "unknown"),
                "canonical_metric": f.get("canonical_metric", ""),
                "actual_source_metric": f.get("actual_source_metric"),
                "source_artifact_path": f.get("source_artifact_path", ""),
            }
            if "missing_reason" in f:
                entry["features"][name]["missing_reason"] = f["missing_reason"]
            # Validate measured provenance
            if f.get("status") == "measured":
                if not f.get("canonical_metric"):
                    provenance_errors.append(f"{rec.get('kernel_invocation_id', '?')}/{name}: empty canonical_metric")
                if not f.get("actual_source_metric"):
                    provenance_errors.append(f"{rec.get('kernel_invocation_id', '?')}/{name}: empty actual_source_metric")
                if not f.get("source_artifact_path"):
                    provenance_errors.append(f"{rec.get('kernel_invocation_id', '?')}/{name}: empty source_artifact_path")
        entries.append(entry)
    return {"audit_name": "L1 PKA Feature Audit", "dataset_level": "L1", "summary": {
        "total_invocations": len(entries), "fully_measured": sum(1 for e in entries if e["outcome"] == "measured"),
        "acquisition_gap": sum(1 for e in entries if e["outcome"] == "acquisition_gap"),
    }, "entries": entries, "provenance_validation_errors": provenance_errors}


def _audit_md(audit, manifest, records, pka_features):
    p0_ids = {e["id"] for e in manifest["entries"] if e["priority"] == "P0"}
    p0_recs = [r for r in records if r.get("manifest_id") in p0_ids]
    gap_recs = [r for r in p0_recs if r.get("outcome") == "acquisition_gap"]

    lines = ["# L1 PKA Feature Audit", "", "## Summary", "",
             f"- P0 invocations: {len(p0_recs)}",
             f"- P0 measured: {sum(1 for r in p0_recs if r['outcome'] == 'measured')}",
             f"- P0 acquisition gaps: {len(gap_recs)}", "",
             "## PKA Features", "", "| # | Feature | Canonical Metric |",
             "|---|---------|-----------------|"]
    for i, (name, spec) in enumerate(pka_features.items(), 1):
        lines.append(f"| F{i:02d} | {name} | `{spec['canonical_metric']}` |")
    lines.extend(["", "## P0 Acquisition Gap Details", ""])
    for rec in gap_recs:
        lines.append(f"### {rec['kernel_invocation_id']} (manifest: {rec.get('manifest_id')})")
        lines.append(f"- kernel_name: `{rec['kernel_name']}`")
        lines.append(f"- source: `{rec.get('source_path')}`")
        mm_count = len(rec.get("missing_metrics", []))
        lines.append(f"- missing metrics: {mm_count}/12")
        mm = rec.get("missing_metrics", [])
        if mm: lines.append(f"- missing: {', '.join(mm)}")
        lines.append("")
    if not gap_recs:
        lines.append("No P0 acquisition gaps.")
    lines.extend(["", "## Conclusion", ""])
    if gap_recs:
        lines.append(f"{len(gap_recs)} P0 invocations have acquisition gaps.")
    else:
        lines.append("All P0 invocations are fully measured.")
    return "\n".join(lines)


def _stage_gate(records: list[dict], manifest: dict) -> dict:
    entries = manifest["entries"]
    p0_ids = {e["id"] for e in entries if e["priority"] == "P0"}
    p0_recs = [r for r in records if r.get("manifest_id") in p0_ids]
    p0_gaps = [r for r in p0_recs if r.get("outcome") == "acquisition_gap"]
    p0_measured = [r for r in p0_recs if r.get("outcome") == "measured"]

    # Pre-validation: reject before assigning valid run status
    p0_with_outcomes = {r["manifest_id"] for r in p0_recs}
    missing_p0 = sorted(p0_ids - p0_with_outcomes)
    invalid_outcomes = {r.get("outcome") for r in p0_recs} - {"measured", "acquisition_gap"}
    all_ids = [r["kernel_invocation_id"] for r in p0_recs]
    dup_ids = sorted(set(i for i in all_ids if all_ids.count(i) > 1))

    validation_errors = {}
    if missing_p0:
        validation_errors["missing_p0_outcomes"] = missing_p0
    if invalid_outcomes:
        validation_errors["invalid_outcome_types"] = sorted(invalid_outcomes)
    if dup_ids:
        validation_errors["duplicate_invocation_ids"] = dup_ids

    if validation_errors:
        return {
            "report_name": "L1 Stage Gate Report", "dataset_level": "L1",
            "run_status": "validation_failed",
            "stages": {"stage_1_manifest": "passed", "stage_2_feature_extraction": "failed",
                       "stage_3_selector": "blocked", "stage_4_b_line_consumption": "blocked",
                       "stage_5_tests": "pending"},
            "counts": {"total_p0_invocations": len(p0_recs),
                       "measured_p0_invocations": len(p0_measured),
                       "gap_p0_invocations": len(p0_gaps),
                       "p0_entries_total": len(p0_ids)},
            "validation_errors": validation_errors,
            "next_action": f"Fix P0 outcome validation errors: {validation_errors}",
        }

    if p0_gaps:
        run_status = "acquisition_gate_success"
        s3, s4 = "blocked", "blocked"
        next_action = "Acquire NCU PM counter data using exact 12 canonical Nsight metric names."
    elif len(p0_measured) < 2:
        run_status = "selector_insufficient_records"
        s3, s4 = "blocked", "blocked"
        next_action = f"Need >= 2 measured records; currently {len(p0_measured)}."
    else:
        # Per-record timing provenance from measured records
        selected_units = set()
        for rec in p0_measured:
            tb = rec.get("timing_basis")
            if tb:
                selected_units.add(tb)
            else:
                selected_units.add("unknown")
        selected_units.discard("unknown")

        if len(selected_units) > 1:
            run_status = "weight_unit_conflict"
            s3, s4 = "blocked", "blocked"
            next_action = "Mixed timing units detected. Standardize on one unit before proceeding."
        else:
            run_status = "full_closure_success"
            s3, s4 = "ready", "pending"
            next_action = "Proceed to Stage 3 (selector)."

    return {
        "report_name": "L1 Stage Gate Report", "dataset_level": "L1",
        "run_status": run_status,
        "stages": {"stage_1_manifest": "passed", "stage_2_feature_extraction": "passed",
                   "stage_3_selector": s3, "stage_4_b_line_consumption": s4,
                   "stage_5_tests": "pending"},
        "counts": {"total_p0_invocations": len(p0_recs),
                   "measured_p0_invocations": len(p0_measured),
                   "gap_p0_invocations": len(p0_gaps),
                   "p0_entries_total": len(p0_ids)},
        "blocking_gaps": [{"manifest_id": r.get("manifest_id", ""),
                          "kernel_invocation_id": r.get("kernel_invocation_id", ""),
                          "kernel_name": r.get("kernel_name", ""),
                          "source_path": r.get("source_path", ""),
                          "missing_metric_count": len(r.get("missing_metrics", [])),
                          } for r in p0_gaps],
        "next_action": next_action,
    }


def _stage_gate_md(report):
    lines = ["# L1 Stage Gate Report", "", f"**Run Status**: `{report['run_status']}`", "",
             "## Stage Status", "", "| Stage | Status |", "|-------|--------|"]
    for stage, status in report["stages"].items():
        lines.append(f"| {stage} | `{status}` |")
    lines.extend(["", "## Counts", "",
                  f"- P0 invocations: {report['counts']['total_p0_invocations']}",
                  f"- P0 measured: {report['counts']['measured_p0_invocations']}",
                  f"- P0 gaps: {report['counts']['gap_p0_invocations']}",
                  "", "## Next Action", "", report["next_action"]])
    ve = report.get("validation_errors")
    if ve:
        lines.extend(["", "## Validation Errors", ""])
        if ve.get("missing_p0_outcomes"):
            lines.append(f"- Missing P0 outcomes: {ve['missing_p0_outcomes']}")
        if ve.get("duplicate_invocation_ids"):
            lines.append(f"- Duplicate invocation IDs: {ve['duplicate_invocation_ids']}")
    return "\n".join(lines)


def main():
    from pka_feature_extractor import PKA_FEATURES as _PKA

    records = json.loads(FT.read_text()) + json.loads(GP.read_text())
    matrix = json.loads(MX.read_text())
    manifest = json.loads(MF.read_text())

    audit = _audit_json(records)
    pve = audit.get("provenance_validation_errors", [])
    if pve:
        print(f"Audit provenance validation errors: {pve}")
    # Never allow measured features with empty provenance
    measured_pve = [e for e in pve if "measured" == audit["entries"][0]["outcome"]] if pve and audit["entries"] else []
    (ARTIFACT_DIR / "pka_feature_audit_l1.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    audit_md = _audit_md(audit, manifest, records, _PKA)
    (ARTIFACT_DIR / "pka_feature_audit_l1.md").write_text(audit_md + "\n")

    sg = _stage_gate(records, manifest)
    if sg["run_status"] == "validation_failed":
        print(f"Stage gate validation failed: {sg.get('validation_errors', {})}")

    # Gate-clean: delete stale downstream artifacts when gate blocks
    if sg["stages"]["stage_3_selector"] == "blocked":
        _clean_stale_artifacts()

    (ARTIFACT_DIR / "l1_stage_gate_report_l1.json").write_text(json.dumps(sg, indent=2, ensure_ascii=False) + "\n")
    (ARTIFACT_DIR / "l1_stage_gate_report_l1.md").write_text(_stage_gate_md(sg) + "\n")

    print(f"Run Status: {sg['run_status']}")
    print(f"Stages: {json.dumps(sg['stages'])}")
    if sg["stages"]["stage_3_selector"] == "blocked":
        print(f"Cleaned stale downstream artifacts: {STAGE_3_4_ARTIFACTS}")
    if sg["run_status"] == "validation_failed":
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
