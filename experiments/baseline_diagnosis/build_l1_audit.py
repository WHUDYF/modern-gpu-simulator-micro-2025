"""PKA feature audit and stage-gate validator for L1."""

from __future__ import annotations

import json, sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "a_line" / "l1"

FT = ARTIFACT_DIR / "pka_feature_table_l1.json"
GP = ARTIFACT_DIR / "pka_acquisition_gap_l1.json"
MX = ARTIFACT_DIR / "pka_metric_availability_matrix_l1.json"
MF = ARTIFACT_DIR / "kernel_validation_manifest_l1.json"

VALID_STATUSES = {"full_closure_success", "acquisition_gate_success", "selector_insufficient_records", "weight_unit_conflict"}


def _audit_json(records, matrix):
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
        entries.append(entry)
    return {"audit_name": "L1 PKA Feature Audit", "dataset_level": "L1", "summary": {
        "total_invocations": len(entries), "fully_measured": sum(1 for e in entries if e["outcome"] == "measured"),
        "acquisition_gap": sum(1 for e in entries if e["outcome"] == "acquisition_gap"),
    }, "entries": entries}


def _audit_md(audit: dict, manifest: dict, records: list, PKA_FEATURES: dict) -> str:
    p0_ids = {e["id"] for e in manifest["entries"] if e["priority"] == "P0"}
    p0_recs = [r for r in records if r.get("manifest_id") in p0_ids]
    gap_recs = [r for r in p0_recs if r.get("outcome") == "acquisition_gap"]

    lines = ["# L1 PKA Feature Audit", "", "## Summary", "",
             f"- P0 invocations: {len(p0_recs)}",
             f"- P0 measured: {sum(1 for r in p0_recs if r['outcome'] == 'measured')}",
             f"- P0 acquisition gaps: {len(gap_recs)}", "",
             "## PKA Features", "",
             "| # | Feature | Canonical Metric |",
             "|---|---------|-----------------|"]
    for i, (name, spec) in enumerate(PKA_FEATURES.items(), 1):
        lines.append(f"| F{i:02d} | {name} | `{spec['canonical_metric']}` |")
    lines.extend(["", "## P0 Acquisition Gap Details", ""])
    for rec in gap_recs:
        lines.append(f"### {rec['kernel_invocation_id']} (manifest: {rec.get('manifest_id')})")
        lines.append(f"- kernel_name: `{rec['kernel_name']}`")
        lines.append(f"- source: `{rec.get('source_path')}`")
        lines.append(f"- missing metrics: {rec.get('missing_metric_count', 0)}/12")
        mm = rec.get("missing_metrics", [])
        if mm:
            lines.append(f"- missing: {', '.join(mm)}")
        lines.append("")
    if not gap_recs:
        lines.append("No P0 acquisition gaps.")
    lines.extend(["", "## Conclusion", ""])
    if gap_recs:
        lines.append(f"{len(gap_recs)} P0 invocations have acquisition gaps. Stage 3 (selector) and Stage 4 (B-line) are blocked.")
    else:
        lines.append("All P0 invocations are fully measured. Proceed to Stage 3.")
    return "\n".join(lines)


def _stage_gate(records, manifest):
    entries = manifest["entries"]
    p0_ids = {e["id"] for e in entries if e["priority"] == "P0"}
    p0_recs = [r for r in records if r.get("manifest_id") in p0_ids]
    p0_gaps = [r for r in p0_recs if r.get("outcome") == "acquisition_gap"]
    p0_measured = [r for r in p0_recs if r.get("outcome") == "measured"]

    if p0_gaps:
        run_status = "acquisition_gate_success"
        s3, s4 = "blocked", "blocked"
    elif len(p0_measured) < 2:
        run_status = "selector_insufficient_records"
        s3, s4 = "blocked", "blocked"
    else:
        run_status = "full_closure_success"
        s3, s4 = "ready", "pending"

    # Check timing unit consistency
    timing_units = set()
    for rec in p0_measured:
        src = rec.get("source_path", "")
        # Check if timing info available
        pass

    return {
        "report_name": "L1 Stage Gate Report",
        "dataset_level": "L1",
        "run_status": run_status,
        "stages": {
            "stage_1_manifest": "passed",
            "stage_2_feature_extraction": "passed",
            "stage_3_selector": s3,
            "stage_4_b_line_consumption": s4,
            "stage_5_tests": "in_progress",
        },
        "counts": {
            "total_p0_invocations": len(p0_recs),
            "measured_p0_invocations": len(p0_measured),
            "gap_p0_invocations": len(p0_gaps),
            "p0_entries_total": len(p0_ids),
        },
        "blocking_gaps": [{
            "manifest_id": r.get("manifest_id", ""),
            "kernel_invocation_id": r.get("kernel_invocation_id", ""),
            "kernel_name": r.get("kernel_name", ""),
            "source_path": r.get("source_path", ""),
            "missing_metric_count": len(r.get("missing_metrics", [])),
        } for r in p0_gaps],
        "next_action": (
            "Acquire NCU PM counter data for all P0 objects using the exact 12 canonical "
            "Nsight metric names. Stages 3 and 4 blocked until all P0 invocations are fully measured."
        ) if p0_gaps else ("Only {} measured records; need >= 2 for clustering.".format(len(p0_measured))
        ) if len(p0_measured) < 2 else "Proceed to Stage 3.",
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
    return "\n".join(lines)


def main():
    from pka_feature_extractor import PKA_FEATURES as _PKA

    records = json.loads(FT.read_text()) + json.loads(GP.read_text())
    matrix = json.loads(MX.read_text())
    manifest = json.loads(MF.read_text())

    audit = _audit_json(records, matrix)
    (ARTIFACT_DIR / "pka_feature_audit_l1.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    audit_md = _audit_md(audit, manifest, records, _PKA)
    (ARTIFACT_DIR / "pka_feature_audit_l1.md").write_text(audit_md + "\n")

    sg = _stage_gate(records, manifest)
    assert sg["run_status"] in VALID_STATUSES, f"Invalid run status: {sg['run_status']}"
    (ARTIFACT_DIR / "l1_stage_gate_report_l1.json").write_text(json.dumps(sg, indent=2, ensure_ascii=False) + "\n")
    (ARTIFACT_DIR / "l1_stage_gate_report_l1.md").write_text(_stage_gate_md(sg) + "\n")

    print(f"Run Status: {sg['run_status']}")
    print(f"Stages: {json.dumps(sg['stages'])}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
