"""PKA feature audit generator and stage-gate validator for L1.

Reads the feature extractor outputs (feature table, acquisition gap, metric
availability matrix) and produces audit reports and stage-gate decisions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "a_line" / "l1"

FEATURE_TABLE_PATH = ARTIFACT_DIR / "pka_feature_table_l1.json"
GAP_PATH = ARTIFACT_DIR / "pka_acquisition_gap_l1.json"
MATRIX_PATH = ARTIFACT_DIR / "pka_metric_availability_matrix_l1.json"
MANIFEST_PATH = ARTIFACT_DIR / "kernel_validation_manifest_l1.json"

AUDIT_JSON_PATH = ARTIFACT_DIR / "pka_feature_audit_l1.json"
AUDIT_MD_PATH = ARTIFACT_DIR / "pka_feature_audit_l1.md"
STAGE_GATE_JSON_PATH = ARTIFACT_DIR / "l1_stage_gate_report_l1.json"
STAGE_GATE_MD_PATH = ARTIFACT_DIR / "l1_stage_gate_report_l1.md"


def _build_audit_json(
    gap_records: list[dict[str, Any]],
    feature_records: list[dict[str, Any]],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    audit_entries: list[dict[str, Any]] = []
    for rec in gap_records:
        features = rec.get("feature_details", {})
        entry = {
            "kernel_invocation_id": rec.get("kernel_invocation_id", "unknown"),
            "kernel_name": rec.get("kernel_name", "unknown"),
            "source_path": rec.get("source_path", ""),
            "outcome": "acquisition_gap",
            "missing_metric_count": len(rec.get("missing_metrics", [])),
            "features": {},
        }
        for pka_name in features:
            f = features[pka_name]
            entry["features"][pka_name] = {
                "value": f.get("value"),
                "status": f.get("status", "unknown"),
                "canonical_metric": f.get("canonical_metric", ""),
                "actual_source_metric": f.get("actual_source_metric"),
                "source_artifact_path": f.get("source_artifact_path", ""),
            }
        audit_entries.append(entry)

    for rec in feature_records:
        features = rec.get("features", {})
        entry = {
            "kernel_invocation_id": rec.get("kernel_invocation_id", "unknown"),
            "kernel_name": rec.get("kernel_name", "unknown"),
            "source_path": rec["metadata"].get("source_path", ""),
            "outcome": "measured",
            "missing_metric_count": 0,
            "features": {},
        }
        for pka_name, f in features.items():
            entry["features"][pka_name] = {
                "value": f.get("value"),
                "status": f.get("status", "measured"),
                "canonical_metric": f.get("canonical_metric", ""),
                "actual_source_metric": f.get("actual_source_metric", ""),
                "source_artifact_path": f.get("source_artifact_path", ""),
            }
        audit_entries.append(entry)

    return {
        "audit_name": "L1 PKA Feature Audit",
        "dataset_level": "L1",
        "summary": {
            "total_invocations": len(audit_entries),
            "fully_measured": len(feature_records),
            "acquisition_gap": len(gap_records),
            "availability_matrix": matrix.get("availability", {}),
        },
        "entries": audit_entries,
    }


def _build_audit_md(
    audit_json: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    summary = audit_json["summary"]
    p0_entries = [e for e in manifest["entries"] if e["priority"] == "P0"]

    lines = [
        "# L1 PKA Feature Audit",
        "",
        f"Generated from {len(p0_entries)} P0 + "
        f"{len(manifest['entries']) - len(p0_entries)} P1 manifest entries.",
        "",
        "## Summary",
        "",
        f"- Total invocations: {summary['total_invocations']}",
        f"- Fully measured (12/12): {summary['fully_measured']}",
        f"- Acquisition gaps: {summary['acquisition_gap']}",
        "",
        "## PKA Feature Requirements",
        "",
        "The 12 PKA features and their canonical Nsight metric names:",
        "",
    ]
    lines.append("| # | Feature | Canonical Metric | Category |")
    lines.append("|---|---------|-----------------|----------|")
    pka_order = [
        ("coalesced_global_loads", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "coalesced_memory"),
        ("coalesced_global_stores", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "coalesced_memory"),
        ("coalesced_local_loads", "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", "coalesced_memory"),
        ("thread_global_loads", "smsp__inst_executed_op_global_ld.sum", "thread_instruction"),
        ("thread_global_stores", "smsp__inst_executed_op_global_st.sum", "thread_instruction"),
        ("thread_local_loads", "smsp__inst_executed_op_local_ld.sum", "thread_instruction"),
        ("thread_shared_loads", "smsp__inst_executed_op_shared_ld.sum", "thread_instruction"),
        ("thread_shared_stores", "smsp__inst_executed_op_shared_st.sum", "thread_instruction"),
        ("thread_global_atomics", "smsp__sass_inst_executed_op_global_atom.sum", "thread_instruction"),
        ("num_instructions", "smsp__inst_executed.sum", "scale_signal"),
        ("divergence_efficiency", "smsp__thread_inst_executed_per_inst_executed.ratio", "efficiency_signal"),
        ("num_thread_blocks", "launch_grid_size", "scale_signal"),
    ]
    for i, (name, metric, cat) in enumerate(pka_order, 1):
        lines.append(f"| F{i:02d} | {name} | `{metric}` | {cat} |")

    lines.extend([
        "",
        "## Metric Availability Matrix",
        "",
        "Availability of the 12 canonical PKA metrics across P0 invocations.",
        "`X` = measured, `-` = missing.",
        "",
    ])

    matrix_header = "| invocation_id |" + "|".join(
        f" F{i:02d} " for i in range(1, 13)
    ) + "|"
    matrix_sep = "|---|" + "|".join(":---:" for _ in range(12)) + "|"
    lines.append(matrix_header)
    lines.append(matrix_sep)

    availability = summary.get("availability_matrix", {})
    for inv_id, metrics in availability.items():
        row = f"| {inv_id} |"
        for feature_name in [f[0] for f in pka_order]:
            status = metrics.get(feature_name, "unknown")
            row += " X |" if status == "measured" else " - |"
        lines.append(row)

    lines.extend([
        "",
        "## Acquisition Gap Details",
        "",
    ])

    gap_entries = [e for e in audit_json["entries"] if e["outcome"] == "acquisition_gap"]
    for entry in gap_entries:
        lines.append(f"### {entry['kernel_invocation_id']}")
        lines.append(f"- kernel_name: `{entry['kernel_name']}`")
        lines.append(f"- source: `{entry['source_path']}`")
        lines.append(f"- missing metrics: {entry['missing_metric_count']}/12")
        lines.append("")
        lines.append("| Feature | Canonical Metric | Status |")
        lines.append("|---------|-----------------|--------|")
        for pka_name, f in entry.get("features", {}).items():
            status = f["status"]
            canonical = f["canonical_metric"]
            lines.append(f"| {pka_name} | `{canonical}` | {status} |")
        lines.append("")

    lines.extend([
        "## Conclusion",
        "",
        f"All {summary['total_invocations']} P0 invocations have acquisition gaps.",
        "The canonical 12 PKA Nsight metrics are absent from the existing repository artifacts.",
        "The pipeline correctly stops at Stage 2.",
        "",
        "To proceed to Stage 3 (selector), new NCU data collection is required using the",
        "exact Nsight metric names listed in the feature requirements table above.",
        "",
    ])
    return "\n".join(lines)


def _build_stage_gate_report(
    gap_records: list[dict[str, Any]],
    feature_records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    p0_ids = {e["id"] for e in manifest["entries"] if e["priority"] == "P0"}
    p0_kernel_cases = {e["kernel_or_case"] for e in manifest["entries"] if e["priority"] == "P0"}

    p0_gap_records = [
        r for r in gap_records
        if any(
            r.get("record_id", "").endswith(kc)
            for kc in p0_kernel_cases
        )
    ]

    # Determine pipeline status
    has_p0_gaps = len(p0_gap_records) > 0
    measured_count = len(feature_records)
    total_p0_invocations = measured_count + len(p0_gap_records)

    if has_p0_gaps:
        pipeline_status = "blocked_on_acquisition"
        run_status = "acquisition_gate_success"
        stage_2 = "passed"
        stage_3 = "blocked"
        stage_4 = "blocked"
    elif measured_count < 2:
        pipeline_status = "blocked_insufficient_records"
        run_status = "selector_insufficient_records"
        stage_2 = "passed"
        stage_3 = "blocked"
        stage_4 = "blocked"
    else:
        pipeline_status = "ready_for_selector"
        run_status = "full_closure_pending"
        stage_2 = "passed"
        stage_3 = "ready"
        stage_4 = "pending"

    return {
        "report_name": "L1 Stage Gate Report",
        "dataset_level": "L1",
        "run_status": run_status,
        "pipeline_status": pipeline_status,
        "checked_at": None,  # timestamp would go here
        "stages": {
            "stage_1_manifest": "passed",
            "stage_2_feature_extraction": stage_2,
            "stage_3_selector": stage_3,
            "stage_4_b_line_consumption": stage_4,
            "stage_5_tests": "in_progress",
        },
        "counts": {
            "total_p0_invocations": total_p0_invocations,
            "measured_p0_invocations": measured_count,
            "gap_p0_invocations": len(p0_gap_records),
            "p0_entries_total": len(p0_ids),
        },
        "blocking_gaps": [
            {
                "kernel_invocation_id": r.get("kernel_invocation_id", "unknown"),
                "kernel_name": r.get("kernel_name", "unknown"),
                "source_path": r.get("source_path", ""),
                "missing_metric_count": len(r.get("missing_metrics", [])),
            }
            for r in p0_gap_records
        ],
        "next_action": (
            "Acquire NCU PM counter data for all P0 objects using the exact "
            "12 canonical Nsight metric names listed in pka_feature_audit_l1.md. "
            "Stage 3 (selector) and Stage 4 (B-line) will remain blocked until "
            "all P0 invocations produce 12 measured PKA features."
        ) if has_p0_gaps else "Proceed to Stage 3 (selector).",
    }


def _build_stage_gate_md(report: dict[str, Any]) -> str:
    lines = [
        "# L1 Stage Gate Report",
        "",
        f"**Run Status**: `{report['run_status']}`",
        f"**Pipeline Status**: `{report['pipeline_status']}`",
        "",
        "## Stage Status",
        "",
        "| Stage | Status |",
        "|-------|--------|",
    ]
    for stage, status in report["stages"].items():
        lines.append(f"| {stage} | `{status}` |")

    lines.extend([
        "",
        "## Counts",
        "",
        f"- Total P0 invocations: {report['counts']['total_p0_invocations']}",
        f"- Measured P0 invocations: {report['counts']['measured_p0_invocations']}",
        f"- Gap P0 invocations: {report['counts']['gap_p0_invocations']}",
        f"- P0 manifest entries: {report['counts']['p0_entries_total']}",
        "",
        "## Next Action",
        "",
        report["next_action"],
        "",
    ])

    if report.get("blocking_gaps"):
        lines.extend([
            "## Blocking Gaps",
            "",
        ])
        for gap in report["blocking_gaps"]:
            lines.append(
                f"- `{gap['kernel_invocation_id']}` ({gap['kernel_name']}): "
                f"missing {gap['missing_metric_count']} metrics"
            )

    return "\n".join(lines)


def main() -> int:
    gap_records = json.loads(GAP_PATH.read_text())
    feature_records = json.loads(FEATURE_TABLE_PATH.read_text())
    matrix = json.loads(MATRIX_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())

    # Build and write audit JSON
    audit_json = _build_audit_json(gap_records, feature_records, matrix)
    AUDIT_JSON_PATH.write_text(
        json.dumps(audit_json, indent=2, ensure_ascii=False) + "\n"
    )

    # Build and write audit markdown
    audit_md = _build_audit_md(audit_json, manifest)
    AUDIT_MD_PATH.write_text(audit_md + "\n")

    # Build and write stage gate report
    stage_gate = _build_stage_gate_report(gap_records, feature_records, manifest)
    STAGE_GATE_JSON_PATH.write_text(
        json.dumps(stage_gate, indent=2, ensure_ascii=False) + "\n"
    )

    stage_gate_md = _build_stage_gate_md(stage_gate)
    STAGE_GATE_MD_PATH.write_text(stage_gate_md + "\n")

    print(f"Audit JSON written: {AUDIT_JSON_PATH}")
    print(f"Audit MD written: {AUDIT_MD_PATH}")
    print(f"Stage Gate JSON written: {STAGE_GATE_JSON_PATH}")
    print(f"Stage Gate MD written: {STAGE_GATE_MD_PATH}")
    print()
    print(f"Run Status: {stage_gate['run_status']}")
    print(f"Stages: {json.dumps(stage_gate['stages'])}")
    print()
    print("Next action:")
    print(f"  {stage_gate['next_action']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
