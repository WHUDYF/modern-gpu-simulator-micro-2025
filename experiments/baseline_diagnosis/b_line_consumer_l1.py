"""B-line consumption check for L1 (parse-and-validate only).

Validates required fields and forbidden fields for every anchor row.
Compression-side fields are also forbidden.
"""

from __future__ import annotations

import json, sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "a_line" / "l1"

ANCHOR_PATH = ARTIFACT_DIR / "representative_anchor_table_l1.json"
SG_PATH = ARTIFACT_DIR / "l1_stage_gate_report_l1.json"
OUTPUT_PATH = ARTIFACT_DIR / "b_line_consumption_report_l1.md"

REQUIRED = frozenset({"rep_kernel_id", "kernel_name", "cluster_id", "member_invocations",
                       "coverage_count", "coverage_weight", "time_weight"})

FORBIDDEN = frozenset({"family_id", "regime_id", "route_primitive", "execution_template",
                        "execution_template_label", "simulator_lane_id",
                        "cross_tb_offset_coverage", "squash_boundary_crossing_flag",
                        "address_override_density", "full_encoding_fallback_rate",
                        "shared_pc_sequence_length",
                        "dominant_format", "format_counts", "instructions_per_warp_mean",
                        "num_tb_files", "num_warps", "total_threadblocks",
                        "kernel_squash_segment_id", "kernel_squash_boundary_count",
                        "kernel_squash_cohesion", "kernel_squash_behavior_summary",
                        "tb_squash_segment_count", "tb_squash_boundary_count"})

def _type_name(t):
    if isinstance(t, tuple):
        return "|".join(x.__name__ for x in t)
    return t.__name__

TYPE_CHECKS = {
    "rep_kernel_id": str,
    "kernel_name": str,
    "cluster_id": str,
    "member_invocations": list,
    "coverage_count": (int, float),
    "coverage_weight": (int, float),
    "time_weight": (int, float),
}


def _check_gate():
    if not SG_PATH.exists():
        return False, "stage gate report not found"
    sg = json.loads(SG_PATH.read_text())
    s4 = sg.get("stages", {}).get("stage_4_b_line_consumption", "unknown")
    if s4 == "blocked":
        return False, sg.get("next_action", "blocked")
    return True, ""


def _validate(table):
    results = []
    for idx, row in enumerate(table):
        if not isinstance(row, dict):
            return [{"row_index": idx, "rep_kernel_id": "unknown",
                     "required_fields_present": False, "missing_required_fields": sorted(REQUIRED),
                     "forbidden_fields_absent": True, "leaked_forbidden_fields": [],
                     "type_errors": ["row is not a dict" if not isinstance(row, dict) else ""]}]
        missing = sorted(REQUIRED - set(row.keys()))
        # Reject None values for required fields
        for f in REQUIRED:
            if f in row and row[f] is None and f not in missing:
                missing.append(f)
        leaked = sorted(FORBIDDEN & set(row.keys()))
        type_errors = []
        for field, expected_type in TYPE_CHECKS.items():
            val = row.get(field)
            if val is None:
                continue  # already caught as missing
            if not isinstance(val, expected_type):
                type_errors.append(f"{field}: expected {_type_name(expected_type)}, got {type(val).__name__}")
        results.append({
            "row_index": idx,
            "rep_kernel_id": row.get("rep_kernel_id", f"row-{idx}"),
            "required_fields_present": len(missing) == 0,
            "missing_required_fields": missing,
            "forbidden_fields_absent": len(leaked) == 0,
            "leaked_forbidden_fields": leaked,
            "type_errors": type_errors,
        })
    return results


def _report(results):
    total = len(results)
    failures = [r for r in results if not r["required_fields_present"] or not r["forbidden_fields_absent"] or r.get("type_errors")]
    all_pass = len(failures) == 0

    lines = ["# B-Line Consumption Report (L1)", "",
             "Mode: parse-and-validate only (no family/regime/writeback generation).", "",
             "## Per-Row Results", "",
             "| Row | rep_kernel_id | Required OK | Forbidden OK | Types OK | Missing | Leaked | Type Errors |",
             "|-----|---------------|-------------|--------------|----------|---------|--------|-------------|"]
    for r in results:
        type_ok = len(r.get("type_errors", [])) == 0
        lines.append(
            f"| {r['row_index']} | {r['rep_kernel_id']} | "
            f"{'PASS' if r['required_fields_present'] else 'FAIL'} | "
            f"{'PASS' if r['forbidden_fields_absent'] else 'FAIL'} | "
            f"{'PASS' if type_ok else 'FAIL'} | "
            f"{'; '.join(r['missing_required_fields']) if r['missing_required_fields'] else '-'} | "
            f"{'; '.join(r['leaked_forbidden_fields']) if r['leaked_forbidden_fields'] else '-'} | "
            f"{'; '.join(r.get('type_errors', [])) if r.get('type_errors') else '-'} |"
        )

    lines.extend(["", "## Summary", "",
                  f"- Anchor rows consumed: {total}",
                  f"- All rows pass: {'yes' if all_pass else 'no'}",
                  f"- Rows with issues: {len(failures)}", "",
                  f"**Overall Interface Status**: {'ALL_PASS' if all_pass else 'ISSUES_FOUND'}", ""])
    return "\n".join(lines)


def main():
    ok, reason = _check_gate()
    if not ok:
        print(f"B-line consumer blocked: {reason}")
        return 2

    if not ANCHOR_PATH.exists():
        print(f"B-line blocked: anchor table not found at {ANCHOR_PATH}")
        return 3

    try:
        table = json.loads(ANCHOR_PATH.read_text())
    except json.JSONDecodeError as e:
        print(f"B-line failed: invalid JSON: {e}")
        return 4

    if not isinstance(table, list) or len(table) == 0:
        print("B-line failed: anchor table must be a non-empty JSON array")
        return 4

    results = _validate(table)
    report = _report(results)
    OUTPUT_PATH.write_text(report + "\n")

    issue_count = sum(1 for r in results if not r["required_fields_present"] or not r["forbidden_fields_absent"] or r.get("type_errors"))
    print(f"B-line consumption: {len(results)} rows, {issue_count} with issues")
    print(f"Report: {OUTPUT_PATH}")
    return 0 if issue_count == 0 else 5


if __name__ == "__main__":
    sys.exit(main())
