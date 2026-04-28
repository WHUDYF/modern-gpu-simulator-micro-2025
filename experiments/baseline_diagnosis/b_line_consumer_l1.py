"""B-line consumption check for L1 (parse-and-validate only).

Reads the representative anchor table and validates:
- Required fields present: rep_kernel_id, kernel_name, cluster_id,
  member_invocations, coverage_count, coverage_weight, time_weight
- Forbidden fields absent: family_id, regime_id, route_primitive,
  execution_template, simulator_lane_id

Does NOT generate family/regime/writeback lineage.
Does NOT depend on artifacts/middle_layer/mini_transformer_v4/bundle.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "a_line" / "l1"

ANCHOR_TABLE_PATH = ARTIFACT_DIR / "representative_anchor_table_l1.json"
STAGE_GATE_PATH = ARTIFACT_DIR / "l1_stage_gate_report_l1.json"
OUTPUT_PATH = ARTIFACT_DIR / "b_line_consumption_report_l1.md"

REQUIRED_FIELDS = frozenset({
    "rep_kernel_id", "kernel_name", "cluster_id", "member_invocations",
    "coverage_count", "coverage_weight", "time_weight",
})

FORBIDDEN_FIELDS = frozenset({
    "family_id", "regime_id", "route_primitive", "execution_template",
    "execution_template_label", "simulator_lane_id",
})


def _check_stage_gate() -> tuple[bool, str]:
    if not STAGE_GATE_PATH.exists():
        return False, "stage gate report not found"
    report = json.loads(STAGE_GATE_PATH.read_text())
    stage_4 = report.get("stages", {}).get("stage_4_b_line_consumption", "unknown")
    if stage_4 == "blocked":
        return False, report.get("next_action", "blocked by stage gate")
    return True, ""


def _validate_anchor_table(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for idx, row in enumerate(table):
        missing = REQUIRED_FIELDS - set(row.keys())
        leaked = FORBIDDEN_FIELDS & set(row.keys())
        results.append({
            "row_index": idx,
            "rep_kernel_id": row.get("rep_kernel_id", f"row-{idx}"),
            "required_fields_present": len(missing) == 0,
            "missing_required_fields": sorted(missing),
            "forbidden_fields_absent": len(leaked) == 0,
            "leaked_forbidden_fields": sorted(leaked),
        })
    return results


def _build_report(validation_results: list[dict[str, Any]]) -> str:
    total = len(validation_results)
    all_pass = all(r["required_fields_present"] and r["forbidden_fields_absent"] for r in validation_results)
    failures = [r for r in validation_results if not r["required_fields_present"] or not r["forbidden_fields_absent"]]

    lines = [
        "# B-Line Consumption Report (L1)",
        "",
        "Consumption mode: parse-and-validate only (no family/regime/writeback generation).",
        "",
        f"## Summary",
        "",
        f"- Anchor rows consumed: {total}",
        f"- All rows pass: {'yes' if all_pass else 'no'}",
        f"- Rows with issues: {len(failures)}",
        "",
    ]

    if failures:
        lines.append("## Schema Validation Failures")
        lines.append("")
        for r in failures:
            lines.append(f"### Row {r['row_index']}: `{r['rep_kernel_id']}`")
            if r["missing_required_fields"]:
                lines.append(f"- Missing required fields: {', '.join(r['missing_required_fields'])}")
            if r["leaked_forbidden_fields"]:
                lines.append(f"- Leaked forbidden fields: {', '.join(r['leaked_forbidden_fields'])}")
            lines.append("")

    lines.extend([
        "## Interface Status",
        "",
        f"Overall: {'ALL_PASS' if all_pass else 'ISSUES_FOUND'}",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    ok, reason = _check_stage_gate()
    if not ok:
        print(f"B-line consumer blocked by stage gate: {reason}")
        return 2

    if not ANCHOR_TABLE_PATH.exists():
        print(f"B-line consumer blocked: anchor table not found at {ANCHOR_TABLE_PATH}")
        return 3

    try:
        anchor_table = json.loads(ANCHOR_TABLE_PATH.read_text())
    except json.JSONDecodeError as e:
        print(f"B-line consumer failed: anchor table is not valid JSON: {e}")
        return 4

    if not isinstance(anchor_table, list):
        print("B-line consumer failed: anchor table must be a JSON array")
        return 4

    validation_results = _validate_anchor_table(anchor_table)
    report = _build_report(validation_results)
    OUTPUT_PATH.write_text(report + "\n")

    total = len(validation_results)
    failures = sum(1 for r in validation_results if not r["required_fields_present"] or not r["forbidden_fields_absent"])
    print(f"B-line consumption complete: {total} rows, {failures} with issues")
    print(f"Report: {OUTPUT_PATH}")
    return 0 if failures == 0 else 5


if __name__ == "__main__":
    sys.exit(main())
