#!/usr/bin/env python3
"""Generate measured-data-only go/no-go verdict."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

OUT_DIR = Path("artifacts/gpu_trace_frontend_difftest_necessity")
THRESHOLD_PCT = 15.0


def load_burden():
    with (OUT_DIR / "complete_flow_burden_ratio.json").open() as f:
        return json.load(f)


def measured_rows(burden):
    return [
        row for row in burden.get("results", [])
        if row.get("claim_bearing") and row.get("data_label") == "measured"
    ]


def build_verdict(burden):
    rows = measured_rows(burden)
    generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    if not rows:
        verdict = "INCONCLUSIVE: measured evidence incomplete"
        status = "Gate D — INCONCLUSIVE: no fully measured claim-bearing row is available."
        basis = {
            "threshold_pct": THRESHOLD_PCT,
            "eligible_measured_rows": 0,
            "rule": burden["go_no_go"]["rule"],
        }
        measured_values = {}
        workload_id = None
        data_label = "placeholder"
        claim_bearing = True
        measurement_unit = None
    else:
        best = max(rows, key=lambda row: row["P_trace_to_sim_pct"])
        go = any(row["P_trace_to_sim_pct"] > THRESHOLD_PCT for row in rows)
        verdict = (
            "GO: frontend prototype investigation justified"
            if go else
            "NO-GO: frontend prototype not justified by current measured evidence"
        )
        status = (
            "Gate D — GO: at least one fully measured claim-bearing row exceeds the threshold."
            if go else
            "Gate D — NO-GO: fully measured claim-bearing rows do not exceed the threshold."
        )
        basis = {
            "threshold_pct": THRESHOLD_PCT,
            "threshold_rule": burden["go_no_go"]["rule"],
            "eligible_measured_rows": len(rows),
            "measured_workload_ids": [row["workload_id"] for row in rows],
            "max_measured_P_trace_to_sim_pct": best["P_trace_to_sim_pct"],
            "satisfied_by": (
                f"{best['workload_id']} P_trace_to_sim={best['P_trace_to_sim_pct']}%"
            ),
        }
        measured_values = {
            "T_trace_to_sim_s": best["T_trace_to_sim_s"],
            "T_kernel_to_sim_done_s": best["T_kernel_to_sim_done_s"],
            "P_trace_to_sim_pct": best["P_trace_to_sim_pct"],
            "components": best["components"],
        }
        workload_id = best["workload_id"]
        data_label = best["data_label"]
        claim_bearing = best["claim_bearing"]
        measurement_unit = best["measurement_unit"]
        source_artifact = best.get("source_artifact", "complete_flow_measurements.json")
        provenance = (
            f"Verdict generated from selected measured row `{workload_id}`. "
            f"Measured source artifact: {source_artifact}. "
            f"Row provenance: {best.get('provenance', 'not recorded')}"
        )

    return {
        "report_name": "GPU Trace Frontend Necessity Go/No-Go Verdict",
        "workload_id": workload_id,
        "data_label": data_label,
        "claim_bearing": claim_bearing,
        "measurement_unit": measurement_unit,
        "source_artifact": source_artifact if rows else "complete_flow_measurements.json",
        "provenance": provenance if rows else "No fully measured claim-bearing row found in complete_flow_burden_ratio.json.",
        "generated_at": generated_at,
        "verdict": verdict,
        "basis": basis,
        "measured_values": measured_values,
        "caveats": [
            "Modeled, placeholder, and control rows are excluded from the go/no-go calculation.",
            "The 15% threshold is an early-stage engineering gate, not a final paper claim threshold.",
        ],
        "status": status,
    }


def build_markdown(verdict):
    basis = verdict["basis"]
    values = verdict["measured_values"]
    lines = [
        "# GPU Trace Frontend Necessity Go/No-Go Verdict",
        "",
        f"**Generated**: {verdict['generated_at']}",
        f"**Status**: {verdict['status']}",
        "",
        f"## Verdict: {verdict['verdict']}",
        "",
        "## Basis",
        "",
        f"- Rule: {basis.get('threshold_rule', basis.get('rule'))}",
        f"- Threshold: {basis['threshold_pct']}%",
        f"- Eligible measured claim-bearing rows: {basis['eligible_measured_rows']}",
        f"- Source artifact: {verdict['source_artifact']}",
    ]
    if values:
        lines += [
            f"- Max measured P_trace_to_sim: {basis['max_measured_P_trace_to_sim_pct']}%",
            f"- Workload: {verdict['workload_id']}",
            "",
            "## Complete-Flow Values",
            "",
            "| Component | Time (s) |",
            "|-----------|---------:|",
        ]
        components = values["components"]
        lines += [
            f"| Trace export | {components['T_kernel_or_trace_export_s']:.2f} |",
            f"| Trace-to-simulator frontend | {components['T_trace_to_sim_s']:.2f} |",
            f"| Simulator backend | {components['T_sim_backend_execution_s']:.2f} |",
            f"| Result analysis | {components['T_result_analysis_s']:.2f} |",
            f"| Total | {values['T_kernel_to_sim_done_s']:.2f} |",
        ]
    lines += [
        "",
        "## Caveats",
        "",
    ]
    lines += [f"- {caveat}" for caveat in verdict["caveats"]]
    return "\n".join(lines) + "\n"


def main():
    burden = load_burden()
    verdict = build_verdict(burden)
    with (OUT_DIR / "go_no_go_verdict.json").open("w") as f:
        json.dump(verdict, f, indent=2)
        f.write("\n")
    with (OUT_DIR / "go_no_go_verdict.md").open("w") as f:
        f.write(build_markdown(verdict))
    print(f"Wrote {OUT_DIR}/go_no_go_verdict.json")
    print(f"Wrote {OUT_DIR}/go_no_go_verdict.md")
    print(f"Verdict: {verdict['verdict']}")


if __name__ == "__main__":
    main()
