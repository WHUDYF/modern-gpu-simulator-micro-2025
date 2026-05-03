#!/usr/bin/env python3
"""Central evidence table generator.

Loads measured and modeled inputs from the artifact pipeline
and merges them into a single evidence table with provenance labels.
"""
import json
import os

DATE = "2026-05-03"
OUT_DIR = "artifacts/gpu_trace_frontend_difftest_necessity"

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_workload_catalog():
    """Load the workload evidence table to get the claim-bearing workload list."""
    data = load_json(f"{OUT_DIR}/workload_evidence_table.json")
    if not data:
        return []
    return data.get("workloads", [])

def load_burden_ratios():
    """Load per-workload burden ratio data."""
    data = load_json(f"{OUT_DIR}/complete_flow_burden_ratio.json")
    if not data:
        return {}
    results = data.get("results", [])
    return {r["workload_id"]: r for r in results}

def load_reduction_estimates():
    """Load DiffTest-style reduction estimates keyed by workload_id."""
    data = load_json(f"{OUT_DIR}/difftest_reduction_model.json")
    if not data:
        return {}
    rows = data.get("rows", [])
    return {r["workload_id"]: r for r in rows}

def load_formula_rows():
    """Load trace-size formula planning rows."""
    data = load_json(f"{OUT_DIR}/trace_to_sim_formula.json")
    if not data:
        return {}
    return {r["trace_label"]: r for r in data.get("rows", [])}

def load_controls():
    """Load measured control workloads from the existing bottleneck map."""
    data = load_json("artifacts/trace_bottleneck_map/benchmark_cost_map.json")
    if not data:
        return []
    controls = []
    for rec in data.get("records", []):
        if rec.get("status") == "measured":
            controls.append(rec)
    return controls

def build_evidence_rows():
    """Merge data from workload catalog, burden ratios, and reduction model."""
    workloads = load_workload_catalog()
    burdens = load_burden_ratios()
    reductions = load_reduction_estimates()

    rows = []
    for w in workloads:
        wid = w["workload_id"]
        row = {
            "workload": w.get("model", wid),
            "slice_type": w.get("slice_type", ""),
            "measurement_unit": w.get("measurement_unit", ""),
            "role": w.get("role", ""),
            "trace_size_GiB": {"value": None, "label": "pending"},
            "kernel_count": {"value": None, "label": "pending"},
            "threadblock_or_warp_count": {"value": None, "label": "pending"},
            "T_trace_to_sim_s": {"value": None, "label": "pending"},
            "T_kernel_to_sim_done_s": {"value": None, "label": "pending"},
            "P_trace_to_sim_pct": {"value": None, "label": "pending"},
            "est_frontend_reduction_s": {"value": None, "label": "pending"},
            "reduced_T_trace_to_sim_s": {"value": None, "label": "pending"},
            "complete_flow_impact": "Pending — no measured data",
            "trace_export_included": True,
            "result_analysis_included": True,
        }

        # Merge burden ratio data if available
        if wid in burdens:
            b = burdens[wid]
            c = b.get("components", {})
            dl = b.get("data_labels", {})
            row["T_trace_to_sim_s"] = {"value": c.get("T_trace_to_sim_s", 0), "label": dl.get("T_trace_to_sim", "pending")}
            row["T_kernel_to_sim_done_s"] = {"value": b.get("T_kernel_to_sim_done_s", 0), "label": dl.get("T_trace_to_sim", "pending")}
            row["P_trace_to_sim_pct"] = {"value": b.get("P_trace_to_sim_pct", 0), "label": dl.get("T_trace_to_sim", "pending")}

        # Merge reduction estimates if available
        if wid in reductions:
            red = reductions[wid]
            expected = red.get("saved_expected_per_run_s", 0)
            reduced_t = red.get("reduced_expected_T_trace_to_sim_s", 0)
            orig_t = red.get("original_T_trace_to_sim_s", 0)
            row["est_frontend_reduction_s"] = {"value": expected, "label": "modeled"}
            row["reduced_T_trace_to_sim_s"] = {"value": reduced_t, "label": "modeled"}
            if expected > 0:
                row["complete_flow_impact"] = f"Expected 30% reduction saves {expected:.1f}s per run"

        # Merge formula-based trace size if available (for scale anchors)
        formula = load_formula_rows()
        for label_key in formula:
            if wid in label_key or w.get("model", "") in label_key:
                row["trace_size_GiB"] = {"value": formula[label_key]["trace_size_GiB"], "label": "modeled"}

        rows.append(row)
    return rows

def evaluate_go_no_go(rows):
    """Go/no-go gate: only evaluate when measured data exists."""
    has_measured = any(
        r["T_trace_to_sim_s"]["label"] == "measured"
        for r in rows
    )
    if not has_measured:
        return {
            "verdict": "PENDING_MEASUREMENT",
            "rule": "P_trace_to_sim_slice > 15% OR P_trace_to_sim_step > 15%",
            "detail": "All inputs are placeholder or modeled. Run simulator instrumentation to obtain measured data."
        }

    slice_rows = [r for r in rows if r["measurement_unit"] == "slice"]
    step_rows = [r for r in rows if r["measurement_unit"] == "step"]
    slice_max = max((r["P_trace_to_sim_pct"]["value"] for r in slice_rows if r["P_trace_to_sim_pct"]["value"] is not None), default=0)
    step_max = max((r["P_trace_to_sim_pct"]["value"] for r in step_rows if r["P_trace_to_sim_pct"]["value"] is not None), default=0)
    go = slice_max > 15.0 or step_max > 15.0
    return {
        "verdict": "GO" if go else "NOT_YET",
        "rule": "P_trace_to_sim_slice > 15% OR P_trace_to_sim_step > 15%",
        "slice_max_pct": slice_max,
        "step_max_pct": step_max,
    }

def build_json(rows, go_no_go, controls):
    return {
        "report_name": "GPU Trace Frontend Necessity: Central Evidence Table",
        "generated_date": DATE,
        "go_no_go": go_no_go,
        "evidence_rows": rows,
        "control_workloads": [
            {
                "suite": c.get("suite", ""),
                "representative_case": c.get("representative_case", ""),
                "trace_size_mib": c.get("trace_size_mib"),
            }
            for c in controls
        ],
        "provenance": "Evidence rows are merged from workload_evidence_table.json, complete_flow_burden_ratio.json, and difftest_reduction_model.json. All data labels reflect the current measurement status.",
    }

def build_markdown(rows, go_no_go, controls):
    lines = [
        "# GPU Trace Frontend Necessity: Central Evidence Table",
        "",
        f"Generated: {DATE}",
        "",
        "## Go/No-Go",
        "",
        f"**Verdict**: {go_no_go['verdict']}",
        f"- Rule: {go_no_go['rule']}",
    ]
    if "slice_max_pct" in go_no_go:
        lines.append(f"- Slice max P_trace_to_sim: {go_no_go['slice_max_pct']:.1f}%")
        lines.append(f"- Step max P_trace_to_sim: {go_no_go['step_max_pct']:.1f}%")
    if "detail" in go_no_go:
        lines.append(f"- Detail: {go_no_go['detail']}")

    lines += [
        "",
        "## Evidence Rows",
        "",
        "| Workload | Unit | Role | Trace Size (GiB) | T_frontend (s) | T_total (s) | P_frontend (%) | Est. Reduction (s) | Impact |",
        "|----------|------|------|-----------------|---------------|------------|---------------|-------------------|--------|",
    ]
    for r in rows:
        ts = r["trace_size_GiB"]
        tfs = r["T_trace_to_sim_s"]
        ttot = r["T_kernel_to_sim_done_s"]
        pct = r["P_trace_to_sim_pct"]
        red = r["est_frontend_reduction_s"]
        lines.append(
            f"| {r['workload']} ({r['slice_type']}) | {r['measurement_unit']} | {r['role']} | "
            f"{ts['value']} ({ts['label']}) | {tfs['value']} ({tfs['label']}) | "
            f"{ttot['value']} ({ttot['label']}) | {pct['value']} ({pct['label']}) | "
            f"{red['value']} ({red['label']}) | {r['complete_flow_impact']} |"
        )

    if controls:
        lines += [
            "",
            "## Control Workloads (Measured, from Existing Bottleneck Map)",
            "",
            "| Suite | Representative Case | Trace Size (MiB) | Export (s) | Sim (s) |",
            "|-------|-------------------|-----------------|-----------|--------|",
        ]
        for c in controls[:10]:
            lines.append(
                f"| {c.get('suite', '')} | {c.get('representative_case', '')} | "
                f"{c.get('trace_size_mib', 'N/A')} | {c.get('export_time_s', 'N/A')} | "
                f"{c.get('sim_time_s', 'N/A')} |"
            )

    lines += [
        "",
        "## Data Provenance",
        "",
        "All evidence rows are merged from:",
        "- `workload_evidence_table.json` — workload definitions",
        "- `complete_flow_burden_ratio.json` — timing and burden ratios",
        "- `difftest_reduction_model.json` — reduction estimates",
        "- `trace_to_sim_formula.json` — formula-based estimates for scale anchors",
        "",
        "Labels reflect whether data is `measured` (from simulator instrumentation), `modeled` (from planning formula), or `pending` (not yet available).",
    ]
    return "\n".join(lines) + "\n"

def main():
    rows = build_evidence_rows()
    go_no_go = evaluate_go_no_go(rows)
    controls = load_controls()
    j = build_json(rows, go_no_go, controls)
    md = build_markdown(rows, go_no_go, controls)

    with open(f"{OUT_DIR}/paper_argument_matrix.json", "w") as f:
        json.dump(j, f, indent=2)
        f.write("\n")
    with open(f"{OUT_DIR}/paper_argument_matrix.md", "w") as f:
        f.write(md)

    print(f"Wrote {OUT_DIR}/paper_argument_matrix.json")
    print(f"Wrote {OUT_DIR}/paper_argument_matrix.md")
    print(f"Go/No-Go: {go_no_go['verdict']}")

    # Verify data-driven: no hardcoded rows
    assert len(rows) > 0, "No evidence rows loaded from artifacts"
    print("Verification PASSED: evidence rows loaded from artifact files")

if __name__ == "__main__":
    main()
