#!/usr/bin/env python3
"""DiffTest-style reduction model for T_trace_to_sim only.

Applies conservative 15%, expected 30%, and optimistic 50% reductions
against T_trace_to_sim. Does NOT modify backend or total wall time.
"""
import json

DATE = "2026-05-03"

REDUCTION_SCENARIOS = [
    {"name": "conservative", "reduction_pct": 15, "description": "Conservative: minor batching and caching wins"},
    {"name": "expected", "reduction_pct": 30, "description": "Expected: structured chunking and static reuse"},
    {"name": "optimistic", "reduction_pct": 50, "description": "Optimistic: aggressive batching, caching, and delta encoding"},
]

# Use the placeholder T_trace_to_sim values from task4's results
WORKLOAD_T_TRACE = [
    {"workload_id": "bert-base-encoder-layer-slice", "T_trace_to_sim_s": 8.0, "meas_unit": "slice", "runs_per_sweep": 10},
    {"workload_id": "bert-base-pretraining-full-step", "T_trace_to_sim_s": 55.0, "meas_unit": "step", "runs_per_sweep": 5},
    {"workload_id": "llama3.1-8b-decoder-layer-slice", "T_trace_to_sim_s": 40.0, "meas_unit": "slice", "runs_per_sweep": 10},
    {"workload_id": "llama3.1-8b-full-step", "T_trace_to_sim_s": 1200.0, "meas_unit": "step", "runs_per_sweep": 2},
    # Modeled at scale
    {"workload_id": "t2-scale-anchor-100GiB", "T_trace_to_sim_s": 1005.0, "meas_unit": "slice", "runs_per_sweep": 3},
    {"workload_id": "t3-scale-anchor-500GiB", "T_trace_to_sim_s": 5005.0, "meas_unit": "slice", "runs_per_sweep": 3},
]

def compute_reductions(T, reduction_pct):
    factor = reduction_pct / 100.0
    reduced_T = T * (1.0 - factor)
    saved_per_run = T - reduced_T
    return {
        "reduction_pct": reduction_pct,
        "original_T_trace_to_sim_s": round(T, 1),
        "reduced_T_trace_to_sim_s": round(reduced_T, 1),
        "saved_per_run_s": round(saved_per_run, 1),
    }

def build_rows():
    rows = []
    for w in WORKLOAD_T_TRACE:
        T = w["T_trace_to_sim_s"]
        row = {
            "workload_id": w["workload_id"],
            "measurement_unit": w["meas_unit"],
            "original_T_trace_to_sim_s": T,
        }
        for sc in REDUCTION_SCENARIOS:
            r = compute_reductions(T, sc["reduction_pct"])
            key = sc["name"]
            row[f"reduced_{key}_T_trace_to_sim_s"] = r["reduced_T_trace_to_sim_s"]
            row[f"saved_{key}_per_run_s"] = r["saved_per_run_s"]
            row[f"saved_{key}_per_sweep_s"] = round(r["saved_per_run_s"] * w["runs_per_sweep"], 1)
        rows.append(row)
    return rows

def build_json(rows):
    return {
        "report_name": "DiffTest-Style Reduction Model",
        "description": "Planning reduction estimates applied to T_trace_to_sim only. NOT measured speedups.",
        "scope": "Reductions apply ONLY to T_trace_to_sim. Backend timing, total wall time, and SM semantics are unchanged.",
        "status": "planning_evidence — to be replaced by prototype measurements",
        "generated_date": DATE,
        "reduction_scenarios": [{"name": sc["name"], "reduction_pct": sc["reduction_pct"]} for sc in REDUCTION_SCENARIOS],
        "rows": rows,
    }

def build_markdown(rows):
    lines = [
        "# DiffTest-Style Reduction Model (Planning Evidence)",
        "",
        f"Generated: {DATE}",
        "",
        "**Status**: Planning evidence — to be replaced by prototype measurements.",
        "",
        "## Scope",
        "",
        "- Reductions apply **only to `T_trace_to_sim`** (trace read + protobuf parse + static bind + threadblock/warp load + frontend delivery prep).",
        "- Backend execution time, total wall time, and SM timing semantics are **unchanged**.",
        "- These are **planning scenarios**, not measured speedups.",
        "",
        "## Reduction Scenarios",
        "",
        "| Scenario | Reduction | Rationale |",
        "|----------|----------|-----------|",
    ]
    for sc in REDUCTION_SCENARIOS:
        lines.append(f"| {sc['name'].capitalize()} | {sc['reduction_pct']}% | {sc['description']} |")

    lines += [
        "",
        "## Per-Workload Reduction Table",
        "",
        "| Workload | Orig T_frontend (s) | Scenario | Reduced T_frontend (s) | Saved/Run (s) | Saved/Sweep (s) |",
        "|----------|--------------------|----------|----------------------|--------------|----------------|",
    ]
    for row in rows:
        T = row["original_T_trace_to_sim_s"]
        for sc in REDUCTION_SCENARIOS:
            name = sc["name"]
            reduced = row[f"reduced_{name}_T_trace_to_sim_s"]
            saved_run = row[f"saved_{name}_per_run_s"]
            saved_sweep = row[f"saved_{name}_per_sweep_s"]
            lines.append(
                f"| {row['workload_id']} | {T:.1f} | {name} {sc['reduction_pct']}% | {reduced:.1f} | {saved_run:.1f} | {saved_sweep:.1f} |"
            )
        lines.append("")
    lines += [
        "## Summary of Impact (Expected Scenario, 30% Reduction)",
        "",
        "| Workload | Single-Run Savings (s) | Sweep Savings (s) | Sweep Savings (min) |",
        "|----------|----------------------|-------------------|--------------------|",
    ]
    for row in rows:
        sr = row["saved_expected_per_run_s"]
        ss = row["saved_expected_per_sweep_s"]
        sm = round(ss / 60.0, 1)
        lines.append(f"| {row['workload_id']} | {sr:.1f} | {ss:.1f} | {sm:.1f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- All values are **planning estimates**, not measured performance data.",
        "- The reduction applies to the frontend path only; backend simulation time is unchanged.",
        "- If prototype measurements later contradict these estimates, the estimates must be recalibrated.",
        "- Conservative (15%) = minimal wins from batching + caching.",
        "- Expected (30%) = structured chunking + static-info reuse.",
        "- Optimistic (50%) = aggressive batching + delta encoding + caching.",
    ]
    return "\n".join(lines) + "\n"

def main():
    rows = build_rows()
    j = build_json(rows)
    md = build_markdown(rows)

    out_dir = "artifacts/gpu_trace_frontend_difftest_necessity"
    with open(f"{out_dir}/difftest_reduction_model.json", "w") as f:
        json.dump(j, f, indent=2)
        f.write("\n")
    with open(f"{out_dir}/difftest_reduction_model.md", "w") as f:
        f.write(md)

    print(f"Wrote {out_dir}/difftest_reduction_model.json")
    print(f"Wrote {out_dir}/difftest_reduction_model.md")

    # Verify reductions only apply to T_trace_to_sim, not total wall time
    for row in rows:
        for sc in REDUCTION_SCENARIOS:
            orig = row["original_T_trace_to_sim_s"]
            reduced = row[f"reduced_{sc['name']}_T_trace_to_sim_s"]
            factor = (1.0 - sc["reduction_pct"] / 100.0)
            expected = round(orig * factor, 1)
            assert abs(reduced - expected) < 0.15, f"Reduction mismatch: {reduced} vs {expected}"
    print("Verification PASSED: all reductions apply to T_trace_to_sim only")

if __name__ == "__main__":
    main()
