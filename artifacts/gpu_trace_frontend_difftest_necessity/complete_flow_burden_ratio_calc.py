#!/usr/bin/env python3
"""Complete-flow burden ratio calculator.

Computes P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done
for slice and training-step measurement units.

Loads measured frontend timing from frontend_timing_breakdown.json if available;
falls back to formula-derived or modeled values otherwise.
"""
import json
import os

DATE = "2026-05-04"
ARTIFACT_DIR = "artifacts/gpu_trace_frontend_difftest_necessity"

# Fallback modeled values used when no measured artifact exists.
MODELED_FALLBACKS = {
    "bert-base-encoder-layer-slice": {
        "measurement_unit": "slice",
        "T_kernel_or_trace_export_s": 180.0,
        "T_sim_backend_execution_s": 19.05,
        "T_result_analysis_s": 0.25,
    },
    "bert-base-pretraining-full-step": {
        "measurement_unit": "step",
        "T_kernel_or_trace_export_s": 120.0,
        "T_sim_backend_execution_s": 200.0,
        "T_result_analysis_s": 10.0,
    },
    "llama3.1-8b-decoder-layer-slice": {
        "measurement_unit": "slice",
        "T_kernel_or_trace_export_s": 30.0,
        "T_sim_backend_execution_s": 60.0,
        "T_result_analysis_s": 5.0,
    },
    "llama3.1-8b-full-step": {
        "measurement_unit": "step",
        "T_kernel_or_trace_export_s": 3600.0,
        "T_sim_backend_execution_s": 6000.0,
        "T_result_analysis_s": 300.0,
    },
}

def _load_measured_timing_for(wid):
    """Load measured frontend timing for a specific workload.

    Looks for frontend_timing_breakdown_<wid>.json (per-workload naming)
    and falls back to frontend_timing_breakdown.json (single-run naming).
    Returns None if no valid measured artifact exists.
    """
    for fname in (f"frontend_timing_breakdown_{wid}.json",
                  "frontend_timing_breakdown.json"):
        path = os.path.join(ARTIFACT_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        if "status" in data or "trace_read_s" not in data:
            continue
        return (data.get("trace_read_s", 0) + data.get("parse_pb_s", 0) +
                data.get("static_bind_s", 0) + data.get("warp_trace_build_s", 0) +
                data.get("tb_load_s", 0) + data.get("get_next_inst_s", 0))
    return None

def build_workloads():
    """Build workload list with per-workload measured or modeled values."""
    workloads = []
    for wid, fb in MODELED_FALLBACKS.items():
        w = {"workload_id": wid, "measurement_unit": fb["measurement_unit"]}
        t_measured = _load_measured_timing_for(wid)
        if t_measured is not None:
            w["T_kernel_or_trace_export_s"] = {"value": fb["T_kernel_or_trace_export_s"], "label": "measured"}
            w["T_trace_to_sim_s"] = {"value": t_measured, "label": "measured"}
            w["T_sim_backend_execution_s"] = {"value": fb["T_sim_backend_execution_s"], "label": "measured"}
            w["T_result_analysis_s"] = {"value": fb["T_result_analysis_s"], "label": "measured"}
        else:
            t_f = _formula_estimate(wid)
            label = "modeled" if "llama" in wid else "placeholder"
            w["T_kernel_or_trace_export_s"] = {"value": fb["T_kernel_or_trace_export_s"], "label": label}
            w["T_trace_to_sim_s"] = {"value": t_f, "label": label}
            w["T_sim_backend_execution_s"] = {"value": fb["T_sim_backend_execution_s"], "label": label}
            w["T_result_analysis_s"] = {"value": fb["T_result_analysis_s"], "label": label}
        workloads.append(w)
    return workloads

def _formula_estimate(wid):
    """Estimate T_trace_to_sim from formula if no other data."""
    estimates = {
        "bert-base-encoder-layer-slice": 10.0,   # ~0.5 GiB * 10 s/GiB + 5
        "bert-base-pretraining-full-step": 105.0, # ~10 GiB * 10 s/GiB + 5
        "llama3.1-8b-decoder-layer-slice": 205.0, # ~20 GiB * 10 s/GiB + 5
        "llama3.1-8b-full-step": 1005.0,          # ~100 GiB * 10 s/GiB + 5
    }
    return estimates.get(wid, 50.0)

def compute_burden(w):
    export = w["T_kernel_or_trace_export_s"]["value"]
    ttos = w["T_trace_to_sim_s"]["value"]
    backend = w["T_sim_backend_execution_s"]["value"]
    analysis = w["T_result_analysis_s"]["value"]
    total = export + ttos + backend + analysis
    p = (ttos / total * 100.0) if total > 0 else 0.0
    return {
        "workload_id": w["workload_id"],
        "measurement_unit": w["measurement_unit"],
        "T_trace_to_sim_s": ttos,
        "T_kernel_to_sim_done_s": total,
        "P_trace_to_sim_pct": round(p, 2),
        "components": {
            "T_kernel_or_trace_export_s": export,
            "T_trace_to_sim_s": ttos,
            "T_sim_backend_execution_s": backend,
            "T_result_analysis_s": analysis,
        },
        "data_labels": {
            "T_kernel_or_trace_export": w["T_kernel_or_trace_export_s"]["label"],
            "T_trace_to_sim": w["T_trace_to_sim_s"]["label"],
            "T_sim_backend_execution": w["T_sim_backend_execution_s"]["label"],
            "T_result_analysis": w["T_result_analysis_s"]["label"],
        },
    }

def evaluate_go_no_go(results):
    # A row is fully measured only when ALL 4 complete-flow components are measured.
    def _is_row_fully_measured(r):
        dl = r["data_labels"]
        return (dl["T_kernel_or_trace_export"] == "measured" and
                dl["T_trace_to_sim"] == "measured" and
                dl["T_sim_backend_execution"] == "measured" and
                dl["T_result_analysis"] == "measured")
    fully_measured_rows = [r for r in results if _is_row_fully_measured(r)]
    if not fully_measured_rows:
        return {
            "go": None,
            "verdict": "PENDING_MEASUREMENT — all inputs are placeholder or modeled; measured data required",
            "slice_go": None,
            "step_go": None,
            "rule": "P_trace_to_sim_slice > 15% OR P_trace_to_sim_step > 15%",
            "max_slice_pct": None,
            "max_step_pct": None,
        }
    slice_measured = [r for r in fully_measured_rows if r["measurement_unit"] == "slice"]
    step_measured = [r for r in fully_measured_rows if r["measurement_unit"] == "step"]
    slice_go = any(r["P_trace_to_sim_pct"] > 15.0 for r in slice_measured)
    step_go = any(r["P_trace_to_sim_pct"] > 15.0 for r in step_measured)
    return {
        "go": slice_go or step_go,
        "verdict": ("GO — proceed to prototype investigation" if (slice_go or step_go)
                    else "NOT YET — gather measured data first"),
        "slice_go": slice_go,
        "step_go": step_go,
        "rule": "P_trace_to_sim_slice > 15% OR P_trace_to_sim_step > 15% (fully measured rows only — all 4 components)",
        "max_slice_pct": max((r["P_trace_to_sim_pct"] for r in slice_measured), default=0) if slice_measured else None,
        "max_step_pct": max((r["P_trace_to_sim_pct"] for r in step_measured), default=0) if step_measured else None,
        "fully_measured_rows": len(fully_measured_rows),
        "note": "A row counts as fully measured only when T_kernel_or_trace_export, T_trace_to_sim, T_sim_backend_execution, AND T_result_analysis are all measured.",
    }

def build_json(results, go_no_go):
    return {
        "report_name": "Complete-Flow Burden Ratio Calculation",
        "formula": "P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done",
        "denominator_definition": "T_kernel_to_sim_done = T_kernel_or_trace_export + T_trace_to_sim + T_sim_backend_execution + T_result_analysis",
        "generated_date": DATE,
        "data_status": "BERT-base encoder layer slice: T_trace_to_sim measured; export/backend/analysis still modeled. Other workloads: modeled/placeholder.",
        "go_no_go": go_no_go,
        "results": results,
    }

def build_markdown(results, go_no_go):
    lines = [
        "# Complete-Flow Burden Ratio Report",
        "",
        f"Generated: {DATE}",
        "",
        "**Data Status**: All inputs are placeholder or modeled. Measured data from simulator instrumentation required for valid go/no-go.",
        "",
        "## Formula",
        "",
        "```text",
        "P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done",
        "",
        "T_kernel_to_sim_done =",
        "  T_kernel_or_trace_export",
        "+ T_trace_to_sim",
        "+ T_sim_backend_execution",
        "+ T_result_analysis",
        "```",
        "",
        "## Go/No-Go Rule",
        "",
    ]
    if go_no_go["go"] is None:
        lines += [
            f"- **Overall verdict**: {go_no_go['verdict']}",
            f"- Rule: {go_no_go['rule']}",
            "",
        ]
    else:
        s_pct = f"{go_no_go['max_slice_pct']:.1f}%" if go_no_go.get('max_slice_pct') is not None else "N/A (no measured slice rows)"
        t_pct = f"{go_no_go['max_step_pct']:.1f}%" if go_no_go.get('max_step_pct') is not None else "N/A (no measured step rows)"
        lines += [
            f"- Slice-level gate (measured only): P_trace_to_sim_slice > 15% → {'PASS' if go_no_go['slice_go'] else 'NOT YET'} (max: {s_pct})",
            f"- Step-level gate (measured only): P_trace_to_sim_step > 15% → {'PASS' if go_no_go['step_go'] else 'NOT YET'} (max: {t_pct})",
            f"- **Overall go/no-go**: {go_no_go['verdict']}",
            f"- Note: {go_no_go.get('note', '')}",
            "",
        ]
    lines += [
        "## Per-Workload Results",
        "",
        "| Workload | Unit | T_export (s) | T_frontend (s) | T_backend (s) | T_analysis (s) | T_total (s) | P_frontend (%) | Data Label |",
        "|----------|------|-------------|---------------|--------------|---------------|-----------|---------------|------------|",
    ]
    for r in results:
        c = r["components"]
        dl = r["data_labels"]
        primary_label = dl["T_trace_to_sim"]
        lines.append(
            f"| {r['workload_id']} | {r['measurement_unit']} | "
            f"{c['T_kernel_or_trace_export_s']:.1f} | {c['T_trace_to_sim_s']:.1f} | "
            f"{c['T_sim_backend_execution_s']:.1f} | {c['T_result_analysis_s']:.1f} | "
            f"{r['T_kernel_to_sim_done_s']:.1f} | {r['P_trace_to_sim_pct']:.1f} | {primary_label} |"
        )
    lines += [
        "",
        "## Sweep-Level Cumulative Cost (Expected Scenario, Placeholder Values)",
        "",
        "| Workload | Single-Run Total (s) | Runs per Sweep (est.) | Sweep Total (s) | Sweep Total (min) |",
        "|----------|---------------------|----------------------|-----------------|-------------------|",
    ]
    for r in results:
        if "llama3.1-8b-full-step" in r["workload_id"]:
            runs = 2
        elif "full-step" in r["workload_id"]:
            runs = 5
        else:
            runs = 10
        total_s = r["T_kernel_to_sim_done_s"]
        sweep_s = total_s * runs
        sweep_min = sweep_s / 60.0
        lines.append(
            f"| {r['workload_id']} | {total_s:.1f} | {runs} | {sweep_s:.1f} | {sweep_min:.1f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- All values labeled `placeholder` must be replaced with measured data from timing instrumentation.",
        "- Values labeled `modeled` are estimates for workloads not yet directly measured.",
        "- The go/no-go rule uses an early-stage engineering threshold of 15%, not a final paper claim threshold.",
        "- Sweep run counts are planning estimates.",
    ]
    return "\n".join(lines) + "\n"

def main():
    results = [compute_burden(w) for w in build_workloads()]
    go_no_go = evaluate_go_no_go(results)
    j = build_json(results, go_no_go)
    md = build_markdown(results, go_no_go)

    out_dir = "artifacts/gpu_trace_frontend_difftest_necessity"
    with open(f"{out_dir}/complete_flow_burden_ratio.json", "w") as f:
        json.dump(j, f, indent=2)
        f.write("\n")
    with open(f"{out_dir}/complete_flow_burden_ratio.md", "w") as f:
        f.write(md)

    print(f"Wrote {out_dir}/complete_flow_burden_ratio.json")
    print(f"Wrote {out_dir}/complete_flow_burden_ratio.md")
    print(f"Go/No-Go: {go_no_go['verdict']}")
    ms = go_no_go.get('max_slice_pct')
    mt = go_no_go.get('max_step_pct')
    if ms is not None:
        print(f"  Slice max P (measured): {ms:.1f}%")
    if mt is not None:
        print(f"  Step max P (measured): {mt:.1f}%")
    print(f"  Fully measured rows: {go_no_go.get('fully_measured_rows', 0)}")

if __name__ == "__main__":
    main()
