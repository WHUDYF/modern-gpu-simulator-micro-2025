#!/usr/bin/env python3
"""Central evidence table generator.

Merges workload catalog, timing breakdown, formula estimates, burden ratios,
redundancy metrics, and reduction estimates into a single evidence table.
Distinguishes measured from modeled values.
"""
import json
import os

DATE = "2026-05-03"
OUT_DIR = "artifacts/gpu_trace_frontend_difftest_necessity"

EVIDENCE_ROWS = [
    {
        "workload": "BERT-base encoder layer slice",
        "measurement_unit": "slice",
        "model_type": "encoder layer",
        "trace_size_GiB": {"value": 0.5, "label": "placeholder"},
        "kernel_count": {"value": None, "label": "pending"},
        "threadblock_or_warp_count": {"value": None, "label": "pending"},
        "T_trace_to_sim_s": {"value": 8.0, "label": "placeholder"},
        "T_kernel_to_sim_done_s": {"value": 29.0, "label": "placeholder"},
        "P_trace_to_sim_pct": {"value": 27.6, "label": "derived_placeholder"},
        "est_frontend_reduction_s": {"value": 2.4, "label": "modeled"},
        "reduced_T_trace_to_sim_s": {"value": 5.6, "label": "modeled"},
        "complete_flow_impact": "Moderate: saves ~2.4s per run, ~24s per 10-run sweep at expected 30% reduction",
        "trace_export_included": True,
        "result_analysis_included": True,
    },
    {
        "workload": "BERT-base pretraining full step",
        "measurement_unit": "step",
        "model_type": "pretraining full step",
        "trace_size_GiB": {"value": 10.0, "label": "placeholder"},
        "kernel_count": {"value": None, "label": "pending"},
        "threadblock_or_warp_count": {"value": None, "label": "pending"},
        "T_trace_to_sim_s": {"value": 55.0, "label": "placeholder"},
        "T_kernel_to_sim_done_s": {"value": 385.0, "label": "placeholder"},
        "P_trace_to_sim_pct": {"value": 14.3, "label": "derived_placeholder"},
        "est_frontend_reduction_s": {"value": 16.5, "label": "modeled"},
        "reduced_T_trace_to_sim_s": {"value": 38.5, "label": "modeled"},
        "complete_flow_impact": "Significant: saves ~16.5s per run, ~82.5s per 5-run sweep at expected 30% reduction",
        "trace_export_included": True,
        "result_analysis_included": True,
    },
    {
        "workload": "Llama 3.1 8B decoder layer slice",
        "measurement_unit": "slice",
        "model_type": "decoder layer",
        "trace_size_GiB": {"value": 20.0, "label": "modeled"},
        "kernel_count": {"value": None, "label": "pending"},
        "threadblock_or_warp_count": {"value": None, "label": "pending"},
        "T_trace_to_sim_s": {"value": 40.0, "label": "modeled"},
        "T_kernel_to_sim_done_s": {"value": 135.0, "label": "modeled"},
        "P_trace_to_sim_pct": {"value": 29.6, "label": "derived_modeled"},
        "est_frontend_reduction_s": {"value": 12.0, "label": "modeled"},
        "reduced_T_trace_to_sim_s": {"value": 28.0, "label": "modeled"},
        "complete_flow_impact": "Significant: saves ~12s per run, ~120s per 10-run sweep at expected 30% reduction",
        "trace_export_included": True,
        "result_analysis_included": True,
    },
    {
        "workload": "T2 scale anchor 100 GiB",
        "measurement_unit": "modeled_anchor",
        "model_type": "scale extrapolation",
        "trace_size_GiB": {"value": 100.0, "label": "modeled"},
        "kernel_count": {"value": None, "label": "not_applicable"},
        "threadblock_or_warp_count": {"value": None, "label": "not_applicable"},
        "T_trace_to_sim_s": {"value": 1005.0, "label": "modeled"},
        "T_kernel_to_sim_done_s": {"value": 4000.0, "label": "modeled"},
        "P_trace_to_sim_pct": {"value": 25.1, "label": "derived_modeled"},
        "est_frontend_reduction_s": {"value": 301.5, "label": "modeled"},
        "reduced_T_trace_to_sim_s": {"value": 703.5, "label": "modeled"},
        "complete_flow_impact": "Major: saves ~301.5s per run, ~904.5s per 3-run sweep at expected 30% reduction",
        "trace_export_included": True,
        "result_analysis_included": True,
    },
    {
        "workload": "T3 scale anchor 500 GiB",
        "measurement_unit": "modeled_anchor",
        "model_type": "scale extrapolation",
        "trace_size_GiB": {"value": 500.0, "label": "modeled"},
        "kernel_count": {"value": None, "label": "not_applicable"},
        "threadblock_or_warp_count": {"value": None, "label": "not_applicable"},
        "T_trace_to_sim_s": {"value": 5005.0, "label": "modeled"},
        "T_kernel_to_sim_done_s": {"value": 20000.0, "label": "modeled"},
        "P_trace_to_sim_pct": {"value": 25.0, "label": "derived_modeled"},
        "est_frontend_reduction_s": {"value": 1501.5, "label": "modeled"},
        "reduced_T_trace_to_sim_s": {"value": 3503.5, "label": "modeled"},
        "complete_flow_impact": "Critical: saves ~1501.5s (25 min) per run, ~4504.5s (75 min) per 3-run sweep",
        "trace_export_included": True,
        "result_analysis_included": True,
    },
]


EXISTING_COST_MAP_PATH = "artifacts/trace_bottleneck_map/benchmark_cost_map.json"

def load_existing_controls():
    if not os.path.exists(EXISTING_COST_MAP_PATH):
        return []
    with open(EXISTING_COST_MAP_PATH) as f:
        data = json.load(f)
    controls = []
    for rec in data.get("records", []):
        if rec.get("status") == "measured":
            controls.append({
                "suite": rec["suite"],
                "representative_case": rec["representative_case"],
                "trace_size_mib": rec.get("trace_size_mib"),
                "export_time_s": rec.get("export_time_s"),
                "sim_time_s": rec.get("sim_time_s"),
                "dominant_bottleneck": rec.get("dominant_bottleneck"),
            })
    return controls

def build_json():
    return {
        "report_name": "GPU Trace Frontend Necessity: Central Evidence Table",
        "description": "Merged evidence from workload catalog, trace-size formula, complete-flow burden ratio, DiffTest reduction model, and existing bottleneck map.",
        "generated_date": DATE,
        "evidence_rows": EVIDENCE_ROWS,
        "control_workloads": load_existing_controls(),
        "label_legend": {
            "measured": "Directly measured from simulator runs",
            "modeled": "Estimated from formula or planning model",
            "derived_measured": "Computed from measured inputs",
            "derived_modeled": "Computed from modeled inputs",
            "placeholder": "Placeholder value pending measurement",
            "derived_placeholder": "Computed from placeholder values",
            "pending": "Measurement not yet available",
            "not_applicable": "Not applicable for this workload",
        },
        "go_no_go": {
            "rule": "P_trace_to_sim_slice > 15% OR P_trace_to_sim_step > 15%",
            "slice_pct": 27.6,
            "step_pct": 14.3,
            "verdict": "GO — slice-level P_trace_to_sim exceeds 15% threshold",
            "caveat": "Placeholder data; must be recalibrated with measured timing.",
        },
    }

def build_markdown():
    lines = [
        "# GPU Trace Frontend Necessity: Central Evidence Table",
        "",
        f"Generated: {DATE}",
        "",
        "## Status",
        "",
        "**All values with `placeholder` or `modeled` labels must be recalibrated with measured timing data from simulator instrumentation (task6).**",
        "",
        "## Go/No-Go Summary",
        "",
        "| Metric | Value | Threshold | Result |",
        "|--------|-------|-----------|--------|",
        "| P_trace_to_sim (slice) | 27.6% (placeholder) | > 15% | GO |",
        "| P_trace_to_sim (step) | 14.3% (placeholder) | > 15% | NOT YET |",
        "| Overall | — | Slice OR Step | GO (pending measured data) |",
        "",
        "## Evidence Table",
        "",
        "| Workload | Unit | Trace Size (GiB) | T_frontend (s) | T_total (s) | P_frontend (%) | Est. Reduction (s) | Reduced T_frontend (s) | Impact |",
        "|----------|------|-----------------|---------------|------------|---------------|-------------------|----------------------|--------|",
    ]
    for r in EVIDENCE_ROWS:
        s = r["trace_size_GiB"]
        t_sim = r["T_trace_to_sim_s"]
        t_total = r["T_kernel_to_sim_done_s"]
        p = r["P_trace_to_sim_pct"]
        red = r["est_frontend_reduction_s"]
        red_t = r["reduced_T_trace_to_sim_s"]
        lines.append(
            f"| {r['workload']} | {r['measurement_unit']} | "
            f"{s['value']} ({s['label']}) | {t_sim['value']} ({t_sim['label']}) | "
            f"{t_total['value']} ({t_total['label']}) | {p['value']} ({p['label']}) | "
            f"{red['value']} ({red['label']}) | {red_t['value']} ({red_t['label']}) | "
            f"{r['complete_flow_impact']} |"
        )

    controls = load_existing_controls()
    if controls:
        lines += [
            "",
            "## Control Workloads (from Existing Bottleneck Map)",
            "",
            "| Suite | Representative Case | Trace Size (MiB) | Export (s) | Sim (s) | Dominant Bottleneck |",
            "|-------|-------------------|-----------------|-----------|--------|--------------------|",
        ]
        for c in controls[:10]:  # Top 10 measured controls
            lines.append(
                f"| {c['suite']} | {c['representative_case']} | "
                f"{c['trace_size_mib'] or 'N/A'} | {c['export_time_s'] or 'N/A'} | "
                f"{c['sim_time_s'] or 'N/A'} | {c['dominant_bottleneck']} |"
            )

    lines += [
        "",
        "## Conclusion (Provisional)",
        "",
        "Based on placeholder data:",
        "- Slice-level P_trace_to_sim exceeds the 15% engineering gate, suggesting frontend restructuring is worth a prototype investigation.",
        "- Step-level P_trace_to_sim is close to but below 15% with placeholder values; measured data may change this.",
        "- Scale-anchor modeling suggests frontend cost grows linearly with trace size, making the optimization increasingly valuable at industrial scale.",
        "- **Next step**: Replace all placeholder values with measured timing data from simulator instrumentation.",
        "",
        "## Label Legend",
        "",
        "| Label | Meaning |",
        "|-------|---------|",
        "| measured | Directly measured from simulator runs |",
        "| modeled | Estimated from formula or planning model |",
        "| derived_measured | Computed from measured inputs |",
        "| derived_modeled | Computed from modeled inputs |",
        "| placeholder | Placeholder value pending measurement |",
        "| derived_placeholder | Computed from placeholder values |",
        "| pending | Measurement not yet available |",
        "| not_applicable | Not applicable for this workload |",
    ]
    return "\n".join(lines) + "\n"

def main():
    j = build_json()
    md = build_markdown()

    with open(f"{OUT_DIR}/paper_argument_matrix.json", "w") as f:
        json.dump(j, f, indent=2)
        f.write("\n")
    with open(f"{OUT_DIR}/paper_argument_matrix.md", "w") as f:
        f.write(md)

    print(f"Wrote {OUT_DIR}/paper_argument_matrix.json")
    print(f"Wrote {OUT_DIR}/paper_argument_matrix.md")

    # Verify: all rows have the required fields
    required = ["workload", "measurement_unit", "trace_size_GiB", "T_trace_to_sim_s",
                "T_kernel_to_sim_done_s", "P_trace_to_sim_pct"]
    for r in EVIDENCE_ROWS:
        for field in required:
            assert field in r, f"Missing required field: {field}"
        assert r["T_trace_to_sim_s"]["value"] is not None, f"Missing T_trace_to_sim for {r['workload']}"
    print("Verification PASSED: all rows have required fields")

if __name__ == "__main__":
    main()
