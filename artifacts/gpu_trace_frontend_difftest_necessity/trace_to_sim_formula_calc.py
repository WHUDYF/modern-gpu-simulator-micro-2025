#!/usr/bin/env python3
"""Trace-size to T_trace_to_sim planning formula calculator.

Implements: T_trace_to_sim ~= C_fixed + S_trace_GiB / R_frontend_GiBps
with fast, expected, and pessimistic scenarios.
"""
import json
import math
import sys

SCENARIOS = {
    "fast": {"C_fixed_s": 2.0, "R_frontend_GiBps": 0.2, "label": "Fast (optimistic)"},
    "expected": {"C_fixed_s": 5.0, "R_frontend_GiBps": 0.1, "label": "Expected (baseline)"},
    "pessimistic": {"C_fixed_s": 10.0, "R_frontend_GiBps": 0.05, "label": "Pessimistic (conservative)"},
}

TRACE_SIZES = [
    {"label": "micro (local)", "size_GiB": 0.01},
    {"label": "small slice (local)", "size_GiB": 0.1},
    {"label": "medium slice (local)", "size_GiB": 0.5},
    {"label": "large slice / small step (local)", "size_GiB": 2.0},
    {"label": "BERT-base full step (local)", "size_GiB": 10.0},
    {"label": "Llama 3.1 8B layer slice (local)", "size_GiB": 20.0},
    {"label": "T2 scale anchor (modeled)", "size_GiB": 100.0},
    {"label": "T3 scale anchor (modeled)", "size_GiB": 500.0},
    {"label": "1 TiB scale anchor (modeled)", "size_GiB": 1024.0},
]

DATE = "2026-05-03"

def compute_t_trace_to_sim(size_GiB, scenario):
    C = scenario["C_fixed_s"]
    R = scenario["R_frontend_GiBps"]
    return C + size_GiB / R

def build_rows():
    rows = []
    for ts in TRACE_SIZES:
        row = {"trace_label": ts["label"], "trace_size_GiB": ts["size_GiB"]}
        for key, sc in SCENARIOS.items():
            t = compute_t_trace_to_sim(ts["size_GiB"], sc)
            row[f"T_trace_to_sim_{key}_s"] = round(t, 2)
            row[f"throughput_{key}_GiBps"] = sc["R_frontend_GiBps"]
        row["T_trace_to_sim_expected_shortcut_s"] = round(5.0 + 10.0 * ts["size_GiB"], 2)
        rows.append(row)
    return rows

def build_json(rows):
    return {
        "report_name": "Trace-Size to T_trace_to_sim Formula Calculator",
        "formula": "T_trace_to_sim ~= C_fixed + S_trace_GiB / R_frontend_GiBps",
        "expected_shortcut": "T_trace_to_sim ~= 5 + 10 * S_trace_GiB seconds",
        "generated_date": DATE,
        "scenarios": {
            key: {"C_fixed_s": sc["C_fixed_s"], "R_frontend_GiBps": sc["R_frontend_GiBps"], "label": sc["label"]}
            for key, sc in SCENARIOS.items()
        },
        "rows": rows,
    }

def build_markdown(rows):
    lines = [
        "# Trace-Size to T_trace_to_sim Formula Calculator",
        "",
        f"Generated: {DATE}",
        "",
        "## Formula",
        "",
        "```text",
        "T_trace_to_sim ~= C_fixed + S_trace_GiB / R_frontend_GiBps",
        "```",
        "",
        "Expected shortcut: `T_trace_to_sim ~= 5 + 10 * S_trace_GiB seconds`",
        "",
        "## Scenario Parameters",
        "",
        "| Scenario | C_fixed (s) | R_frontend (GiB/s) | Time per GiB (s/GiB) |",
        "|----------|------------|-------------------|---------------------|",
    ]
    for key, sc in SCENARIOS.items():
        lines.append(
            f"| {sc['label']} | {sc['C_fixed_s']:.0f} | {sc['R_frontend_GiBps']:.2f} | {1.0/sc['R_frontend_GiBps']:.0f} |"
        )
    lines += [
        "",
        "## Planning Table",
        "",
        "| Trace Label | Size (GiB) | T (Fast, s) | T (Expected, s) | T (Pessimistic, s) | Expected Shortcut (s) |",
        "|-------------|-----------|------------|----------------|-------------------|----------------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['trace_label']} | {row['trace_size_GiB']:.3f} | "
            f"{row['T_trace_to_sim_fast_s']:.1f} | {row['T_trace_to_sim_expected_s']:.1f} | "
            f"{row['T_trace_to_sim_pessimistic_s']:.1f} | {row['T_trace_to_sim_expected_shortcut_s']:.1f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- All estimates are **planning and modeling thresholds**, not hard performance guarantees.",
        "- The expected shortcut `5 + 10 * S_trace_GiB` reproduces the formula requirement from the acceptance criteria.",
        "- Fast scenario assumes optimized frontend throughput (0.2 GiB/s, 5 s/GiB).",
        "- Pessimistic scenario accounts for slow I/O, large metadata, or contention (0.05 GiB/s, 20 s/GiB).",
        "- Values for T2/T3 scale anchors are **modeled**, not measured.",
        "- Actual measured data will calibrate these parameters.",
        "",
        "## Sweep-Level Cumulative Cost (Expected Scenario)",
        "",
        "| Sweep Type | Runs per Sweep | Trace Size per Run (GiB) | T per Run (s) | Total Sweep T (s) | Total Sweep T (min) |",
        "|-----------|---------------|------------------------|--------------|------------------|-------------------|",
    ]
    sweep_examples = [
        ("micro slice sweep", 50, 0.1),
        ("medium slice sweep", 20, 0.5),
        ("large slice sweep", 10, 2.0),
        ("BERT-base full step sweep", 5, 10.0),
        ("scale-anchor modeled sweep", 3, 100.0),
    ]
    for label, runs, size in sweep_examples:
        t_per = round(5.0 + 10.0 * size, 1)
        total = round(runs * t_per, 1)
        total_min = round(total / 60.0, 1)
        lines.append(f"| {label} | {runs} | {size:.1f} | {t_per:.1f} | {total:.1f} | {total_min:.1f} |")
    return "\n".join(lines) + "\n"

def main():
    rows = build_rows()
    j = build_json(rows)
    md = build_markdown(rows)

    out_dir = "artifacts/gpu_trace_frontend_difftest_necessity"
    with open(f"{out_dir}/trace_to_sim_formula.json", "w") as f:
        json.dump(j, f, indent=2)
        f.write("\n")
    with open(f"{out_dir}/trace_to_sim_formula.md", "w") as f:
        f.write(md)

    print(f"Wrote {out_dir}/trace_to_sim_formula.json")
    print(f"Wrote {out_dir}/trace_to_sim_formula.md")

    # Verify expected shortcut
    for size_GiB in [0.1, 1.0, 10.0]:
        computed = compute_t_trace_to_sim(size_GiB, SCENARIOS["expected"])
        shortcut = 5.0 + 10.0 * size_GiB
        assert abs(computed - shortcut) < 0.01, f"Shortcut mismatch: {computed} vs {shortcut}"
    print("Verification PASSED: expected shortcut matches formula")

if __name__ == "__main__":
    main()
