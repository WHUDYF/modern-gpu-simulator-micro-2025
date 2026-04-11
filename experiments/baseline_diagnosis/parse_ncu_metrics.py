#!/usr/bin/env python3
"""Parse Nsight Compute CSV output into structured JSON with hardware stats."""

import argparse
import csv
import json
import io
import sys


METRIC_MAP = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_utilization",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "sm__inst_executed.avg.per_cycle_active": "ipc",
    "l1tex__data_bank_conflicts_pipe_lsu.sum": "l1_bank_conflicts",
    "sm__sass_thread_inst_executed_op_branch_pred_on.sum": "branch_instructions",
    "sm__sass_thread_inst_executed_op_branch_pred_on_diverged.sum": "divergent_branches",
    "dram__bytes_read.sum": "dram_bytes_read",
    "dram__bytes_write.sum": "dram_bytes_write",
}


def parse_value(raw: str) -> float:
    """Convert a metric value string to float, stripping commas and units."""
    cleaned = raw.strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def compute_derived_metrics(metrics: dict) -> dict:
    """Add derived metrics based on raw extracted values."""
    l1_hit = metrics.get("l1_hit_rate")
    if l1_hit is not None:
        metrics["l1_miss_rate"] = 1.0 - l1_hit / 100.0

    l2_hit = metrics.get("l2_hit_rate")
    if l2_hit is not None:
        metrics["l2_miss_rate"] = 1.0 - l2_hit / 100.0

    branch = metrics.get("branch_instructions", 0)
    divergent = metrics.get("divergent_branches", 0)
    if branch > 0:
        metrics["warp_divergence_rate"] = divergent / branch

    return metrics


def parse_ncu_csv(input_path: str) -> dict:
    """Read NCU CSV, extract metrics per kernel, return structured dict."""
    with open(input_path, "r") as f:
        lines = f.readlines()

    # Skip NCU header lines that start with "=="
    data_lines = [line for line in lines if not line.startswith("==")]
    if not data_lines:
        return {"hardware_stats": {}}

    reader = csv.DictReader(io.StringIO("".join(data_lines)))

    kernels: dict = {}
    for row in reader:
        kernel_name = row.get("Kernel Name", "").strip()
        metric_name = row.get("Metric Name", "").strip()
        metric_value = row.get("Metric Value", "").strip()

        if not kernel_name or metric_name not in METRIC_MAP:
            continue

        friendly = METRIC_MAP[metric_name]
        kernels.setdefault(kernel_name, {})[friendly] = parse_value(metric_value)

    for kernel_name in kernels:
        kernels[kernel_name] = compute_derived_metrics(kernels[kernel_name])

    return {"hardware_stats": kernels}


def main():
    parser = argparse.ArgumentParser(
        description="Parse NCU CSV output into structured JSON"
    )
    parser.add_argument("--input", required=True, help="Path to NCU CSV file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    result = parse_ncu_csv(args.input)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    kernel_count = len(result["hardware_stats"])
    print(f"Parsed {kernel_count} kernel(s). Output written to {args.output}")


if __name__ == "__main__":
    main()
