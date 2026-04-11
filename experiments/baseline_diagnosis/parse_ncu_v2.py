#!/usr/bin/env python3
"""Parse NCU --set full CSV into the same format as mini_transformer_hw.json.

Aggregates multiple launches of the same kernel by name, computing
mean values across all instances (per-kernel type, not per-launch).
"""
import csv, io, json, sys, re
from collections import defaultdict

METRIC_MAP = {
    "DRAM Frequency":                    "dram_freq_hz",
    "SM Frequency":                      "sm_freq_hz",
    "Elapsed Cycles":                    "elapsed_cycles",
    "Duration":                          "duration_ns",
    "SM Active Cycles":                  "sm_active_cycles",
    "DRAM Throughput":                   "dram_throughput_pct",
    "Memory Throughput":                 "memory_throughput_bps",
    "Compute (SM) Throughput":           "compute_throughput_pct",
    "L1/TEX Cache Throughput":           "l1_throughput_pct",
    "L2 Cache Throughput":               "l2_throughput_pct",
    "Achieved Occupancy":                "achieved_occupancy_pct",
    "Theoretical Occupancy":             "theoretical_occupancy_pct",
    "Achieved Active Warps Per SM":      "achieved_warps_per_sm",
    "Executed Ipc Active":               "ipc_active",
    "Executed Ipc Elapsed":              "ipc_elapsed",
    "Issued Ipc Active":                 "issued_ipc_active",
    "L1/TEX Hit Rate":                   "l1_hit_rate_pct",
    "L2 Hit Rate":                       "l2_hit_rate_pct",
    "Mem Pipes Busy":                    "mem_pipes_busy_pct",
    "Branch Instructions Ratio":         "branch_inst_ratio",
    "Branch Efficiency":                 "branch_efficiency_pct",
    "Avg. Divergent Branches":           "avg_divergent_branches",
    "Block Limit SM":                    "block_limit_sm",
    "Block Limit Registers":             "block_limit_registers",
    "Block Limit Shared Mem":            "block_limit_shmem",
    "Block Limit Warps":                 "block_limit_warps",
    "Theoretical Active Warps per SM":   "theoretical_active_warps_per_sm",
    "Registers Per Thread":              "registers_per_thread",
    "Static Shared Memory Per Block":    "static_shmem_per_block",
    "Dynamic Shared Memory Per Block":   "dynamic_shmem_per_block",
    "Threads":                           "threads",
    "Waves Per SM":                      "waves_per_sm",
    "Warp Cycles Per Issued Instruction":   "warp_cycles_per_issued_inst",
    "Warp Cycles Per Executed Instruction": "warp_cycles_per_executed_inst",
    "Avg. Active Threads Per Warp":      "avg_active_threads_per_warp",
    "Avg. Not Predicated Off Threads Per Warp": "avg_not_predicated_off_threads_per_warp",
    "Mem Busy":                          "mem_busy_pct",
    "Max Bandwidth":                     "max_bandwidth_pct",
}

def parse_value(raw):
    cleaned = raw.strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None

def short_name(full_name):
    m = re.match(r'(\w+)\(', full_name)
    return m.group(1) if m else full_name.split('(')[0].strip()

def main(csv_path, out_path):
    with open(csv_path) as f:
        lines = f.readlines()
    data_lines = [l for l in lines if l.startswith('"')]
    reader = csv.DictReader(io.StringIO("".join(data_lines)))

    # {short_name: {metric_key: [values]}}
    kernel_data = defaultdict(lambda: {"_full_name": "", "_block": "", "_grid": "", "_cc": "",
                                        "_metrics": defaultdict(list), "_count": 0})

    for row in reader:
        full_name = row.get("Kernel Name", "").strip()
        metric_name = row.get("Metric Name", "").strip()
        metric_value = row.get("Metric Value", "").strip()
        block = row.get("Block Size", "").strip()
        grid  = row.get("Grid Size", "").strip()
        cc    = row.get("CC", "").strip()

        if metric_name not in METRIC_MAP:
            continue

        sname = short_name(full_name)
        kd = kernel_data[sname]
        kd["_full_name"] = full_name
        kd["_block"] = block
        kd["_grid"] = grid
        kd["_cc"] = cc

        val = parse_value(metric_value)
        if val is not None:
            friendly = METRIC_MAP[metric_name]
            kd["_metrics"][friendly].append(val)

    # Aggregate: mean per metric per kernel type
    result = {}
    for sname, kd in sorted(kernel_data.items()):
        metrics = {}
        for k, vals in kd["_metrics"].items():
            metrics[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
        result[sname] = {
            "full_name": kd["_full_name"],
            "block_size": kd["_block"],
            "grid_size": kd["_grid"],
            "cc": kd["_cc"],
            "key_metrics": metrics,
            "num_launches": len(kd["_metrics"].get("waves_per_sm", [1])),
        }

    out = {"hardware_stats": result, "_full_metrics_count": len(METRIC_MAP),
           "_kernels": list(result.keys())}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Parsed {len(result)} kernel types → {out_path}", file=sys.stderr)
    for k, v in result.items():
        waves = v["key_metrics"].get("waves_per_sm", "?")
        occ   = v["key_metrics"].get("achieved_occupancy_pct", "?")
        wcpi  = v["key_metrics"].get("warp_cycles_per_issued_inst", "?")
        comp  = v["key_metrics"].get("compute_throughput_pct", "?")
        print(f"  {k:25s}  waves={waves:6}  occ={occ:5}%  warp_cyc={wcpi:6}  compute={comp:5}%", file=sys.stderr)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
