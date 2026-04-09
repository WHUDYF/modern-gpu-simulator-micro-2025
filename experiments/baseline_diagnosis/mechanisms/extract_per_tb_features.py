#!/usr/bin/env python3
"""Extract unified per-TB feature vectors from existing full features JSON.

Reads an existing `<workload>_full.json` produced by the baseline diagnosis
pipeline and produces a new JSON conforming to per_tb_features_schema.json.
"""
import argparse
import json
import sys
from pathlib import Path


FP64_OPCODES = {"DMUL", "DFMA", "F2F.F64.F32", "F2F.F32.F64", "DADD", "DSUB"}


def classify_opcode(opcode):
    op = opcode.upper()
    if "FFMA" in op:
        return "ffma"
    if any(x in op for x in ["DFMA", "DMUL", "DADD", "DSUB"]):
        return "dfma"
    if "LDG" in op:
        return "ldg"
    if "STG" in op:
        return "stg"
    if "LDS" in op:
        return "lds"
    if "STS" in op:
        return "sts"
    if "IADD" in op:
        return "iadd"
    if "BAR" in op:
        return "bar"
    return "other"


def compute_opcode_ratios(top_opcodes):
    total = sum(entry["count"] for entry in top_opcodes)
    if total == 0:
        return {}
    categories = {}
    for entry in top_opcodes:
        cat = classify_opcode(entry["opcode"])
        categories[cat] = categories.get(cat, 0) + entry["count"]
    return {f"opcode_{cat}_ratio": count / total for cat, count in categories.items()}


def build_kernel_summary(kernel_data):
    static = kernel_data.get("static_info", {})
    dynamic = kernel_data.get("dynamic_stats", {}) or {}
    compression = kernel_data.get("compression_features", {}) or {}
    top_opcodes = static.get("top_opcodes", [])

    uses_fp64 = any(
        any(fp64 in entry["opcode"].upper() for fp64 in FP64_OPCODES)
        for entry in top_opcodes
    )
    uses_shared_memory = any(
        "LDS" in entry["opcode"].upper() or "STS" in entry["opcode"].upper()
        for entry in top_opcodes
    )
    num_barriers = sum(
        entry["count"] for entry in top_opcodes if "BAR" in entry["opcode"].upper()
    )

    return {
        "top_opcodes": top_opcodes[:10],
        "total_static_instructions": static.get("total_static_instructions", 0),
        "total_dynamic_instructions": dynamic.get("total_dynamic_insts", 0),
        "uses_fp64": uses_fp64,
        "uses_shared_memory": uses_shared_memory,
        "num_barriers": num_barriers,
        "grid_dim": dynamic.get("grid_dim", ""),
        "block_dim": dynamic.get("block_dim", ""),
        "num_tbs": compression.get("num_tb_files", 0),
    }


def build_per_tb_entries(kernel_data):
    compression = kernel_data.get("compression_features", {}) or {}
    static = kernel_data.get("static_info", {})
    top_opcodes = static.get("top_opcodes", [])
    opcode_ratios = compute_opcode_ratios(top_opcodes)

    num_tbs = compression.get("num_tb_files", 0)
    num_warps_stats = compression.get("num_warps", {}) or {}
    inst_stats = compression.get("instructions_per_warp_mean", {}) or {}

    num_warps_mean = num_warps_stats.get("mean", 0) or 0
    inst_per_warp_mean = inst_stats.get("mean", 0) or 0
    inst_per_warp_std = inst_stats.get("std", 0) or 0

    base_features = {
        "num_warps": num_warps_mean,
        "instructions_per_warp_mean": inst_per_warp_mean,
        "instructions_per_warp_std": inst_per_warp_std,
        "compression_format": compression.get("dominant_format", "unknown"),
        "address_override_count": 0,
        "is_full_encoding": False,
    }
    for field, value in opcode_ratios.items():
        base_features[field] = value

    return [
        {"tb_index": i, "features": dict(base_features)}
        for i in range(num_tbs)
    ]


def extract_per_tb_features(full_json_path):
    with open(full_json_path) as f:
        full = json.load(f)

    workload = full.get("workload", "unknown")
    per_kernel = full.get("per_kernel", {})

    kernels = []
    for idx, (kname, kdata) in enumerate(per_kernel.items(), start=1):
        kernels.append({
            "kernel_id": idx,
            "kernel_name": kname,
            "kernel_summary": build_kernel_summary(kdata),
            "per_tb": build_per_tb_entries(kdata),
        })

    return {
        "workload": workload,
        "kernels": kernels,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract unified per-TB features.")
    parser.add_argument("--input", required=True, help="Path to <workload>_full.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = extract_per_tb_features(args.input)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    total_tbs = sum(len(k["per_tb"]) for k in result["kernels"])
    print(
        f"[per_tb] wrote {output_path} "
        f"(workload={result['workload']}, kernels={len(result['kernels'])}, total_tbs={total_tbs})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
