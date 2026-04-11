#!/usr/bin/env python3
"""Extract compression-level features from GPU simulator trace files.

Parses protobuf trace data, static instruction metadata, and kernel stats
to produce a structured JSON feature summary for architecture diagnosis.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proto_gen import (
    compressed_threadblock_pb2,
    threadblock_pb2,
    trace_pb2,
)


def compute_stats(values):
    """Compute summary statistics for a list of numeric values."""
    if not values:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0,
                "p25": 0, "p50": 0, "p75": 0}
    arr = np.array(values, dtype=float)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def parse_dynamic_trace(trace_dir):
    """Parse dynamic_trace.pb for top-level trace metadata."""
    pb_path = os.path.join(trace_dir, "dynamic_trace.pb")
    if not os.path.exists(pb_path):
        return None

    with open(pb_path, "rb") as f:
        data = f.read()

    trace = trace_pb2.Trace()
    trace.ParseFromString(data)

    kernels = []
    for _dev_id, device in sorted(trace.gpu_device.items()):
        for _stream_id, stream in sorted(device.streams.items()):
            for k in stream.kernels:
                kernels.append({
                    "id": k.id,
                    "name": k.name,
                    "function_unique_id": k.function_unique_id,
                    "grid_dim": {"x": k.grid_dim.x, "y": k.grid_dim.y,
                                 "z": k.grid_dim.z},
                    "block_dim": {"x": k.block_dim.x, "y": k.block_dim.y,
                                  "z": k.block_dim.z},
                    "number_of_registers": k.number_of_registers,
                    "size_shared_memory": k.size_shared_memory,
                })

    return {
        "name": trace.name,
        "binary_version": trace.binary_version,
        "nvbit_version": trace.nvbit_version,
        "accelsim_version": trace.accelsim_version,
        "kernels": kernels,
    }


def parse_stats_csv(trace_dir):
    """Parse stats.csv for raw kernel statistics."""
    csv_path = os.path.join(trace_dir, "stats.csv")
    if not os.path.exists(csv_path):
        return []

    results = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            stripped = [c.strip() for c in row]
            if header is None:
                header = stripped
                continue
            if not stripped or not stripped[0]:
                continue
            entry = {}
            for i, col in enumerate(header):
                if i < len(stripped):
                    val = stripped[i]
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                    entry[col] = val
            results.append(entry)
    return results


def parse_enhanced_execution_info(trace_dir):
    """Parse enhanced_execution_info.json for static instruction metadata."""
    json_path = os.path.join(trace_dir, "extra_info", "enhanced_execution_info.json")
    if not os.path.exists(json_path):
        return None

    with open(json_path) as f:
        data = json.load(f)

    result = {"benchmark_name": data.get("benchmark_name", "")}
    kernel_summaries = []

    for kernel in data.get("kernels", []):
        instructions = kernel.get("instructions", [])
        opcode_counts = Counter(inst.get("op_code", "") for inst in instructions)
        top_opcodes = opcode_counts.most_common(10)

        stall_counts = []
        barrier_waits = 0
        yields = 0
        write_barriers = 0
        read_barriers = 0

        for inst in instructions:
            cb = inst.get("control_bits", {})
            stall_counts.append(cb.get("stall_count", 0))
            if cb.get("wait_barrier_bits", 0) != 0:
                barrier_waits += 1
            if cb.get("is_yield", False):
                yields += 1
            if cb.get("is_new_write_barrier", False):
                write_barriers += 1
            if cb.get("is_new_read_barrier", False):
                read_barriers += 1

        kernel_summaries.append({
            "kernel_name": kernel.get("kernel_name", ""),
            "unique_function_id": kernel.get("unique_function_id", 0),
            "total_static_instructions": len(instructions),
            "top_opcodes": [{"opcode": op, "count": cnt} for op, cnt in top_opcodes],
            "control_bits_summary": {
                "stall_count_stats": compute_stats(stall_counts),
                "barrier_waits": barrier_waits,
                "yields": yields,
                "write_barriers": write_barriers,
                "read_barriers": read_barriers,
            },
        })

    result["kernels"] = kernel_summaries
    return result


def detect_and_parse_tb(data):
    """Auto-detect threadblock protobuf format and parse accordingly.

    Attempts formats from newest to oldest. Uses content heuristics
    to distinguish between formats that share compatible wire layouts.

    Returns (format_name, parsed_message) or (None, None) on failure.
    """
    # Try compressed_kernel_v8: meaningful if base_threadblock has warps
    # or shared_pc_sequence, or if delta TBs have reference block IDs with
    # non-zero coordinates alongside a non-empty base.
    try:
        msg = compressed_threadblock_pb2.compressed_kernel_v8()
        msg.ParseFromString(data)
        base = msg.base_threadblock
        has_base_content = (len(base.warps) > 0
                           or len(base.shared_pc_sequence) > 0
                           or base.function_unique_id != 0)
        if has_base_content:
            return "v8", msg
    except Exception:
        pass

    # Try compressed_threadblock_v7: meaningful if shared_pc_sequence is populated
    try:
        msg = compressed_threadblock_pb2.compressed_threadblock_v7()
        msg.ParseFromString(data)
        if len(msg.shared_pc_sequence) > 0:
            return "v7", msg
    except Exception:
        pass

    # Try compressed_threadblock_v6: meaningful if any warp has runs
    try:
        msg = compressed_threadblock_pb2.compressed_threadblock_v6()
        msg.ParseFromString(data)
        if len(msg.warps) > 0:
            has_runs = any(len(w.runs) > 0 for w in msg.warps.values())
            if has_runs:
                return "v6", msg
    except Exception:
        pass

    # Try original compressed_threadblock
    try:
        msg = compressed_threadblock_pb2.compressed_threadblock()
        msg.ParseFromString(data)
        if len(msg.warps) > 0:
            return "compressed", msg
    except Exception:
        pass

    # Try uncompressed threadblock
    try:
        msg = threadblock_pb2.threadblock()
        msg.ParseFromString(data)
        if len(msg.warps) > 0:
            return "uncompressed", msg
    except Exception:
        pass

    return None, None


def extract_v8_features(msg):
    """Extract features from compressed_kernel_v8."""
    base = msg.base_threadblock
    deltas = msg.delta_threadblocks
    total_tbs = 1 + len(deltas)

    shared_pc_len = len(base.shared_pc_sequence)

    offset_only_count = 0
    override_counts = []
    full_encoding_count = 0

    for delta in deltas:
        if delta.is_full_encoding:
            full_encoding_count += 1
        elif len(delta.address_overrides) == 0:
            offset_only_count += 1
        override_counts.append(len(delta.address_overrides))

    cross_tb_offset_coverage = (
        offset_only_count / len(deltas) if deltas else 0.0
    )
    address_override_density = (
        float(np.mean(override_counts)) if override_counts else 0.0
    )
    full_encoding_fallback_rate = (
        full_encoding_count / len(deltas) if deltas else 0.0
    )

    warp_diff_sizes = []
    for warp in base.warps.values():
        warp_diff_sizes.append(len(warp.instructions) + len(warp.pc_overrides))

    return {
        "format": "compressed_kernel_v8",
        "total_threadblocks": total_tbs,
        "shared_pc_sequence_length": shared_pc_len,
        "cross_tb_offset_coverage": cross_tb_offset_coverage,
        "address_override_density": address_override_density,
        "full_encoding_fallback_rate": full_encoding_fallback_rate,
        "warp_diff_distribution": compute_stats(warp_diff_sizes),
        "num_warps_in_base": len(base.warps),
    }


def extract_v7_features(msg):
    """Extract features from compressed_threadblock_v7."""
    warp_diff_sizes = []
    for warp in msg.warps.values():
        warp_diff_sizes.append(len(warp.instructions) + len(warp.pc_overrides))

    return {
        "format": "compressed_threadblock_v7",
        "shared_pc_sequence_length": len(msg.shared_pc_sequence),
        "function_unique_id": msg.function_unique_id,
        "num_warps": len(msg.warps),
        "warp_diff_distribution": compute_stats(warp_diff_sizes),
    }


def extract_v6_features(msg):
    """Extract features from compressed_threadblock_v6."""
    total_instructions = 0
    total_run_instructions = 0
    run_counts = []

    for warp in msg.warps.values():
        num_explicit = len(warp.instructions)
        run_inst_count = sum(r.count for r in warp.runs)
        total_instructions += num_explicit + run_inst_count
        total_run_instructions += run_inst_count
        run_counts.append(len(warp.runs))

    rle_coverage = (
        total_run_instructions / total_instructions
        if total_instructions > 0 else 0.0
    )

    return {
        "format": "compressed_threadblock_v6",
        "function_unique_id": msg.function_unique_id,
        "num_warps": len(msg.warps),
        "rle_coverage": rle_coverage,
        "rle_length_distribution": compute_stats(run_counts),
    }


def extract_compressed_features(msg):
    """Extract features from original compressed_threadblock."""
    inst_counts = []
    for warp in msg.warps.values():
        inst_counts.append(len(warp.instructions))

    return {
        "format": "compressed_threadblock",
        "function_unique_id": msg.function_unique_id,
        "num_warps": len(msg.warps),
        "instructions_per_warp": compute_stats(inst_counts),
    }


def extract_uncompressed_features(msg):
    """Extract features from uncompressed threadblock."""
    inst_counts = []
    for warp in msg.warps.values():
        inst_counts.append(len(warp.instructions))

    return {
        "format": "uncompressed_threadblock",
        "num_warps": len(msg.warps),
        "instructions_per_warp": compute_stats(inst_counts),
    }


FEATURE_EXTRACTORS = {
    "v8": extract_v8_features,
    "v7": extract_v7_features,
    "v6": extract_v6_features,
    "compressed": extract_compressed_features,
    "uncompressed": extract_uncompressed_features,
}


def find_tb_files(trace_dir):
    """Find all threadblock .pb files grouped by kernel directory."""
    tb_root = os.path.join(trace_dir, "threadblocks")
    if not os.path.isdir(tb_root):
        return {}

    kernel_files = {}
    for dirpath, _dirnames, filenames in os.walk(tb_root):
        for fname in filenames:
            if fname.endswith(".pb"):
                kernel_dir = os.path.basename(dirpath)
                kernel_files.setdefault(kernel_dir, []).append(
                    os.path.join(dirpath, fname)
                )

    for kernel_dir in kernel_files:
        kernel_files[kernel_dir].sort()

    return kernel_files


def parse_kernel_threadblocks(tb_files):
    """Parse all threadblock files for a single kernel."""
    format_counts = Counter()
    all_features = []

    for tb_path in tb_files:
        with open(tb_path, "rb") as f:
            data = f.read()

        fmt, msg = detect_and_parse_tb(data)
        if fmt is None:
            format_counts["unknown"] += 1
            continue

        format_counts[fmt] += 1
        features = FEATURE_EXTRACTORS[fmt](msg)
        features["file"] = os.path.basename(tb_path)
        all_features.append(features)

    return format_counts, all_features


def aggregate_kernel_tb_features(all_features, format_counts):
    """Aggregate per-TB features into kernel-level summary."""
    if not all_features:
        return {"format_counts": dict(format_counts), "num_tb_files": 0}

    dominant_format = format_counts.most_common(1)[0][0]
    summary = {
        "num_tb_files": len(all_features),
        "format_counts": dict(format_counts),
        "dominant_format": dominant_format,
    }

    if dominant_format == "v8":
        total_tbs = [f["total_threadblocks"] for f in all_features
                     if f.get("format") == "compressed_kernel_v8"]
        shared_pc_lens = [f["shared_pc_sequence_length"] for f in all_features
                          if f.get("format") == "compressed_kernel_v8"]
        offset_coverages = [f["cross_tb_offset_coverage"] for f in all_features
                            if f.get("format") == "compressed_kernel_v8"]
        override_densities = [f["address_override_density"] for f in all_features
                              if f.get("format") == "compressed_kernel_v8"]
        fallback_rates = [f["full_encoding_fallback_rate"] for f in all_features
                          if f.get("format") == "compressed_kernel_v8"]

        summary.update({
            "total_threadblocks": compute_stats(total_tbs),
            "shared_pc_sequence_length": compute_stats(shared_pc_lens),
            "cross_tb_offset_coverage": compute_stats(offset_coverages),
            "address_override_density": compute_stats(override_densities),
            "full_encoding_fallback_rate": compute_stats(fallback_rates),
        })

    elif dominant_format == "compressed":
        warp_counts = [f["num_warps"] for f in all_features
                       if f.get("format") == "compressed_threadblock"]
        all_inst_means = [f["instructions_per_warp"]["mean"] for f in all_features
                          if f.get("format") == "compressed_threadblock"]
        summary.update({
            "num_warps": compute_stats(warp_counts),
            "instructions_per_warp_mean": compute_stats(all_inst_means),
        })

    elif dominant_format == "v6":
        rle_coverages = [f["rle_coverage"] for f in all_features
                         if f.get("format") == "compressed_threadblock_v6"]
        summary.update({
            "rle_coverage": compute_stats(rle_coverages),
        })

    elif dominant_format == "v7":
        shared_pc_lens = [f["shared_pc_sequence_length"] for f in all_features
                          if f.get("format") == "compressed_threadblock_v7"]
        summary.update({
            "shared_pc_sequence_length": compute_stats(shared_pc_lens),
        })

    elif dominant_format == "uncompressed":
        all_inst_means = [f["instructions_per_warp"]["mean"] for f in all_features
                          if f.get("format") == "uncompressed_threadblock"]
        summary.update({
            "instructions_per_warp_mean": compute_stats(all_inst_means),
        })

    return summary


def extract_features(trace_dir):
    """Extract all features from a trace directory."""
    print(f"Processing trace directory: {trace_dir}")

    result = {"trace_dir": trace_dir}

    # Dynamic trace info
    trace_info = parse_dynamic_trace(trace_dir)
    if trace_info:
        result["dynamic_trace"] = trace_info
        print(f"  Trace: {trace_info['name']}, "
              f"version={trace_info['binary_version']}, "
              f"kernels={len(trace_info['kernels'])}")
    else:
        print("  WARNING: dynamic_trace.pb not found")

    # Stats CSV
    stats = parse_stats_csv(trace_dir)
    result["stats_csv"] = stats
    print(f"  Stats CSV: {len(stats)} kernel entries")

    # Enhanced execution info
    exec_info = parse_enhanced_execution_info(trace_dir)
    if exec_info:
        result["enhanced_execution_info"] = exec_info
        for ks in exec_info["kernels"]:
            print(f"  Static info for {ks['kernel_name']}: "
                  f"{ks['total_static_instructions']} instructions, "
                  f"top opcode: {ks['top_opcodes'][0]['opcode'] if ks['top_opcodes'] else 'N/A'}")
    else:
        print("  WARNING: enhanced_execution_info.json not found")

    # Threadblock protobuf files
    kernel_tb_files = find_tb_files(trace_dir)
    print(f"  Found {sum(len(v) for v in kernel_tb_files.values())} TB files "
          f"across {len(kernel_tb_files)} kernel directories")

    kernel_tb_features = {}
    for kernel_dir, tb_files in sorted(kernel_tb_files.items()):
        print(f"    Parsing {kernel_dir}: {len(tb_files)} TB files...", end="")
        format_counts, all_features = parse_kernel_threadblocks(tb_files)
        summary = aggregate_kernel_tb_features(all_features, format_counts)
        kernel_tb_features[kernel_dir] = summary
        print(f" formats: {dict(format_counts)}")

    result["threadblock_features"] = kernel_tb_features

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract compression-level features from GPU trace files.")
    parser.add_argument("--trace-dir", required=True,
                        help="Path to the trace directory")
    parser.add_argument("--output", required=True,
                        help="Output JSON file path")
    args = parser.parse_args()

    if not os.path.isdir(args.trace_dir):
        print(f"ERROR: Trace directory not found: {args.trace_dir}")
        sys.exit(1)

    features = extract_features(args.trace_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(features, f, indent=2)

    print(f"\nFeatures written to: {args.output}")


if __name__ == "__main__":
    main()
