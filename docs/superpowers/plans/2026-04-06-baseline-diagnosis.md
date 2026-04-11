# Baseline Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate whether AI can produce useful architecture diagnoses from existing GPU trace compression features + Nsight Compute hardware stats, and identify blind spots for subsequent semantic enhancement.

**Architecture:** A Python feature extraction pipeline reads protobuf trace files and Nsight Compute CSV output, produces structured JSON feature packages, which are then fed to an LLM for architecture diagnosis. The pipeline runs entirely offline, post-simulation.

**Tech Stack:** Python 3, protobuf (7.34.1), protoc (3.21.12), Nsight Compute CLI (2025.1.0), JSON

---

## File Structure

```
experiments/baseline_diagnosis/
  proto_gen/                     # Generated Python protobuf modules
  extract_trace_features.py      # Reads .pb trace files, outputs compression features JSON
  parse_ncu_metrics.py           # Reads Nsight Compute CSV, outputs hardware stats JSON
  merge_features.py              # Merges trace features + hardware stats into one JSON
  diagnosis_prompt.md            # System prompt template for AI diagnosis
  run_ncu_microbench.sh          # Nsight Compute profiling script for microbenchmarks
  run_ncu_gpt2.sh                # Nsight Compute profiling script for GPT-2
  results/                       # Output directory for feature JSONs and diagnosis reports
    microbench/
    gpt2/
  evaluation_template.md         # Template for human evaluation of AI diagnosis
```

---

### Task 1: Generate Python Protobuf Modules

**Files:**
- Create: `experiments/baseline_diagnosis/compile_proto.sh`
- Create: `experiments/baseline_diagnosis/proto_gen/` (generated)

- [ ] **Step 1: Write the proto compilation script**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROTO_DIR="$SCRIPT_DIR/../../simulator-remodeled/util/traces_enhanced/dynamic_trace"
OUT_DIR="$SCRIPT_DIR/proto_gen"

mkdir -p "$OUT_DIR"

protoc \
  --proto_path="$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  "$PROTO_DIR"/trace.proto \
  "$PROTO_DIR"/gpu_device.proto \
  "$PROTO_DIR"/cuda_stream.proto \
  "$PROTO_DIR"/kernel.proto \
  "$PROTO_DIR"/threadblock.proto \
  "$PROTO_DIR"/warp.proto \
  "$PROTO_DIR"/instruction.proto \
  "$PROTO_DIR"/compressed_threadblock.proto \
  "$PROTO_DIR"/compressed_instruction.proto \
  "$PROTO_DIR"/address.proto \
  "$PROTO_DIR"/dim3d.proto

touch "$OUT_DIR/__init__.py"
echo "Proto compilation done. Output in $OUT_DIR"
```

- [ ] **Step 2: Run proto compilation and verify**

Run: `cd /home/dyf/modern-gpu-simulator-micro-2025 && bash experiments/baseline_diagnosis/compile_proto.sh`

Expected: `Proto compilation done.` and Python `_pb2.py` files in `proto_gen/`.

Verify: `ls experiments/baseline_diagnosis/proto_gen/*_pb2.py | wc -l`

Expected: 11 files (one per .proto)

- [ ] **Step 3: Verify proto modules are importable**

Run:
```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
python3 -c "
import sys
sys.path.insert(0, 'experiments/baseline_diagnosis')
from proto_gen import trace_pb2, compressed_threadblock_pb2
print('trace_pb2 fields:', [f.name for f in trace_pb2.Trace.DESCRIPTOR.fields])
print('compressed_kernel_v8 fields:', [f.name for f in compressed_threadblock_pb2.compressed_kernel_v8.DESCRIPTOR.fields])
"
```

Expected: field names printed without import errors.

- [ ] **Step 4: Commit**

```bash
git add experiments/baseline_diagnosis/compile_proto.sh
git add experiments/baseline_diagnosis/proto_gen/
git commit -m "add proto compilation script for baseline diagnosis"
```

---

### Task 2: Trace Feature Extraction Script

**Files:**
- Create: `experiments/baseline_diagnosis/extract_trace_features.py`

This script reads the protobuf trace files for a given workload and extracts
compression-level features. It handles both compressed (v7/v8) and
uncompressed threadblock formats.

- [ ] **Step 1: Write the feature extraction script**

```python
#!/usr/bin/env python3
"""Extract compression features from GPU trace protobuf files.

Usage:
    python extract_trace_features.py --trace-dir <path-to-traces-dir> --output <output.json>

The trace-dir should contain:
    dynamic_trace.pb
    extra_info/enhanced_execution_info.json
    threadblocks/device_*/stream_*/kernel_*/*.pb
"""
import argparse
import glob
import json
import os
import struct
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from proto_gen import (
    trace_pb2,
    compressed_threadblock_pb2 as ctb_pb2,
    threadblock_pb2 as tb_pb2,
)


def read_pb_file(path, msg_type):
    """Read a protobuf file. Try msg_type first; return None on parse failure."""
    with open(path, "rb") as f:
        data = f.read()
    msg = msg_type()
    try:
        msg.ParseFromString(data)
        return msg
    except Exception:
        return None


def try_parse_threadblock(path):
    """Try parsing a threadblock .pb file in multiple formats.

    Returns (format_name, parsed_message) or ("unknown", None).
    """
    for fmt, msg_cls in [
        ("compressed_kernel_v8", ctb_pb2.compressed_kernel_v8),
        ("compressed_threadblock_v7", ctb_pb2.compressed_threadblock_v7),
        ("compressed_threadblock_v6", ctb_pb2.compressed_threadblock_v6),
        ("compressed_threadblock", ctb_pb2.compressed_threadblock),
        ("threadblock", tb_pb2.threadblock),
    ]:
        msg = read_pb_file(path, msg_cls)
        if msg is not None and msg.ByteSize() > 0:
            return fmt, msg
    return "unknown", None


def extract_v8_features(msg):
    """Extract compression features from a compressed_kernel_v8 message."""
    base_tb = msg.base_threadblock
    deltas = list(msg.delta_threadblocks)
    total_tbs = 1 + len(deltas)

    # Cross-TB delta analysis
    full_encoding_count = sum(1 for d in deltas if d.is_full_encoding)
    offset_only_count = sum(
        1 for d in deltas
        if not d.is_full_encoding and len(d.address_overrides) == 0
    )
    total_overrides = sum(len(d.address_overrides) for d in deltas)

    # Shared PC sequence from base threadblock
    shared_pc_len = len(base_tb.shared_pc_sequence)

    # Warp diff analysis from base threadblock
    warp_diff_counts = []
    for warp_id, warp in base_tb.warps.items():
        diff_count = len(warp.instructions) + len(warp.pc_overrides)
        warp_diff_counts.append(diff_count)

    warp_diff_arr = np.array(warp_diff_counts) if warp_diff_counts else np.array([0])

    return {
        "format": "compressed_kernel_v8",
        "total_threadblocks": total_tbs,
        "shared_pc_sequence_length": shared_pc_len,
        "cross_tb_offset_coverage": offset_only_count / max(len(deltas), 1),
        "address_override_density": total_overrides / max(len(deltas), 1),
        "full_encoding_fallback_rate": full_encoding_count / max(len(deltas), 1),
        "warp_diff_distribution": {
            "count": len(warp_diff_counts),
            "mean": float(warp_diff_arr.mean()),
            "std": float(warp_diff_arr.std()),
            "min": int(warp_diff_arr.min()),
            "p50": float(np.percentile(warp_diff_arr, 50)),
            "p95": float(np.percentile(warp_diff_arr, 95)),
            "max": int(warp_diff_arr.max()),
        },
    }


def extract_v7_features(msg):
    """Extract compression features from a compressed_threadblock_v7 message."""
    shared_pc_len = len(msg.shared_pc_sequence)
    warp_diff_counts = []
    for warp_id, warp in msg.warps.items():
        diff_count = len(warp.instructions) + len(warp.pc_overrides)
        warp_diff_counts.append(diff_count)

    warp_diff_arr = np.array(warp_diff_counts) if warp_diff_counts else np.array([0])

    return {
        "format": "compressed_threadblock_v7",
        "total_threadblocks": 1,
        "shared_pc_sequence_length": shared_pc_len,
        "warp_diff_distribution": {
            "count": len(warp_diff_counts),
            "mean": float(warp_diff_arr.mean()),
            "std": float(warp_diff_arr.std()),
            "min": int(warp_diff_arr.min()),
            "p50": float(np.percentile(warp_diff_arr, 50)),
            "p95": float(np.percentile(warp_diff_arr, 95)),
            "max": int(warp_diff_arr.max()),
        },
    }


def extract_v6_features(msg):
    """Extract features from compressed_threadblock_v6 (has RLE runs)."""
    rle_counts = []
    non_rle_counts = []
    for warp_id, warp in msg.warps.items():
        rle_counts.append(sum(r.count for r in warp.runs))
        non_rle_counts.append(len(warp.instructions))

    total_rle = sum(rle_counts)
    total_non_rle = sum(non_rle_counts)
    total = total_rle + total_non_rle

    run_lengths = []
    for warp_id, warp in msg.warps.items():
        for r in warp.runs:
            run_lengths.append(r.count)

    run_arr = np.array(run_lengths) if run_lengths else np.array([0])

    return {
        "format": "compressed_threadblock_v6",
        "total_threadblocks": 1,
        "rle_coverage": total_rle / max(total, 1),
        "rle_length_distribution": {
            "count": len(run_lengths),
            "mean": float(run_arr.mean()),
            "std": float(run_arr.std()),
            "min": int(run_arr.min()),
            "p50": float(np.percentile(run_arr, 50)),
            "p95": float(np.percentile(run_arr, 95)),
            "max": int(run_arr.max()),
        },
    }


def extract_uncompressed_features(msg):
    """Extract basic features from an uncompressed threadblock."""
    total_insts = 0
    warp_inst_counts = []
    for warp_id, warp in msg.warps.items():
        n = len(warp.instructions)
        warp_inst_counts.append(n)
        total_insts += n

    return {
        "format": "uncompressed",
        "total_threadblocks": 1,
        "total_instructions": total_insts,
        "num_warps": len(warp_inst_counts),
    }


def extract_static_metadata(json_path):
    """Extract summary from enhanced_execution_info.json."""
    with open(json_path) as f:
        data = json.load(f)

    kernels_info = []
    for kernel in data.get("kernels", []):
        instructions = kernel.get("instructions", [])
        opcodes = [inst.get("op_code", "") for inst in instructions]
        opcode_counts = {}
        for op in opcodes:
            opcode_counts[op] = opcode_counts.get(op, 0) + 1

        # Control bits summary
        stall_counts = []
        barrier_waits = 0
        yields = 0
        for inst in instructions:
            cb = inst.get("control_bits", {})
            stall_counts.append(cb.get("stall_count", 0))
            if cb.get("wait_barrier_bits", 0) != 0:
                barrier_waits += 1
            if cb.get("is_yield", False):
                yields += 1

        stall_arr = np.array(stall_counts) if stall_counts else np.array([0])

        kernels_info.append({
            "kernel_name": kernel.get("kernel_name", "unknown"),
            "unique_function_id": kernel.get("unique_function_id", -1),
            "total_static_instructions": len(instructions),
            "top_opcodes": dict(sorted(opcode_counts.items(), key=lambda x: -x[1])[:10]),
            "control_bits_summary": {
                "stall_count_mean": float(stall_arr.mean()),
                "stall_count_max": int(stall_arr.max()),
                "barrier_wait_instructions": barrier_waits,
                "yield_instructions": yields,
            },
        })

    return {"static_metadata": kernels_info}


def extract_trace_info(dynamic_trace_path):
    """Extract high-level trace info from dynamic_trace.pb."""
    msg = read_pb_file(dynamic_trace_path, trace_pb2.Trace)
    if msg is None:
        return {}

    info = {
        "trace_name": msg.name,
        "binary_version": msg.binary_version,
        "accelsim_version": msg.accelsim_version,
    }

    kernels = []
    for dev_id, dev in msg.gpu_device.items():
        for stream_id, stream in dev.streams.items():
            for kernel in stream.kernels:
                kernels.append({
                    "kernel_id": kernel.id,
                    "kernel_name": kernel.name,
                    "grid_dim": f"{kernel.grid_dim.x}x{kernel.grid_dim.y}x{kernel.grid_dim.z}",
                    "block_dim": f"{kernel.block_dim.x}x{kernel.block_dim.y}x{kernel.block_dim.z}",
                    "shared_memory": kernel.size_shared_memory,
                    "registers": kernel.number_of_registers,
                })
    info["kernels"] = kernels
    return info


def main():
    parser = argparse.ArgumentParser(description="Extract trace compression features")
    parser.add_argument("--trace-dir", required=True, help="Path to traces/ directory")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    trace_dir = args.trace_dir
    result = {}

    # 1. Parse dynamic_trace.pb
    dt_path = os.path.join(trace_dir, "dynamic_trace.pb")
    if os.path.exists(dt_path):
        result["trace_info"] = extract_trace_info(dt_path)

    # 2. Parse enhanced_execution_info.json
    eei_path = os.path.join(trace_dir, "extra_info", "enhanced_execution_info.json")
    if os.path.exists(eei_path):
        result.update(extract_static_metadata(eei_path))

    # 3. Parse threadblock .pb files
    tb_pattern = os.path.join(trace_dir, "threadblocks", "device_*", "stream_*", "kernel_*", "*.pb")
    tb_files = sorted(glob.glob(tb_pattern))

    kernel_features = {}
    for tb_path in tb_files:
        # Extract kernel id from path: .../kernel_N/file.pb
        kernel_dir = os.path.basename(os.path.dirname(tb_path))

        fmt, msg = try_parse_threadblock(tb_path)
        if msg is None:
            continue

        if fmt == "compressed_kernel_v8":
            features = extract_v8_features(msg)
        elif fmt == "compressed_threadblock_v7":
            features = extract_v7_features(msg)
        elif fmt == "compressed_threadblock_v6":
            features = extract_v6_features(msg)
        elif fmt == "threadblock":
            features = extract_uncompressed_features(msg)
        else:
            continue

        if kernel_dir not in kernel_features:
            kernel_features[kernel_dir] = features
        else:
            # Multiple TB files for same kernel: aggregate
            existing = kernel_features[kernel_dir]
            existing["total_threadblocks"] = existing.get("total_threadblocks", 0) + features.get("total_threadblocks", 0)

    result["kernel_compression_features"] = kernel_features

    # 4. Parse stats.csv if present
    stats_path = os.path.join(trace_dir, "stats.csv")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            lines = f.readlines()
        if len(lines) >= 2:
            headers = [h.strip() for h in lines[0].split(",")]
            values = [v.strip() for v in lines[1].split(",")]
            result["trace_stats"] = dict(zip(headers, values))

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Features written to {args.output}")
    print(f"  Kernels found: {len(kernel_features)}")
    for k, v in kernel_features.items():
        print(f"    {k}: format={v.get('format', '?')}, TBs={v.get('total_threadblocks', '?')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test on l1_bw_32f microbench trace (single TB)**

Run:
```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
python3 experiments/baseline_diagnosis/extract_trace_features.py \
  --trace-dir hw_run/traces/device-0/12.8/benchstudy-20260403/l1_bw_32f/traces \
  --output experiments/baseline_diagnosis/results/microbench/l1_bw_32f.json
```

Expected: JSON file written, 1 kernel found. Inspect the output to verify
feature values are reasonable.

- [ ] **Step 3: Test on mem_bw microbench trace (160 TBs)**

Run:
```bash
python3 experiments/baseline_diagnosis/extract_trace_features.py \
  --trace-dir hw_run/traces/device-0/12.8/benchstudy-20260403/mem_bw/traces \
  --output experiments/baseline_diagnosis/results/microbench/mem_bw.json
```

Expected: JSON file written, 1 kernel, 160 TBs. If format is v8, expect
cross-TB delta features.

- [ ] **Step 4: Run extraction on all microbenchmarks**

```bash
BENCH_ROOT=hw_run/traces/device-0/12.8/benchstudy-20260403
OUT_ROOT=experiments/baseline_diagnosis/results/microbench

for bench in "$BENCH_ROOT"/*/; do
  name=$(basename "$bench")
  python3 experiments/baseline_diagnosis/extract_trace_features.py \
    --trace-dir "$bench/traces" \
    --output "$OUT_ROOT/${name}.json"
done
```

Expected: One JSON per microbenchmark in results/microbench/.

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline_diagnosis/extract_trace_features.py
git add experiments/baseline_diagnosis/results/microbench/
git commit -m "add trace feature extraction script with microbench results"
```

---

### Task 3: Nsight Compute Profiling Script for Microbenchmarks

**Files:**
- Create: `experiments/baseline_diagnosis/run_ncu_microbench.sh`
- Create: `experiments/baseline_diagnosis/parse_ncu_metrics.py`

- [ ] **Step 1: Write the Nsight Compute profiling script**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Profile microbenchmarks with Nsight Compute to collect hardware stats.
# Prerequisites: microbench binaries must be available.
#
# Usage: bash run_ncu_microbench.sh <microbench_binary> <output_csv>

BINARY=${1:?Usage: run_ncu_microbench.sh <binary> <output_csv>}
OUTPUT=${2:?Usage: run_ncu_microbench.sh <binary> <output_csv>}

ncu --set full \
    --csv \
    --target-processes all \
    "$BINARY" \
    > "$OUTPUT" 2>/dev/null

echo "NCU profiling done: $OUTPUT"
```

- [ ] **Step 2: Write the NCU CSV parser**

```python
#!/usr/bin/env python3
"""Parse Nsight Compute CSV output into structured JSON hardware stats.

Usage:
    python parse_ncu_metrics.py --input <ncu_output.csv> --output <stats.json>

If no NCU CSV is available, can also accept manually provided metrics via --manual flag.
"""
import argparse
import csv
import json
import os
import sys


# Metrics we care about and their friendly names
METRIC_MAP = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_utilization",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "smsp__average_inst_executed_pipe_lsu_pred_on.pct": "memory_pipe_utilization",
    "smsp__average_inst_executed_pipe_alu_pred_on.pct": "alu_pipe_utilization",
    "smsp__average_inst_executed_pipe_fma_pred_on.pct": "fma_pipe_utilization",
    "dram__bytes_read.sum": "dram_bytes_read",
    "dram__bytes_write.sum": "dram_bytes_write",
    "sm__inst_executed.avg.per_cycle_active": "ipc",
    "l1tex__data_bank_conflicts_pipe_lsu.sum": "l1_bank_conflicts",
    "sm__sass_thread_inst_executed_op_branch_pred_on.sum": "branch_instructions",
    "sm__sass_thread_inst_executed_op_branch_pred_on_diverged.sum": "divergent_branches",
}


def parse_ncu_csv(csv_path):
    """Parse NCU CSV output and extract relevant metrics per kernel."""
    kernels = {}

    with open(csv_path) as f:
        # Skip NCU header lines (start with ==)
        lines = []
        for line in f:
            if not line.startswith("==") and line.strip():
                lines.append(line)

    if not lines:
        return kernels

    reader = csv.DictReader(lines)
    for row in reader:
        kernel_name = row.get("Kernel Name", "unknown")
        metric_name = row.get("Metric Name", "")
        metric_value = row.get("Metric Value", "")

        if metric_name in METRIC_MAP:
            friendly_name = METRIC_MAP[metric_name]
            if kernel_name not in kernels:
                kernels[kernel_name] = {}
            try:
                kernels[kernel_name][friendly_name] = float(metric_value.replace(",", ""))
            except ValueError:
                kernels[kernel_name][friendly_name] = metric_value

    # Compute derived metrics
    for kname, metrics in kernels.items():
        if "l1_hit_rate" in metrics:
            metrics["l1_miss_rate"] = round(1.0 - metrics["l1_hit_rate"] / 100.0, 4)
        if "l2_hit_rate" in metrics:
            metrics["l2_miss_rate"] = round(1.0 - metrics["l2_hit_rate"] / 100.0, 4)
        if "branch_instructions" in metrics and "divergent_branches" in metrics:
            total = metrics["branch_instructions"]
            if total > 0:
                metrics["warp_divergence_rate"] = round(
                    metrics["divergent_branches"] / total, 4
                )

    return kernels


def main():
    parser = argparse.ArgumentParser(description="Parse NCU CSV to JSON")
    parser.add_argument("--input", required=True, help="NCU CSV file path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    kernels = parse_ncu_csv(args.input)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"hardware_stats": kernels}, f, indent=2)

    print(f"Hardware stats written to {args.output}")
    for kname, metrics in kernels.items():
        print(f"  {kname}: {len(metrics)} metrics")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test NCU profiling on one microbench**

Run (adjust binary path to your actual microbench binary):
```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
# Identify a runnable microbench binary first:
ls simulator-remodeled/gpu-app-collection/bin/

# Then profile it:
bash experiments/baseline_diagnosis/run_ncu_microbench.sh \
  <path-to-l1_bw_32f-binary> \
  experiments/baseline_diagnosis/results/microbench/l1_bw_32f_ncu.csv
```

If no pre-built binary is immediately available, create a manual metrics
JSON for l1_bw_32f based on expected behavior:

```bash
cat > experiments/baseline_diagnosis/results/microbench/l1_bw_32f_hw.json << 'EOF'
{
  "hardware_stats": {
    "_Z5l1_bwPjS_PfS0_": {
      "compute_utilization": 15.0,
      "memory_throughput_pct": 85.0,
      "l1_miss_rate": 0.05,
      "l2_miss_rate": 0.02,
      "occupancy_pct": 50.0,
      "ipc": 12.0,
      "warp_divergence_rate": 0.0,
      "note": "manually estimated for l1 bandwidth benchmark"
    }
  }
}
EOF
```

- [ ] **Step 4: Test NCU CSV parser**

Run:
```bash
python3 experiments/baseline_diagnosis/parse_ncu_metrics.py \
  --input experiments/baseline_diagnosis/results/microbench/l1_bw_32f_ncu.csv \
  --output experiments/baseline_diagnosis/results/microbench/l1_bw_32f_hw.json
```

Expected: JSON file with per-kernel hardware stats.

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline_diagnosis/run_ncu_microbench.sh
git add experiments/baseline_diagnosis/parse_ncu_metrics.py
git add experiments/baseline_diagnosis/results/microbench/
git commit -m "add Nsight Compute profiling and parsing scripts"
```

---

### Task 4: Feature Merge Script

**Files:**
- Create: `experiments/baseline_diagnosis/merge_features.py`

- [ ] **Step 1: Write the merge script**

```python
#!/usr/bin/env python3
"""Merge trace compression features and hardware stats into a single JSON.

Usage:
    python merge_features.py \
        --trace-features <trace.json> \
        --hw-stats <hw.json> \
        --output <merged.json>
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Merge feature sources")
    parser.add_argument("--trace-features", required=True)
    parser.add_argument("--hw-stats", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.trace_features) as f:
        trace = json.load(f)
    with open(args.hw_stats) as f:
        hw = json.load(f)

    merged = {
        "trace_info": trace.get("trace_info", {}),
        "static_metadata": trace.get("static_metadata", []),
        "trace_stats": trace.get("trace_stats", {}),
        "kernel_compression_features": trace.get("kernel_compression_features", {}),
        "hardware_stats": hw.get("hardware_stats", {}),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged features written to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test merge on l1_bw_32f**

Run:
```bash
python3 experiments/baseline_diagnosis/merge_features.py \
  --trace-features experiments/baseline_diagnosis/results/microbench/l1_bw_32f.json \
  --hw-stats experiments/baseline_diagnosis/results/microbench/l1_bw_32f_hw.json \
  --output experiments/baseline_diagnosis/results/microbench/l1_bw_32f_merged.json
```

Expected: merged JSON containing both trace compression features and
hardware stats.

- [ ] **Step 3: Commit**

```bash
git add experiments/baseline_diagnosis/merge_features.py
git commit -m "add feature merge script"
```

---

### Task 5: AI Diagnosis Prompt Template

**Files:**
- Create: `experiments/baseline_diagnosis/diagnosis_prompt.md`

- [ ] **Step 1: Write the prompt template**

```markdown
# GPU Architecture Diagnosis System Prompt

You are a GPU architecture analyst. You will receive structured feature data
from a GPU workload's execution trace and hardware profiling. Your task is
to produce a diagnostic report identifying architectural issues and
optimization opportunities.

## Background Knowledge

**GPU SM Structure:**
- Modern NVIDIA GPUs have multiple Streaming Multiprocessors (SMs)
- Each SM contains sub-cores with warp schedulers, register files, and
  execution pipelines
- Warps are groups of 32 threads executing in SIMT fashion
- Threadblocks (TBs) are groups of warps assigned to an SM

**Memory Hierarchy:**
- Registers > Shared Memory > L1 Cache > L2 Cache > HBM (device memory)
- L1 and Shared Memory share the same on-chip SRAM
- Memory coalescing: threads in a warp accessing contiguous addresses
  can be served in fewer transactions

**Trace Compression Features — What They Mean:**
- `rle_coverage`: fraction of instructions compressible by run-length
  encoding. High = regular loop-heavy computation. Low = irregular control
  flow.
- `cross_tb_offset_coverage`: fraction of threadblocks whose memory
  addresses differ from the base TB only by a constant offset. High = very
  regular data parallelism. Low = data-dependent access patterns.
- `address_override_density`: average number of per-TB address overrides
  needed. High = significant data-dependent address divergence.
- `warp_diff_distribution`: how many instructions each warp differs from
  the shared PC sequence. Low mean = high SIMT convergence. High max with
  low mean = one or few outlier warps with divergent control flow.
- `shared_pc_sequence_length`: length of the PC sequence shared across all
  warps. Longer = more instruction-level uniformity.
- `full_encoding_fallback_rate`: fraction of TBs that cannot be
  delta-encoded at all. High = some TBs have fundamentally different
  behavior.

**Hardware Stats — What They Mean:**
- `compute_utilization`: percentage of peak compute throughput achieved
- `memory_throughput_pct`: percentage of peak memory bandwidth achieved
- `l1_miss_rate` / `l2_miss_rate`: cache miss rates
- `occupancy_pct`: percentage of peak warp occupancy achieved
- `ipc`: instructions executed per cycle per SM
- `warp_divergence_rate`: fraction of branch instructions where threads
  within a warp take different paths

## Your Task

Analyze the provided feature data and produce a diagnostic report with
exactly these four sections:

### 1. Behavioral Summary
- Classify the workload: compute-bound, memory-bound, or mixed
- Identify the dominant behavioral characteristics

### 2. Anomaly Findings
- Identify metrics that deviate from expected patterns or are inconsistent
  with each other
- Rate each anomaly: HIGH / MEDIUM / LOW severity

### 3. Causal Hypotheses
- For each anomaly, propose the most likely root cause
- Explain your reasoning
- Rate confidence: HIGH / MEDIUM / LOW

### 4. Suggested Exploration Directions
- List architecture dimensions worth investigating
- For each, explain what change you would test and what improvement you
  would expect

## Important Rules
- Do not restate raw numbers without interpretation
- Focus on cross-feature correlations, not single-metric observations
- Flag contradictions between features (e.g., high regularity in
  compression but high cache miss in hardware)
- Be specific: "L1 miss is high" is not useful; "L1 miss rate of 0.32
  combined with cross_tb_offset_coverage of 0.92 suggests that while
  inter-TB access patterns are regular, intra-TB locality is poor" is useful

---

## Feature Data

[INSERT MERGED JSON HERE]
```

- [ ] **Step 2: Commit**

```bash
git add experiments/baseline_diagnosis/diagnosis_prompt.md
git commit -m "add AI diagnosis prompt template"
```

---

### Task 6: Evaluation Template

**Files:**
- Create: `experiments/baseline_diagnosis/evaluation_template.md`

- [ ] **Step 1: Write the evaluation template**

```markdown
# AI Diagnosis Evaluation

## Workload: [name]
## Date: [date]

## Findings Evaluation

| # | AI Finding (summary) | Category | Notes |
|---|---------------------|----------|-------|
| 1 | | correct-nontrivial / correct-trivial / wrong / blind-spot | |
| 2 | | | |
| 3 | | | |

## Blind Spots (issues AI missed)

| # | Known Issue | Why AI Missed It | What Feature Would Help |
|---|------------|-----------------|----------------------|
| 1 | | | |
| 2 | | | |

## Summary Metrics

- Total findings: __
- Correct and non-trivial: __
- Correct but trivial: __
- Wrong: __
- Blind spots identified: __
- Diagnostic value score (1-5): __

## Conclusion

[Does this justify proceeding to Step 2 (semantic enhancement)?]
```

- [ ] **Step 2: Commit**

```bash
git add experiments/baseline_diagnosis/evaluation_template.md
git commit -m "add diagnosis evaluation template"
```

---

### Task 7: GPT-2 Trace Generation

**Files:**
- Modify: `experiments/gpt2_decode/run_trace.sh` (no changes needed,
  already correct)

This task requires running on the actual 5090 hardware.

- [ ] **Step 1: Verify trace_gen conda environment**

Run:
```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate trace_gen
python -c "import torch; from transformers import AutoModelForCausalLM; print('OK')"
```

Expected: `OK`

- [ ] **Step 2: Verify tracer_tool.so exists**

Run:
```bash
ls -la /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.so
```

Expected: file exists. If not, build it first:
```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit/
```

- [ ] **Step 3: Run GPT-2 trace generation (start with single config)**

Run:
```bash
cd /home/dyf/modern-gpu-simulator-micro-2025
CONTEXTS="128" RUNS="1" bash experiments/gpt2_decode/run_trace.sh
```

Expected: trace files written to
`experiments/gpt2_decode/results/model_gpt2_ctx128_gen1_run1/traces/`

- [ ] **Step 4: Verify trace output structure**

Run:
```bash
ls -R experiments/gpt2_decode/results/model_gpt2_ctx128_gen1_run1/traces/ | head -30
```

Expected: `dynamic_trace.pb`, `extra_info/`, `threadblocks/` with
kernel subdirectories.

- [ ] **Step 5: Extract trace features for GPT-2**

Run:
```bash
python3 experiments/baseline_diagnosis/extract_trace_features.py \
  --trace-dir experiments/gpt2_decode/results/model_gpt2_ctx128_gen1_run1/traces \
  --output experiments/baseline_diagnosis/results/gpt2/ctx128.json
```

Expected: JSON with multiple kernels (GPT-2 decode involves GEMM,
attention, layernorm, etc.)

- [ ] **Step 6: Run remaining context lengths**

Run:
```bash
CONTEXTS="512 1024" RUNS="1" bash experiments/gpt2_decode/run_trace.sh
```

Then extract features for each:
```bash
for ctx in 512 1024; do
  python3 experiments/baseline_diagnosis/extract_trace_features.py \
    --trace-dir experiments/gpt2_decode/results/model_gpt2_ctx${ctx}_gen1_run1/traces \
    --output experiments/baseline_diagnosis/results/gpt2/ctx${ctx}.json
done
```

- [ ] **Step 7: Commit**

```bash
git add experiments/baseline_diagnosis/results/gpt2/
git commit -m "add GPT-2 trace features for ctx 128/512/1024"
```

---

### Task 8: Nsight Compute Profiling for GPT-2

**Files:**
- Create: `experiments/baseline_diagnosis/run_ncu_gpt2.sh`

- [ ] **Step 1: Write the GPT-2 NCU profiling script**

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/dyf/modern-gpu-simulator-micro-2025
OUT_ROOT=$PROJECT_ROOT/experiments/baseline_diagnosis/results/gpt2
CONTEXTS=${CONTEXTS:-"128 512 1024"}

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate trace_gen

mkdir -p "$OUT_ROOT"

for CTX in $CONTEXTS; do
  echo "Profiling GPT-2 ctx=$CTX..."
  ncu --set full \
      --csv \
      --target-processes all \
      python $PROJECT_ROOT/experiments/gpt2_decode/run_decode.py \
        --model gpt2 \
        --context-len "$CTX" \
        --gen-tokens 1 \
      > "$OUT_ROOT/ctx${CTX}_ncu.csv" 2>/dev/null

  python3 $PROJECT_ROOT/experiments/baseline_diagnosis/parse_ncu_metrics.py \
    --input "$OUT_ROOT/ctx${CTX}_ncu.csv" \
    --output "$OUT_ROOT/ctx${CTX}_hw.json"

  python3 $PROJECT_ROOT/experiments/baseline_diagnosis/merge_features.py \
    --trace-features "$OUT_ROOT/ctx${CTX}.json" \
    --hw-stats "$OUT_ROOT/ctx${CTX}_hw.json" \
    --output "$OUT_ROOT/ctx${CTX}_merged.json"

  echo "Done: $OUT_ROOT/ctx${CTX}_merged.json"
done
```

- [ ] **Step 2: Run NCU profiling for ctx=128 first**

Run:
```bash
CONTEXTS="128" bash experiments/baseline_diagnosis/run_ncu_gpt2.sh
```

Expected: `ctx128_merged.json` containing both trace features and
hardware stats.

Note: NCU profiling is slow (may take several minutes per context length).
Start with ctx=128 to validate the pipeline, then run 512/1024.

- [ ] **Step 3: Verify merged output**

Run:
```bash
python3 -c "
import json
with open('experiments/baseline_diagnosis/results/gpt2/ctx128_merged.json') as f:
    d = json.load(f)
print('Keys:', list(d.keys()))
print('Kernels in compression:', list(d.get('kernel_compression_features', {}).keys()))
print('Kernels in hw stats:', list(d.get('hardware_stats', {}).keys()))
"
```

Expected: both `kernel_compression_features` and `hardware_stats` populated
with matching kernel names.

- [ ] **Step 4: Commit**

```bash
git add experiments/baseline_diagnosis/run_ncu_gpt2.sh
git add experiments/baseline_diagnosis/results/gpt2/
git commit -m "add GPT-2 NCU profiling pipeline and merged features"
```

---

### Task 9: Run Sanity Check Diagnosis (Microbenchmarks)

This task is performed manually in conversation with the AI.

- [ ] **Step 1: Select 3 representative microbenchmarks**

Choose microbenchmarks with known, distinct characteristics:
- `l1_bw_32f`: L1 bandwidth bound, minimal cache misses
- `mem_bw`: HBM bandwidth bound, high L1/L2 miss rates
- `MaxFlops`: compute bound, minimal memory pressure

- [ ] **Step 2: Prepare merged feature JSONs**

For each selected microbenchmark, ensure you have a merged JSON
(trace features + hardware stats). If NCU data is not available,
use manually estimated hardware stats based on known behavior.

- [ ] **Step 3: Feed each JSON to AI using diagnosis_prompt.md**

Copy the system prompt from `diagnosis_prompt.md`, replace
`[INSERT MERGED JSON HERE]` with the merged JSON content,
and submit to the AI in conversation.

- [ ] **Step 4: Evaluate each diagnosis using evaluation_template.md**

For each microbenchmark, fill in the evaluation template.
Key question: did the AI correctly identify the dominant bottleneck
(L1 BW / HBM BW / compute)?

- [ ] **Step 5: Save evaluation results**

Save completed evaluations to:
```
experiments/baseline_diagnosis/results/microbench/eval_l1_bw_32f.md
experiments/baseline_diagnosis/results/microbench/eval_mem_bw.md
experiments/baseline_diagnosis/results/microbench/eval_MaxFlops.md
```

- [ ] **Step 6: Commit**

```bash
git add experiments/baseline_diagnosis/results/microbench/eval_*.md
git commit -m "add microbench sanity check evaluation results"
```

---

### Task 10: Run Formal Diagnosis (GPT-2)

- [ ] **Step 1: Feed ctx128 merged JSON to AI diagnosis**

Use the same process as Task 9, but with GPT-2 data.

- [ ] **Step 2: Evaluate the diagnosis**

Fill in evaluation template. Pay special attention to:
- Does AI identify which kernels dominate execution time?
- Does AI notice differences between kernel types (GEMM vs attention
  vs layernorm)?
- Does AI find any cross-kernel patterns?

- [ ] **Step 3: Compare across context lengths**

Feed ctx512 and ctx1024 to AI. Evaluate whether AI notices behavioral
changes as context length increases (e.g., attention becoming more
memory-bound at longer sequences).

- [ ] **Step 4: Compile blind spot analysis**

Across all evaluations, compile a list of blind spots:

```markdown
# Blind Spot Summary

| # | Blind Spot | Affected Workloads | Feature Category Needed |
|---|-----------|-------------------|----------------------|
| 1 | ... | ... | behavioral segmentation / anomaly detection / causal chain |
```

- [ ] **Step 5: Save all results and commit**

```bash
git add experiments/baseline_diagnosis/results/gpt2/eval_*.md
git add experiments/baseline_diagnosis/results/blind_spots.md
git commit -m "add GPT-2 diagnosis evaluation and blind spot analysis"
```

---

## Decision Gate

After Task 10, review the blind spot analysis to determine next steps:

- If **correct non-trivial findings >= 2** AND **identifiable blind spots
  exist**: proceed to Step 2 (Squash/Delta semantic enhancement).
  The blind spots define what to build.

- If **correct non-trivial findings == 0**: the approach needs
  rethinking. Consider whether the feature extraction is too lossy,
  the prompt needs improvement, or the fundamental premise is flawed.

- If **no blind spots identified** (AI found everything): the existing
  trace features are sufficient. Skip Step 2 compression work and
  proceed directly to Step 3 (closed-loop evaluation).
