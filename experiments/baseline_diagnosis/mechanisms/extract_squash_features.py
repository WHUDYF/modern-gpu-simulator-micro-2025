#!/usr/bin/env python3
"""Squash mechanism: temporal segmentation via sliding-window similarity.

Operates at two levels:
  kernel_level: segment the kernel sequence of a workload
  tb_level: segment the TB sequence within each kernel
"""
import argparse
import json
import math
import sys
from pathlib import Path


def to_feature_vector(features_dict, key_order):
    vec = []
    for key in key_order:
        val = features_dict.get(key, 0)
        if isinstance(val, bool):
            vec.append(1.0 if val else 0.0)
        elif isinstance(val, (int, float)):
            vec.append(float(val))
        else:
            vec.append(0.0)
    return vec


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0 if na == 0 and nb == 0 else 0.0
    return dot / (na * nb)


def segment_sequence(vectors, threshold):
    if len(vectors) == 0:
        return []
    if len(vectors) == 1:
        return [(0, 0)]
    segments = []
    start = 0
    for i in range(1, len(vectors)):
        sim = cosine_similarity(vectors[i - 1], vectors[i])
        if sim < threshold:
            segments.append((start, i - 1))
            start = i
    segments.append((start, len(vectors) - 1))
    return segments


def cohesion_score(vectors):
    if len(vectors) <= 1:
        return 1.0
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(cosine_similarity(vectors[i], vectors[j]))
    return sum(sims) / len(sims) if sims else 1.0


def kernel_summary_to_vector(summary):
    """Convert kernel_summary to a numeric vector for kernel-level comparison.

    Uses opcode ratios when available. Falls back to / augments with NCU
    hardware metrics (compute_throughput_pct, l1_throughput_pct, etc.) when
    they are present in the summary. This ensures the vector captures
    meaningful behavioral differences even when opcode data is missing
    (e.g., when enhanced_execution_info was not generated).
    """
    opcodes = {entry["opcode"].upper(): entry["count"] for entry in summary.get("top_opcodes", [])}
    total_ops = sum(opcodes.values()) or 1

    vec = [
        opcodes.get("FFMA", 0) / total_ops,
        sum(v for k, v in opcodes.items() if k.startswith("DFMA") or k.startswith("DMUL") or k.startswith("DADD")) / total_ops,
        sum(v for k, v in opcodes.items() if "LDG" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "STG" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "LDS" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "STS" in k) / total_ops,
        sum(v for k, v in opcodes.items() if "BAR" in k) / total_ops,
        1.0 if summary.get("uses_fp64") else 0.0,
        1.0 if summary.get("uses_shared_memory") else 0.0,
    ]

    # Augment with NCU hardware metrics when available (normalized to 0-1)
    for metric in ["compute_throughput_pct", "l1_throughput_pct",
                    "l2_throughput_pct", "dram_throughput_pct",
                    "ipc_active", "mem_pipes_busy_pct",
                    "l1_hit_rate_pct", "achieved_occupancy_pct"]:
        val = summary.get(metric)
        if val is not None:
            vec.append(float(val) / 100.0)  # normalize pct to 0-1

    return vec


def dominant_opcodes(summary, top_n=3):
    ops = sorted(summary.get("top_opcodes", []), key=lambda e: -e.get("count", 0))
    return [e["opcode"] for e in ops[:top_n]]


def squash_kernel_level(workload_data, threshold):
    kernels = workload_data["kernels"]
    vectors = [kernel_summary_to_vector(k["kernel_summary"]) for k in kernels]
    segments_idx = segment_sequence(vectors, threshold)

    segments = []
    for seg_id, (start, end) in enumerate(segments_idx):
        seg_vectors = vectors[start : end + 1]
        segments.append({
            "segment_id": seg_id,
            "kernel_range": [start, end],
            "kernel_count": end - start + 1,
            "dominant_opcodes": dominant_opcodes(kernels[start]["kernel_summary"]),
            "cohesion_score": cohesion_score(seg_vectors),
            "representative_kernel": start,
            "behavior_summary": f"kernels {start}..{end}: {kernels[start]['kernel_name']}"
                               + (f" ... {kernels[end]['kernel_name']}" if end > start else ""),
        })

    return {
        "squash_segments": segments,
        "boundary_count": max(0, len(segments) - 1),
        "total_kernels": len(kernels),
    }


def squash_tb_level(workload_data, threshold):
    result = {}
    for kernel in workload_data["kernels"]:
        tbs = kernel.get("per_tb", [])
        if not tbs:
            continue

        first_features = tbs[0]["features"]
        key_order = sorted(first_features.keys())

        vectors = [to_feature_vector(tb["features"], key_order) for tb in tbs]
        segments_idx = segment_sequence(vectors, threshold)

        segments = []
        for seg_id, (start, end) in enumerate(segments_idx):
            seg_vectors = vectors[start : end + 1]
            segments.append({
                "segment_id": seg_id,
                "tb_range": [start, end],
                "tb_count": end - start + 1,
                "cohesion_score": cohesion_score(seg_vectors),
                "representative_tb": start,
                "behavior_summary": f"TBs {start}..{end}",
            })

        result[str(kernel["kernel_id"])] = {
            "squash_segments": segments,
            "boundary_count": max(0, len(segments) - 1),
            "total_tbs": len(tbs),
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Squash mechanism")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workload_data = json.loads(Path(args.input).read_text())
    config = json.loads(Path(args.config).read_text())["squash"]

    kernel_level = squash_kernel_level(
        workload_data, config["kernel_level"]["similarity_threshold"]
    )
    tb_level = squash_tb_level(
        workload_data, config["tb_level"]["similarity_threshold"]
    )

    reuse_hint = {
        "kernel_level_representatives": [
            seg["representative_kernel"] for seg in kernel_level["squash_segments"]
        ],
        "tb_level_representatives": {
            kid: [seg["representative_tb"] for seg in data["squash_segments"]]
            for kid, data in tb_level.items()
        },
    }

    output = {
        "mechanism": "squash",
        "workload": workload_data.get("workload", "unknown"),
        "kernel_level": kernel_level,
        "tb_level": tb_level,
        "_simulation_reuse_hint": reuse_hint,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(
        f"[squash] wrote {out_path} "
        f"(kernel_segments={len(kernel_level['squash_segments'])}, "
        f"kernel_boundaries={kernel_level['boundary_count']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
