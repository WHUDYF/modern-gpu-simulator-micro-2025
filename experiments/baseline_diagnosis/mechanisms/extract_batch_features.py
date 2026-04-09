#!/usr/bin/env python3
"""Batch mechanism: spatial homogeneity clustering via DBSCAN.

Operates at two levels:
  kernel_level: cluster the kernels of a workload
  tb_level: cluster the TBs within each kernel
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def to_matrix(vectors):
    """Convert list of list to numpy matrix, handle empty case."""
    if not vectors:
        return np.zeros((0, 1))
    return np.array(vectors, dtype=float)


def normalize(matrix):
    """Standardize features to zero mean, unit variance."""
    if matrix.shape[0] < 2:
        return matrix
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)


def cluster_with_dbscan(matrix, eps, min_samples):
    """Run DBSCAN, return labels array. -1 indicates outlier."""
    if matrix.shape[0] == 0:
        return np.array([], dtype=int)
    if matrix.shape[0] == 1:
        return np.array([0])
    normalized = normalize(matrix)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    return db.fit_predict(normalized)


def homogeneity_from_labels(labels):
    """Homogeneity = largest cluster size / total items. 1.0 means one cluster."""
    if len(labels) == 0:
        return 1.0
    counts = Counter(labels)
    cluster_counts = [c for lbl, c in counts.items() if lbl != -1]
    if not cluster_counts:
        return 0.0
    return max(cluster_counts) / len(labels)


def build_clusters(labels, items, centroid_field_fn):
    """Build the batch_clusters list from DBSCAN labels."""
    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[int(label)].append(idx)

    clusters = []
    outliers = []
    total = len(items)

    for label, indices in groups.items():
        if label == -1:
            outliers.extend(indices)
            continue
        clusters.append({
            "cluster_id": int(label),
            "cluster_size": len(indices),
            "cluster_pct": len(indices) / total * 100 if total else 0.0,
            "centroid_summary": centroid_field_fn([items[i] for i in indices]),
            "cohesion": 1.0,
            "_members": indices,
        })

    return clusters, outliers


def kernel_centroid_summary(kernels):
    """Summarize a group of kernels."""
    if not kernels:
        return {}
    names = [k["kernel_name"] for k in kernels]
    uses_fp64 = any(k["kernel_summary"].get("uses_fp64", False) for k in kernels)
    return {
        "kernel_names": names,
        "count": len(kernels),
        "any_fp64": uses_fp64,
    }


def tb_centroid_summary(tbs):
    """Summarize a group of TBs."""
    if not tbs:
        return {}
    insts = [tb["features"].get("instructions_per_warp_mean", 0) for tb in tbs]
    warps = [tb["features"].get("num_warps", 0) for tb in tbs]
    return {
        "count": len(tbs),
        "avg_inst_per_warp": float(np.mean(insts)) if insts else 0.0,
        "avg_num_warps": float(np.mean(warps)) if warps else 0.0,
    }


def kernel_summary_vector(summary):
    """Same vectorization as squash: key opcode ratios + flags."""
    opcodes = {entry["opcode"].upper(): entry["count"] for entry in summary.get("top_opcodes", [])}
    total_ops = sum(opcodes.values()) or 1
    return [
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


def tb_feature_vector(features, key_order):
    """Convert TB features dict to numeric vector."""
    vec = []
    for key in key_order:
        val = features.get(key, 0)
        if isinstance(val, (int, float)):
            vec.append(float(val))
        elif isinstance(val, bool):
            vec.append(1.0 if val else 0.0)
        else:
            vec.append(0.0)
    return vec


def batch_kernel_level(workload_data, config):
    kernels = workload_data["kernels"]
    vectors = [kernel_summary_vector(k["kernel_summary"]) for k in kernels]
    matrix = to_matrix(vectors)
    labels = cluster_with_dbscan(
        matrix, config["dbscan_eps"], config["dbscan_min_samples"]
    )
    clusters, outliers = build_clusters(
        labels, kernels, kernel_centroid_summary
    )

    for cluster in clusters:
        cluster["kernel_ids"] = [kernels[i]["kernel_id"] for i in cluster.pop("_members")]

    return {
        "batch_clusters": clusters,
        "outlier_kernels": [kernels[i]["kernel_id"] for i in outliers],
        "homogeneity_score": homogeneity_from_labels(labels),
    }


def batch_tb_level(workload_data, config):
    result = {}
    for kernel in workload_data["kernels"]:
        tbs = kernel.get("per_tb", [])
        if not tbs:
            continue
        key_order = sorted(tbs[0]["features"].keys())
        vectors = [tb_feature_vector(tb["features"], key_order) for tb in tbs]
        matrix = to_matrix(vectors)
        labels = cluster_with_dbscan(
            matrix, config["dbscan_eps"], config["dbscan_min_samples"]
        )
        clusters, outliers = build_clusters(
            labels, tbs, tb_centroid_summary
        )
        for cluster in clusters:
            cluster["tb_ids"] = [tbs[i]["tb_index"] for i in cluster.pop("_members")]

        result[str(kernel["kernel_id"])] = {
            "batch_clusters": clusters,
            "outlier_tbs": [tbs[i]["tb_index"] for i in outliers],
            "homogeneity_score": homogeneity_from_labels(labels),
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch mechanism")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workload_data = json.loads(Path(args.input).read_text())
    config = json.loads(Path(args.config).read_text())["batch"]

    kernel_level = batch_kernel_level(workload_data, config["kernel_level"])
    tb_level = batch_tb_level(workload_data, config["tb_level"])

    reuse_hint = {
        "kernel_cluster_representatives": {
            str(c["cluster_id"]): c["kernel_ids"][0] if c.get("kernel_ids") else None
            for c in kernel_level["batch_clusters"]
        },
        "tb_cluster_representatives": {
            kid: {
                str(c["cluster_id"]): c["tb_ids"][0] if c.get("tb_ids") else None
                for c in data["batch_clusters"]
            }
            for kid, data in tb_level.items()
        },
    }

    output = {
        "mechanism": "batch",
        "workload": workload_data.get("workload", "unknown"),
        "kernel_level": kernel_level,
        "tb_level": tb_level,
        "_simulation_reuse_hint": reuse_hint,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(
        f"[batch] wrote {out_path} "
        f"(kernel_clusters={len(kernel_level['batch_clusters'])}, "
        f"outliers={len(kernel_level['outlier_kernels'])})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
