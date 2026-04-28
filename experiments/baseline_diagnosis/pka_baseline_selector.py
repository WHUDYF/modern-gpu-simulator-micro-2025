"""PKA baseline selector for L1.

Reads PkaFeatureTable, runs PCA-like dimensionality reduction + k-means
clustering on the 12-D feature space. Forbidden fields must not enter
the grouping key. Emits selector config, dimensionality reduction report,
reduced feature table, cluster assignment, and anchor table.
"""

from __future__ import annotations

import json, math, sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "a_line" / "l1"

FT_PATH = ARTIFACT_DIR / "pka_feature_table_l1.json"
SG_PATH = ARTIFACT_DIR / "l1_stage_gate_report_l1.json"

FORBIDDEN = frozenset({
    "kernel_name", "grid_dim", "block_dim", "shape_hint", "trace_order",
    "cross_tb_offset_coverage", "squash_boundary_crossing_flag",
    "family_id", "regime_id", "route_primitive", "execution_template",
    "execution_template_label", "simulator_lane_id",
    "address_override_density", "full_encoding_fallback_rate",
    "shared_pc_sequence_length",
})

ALLOWED_FEATURES = [
    "coalesced_global_loads", "coalesced_global_stores", "coalesced_local_loads",
    "thread_global_loads", "thread_global_stores", "thread_local_loads",
    "thread_shared_loads", "thread_shared_stores", "thread_global_atomics",
    "num_instructions", "divergence_efficiency", "num_thread_blocks",
]

COUNT_FEATURES = {
    "coalesced_global_loads", "coalesced_global_stores", "coalesced_local_loads",
    "thread_global_loads", "thread_global_stores", "thread_local_loads",
    "thread_shared_loads", "thread_shared_stores", "thread_global_atomics",
    "num_instructions", "num_thread_blocks",
}

RATIO_FEATURES = {"divergence_efficiency"}


def _check_gate() -> tuple[bool, str]:
    if not SG_PATH.exists():
        return False, "stage gate report not found"
    sg = json.loads(SG_PATH.read_text())
    s3 = sg.get("stages", {}).get("stage_3_selector", "unknown")
    if s3 != "ready":
        return False, sg.get("next_action", f"stage_3 is {s3}, not ready")
    return True, ""


def _validate_allowlist(feature_list: list[str]) -> list[str]:
    errors = []
    # Exact match: length, order, and members must equal ALLOWED_FEATURES
    if len(feature_list) != 12:
        errors.append(f"Allowlist length {len(feature_list)} != 12")
    if feature_list != ALLOWED_FEATURES:
        for i, (a, b) in enumerate(zip(feature_list, ALLOWED_FEATURES)):
            if a != b:
                errors.append(f"Allowlist mismatch at position {i}: got '{a}', expected '{b}'")
                break
    forbidden = sorted(set(feature_list) & FORBIDDEN)
    if forbidden:
        errors.append(f"Forbidden fields in allowlist: {forbidden}")
    dups = [f for f in feature_list if feature_list.count(f) > 1]
    if dups:
        errors.append(f"Duplicate features in allowlist: {sorted(set(dups))}")
    extra = sorted(set(feature_list) - set(ALLOWED_FEATURES))
    if extra:
        errors.append(f"Extra features not in PKA spec: {extra}")
    return errors


def _build_matrix(records: list[dict], timing_unit: str | None = None) -> tuple[list[list[float]], list[dict], list[str]]:
    matrix, meta = [], []
    for rec in records:
        features = rec.get("features", {})
        row = []
        ok = True
        for fn in ALLOWED_FEATURES:
            f = features.get(fn, {})
            val = f.get("value")
            if val is None or f.get("status") != "measured":
                ok = False
                break
            row.append(float(val))
        if ok and len(row) == 12:
            matrix.append(row)
            meta.append(rec)
    return matrix, meta, []


def _mean(vals):
    return sum(vals) / len(vals)


def _std(vals, m):
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def _standardize(matrix):
    N = len(matrix)
    if N == 0:
        return matrix, [], []
    D = len(matrix[0])
    means = [_mean([matrix[i][j] for i in range(N)]) for j in range(D)]
    stds = [_std([matrix[i][j] for i in range(N)], means[j]) for j in range(D)]
    zero_var = [j for j in range(D) if stds[j] == 0]
    for j in zero_var:
        stds[j] = 1.0
    result = [[(matrix[i][j] - means[j]) / stds[j] for j in range(D)] for i in range(N)]
    return result, means, stds, zero_var


def _covariance(matrix):
    N, D = len(matrix), len(matrix[0]) if matrix else 0
    cov = [[0.0] * D for _ in range(D)]
    for i in range(D):
        for j in range(D):
            cov[i][j] = sum(matrix[k][i] * matrix[k][j] for k in range(N)) / (N - 1) if N > 1 else 0.0
    return cov


def _power_iteration(cov, iters=100):
    D = len(cov)
    v = [1.0 / math.sqrt(D)] * D
    for _ in range(iters):
        mv = [sum(cov[i][j] * v[j] for j in range(D)) for i in range(D)]
        norm = math.sqrt(sum(x * x for x in mv))
        if norm < 1e-12:
            break
        v = [x / norm for x in mv]
    eigval = sum(v[i] * sum(cov[i][j] * v[j] for j in range(D)) for i in range(D))
    return eigval, v


def _pca_reduce(matrix, n_components):
    N, D = len(matrix), len(matrix[0]) if matrix else 0
    if N < 2 or D == 0:
        return matrix, [], [], 0.0
    cov = _covariance(matrix)
    components = []
    eigvals = []
    residual = [row[:] for row in cov]
    for _ in range(min(n_components, D)):
        eigval, vec = _power_iteration(residual)
        eigvals.append(eigval)
        components.append(vec)
        for i in range(D):
            for j in range(D):
                residual[i][j] -= eigval * vec[i] * vec[j]
    total_var = sum(eigvals)
    explained = []
    for ev in eigvals:
        explained.append(ev / total_var if total_var > 0 else 0.0)
    reduced = [[sum(matrix[i][j] * components[c][j] for j in range(D)) for c in range(len(components))] for i in range(N)]
    return reduced, components, explained, sum(explained)


def _kmeans(data, k, seed=42, max_iters=100):
    import random
    rng = random.Random(seed)
    N = len(data)
    if N == 0:
        return []
    k = min(k, N)
    centers = [data[i][:] for i in rng.sample(range(N), k)]
    assignments = [0] * N
    for _ in range(max_iters):
        changed = False
        for i in range(N):
            best_c, best_d = -1, float("inf")
            for c in range(k):
                d = sum((data[i][j] - centers[c][j]) ** 2 for j in range(len(data[i])))
                if d < best_d:
                    best_d = d
                    best_c = c
            if assignments[i] != best_c:
                assignments[i] = best_c
                changed = True
        if not changed:
            break
        counts = [0] * k
        new_centers = [[0.0] * len(data[0]) for _ in range(k)]
        for i in range(N):
            c = assignments[i]
            counts[c] += 1
            for j in range(len(data[i])):
                new_centers[c][j] += data[i][j]
        for c in range(k):
            if counts[c] > 0:
                new_centers[c] = [x / counts[c] for x in new_centers[c]]
            else:
                new_centers[c] = centers[c][:]
        centers = new_centers
    return assignments


def _select_representative(members, meta):
    """first_chronological: earliest by trace_order in metadata."""
    best = None
    best_order = float("inf")
    for m in members:
        order = m.get("metadata", {}).get("trace_order", float("inf"))
        if order < best_order:
            best_order = order
            best = m
    return best if best else members[0]


def main() -> int:
    ok, reason = _check_gate()
    if not ok:
        print(f"Selector blocked: {reason}")
        return 2

    records = json.loads(FT_PATH.read_text())
    if len(records) < 2:
        print(f"selector_insufficient_records: {len(records)} measured records")
        if SG_PATH.exists():
            sg = json.loads(SG_PATH.read_text())
            sg["run_status"] = "selector_insufficient_records"
            sg["stages"]["stage_3_selector"] = "blocked"
            sg["stages"]["stage_4_b_line_consumption"] = "blocked"
            SG_PATH.write_text(json.dumps(sg, indent=2) + "\n")
        return 3

    # Validate allowlist
    errs = _validate_allowlist(ALLOWED_FEATURES)
    if errs:
        print(f"Allowlist validation failed: {errs}")
        return 4

    # Build matrix with log1p for count features
    matrix, meta, _ = _build_matrix(records)
    N, D = len(matrix), len(matrix[0])
    if N < 2:
        print(f"selector_insufficient_records: {N} valid rows")
        return 3

    # log1p transform count features, keep ratio as-is
    for i in range(N):
        for j, fn in enumerate(ALLOWED_FEATURES):
            if fn in COUNT_FEATURES:
                matrix[i][j] = math.log1p(matrix[i][j])

    # Standardize
    std_matrix, means, stds, zero_var = _standardize(matrix)

    # PCA
    n_comp = min(3, N - 1, D)
    reduced, components, explained_var, total_explained = _pca_reduce(std_matrix, n_comp)

    # k-means
    k = min(3, N)
    seed = 42
    assignments = _kmeans(reduced, k, seed)

    # Build clusters
    clusters: dict[int, list[int]] = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    # Select representatives
    anchor_table = []
    clist = sorted(clusters.items())
    for cluster_idx, (_, member_idxs) in enumerate(clist, 1):
        members = [meta[i] for i in member_idxs]
        rep = _select_representative(members, meta)
        member_ids = [m["kernel_invocation_id"] for m in members]
        anchor_table.append({
            "output_role": "mainline_anchor",
            "rep_kernel_id": f"rep-pka-baseline-{cluster_idx}",
            "kernel_name": rep.get("kernel_name", ""),
            "cluster_id": f"pka-baseline-{cluster_idx}",
            "member_invocations": member_ids,
            "coverage_count": len(member_ids),
            "coverage_weight": len(member_ids) / N,
            "coverage_weight_source": "derived_from_member_count",
            "time_weight": 1.0 / len(clist),
            "time_weight_source": "uniform_no_timing_data",
        })

    # Emit selector config
    config = {
        "feature_allowlist": ALLOWED_FEATURES,
        "allowlist_validated": True,
        "preprocessing": {"log1p_features": sorted(COUNT_FEATURES), "ratio_features": sorted(RATIO_FEATURES)},
        "standardization": {"zero_variance_columns": zero_var},
        "pca": {"n_components": n_comp, "explained_variance_ratio": explained_var, "total_explained_variance": total_explained},
        "kmeans": {"k": k, "seed": seed},
        "representative_selection": "first_chronological",
    }
    (ARTIFACT_DIR / "pka_selector_config_l1.json").write_text(json.dumps(config, indent=2) + "\n")

    # Dimensionality reduction report
    dim_report = {
        "n_components": n_comp, "explained_variance_ratio": explained_var,
        "total_explained_variance": total_explained, "component_matrix": components,
        "zero_variance_columns": zero_var, "feature_order": ALLOWED_FEATURES,
        "mean": means, "std_deviation": stds,
    }
    (ARTIFACT_DIR / "pka_dimensionality_reduction_report_l1.json").write_text(json.dumps(dim_report, indent=2) + "\n")
    dim_md = ["# PKA Dimensionality Reduction", "", f"Components: {n_comp}",
              f"Total explained variance: {total_explained:.4f}", "",
              "| PC | Explained Variance |", "|----|-------------------|"]
    for i, ev in enumerate(explained_var, 1):
        dim_md.append(f"| PC{i} | {ev:.4f} |")
    if zero_var:
        dim_md.extend(["", f"Zero-variance columns (kept as zero): {zero_var}"])
    (ARTIFACT_DIR / "pka_dimensionality_reduction_report_l1.md").write_text("\n".join(dim_md) + "\n")

    # Reduced feature table
    reduced_table = []
    for i in range(N):
        row = {"kernel_invocation_id": meta[i]["kernel_invocation_id"]}
        for c in range(n_comp):
            row[f"PC{c+1}"] = reduced[i][c]
        reduced_table.append(row)
    (ARTIFACT_DIR / "pka_reduced_feature_table_l1.json").write_text(json.dumps(reduced_table, indent=2) + "\n")

    # Cluster assignments
    cluster_assign = []
    for i in range(N):
        cluster_assign.append({
            "kernel_invocation_id": meta[i]["kernel_invocation_id"],
            "cluster_id": f"pka-baseline-{assignments[i]+1}",
        })
    (ARTIFACT_DIR / "pka_cluster_assignment_l1.json").write_text(json.dumps(cluster_assign, indent=2) + "\n")

    # Anchor table
    (ARTIFACT_DIR / "representative_anchor_table_l1.json").write_text(json.dumps(anchor_table, indent=2) + "\n")

    print(f"Selector complete: {N} records -> {k} clusters ({n_comp} PCA components)")
    print(f"Explained variance: {total_explained:.4f}")
    print(f"Clusters: {sorted([len(v) for v in clusters.values()])} members each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
