#!/usr/bin/env python3
"""Delta mechanism: field-level change pattern analysis.

Operates at two levels:
  kernel_level: diff between adjacent kernels
  tb_level: diff between adjacent TBs within each kernel
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def extract_numeric_fields(obj):
    """Walk a dict recursively; return {flat_key: numeric_value}.
    Bool → 0/1. Nested dicts → dot-joined keys."""
    result = {}
    def walk(d, prefix):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                walk(v, key)
            elif isinstance(v, bool):
                result[key] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                result[key] = float(v)
    walk(obj, "")
    return result


def compute_temperature(value_series):
    """Given a list of values over a sequence, compute a temperature score.
    Score = stddev / (|mean| + 1e-9), clipped to [0, 1] via tanh.
    Hot fields have high variation relative to mean."""
    if len(value_series) < 2:
        return 0.0
    arr = np.array(value_series, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    denom = abs(mean) + 1e-9
    raw = std / denom
    return float(np.tanh(raw))


def classify_fields(field_temps, hot_threshold, cold_threshold):
    """Partition fields into hot/cold/warm."""
    hot = [f for f, t in field_temps.items() if t >= hot_threshold]
    cold = [f for f, t in field_temps.items() if t <= cold_threshold]
    return hot, cold


def pairwise_correlation(field_series_dict, threshold):
    """Compute correlations between field pairs that change together."""
    fields = list(field_series_dict.keys())
    correlations = []
    for i, f1 in enumerate(fields):
        for f2 in fields[i + 1:]:
            s1 = np.array(field_series_dict[f1])
            s2 = np.array(field_series_dict[f2])
            if len(s1) < 3:
                continue
            if np.std(s1) == 0 or np.std(s2) == 0:
                continue
            corr = float(np.corrcoef(s1, s2)[0, 1])
            if abs(corr) >= threshold:
                correlations.append({
                    "fields": [f1, f2],
                    "correlation": corr,
                    "interpretation": (
                        f"{f1} and {f2} " +
                        ("covary together" if corr > 0 else "move inversely")
                    ),
                })
    return correlations


def detect_outlier_diffs(field_series_dict, zscore_threshold):
    """Find adjacent pairs where total delta magnitude is an outlier."""
    fields = list(field_series_dict.keys())
    if not fields:
        return []
    length = len(next(iter(field_series_dict.values())))
    if length < 3:
        return []

    magnitudes = []
    for i in range(length - 1):
        total = 0.0
        dominant_fields = []
        for f in fields:
            series = field_series_dict[f]
            diff = abs(series[i + 1] - series[i])
            std = np.std(series)
            if std > 0:
                norm_diff = diff / std
                total += norm_diff
                if norm_diff > 1.0:
                    dominant_fields.append((f, norm_diff))
        magnitudes.append((i, total, dominant_fields))

    if not magnitudes:
        return []

    total_values = [m[1] for m in magnitudes]
    mean_mag = np.mean(total_values)
    std_mag = np.std(total_values)
    if std_mag == 0:
        return []

    outliers = []
    for idx, mag, dominant in magnitudes:
        z = (mag - mean_mag) / std_mag
        if z >= zscore_threshold:
            sorted_dom = sorted(dominant, key=lambda p: -p[1])[:5]
            outliers.append({
                "pair": [idx, idx + 1],
                "magnitude": float(mag),
                "dominant_changing_fields": [f for f, _ in sorted_dom],
                "interpretation": f"z-score {z:.2f} above mean magnitude",
            })

    return outliers


def delta_on_sequence(numeric_dicts, config):
    """Run delta analysis on a sequence of numeric field dicts."""
    if not numeric_dicts:
        return {
            "field_temperature": {},
            "hot_fields": [],
            "cold_fields": [],
            "field_correlations": [],
            "outlier_diffs": [],
        }

    all_fields = set()
    for d in numeric_dicts:
        all_fields.update(d.keys())

    field_series = {}
    for f in all_fields:
        field_series[f] = [d.get(f, 0.0) for d in numeric_dicts]

    field_temps = {f: compute_temperature(series) for f, series in field_series.items()}

    hot, cold = classify_fields(
        field_temps, config["hot_threshold"], config["cold_threshold"]
    )

    correlations = pairwise_correlation(field_series, config["correlation_threshold"])
    outliers = detect_outlier_diffs(field_series, config["outlier_zscore"])

    return {
        "field_temperature": field_temps,
        "hot_fields": sorted(hot),
        "cold_fields": sorted(cold),
        "field_correlations": correlations,
        "outlier_diffs": outliers,
    }


def delta_kernel_level(workload_data, config):
    kernels = workload_data["kernels"]
    kernel_numeric = []
    for k in kernels:
        flat = extract_numeric_fields(k.get("kernel_summary", {}))
        kernel_numeric.append(flat)
    return delta_on_sequence(kernel_numeric, config)


def delta_tb_level(workload_data, config):
    result = {}
    for kernel in workload_data["kernels"]:
        tbs = kernel.get("per_tb", [])
        if len(tbs) < 2:
            continue
        tb_numeric = [extract_numeric_fields(tb.get("features", {})) for tb in tbs]
        result[str(kernel["kernel_id"])] = delta_on_sequence(tb_numeric, config)
    return result


def main():
    parser = argparse.ArgumentParser(description="Delta mechanism")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workload_data = json.loads(Path(args.input).read_text())
    config = json.loads(Path(args.config).read_text())["delta"]

    kernel_level = delta_kernel_level(workload_data, config["kernel_level"])
    tb_level = delta_tb_level(workload_data, config["tb_level"])

    output = {
        "mechanism": "delta",
        "workload": workload_data.get("workload", "unknown"),
        "kernel_level": kernel_level,
        "tb_level": tb_level,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(
        f"[delta] wrote {out_path} "
        f"(kernel_hot={len(kernel_level['hot_fields'])}, "
        f"kernel_cold={len(kernel_level['cold_fields'])})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
