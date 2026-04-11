#!/usr/bin/env python3
"""Merge trace features and hardware stats into a single JSON file.

Combines output from extract_trace_features.py and parse_ncu_metrics.py
into one unified feature JSON for downstream analysis.
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge trace feature JSON and hardware stats JSON into one file."
    )
    parser.add_argument(
        "--trace-features",
        required=True,
        help="Path to JSON from extract_trace_features.py",
    )
    parser.add_argument(
        "--hw-stats",
        required=True,
        help="Path to JSON from parse_ncu_metrics.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for merged output JSON",
    )
    return parser.parse_args()


def load_json(path, label):
    if not os.path.isfile(path):
        print(f"Error: {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()

    trace_data = load_json(args.trace_features, "trace-features")
    hw_data = load_json(args.hw_stats, "hw-stats")

    # hw-stats as base, trace-features overwrites on conflict
    merged = {}
    merged.update(hw_data)
    merged.update(trace_data)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    trace_keys = sorted(trace_data.keys())
    hw_keys = sorted(hw_data.keys())
    overlap = sorted(set(trace_keys) & set(hw_keys))

    print(f"Trace-features keys ({len(trace_keys)}): {trace_keys}")
    print(f"HW-stats keys ({len(hw_keys)}): {hw_keys}")
    if overlap:
        print(f"Overlapping keys (trace-features takes precedence): {overlap}")
    print(f"Merged keys ({len(merged)}): {sorted(merged.keys())}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
