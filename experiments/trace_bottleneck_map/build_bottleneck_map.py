#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.trace_bottleneck_map.catalog import ESTIMATED_SUITE_OBSERVATIONS
from experiments.trace_bottleneck_map.local_evidence import load_trace_benchmark_table
from experiments.trace_bottleneck_map.model import BenchmarkObservation, record_from_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trace bottleneck cost map artifacts.")
    parser.add_argument("--trace-benchmark-md", required=True, help="Path to docs/trace-benchmark-2026-04-03.md")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and Markdown outputs")
    return parser.parse_args()


def _records_from_observations(observations: list[BenchmarkObservation]) -> list[dict]:
    return [record_from_observation(obs) for obs in observations]


def _stable_trace_benchmark_source(raw_path: str) -> str:
    path = Path(raw_path)
    if not path.is_absolute():
        return raw_path

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _build_markdown(records: list[dict]) -> str:
    counts = Counter(row["dominant_bottleneck"] for row in records if row["status"] == "measured")
    lines = [
        "# Trace Bottleneck Cost Map",
        "",
        "## Summary",
        "",
        f"- Total records: {len(records)}",
        f"- Measured records: {sum(1 for row in records if row['status'] == 'measured')}",
        f"- Estimated records: {sum(1 for row in records if row['status'] == 'estimated')}",
        f"- Appendix/excluded records: {sum(1 for row in records if row['status'] in {'excluded', 'appendix_only'})}",
        "",
        "## Measured Bottleneck Counts",
        "",
    ]
    if counts:
        for name, count in sorted(counts.items()):
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- No measured rows found.")

    lines.extend(
        [
            "",
            "## Cost Map",
            "",
            "| Suite | Case | Category | Trace Size | Export | Sim Proxy | Status | Dominant Bottleneck | Evidence |",
            "|---|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in records:
        lines.append(
            "| {suite} | {case} | {category} | {trace_size} | {export_time} | {sim_time} | {status} | {bottleneck} | {evidence} |".format(
                suite=row["suite"],
                case=row["representative_case"],
                category=row["category"],
                trace_size=row["trace_size_mib"] if row["trace_size_mib"] is not None else "-",
                export_time=row["export_time_s"] if row["export_time_s"] is not None else "-",
                sim_time=row["sim_time_s"] if row["sim_time_s"] is not None else "-",
                status=row["status"],
                bottleneck=row["dominant_bottleneck"],
                evidence=row["evidence"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        measured = load_trace_benchmark_table(args.trace_benchmark_md)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    trace_benchmark_source = _stable_trace_benchmark_source(args.trace_benchmark_md)
    measured = [replace(obs, evidence=trace_benchmark_source) for obs in measured]
    all_observations = measured + ESTIMATED_SUITE_OBSERVATIONS
    records = _records_from_observations(all_observations)
    payload = {
        "report_name": "Trace Bottleneck Cost Map",
        "trace_benchmark_source": trace_benchmark_source,
        "records": records,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_cost_map.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "benchmark_cost_map.md").write_text(_build_markdown(records))

    print(f"Wrote {output_dir / 'benchmark_cost_map.json'}")
    print(f"Wrote {output_dir / 'benchmark_cost_map.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
