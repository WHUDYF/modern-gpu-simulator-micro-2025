#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


RUN_RE = re.compile(r"model_(?P<model>.+?)_ctx(?P<context_len>\d+)_gen(?P<gen_tokens>\d+)_run(?P<run_id>\d+)$")
RUN_LOG_PATTERNS = {
    "model": re.compile(r"^model=(.+)$"),
    "context_len": re.compile(r"^context_len=(\d+)$"),
    "gen_tokens": re.compile(r"^gen_tokens=(\d+)$"),
    "decode_time_s": re.compile(r"^decode_time_s=([0-9.]+)$"),
}


def parse_run_log(run_log: Path):
    values = {}
    if not run_log.exists():
        return values
    for line in run_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        for key, pattern in RUN_LOG_PATTERNS.items():
            match = pattern.match(line.strip())
            if match:
                values[key] = match.group(1)
    return values


def count_kernel_rows(stats_csv: Path):
    if not stats_csv.exists():
        return 0
    with stats_csv.open(newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.reader(f))
    return max(len(rows) - 1, 0)


def dir_size_bytes(path: Path):
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def summarize_run(run_dir: Path):
    match = RUN_RE.match(run_dir.name)
    if not match:
        return None

    meta = match.groupdict()
    traces_dir = run_dir / "traces"
    dynamic_trace = traces_dir / "dynamic_trace.pb"
    stats_csv = traces_dir / "stats.csv"
    threadblocks_dir = traces_dir / "threadblocks"
    extra_info = traces_dir / "extra_info" / "enhanced_execution_info.json"
    log_values = parse_run_log(run_dir / "run.log")

    return {
        "run_dir": str(run_dir),
        "model": log_values.get("model", meta["model"]),
        "context_len": log_values.get("context_len", meta["context_len"]),
        "gen_tokens": log_values.get("gen_tokens", meta["gen_tokens"]),
        "run_id": meta["run_id"],
        "decode_time_s": log_values.get("decode_time_s", ""),
        "trace_size_mb": f"{dir_size_bytes(traces_dir) / (1024 * 1024):.6f}",
        "dynamic_trace_pb_mb": f"{dir_size_bytes(dynamic_trace) / (1024 * 1024):.6f}",
        "threadblocks_mb": f"{dir_size_bytes(threadblocks_dir) / (1024 * 1024):.6f}",
        "extra_info_kb": f"{dir_size_bytes(extra_info) / 1024:.3f}",
        "num_pb_files": str(sum(1 for _ in traces_dir.rglob('*.pb'))) if traces_dir.exists() else "0",
        "num_kernel_rows": str(count_kernel_rows(stats_csv)),
        "has_dynamic_trace_pb": "1" if dynamic_trace.exists() else "0",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(p for p in args.results_root.iterdir() if p.is_dir()):
        row = summarize_run(run_dir)
        if row is not None:
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "model",
        "context_len",
        "gen_tokens",
        "run_id",
        "decode_time_s",
        "trace_size_mb",
        "dynamic_trace_pb_mb",
        "threadblocks_mb",
        "extra_info_kb",
        "num_pb_files",
        "num_kernel_rows",
        "has_dynamic_trace_pb",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
