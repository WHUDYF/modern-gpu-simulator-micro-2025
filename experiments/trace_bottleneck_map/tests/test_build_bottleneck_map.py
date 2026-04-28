import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "build_bottleneck_map.py"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_trace_benchmark(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "| benchmark | trace status | trace size (MiB) | export time (s) | export speed | dynamic insts | sim status | sim time (s) |",
                "|---|---|---:|---:|---:|---:|---|---:|",
                "| l2_bw_32f | ok | 568.192 | 69.41 | 8.186 | 56820736 | ok | about 17 |",
                "| atomic_add_bw | ok | 5.423 | 2.27 | 2.389 | 378880 | ok | 10.13 |",
            ]
        )
    )


def test_builder_writes_json_and_markdown(tmp_path):
    trace_md = tmp_path / "trace-benchmark.md"
    _write_trace_benchmark(trace_md)
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-benchmark-md",
            "trace-benchmark.md",
            "--output-dir",
            "out",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert proc.returncode == 0, proc.stderr
    json_path = out_dir / "benchmark_cost_map.json"
    md_path = out_dir / "benchmark_cost_map.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["trace_benchmark_source"] == "trace-benchmark.md"
    measured = [row for row in payload["records"] if row["status"] == "measured"]
    assert any(row["representative_case"] == "l2_bw_32f" for row in measured)
    assert any(row["suite"] == "NCCL-tests" and row["status"] == "excluded" for row in payload["records"])
    assert "trace export / I/O" in md_path.read_text()


def test_builder_normalizes_repo_absolute_provenance(tmp_path):
    trace_md = REPO_ROOT / "docs" / "trace-benchmark-2026-04-03.md"
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-benchmark-md",
            str(trace_md),
            "--output-dir",
            "out",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads((out_dir / "benchmark_cost_map.json").read_text())
    assert payload["trace_benchmark_source"] == "docs/trace-benchmark-2026-04-03.md"

    measured = [row for row in payload["records"] if row["status"] == "measured"]
    assert any(
        row["representative_case"] == "l2_bw_32f"
        and row["evidence"] == "docs/trace-benchmark-2026-04-03.md"
        for row in measured
    )


def test_builder_canonicalizes_dot_prefixed_repo_provenance(tmp_path):
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-benchmark-md",
            "./docs/trace-benchmark-2026-04-03.md",
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads((out_dir / "benchmark_cost_map.json").read_text())
    assert payload["trace_benchmark_source"] == "docs/trace-benchmark-2026-04-03.md"

    measured = [row for row in payload["records"] if row["status"] == "measured"]
    assert any(
        row["representative_case"] == "l2_bw_32f"
        and row["evidence"] == payload["trace_benchmark_source"]
        for row in measured
    )


def test_builder_rejects_missing_trace_benchmark(tmp_path):
    out_dir = tmp_path / "out"
    json_path = out_dir / "benchmark_cost_map.json"
    md_path = out_dir / "benchmark_cost_map.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-benchmark-md",
            "missing.md",
            "--output-dir",
            "out",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert proc.returncode == 2
    assert "trace benchmark file not found" in proc.stderr
    assert not json_path.exists()
    assert not md_path.exists()


def test_generated_report_keeps_communication_suites_out_of_measured_main_table(tmp_path):
    trace_md = tmp_path / "trace-benchmark.md"
    _write_trace_benchmark(trace_md)
    out_dir = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-benchmark-md",
            str(trace_md),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads((out_dir / "benchmark_cost_map.json").read_text())
    communication_rows = [
        row for row in payload["records"]
        if row["suite"] in {"NCCL-tests", "OSU micro-benchmarks"}
    ]
    assert communication_rows
    assert all(row["status"] == "excluded" for row in communication_rows)
