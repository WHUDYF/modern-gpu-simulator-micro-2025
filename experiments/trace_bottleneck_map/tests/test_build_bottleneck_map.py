import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "build_bottleneck_map.py"


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


def test_builder_rejects_missing_trace_benchmark(tmp_path):
    out_dir = tmp_path / "out"

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
    assert not (out_dir / "benchmark_cost_map.json").exists()
    assert not (out_dir / "benchmark_cost_map.md").exists()
