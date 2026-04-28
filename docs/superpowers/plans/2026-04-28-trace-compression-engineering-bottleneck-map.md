# Trace Compression Engineering Bottleneck Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible benchmark cost-map generator that identifies whether trace export/I/O, trace reading, benchmark sweep size, or simulator replay is the dominant bottleneck.

**Architecture:** Add a small isolated package under `experiments/trace_bottleneck_map/` so this engineering line does not touch L1 selector files. The package reads the existing measured trace benchmark markdown table, combines it with a static catalog of mainstream benchmark suites, classifies cost levels, infers dominant bottlenecks, and emits JSON plus Markdown reports under `artifacts/trace_bottleneck_map/`.

**Tech Stack:** Python standard library, pytest, existing `docs/trace-benchmark-2026-04-03.md`, generated JSON/Markdown artifacts, no new runtime dependency.

---

## Normative References

1. `docs/superpowers/specs/2026-04-28-trace-compression-engineering-bottleneck-map-design.md`
2. `docs/trace-benchmark-2026-04-03.md`
3. `docs/microbenchmark-runtime-related-work-2026-04-26.md`
4. `docs/a-line-kernel-validation-dataset-recommendation-2026-04-26.md`

If this plan and the design spec disagree, the design spec is authoritative.

## Scope Check

This plan covers one subsystem: the bottleneck cost-map generator and its report artifacts. It does not implement streaming compression, simulator acceleration, trace format changes, or new benchmark runners. Those decisions come after the cost map identifies the dominant bottleneck.

## File Structure

- Create `experiments/trace_bottleneck_map/__init__.py`
  - Marks the engineering-line package.
- Create `experiments/trace_bottleneck_map/model.py`
  - Defines cost records, cost classes, and bottleneck inference.
- Create `experiments/trace_bottleneck_map/local_evidence.py`
  - Parses existing local markdown evidence, starting with `docs/trace-benchmark-2026-04-03.md`.
- Create `experiments/trace_bottleneck_map/catalog.py`
  - Stores mainstream suite entries and appendix exclusions.
- Create `experiments/trace_bottleneck_map/build_bottleneck_map.py`
  - CLI that writes `benchmark_cost_map.json` and `benchmark_cost_map.md`.
- Create `experiments/trace_bottleneck_map/tests/test_model.py`
  - Unit tests for classification and bottleneck inference.
- Create `experiments/trace_bottleneck_map/tests/test_local_evidence.py`
  - Unit tests for markdown parsing.
- Create `experiments/trace_bottleneck_map/tests/test_catalog.py`
  - Unit tests for main-table vs appendix boundaries.
- Create `experiments/trace_bottleneck_map/tests/test_build_bottleneck_map.py`
  - CLI smoke tests for JSON and Markdown outputs.
- Generate `artifacts/trace_bottleneck_map/benchmark_cost_map.json`
  - Machine-readable cost map.
- Generate `artifacts/trace_bottleneck_map/benchmark_cost_map.md`
  - Reviewable report table and conclusion.

---

### Task 1: Cost Model Core

**Files:**
- Create: `experiments/trace_bottleneck_map/__init__.py`
- Create: `experiments/trace_bottleneck_map/model.py`
- Test: `experiments/trace_bottleneck_map/tests/test_model.py`

- [ ] **Step 1: Create package marker**

Create `experiments/trace_bottleneck_map/__init__.py`:

```python
"""Engineering-line tools for GPU trace bottleneck mapping."""
```

- [ ] **Step 2: Write failing model tests**

Create `experiments/trace_bottleneck_map/tests/test_model.py`:

```python
from experiments.trace_bottleneck_map.model import (
    BenchmarkObservation,
    classify_seconds,
    classify_trace_size_mib,
    infer_bottleneck,
    record_from_observation,
)


def test_classify_seconds_boundaries():
    assert classify_seconds(None) == "unknown"
    assert classify_seconds(0.4) == "sub-second"
    assert classify_seconds(2.0) == "seconds"
    assert classify_seconds(34.0) == "tens of seconds"
    assert classify_seconds(120.0) == "minutes"
    assert classify_seconds(7200.0) == "hours"
    assert classify_seconds(200000.0) == "infeasible"


def test_classify_trace_size_mib_boundaries():
    assert classify_trace_size_mib(None) == "unknown"
    assert classify_trace_size_mib(0.3) == "tiny"
    assert classify_trace_size_mib(4.0) == "small"
    assert classify_trace_size_mib(47.0) == "medium"
    assert classify_trace_size_mib(568.0) == "large"
    assert classify_trace_size_mib(2048.0) == "huge"


def test_infer_bottleneck_export_dominates_large_trace():
    assert infer_bottleneck(trace_size_mib=568.192, export_time_s=69.41, sim_time_s=17.0) == "trace export / I/O"


def test_infer_bottleneck_simulator_dominates():
    assert infer_bottleneck(trace_size_mib=5.423, export_time_s=2.27, sim_time_s=10.13) == "simulator throughput"


def test_infer_bottleneck_fixed_overhead_for_tiny_trace():
    assert infer_bottleneck(trace_size_mib=0.288, export_time_s=2.11, sim_time_s=1.26) == "capture / fixed overhead"


def test_record_from_observation_classifies_fields():
    obs = BenchmarkObservation(
        suite="GPU_Microbenchmark",
        representative_case="l2_bw_32f",
        category="bandwidth",
        trace_size_mib=568.192,
        export_time_s=69.41,
        sim_time_s=17.0,
        status="measured",
        evidence="docs/trace-benchmark-2026-04-03.md",
    )
    record = record_from_observation(obs)
    assert record["trace_size_class"] == "large"
    assert record["export_cost_class"] == "minutes"
    assert record["simulator_cost_class"] == "tens of seconds"
    assert record["dominant_bottleneck"] == "trace export / I/O"


def test_record_from_observation_uses_catalog_hint():
    obs = BenchmarkObservation(
        suite="nvbench",
        representative_case="runtime and compile-time parameter sweeps",
        category="generic kernel benchmark",
        status="estimated",
        evidence="https://github.com/NVIDIA/nvbench",
        dominant_bottleneck_hint="benchmark sweep explosion",
    )
    record = record_from_observation(obs)
    assert record["dominant_bottleneck"] == "benchmark sweep explosion"
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests/test_model.py -q
```

Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `experiments.trace_bottleneck_map.model`.

- [ ] **Step 4: Implement the minimal cost model**

Create `experiments/trace_bottleneck_map/model.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkObservation:
    suite: str
    representative_case: str
    category: str
    trace_size_mib: float | None = None
    export_time_s: float | None = None
    sim_time_s: float | None = None
    native_runtime_s: float | None = None
    status: str = "estimated"
    evidence: str = ""
    dominant_bottleneck_hint: str | None = None


def classify_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 1:
        return "sub-second"
    if seconds < 10:
        return "seconds"
    if seconds < 60:
        return "tens of seconds"
    if seconds < 3600:
        return "minutes"
    if seconds < 86400:
        return "hours"
    return "infeasible"


def classify_trace_size_mib(size_mib: float | None) -> str:
    if size_mib is None:
        return "unknown"
    if size_mib < 1:
        return "tiny"
    if size_mib < 10:
        return "small"
    if size_mib < 128:
        return "medium"
    if size_mib < 1024:
        return "large"
    return "huge"


def infer_bottleneck(
    *,
    trace_size_mib: float | None,
    export_time_s: float | None,
    sim_time_s: float | None,
) -> str:
    if trace_size_mib is not None and trace_size_mib < 1 and max(export_time_s or 0, sim_time_s or 0) <= 3:
        return "capture / fixed overhead"
    if export_time_s is None and sim_time_s is None:
        return "insufficient evidence"
    if export_time_s is None:
        return "simulator throughput"
    if sim_time_s is None:
        return "trace export / I/O"
    if export_time_s >= sim_time_s:
        return "trace export / I/O"
    if sim_time_s >= 1.5 * export_time_s:
        return "simulator throughput"
    return "balanced / mixed"


def record_from_observation(observation: BenchmarkObservation) -> dict[str, Any]:
    record = asdict(observation)
    record.update(
        {
            "native_runtime_class": classify_seconds(observation.native_runtime_s),
            "trace_size_class": classify_trace_size_mib(observation.trace_size_mib),
            "export_cost_class": classify_seconds(observation.export_time_s),
            "simulator_cost_class": classify_seconds(observation.sim_time_s),
            "dominant_bottleneck": observation.dominant_bottleneck_hint or infer_bottleneck(
                trace_size_mib=observation.trace_size_mib,
                export_time_s=observation.export_time_s,
                sim_time_s=observation.sim_time_s,
            ),
        }
    )
    return record
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests/test_model.py -q
```

Expected: PASS, 7 tests.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add experiments/trace_bottleneck_map/__init__.py \
  experiments/trace_bottleneck_map/model.py \
  experiments/trace_bottleneck_map/tests/test_model.py
git commit -m "feat: add bottleneck cost model"
```

---

### Task 2: Local Evidence Parser

**Files:**
- Create: `experiments/trace_bottleneck_map/local_evidence.py`
- Test: `experiments/trace_bottleneck_map/tests/test_local_evidence.py`

- [ ] **Step 1: Write failing parser tests**

Create `experiments/trace_bottleneck_map/tests/test_local_evidence.py`:

```python
from pathlib import Path

from experiments.trace_bottleneck_map.local_evidence import load_trace_benchmark_table


def test_load_trace_benchmark_table_parses_measured_rows(tmp_path):
    source = tmp_path / "trace-benchmark.md"
    source.write_text(
        "\n".join(
            [
                "| benchmark | trace status | trace size (MiB) | export time (s) | export speed | dynamic insts | sim status | sim time (s) |",
                "|---|---|---:|---:|---:|---:|---|---:|",
                "| l2_bw_32f | ok | 568.192 | 69.41 | 8.186 | 56820736 | ok | about 17 |",
                "| shared_lat | ok | 0.340 | 2.18 | 0.156 | 17344 | ok | 1.29 |",
                "| l2_bw_64f | partial_or_timeout | 0.000 | 90.18 | 0.000 | 0 | missing |  |",
            ]
        )
    )

    observations = load_trace_benchmark_table(source)

    cases = {obs.representative_case: obs for obs in observations}
    assert cases["l2_bw_32f"].trace_size_mib == 568.192
    assert cases["l2_bw_32f"].export_time_s == 69.41
    assert cases["l2_bw_32f"].sim_time_s == 17.0
    assert cases["shared_lat"].category == "latency"
    assert "l2_bw_64f" not in cases


def test_load_trace_benchmark_table_rejects_missing_file(tmp_path):
    missing = tmp_path / "missing.md"
    try:
        load_trace_benchmark_table(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing trace benchmark file was accepted")
```

- [ ] **Step 2: Run the parser tests and confirm they fail**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests/test_local_evidence.py -q
```

Expected: FAIL with missing module or missing function errors.

- [ ] **Step 3: Implement the local evidence parser**

Create `experiments/trace_bottleneck_map/local_evidence.py`:

```python
from __future__ import annotations

from pathlib import Path

from experiments.trace_bottleneck_map.model import BenchmarkObservation


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_float(text: str) -> float | None:
    cleaned = text.strip().replace("about", "").strip()
    if cleaned == "":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _category_for_case(name: str) -> str:
    lowered = name.lower()
    if "lat" in lowered:
        return "latency"
    if "atomic" in lowered:
        return "atomic / bandwidth"
    if "flop" in lowered:
        return "compute"
    if "bw" in lowered:
        return "bandwidth"
    return "mixed"


def load_trace_benchmark_table(path: str | Path) -> list[BenchmarkObservation]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"trace benchmark file not found: {source}")

    observations: list[BenchmarkObservation] = []
    for line in source.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = _cells(stripped)
        if not cells or cells[0] in {"benchmark", "---"} or set(cells[0]) == {"-"}:
            continue
        if len(cells) < 8:
            continue

        benchmark = cells[0].strip("` ")
        trace_status = cells[1]
        sim_status = cells[6]
        if trace_status != "ok" or sim_status != "ok":
            continue

        observations.append(
            BenchmarkObservation(
                suite="GPU_Microbenchmark",
                representative_case=benchmark,
                category=_category_for_case(benchmark),
                trace_size_mib=_parse_float(cells[2]),
                export_time_s=_parse_float(cells[3]),
                sim_time_s=_parse_float(cells[7]),
                status="measured",
                evidence=str(source),
            )
        )
    return observations
```

- [ ] **Step 4: Run the parser tests and confirm they pass**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests/test_local_evidence.py -q
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add experiments/trace_bottleneck_map/local_evidence.py \
  experiments/trace_bottleneck_map/tests/test_local_evidence.py
git commit -m "feat: parse local trace benchmark evidence"
```

---

### Task 3: Benchmark Catalog and Boundaries

**Files:**
- Create: `experiments/trace_bottleneck_map/catalog.py`
- Test: `experiments/trace_bottleneck_map/tests/test_catalog.py`

- [ ] **Step 1: Write failing catalog tests**

Create `experiments/trace_bottleneck_map/tests/test_catalog.py`:

```python
from experiments.trace_bottleneck_map.catalog import ESTIMATED_SUITE_OBSERVATIONS


def test_catalog_keeps_multigpu_suites_out_of_main_table():
    by_suite = {entry.suite: entry for entry in ESTIMATED_SUITE_OBSERVATIONS}
    assert by_suite["NCCL-tests"].status == "excluded"
    assert by_suite["OSU micro-benchmarks"].status == "excluded"


def test_catalog_marks_mlperf_as_appendix_anchor():
    by_suite = {entry.suite: entry for entry in ESTIMATED_SUITE_OBSERVATIONS}
    assert by_suite["MLPerf Inference / Training"].status == "appendix_only"
    assert by_suite["MLPerf Inference / Training"].category == "full workload anchor"


def test_catalog_includes_main_single_gpu_suites():
    suites = {entry.suite for entry in ESTIMATED_SUITE_OBSERVATIONS if entry.status == "estimated"}
    assert {"BabelStream", "nvbandwidth", "nvbench", "CUTLASS profiler", "Rodinia", "Parboil", "PolyBench/GPU"} <= suites
```

- [ ] **Step 2: Run catalog tests and confirm they fail**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests/test_catalog.py -q
```

Expected: FAIL with missing module or missing symbol errors.

- [ ] **Step 3: Implement the catalog**

Create `experiments/trace_bottleneck_map/catalog.py`:

```python
from __future__ import annotations

from experiments.trace_bottleneck_map.model import BenchmarkObservation


ESTIMATED_SUITE_OBSERVATIONS = [
    BenchmarkObservation(
        suite="BabelStream",
        representative_case="copy/scale/add/triad/dot",
        category="bandwidth",
        sim_time_s=20.0,
        status="estimated",
        evidence="https://github.com/UoB-HPC/BabelStream",
        dominant_bottleneck_hint="trace export or simulator depending on array size",
    ),
    BenchmarkObservation(
        suite="nvbandwidth",
        representative_case="memcpy and link bandwidth patterns",
        category="bandwidth / link copy",
        sim_time_s=20.0,
        status="estimated",
        evidence="https://github.com/NVIDIA/nvbandwidth",
        dominant_bottleneck_hint="export / I/O for large sweeps; communication path for multi-link modes",
    ),
    BenchmarkObservation(
        suite="nvbench",
        representative_case="runtime and compile-time parameter sweeps",
        category="generic kernel benchmark",
        sim_time_s=120.0,
        status="estimated",
        evidence="https://github.com/NVIDIA/nvbench",
        dominant_bottleneck_hint="benchmark sweep explosion",
    ),
    BenchmarkObservation(
        suite="CUTLASS profiler",
        representative_case="GEMM and convolution configs",
        category="dense compute",
        sim_time_s=120.0,
        status="estimated",
        evidence="https://github.com/NVIDIA/cutlass/wiki/Performance-Profiling",
        dominant_bottleneck_hint="simulator throughput and parameter sweep explosion",
    ),
    BenchmarkObservation(
        suite="Rodinia",
        representative_case="nn/backprop/bfs/lud/nw",
        category="irregular / mixed",
        sim_time_s=30.0,
        status="estimated",
        evidence="experiments/baseline_diagnosis/results/rodinia",
        dominant_bottleneck_hint="mixed control / trace depth",
    ),
    BenchmarkObservation(
        suite="Parboil",
        representative_case="sgemm/stencil/cutcp/mri-q/histo/bfs",
        category="mixed dense / irregular",
        sim_time_s=30.0,
        status="estimated",
        evidence="APEs/*/parboil.md",
        dominant_bottleneck_hint="trace export plus irregularity",
    ),
    BenchmarkObservation(
        suite="PolyBench/GPU",
        representative_case="gemm/3mm/3DConvolution/atax/bicg/syrk",
        category="dense / regular",
        sim_time_s=5.0,
        status="estimated",
        evidence="APEs/*/polybench.md",
        dominant_bottleneck_hint="simulator throughput for compute-heavy configs",
    ),
    BenchmarkObservation(
        suite="NCCL-tests",
        representative_case="all_reduce_perf and related collectives",
        category="multi-GPU communication",
        status="excluded",
        evidence="https://github.com/NVIDIA/nccl-tests",
        dominant_bottleneck_hint="different problem class",
    ),
    BenchmarkObservation(
        suite="OSU micro-benchmarks",
        representative_case="MPI latency and bandwidth microbenchmarks",
        category="communication / network",
        status="excluded",
        evidence="https://github.com/forresti/osu-micro-benchmarks",
        dominant_bottleneck_hint="different problem class",
    ),
    BenchmarkObservation(
        suite="MLPerf Inference / Training",
        representative_case="BERT/ResNet/DLRM/Llama2/Mixtral",
        category="full workload anchor",
        sim_time_s=200000.0,
        status="appendix_only",
        evidence="https://docs.mlcommons.org/inference/index_gh/",
        dominant_bottleneck_hint="full workload scale / trace explosion",
    ),
]
```

- [ ] **Step 4: Run catalog tests and confirm they pass**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests/test_catalog.py -q
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add experiments/trace_bottleneck_map/catalog.py \
  experiments/trace_bottleneck_map/tests/test_catalog.py
git commit -m "feat: add benchmark suite catalog"
```

---

### Task 4: Cost Map Builder CLI

**Files:**
- Create: `experiments/trace_bottleneck_map/build_bottleneck_map.py`
- Test: `experiments/trace_bottleneck_map/tests/test_build_bottleneck_map.py`

- [ ] **Step 1: Write failing CLI tests**

Create `experiments/trace_bottleneck_map/tests/test_build_bottleneck_map.py`:

```python
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("experiments/trace_bottleneck_map/build_bottleneck_map.py")


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
            str(trace_md),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    json_path = out_dir / "benchmark_cost_map.json"
    md_path = out_dir / "benchmark_cost_map.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text())
    measured = [row for row in payload["records"] if row["status"] == "measured"]
    assert any(row["representative_case"] == "l2_bw_32f" for row in measured)
    assert any(row["suite"] == "NCCL-tests" and row["status"] == "excluded" for row in payload["records"])
    assert "trace export / I/O" in md_path.read_text()


def test_builder_rejects_missing_trace_benchmark(tmp_path):
    out_dir = tmp_path / "out"
    missing = tmp_path / "missing.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-benchmark-md",
            str(missing),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "trace benchmark file not found" in proc.stderr
```

- [ ] **Step 2: Run CLI tests and confirm they fail**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests/test_build_bottleneck_map.py -q
```

Expected: FAIL with missing script or missing output errors.

- [ ] **Step 3: Implement the CLI**

Create `experiments/trace_bottleneck_map/build_bottleneck_map.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

    all_observations = measured + ESTIMATED_SUITE_OBSERVATIONS
    records = _records_from_observations(all_observations)
    payload = {
        "report_name": "Trace Bottleneck Cost Map",
        "trace_benchmark_source": str(Path(args.trace_benchmark_md).resolve()),
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
```

- [ ] **Step 4: Run CLI tests and confirm they pass**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests/test_build_bottleneck_map.py -q
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Run all trace bottleneck tests**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests -q
```

Expected: PASS, 14 tests.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add experiments/trace_bottleneck_map/build_bottleneck_map.py \
  experiments/trace_bottleneck_map/tests/test_build_bottleneck_map.py
git commit -m "feat: build trace bottleneck cost map"
```

---

### Task 5: Generate Review Artifacts

**Files:**
- Generate: `artifacts/trace_bottleneck_map/benchmark_cost_map.json`
- Generate: `artifacts/trace_bottleneck_map/benchmark_cost_map.md`

- [ ] **Step 1: Generate artifacts from current local evidence**

Run:

```bash
python experiments/trace_bottleneck_map/build_bottleneck_map.py \
  --trace-benchmark-md docs/trace-benchmark-2026-04-03.md \
  --output-dir artifacts/trace_bottleneck_map
```

Expected output:

```text
Wrote artifacts/trace_bottleneck_map/benchmark_cost_map.json
Wrote artifacts/trace_bottleneck_map/benchmark_cost_map.md
```

- [ ] **Step 2: Inspect the JSON for required sections**

Run:

```bash
jq '.report_name, (.records | length), [.records[] | select(.status=="measured")] | length' \
  artifacts/trace_bottleneck_map/benchmark_cost_map.json
```

Expected output shape:

```text
"Trace Bottleneck Cost Map"
28
18
```

The exact counts above assume the current `docs/trace-benchmark-2026-04-03.md` table with 18 measured rows and 10 catalog rows.

- [ ] **Step 3: Inspect the Markdown summary**

Run:

```bash
sed -n '1,80p' artifacts/trace_bottleneck_map/benchmark_cost_map.md
```

Expected: the report contains `## Summary`, `## Measured Bottleneck Counts`, and a cost-map table with `GPU_Microbenchmark`, `BabelStream`, `CUTLASS profiler`, `NCCL-tests`, and `MLPerf Inference / Training` rows.

- [ ] **Step 4: Run regression tests before committing artifacts**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests -q
```

Expected: PASS, 14 tests.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add artifacts/trace_bottleneck_map/benchmark_cost_map.json \
  artifacts/trace_bottleneck_map/benchmark_cost_map.md
git commit -m "data: add trace bottleneck cost map artifacts"
```

---

### Task 6: End-to-End Guardrails

**Files:**
- Modify: `experiments/trace_bottleneck_map/tests/test_build_bottleneck_map.py`

- [ ] **Step 1: Add a guardrail test for communication exclusions**

Append this test to `experiments/trace_bottleneck_map/tests/test_build_bottleneck_map.py`:

```python
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
```

- [ ] **Step 2: Run the guardrail test**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests/test_build_bottleneck_map.py::test_generated_report_keeps_communication_suites_out_of_measured_main_table -q
```

Expected: PASS.

- [ ] **Step 3: Run full verification**

Run:

```bash
pytest experiments/trace_bottleneck_map/tests -q
python experiments/trace_bottleneck_map/build_bottleneck_map.py \
  --trace-benchmark-md docs/trace-benchmark-2026-04-03.md \
  --output-dir artifacts/trace_bottleneck_map
git diff --check -- experiments/trace_bottleneck_map artifacts/trace_bottleneck_map
```

Expected:

```text
15 passed
Wrote artifacts/trace_bottleneck_map/benchmark_cost_map.json
Wrote artifacts/trace_bottleneck_map/benchmark_cost_map.md
```

The path-limited `git diff --check` must produce no whitespace errors.

- [ ] **Step 4: Commit Task 6**

Run:

```bash
git add experiments/trace_bottleneck_map/tests/test_build_bottleneck_map.py \
  artifacts/trace_bottleneck_map/benchmark_cost_map.json \
  artifacts/trace_bottleneck_map/benchmark_cost_map.md
git commit -m "test: guard bottleneck map suite boundaries"
```

---

## Final Verification

Run:

```bash
pytest experiments/trace_bottleneck_map/tests -q
python experiments/trace_bottleneck_map/build_bottleneck_map.py \
  --trace-benchmark-md docs/trace-benchmark-2026-04-03.md \
  --output-dir artifacts/trace_bottleneck_map
jq '.records | length' artifacts/trace_bottleneck_map/benchmark_cost_map.json
git status --short
```

Expected:

- pytest passes.
- The builder writes both cost-map artifacts.
- The JSON record count is greater than the measured-only count because estimated, excluded, and appendix rows are included.
- `git status --short` shows the plan-owned files plus any pre-existing unrelated worktree changes. Commit only the plan-owned paths listed in each task.

## Self-Review Checklist

- Spec coverage: Tasks 1-6 cover cost model, mainstream suite catalog, local measured anchors, main/appendix boundaries, generated JSON, generated Markdown, and verification.
- Type consistency: `BenchmarkObservation` fields match all parser, catalog, and builder uses.
- Scope control: no task touches L1 selector, B-line consumer, simulator source, tracer source, or streaming compression implementation.
- Artifact boundary: generated cost-map artifacts live under `artifacts/trace_bottleneck_map/`, not under L1 artifacts.
