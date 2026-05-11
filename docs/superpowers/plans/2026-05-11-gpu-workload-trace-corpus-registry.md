# GPU Workload Trace Corpus Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Gate C0/C1 registry tooling for the GPU Workload Trace Corpus: machine-readable source registry plus a first workload registry draft from local downloaded sources.

**Architecture:** Add two small Python CLI tools under `scripts/`: one converts `clone_status.tsv` into `registry/source_registry.json` and `.md`, the other scans known source trees and emits `registry/workload_registry.json` and `.md`. Keep all large third-party source trees outside git under `/home/dyf/workloads/...`; repository files contain only scripts, schemas, and reports.

**Tech Stack:** Python standard library, JSON, Markdown, pytest.

---

## File Structure

- Create: `scripts/generate_source_registry.py`
  - Reads external `clone_status.tsv`, probes each local git checkout, writes source registry JSON and Markdown.
- Create: `scripts/generate_workload_registry.py`
  - Reads source registry JSON, applies source-specific scan rules, writes candidate workload registry JSON and Markdown.
- Create: `tests/test_workload_registry_tools.py`
  - Unit tests for parsing clone status, source record generation, and workload scan behavior using tiny fixture directories.
- Create: `registry/.gitkeep`
  - Keeps the registry output directory present in git.
- Generated: `registry/source_registry.json`
- Generated: `registry/source_registry.md`
- Generated: `registry/workload_registry.json`
- Generated: `registry/workload_registry.md`
- Modify: `docs/workload_download_status.md`
  - Add links to generated registry artifacts after generation.
- Modify: `progress.md`
  - Record implementation and verification.

## Task 1: Source Registry Tool

**Files:**
- Create: `scripts/generate_source_registry.py`
- Test: `tests/test_workload_registry_tools.py`
- Create: `registry/.gitkeep`

- [ ] **Step 1: Write failing tests for clone status parsing and source registry generation**

Add this to `tests/test_workload_registry_tools.py`:

```python
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_source_registry import (
    build_source_registry,
    infer_clone_mode,
    parse_clone_status,
)


def init_git_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("# fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=path, text=True).strip()


def test_parse_clone_status_reads_tsv_rows(tmp_path):
    status = tmp_path / "clone_status.tsv"
    status.write_text(
        "name\tstatus\tcommit\tpath\turl\n"
        "gpu-rodinia\texists\tabc123\t/tmp/gpu-rodinia\thttps://example/rodinia.git\n"
    )

    rows = parse_clone_status(status)

    assert rows == [
        {
            "name": "gpu-rodinia",
            "status": "exists",
            "commit": "abc123",
            "path": "/tmp/gpu-rodinia",
            "url": "https://example/rodinia.git",
        }
    ]


def test_infer_clone_mode_detects_sparse_checkout(tmp_path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    subprocess.run(["git", "config", "core.sparseCheckout", "true"], cwd=repo, check=True)

    assert infer_clone_mode(repo) == "sparse_partial"


def test_build_source_registry_uses_local_git_commit(tmp_path):
    source = tmp_path / "sources" / "gpu-rodinia"
    commit = init_git_repo(source)
    status = tmp_path / "clone_status.tsv"
    status.write_text(
        "name\tstatus\tcommit\tpath\turl\n"
        f"gpu-rodinia\texists\told\t{source}\thttps://example/rodinia.git\n"
    )

    registry = build_source_registry(status)

    assert registry["schema_version"] == "source_registry_v1"
    assert registry["sources"][0]["source_id"] == "gpu-rodinia"
    assert registry["sources"][0]["commit"] == commit
    assert registry["sources"][0]["availability_status"] == "source_available"
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest -q tests/test_workload_registry_tools.py
```

Expected result:

```text
ModuleNotFoundError: No module named 'scripts.generate_source_registry'
```

- [ ] **Step 3: Implement `scripts/generate_source_registry.py`**

Create `scripts/generate_source_registry.py` with:

```python
#!/usr/bin/env python3
"""Generate GPU workload corpus source registry artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("/home/dyf/workloads/trace-compressions-industrial-codex-workload")

SOURCE_TYPES = {
    "gpu-rodinia": ("benchmark_suite", "control_and_candidate"),
    "gpu-parboil": ("benchmark_suite", "control_and_candidate"),
    "shoc": ("benchmark_suite", "control_and_candidate"),
    "altis": ("benchmark_suite", "candidate"),
    "deepbench": ("dnn_primitive_suite", "candidate"),
    "cutlass": ("kernel_generator", "candidate"),
    "mlperf-inference": ("full_network_suite", "candidate"),
    "gunrock": ("graph_suite", "candidate"),
    "pannotia": ("graph_suite", "candidate"),
    "hecbench": ("heterogeneous_suite", "candidate"),
    "lammps": ("hpc_full_application", "candidate"),
    "gromacs": ("hpc_full_application", "candidate"),
}


def run_git(path: Path, args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        return None


def parse_clone_status(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def infer_clone_mode(path: Path) -> str:
    sparse = run_git(path, ["config", "--get", "core.sparseCheckout"])
    partial = run_git(path, ["config", "--get", "remote.origin.promisor"])
    if sparse == "true" or partial == "true":
        return "sparse_partial"
    return "shallow_or_full"


def infer_availability(path: Path, status: str) -> str:
    if not path.exists():
        return "source_unavailable"
    if not (path / ".git").exists():
        return "source_unavailable"
    if infer_clone_mode(path) == "sparse_partial":
        return "source_sparse_available"
    if status.startswith("failed"):
        return "source_sparse_available"
    return "source_available"


def build_source_registry(status_path: Path) -> dict:
    rows = parse_clone_status(status_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    sources = []
    for row in rows:
        source_id = row["name"]
        local_path = Path(row["path"])
        commit = run_git(local_path, ["rev-parse", "--short", "HEAD"]) or row["commit"]
        source_type, corpus_role = SOURCE_TYPES.get(source_id, ("unknown", "candidate"))
        sources.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "corpus_role": corpus_role,
                "url": row["url"],
                "local_path": str(local_path),
                "commit": commit,
                "clone_status": row["status"],
                "clone_mode": infer_clone_mode(local_path) if local_path.exists() else "unavailable",
                "availability_status": infer_availability(local_path, row["status"]),
                "license_status": "needs_review",
            }
        )
    return {"schema_version": "source_registry_v1", "generated_at": generated_at, "sources": sources}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def render_markdown(registry: dict) -> str:
    lines = [
        "# GPU Workload Trace Corpus Source Registry",
        "",
        f"Generated at: `{registry['generated_at']}`",
        "",
        "| Source | Type | Role | Availability | Clone Mode | Commit | License |",
        "|--------|------|------|--------------|------------|--------|---------|",
    ]
    for source in registry["sources"]:
        lines.append(
            "| `{source_id}` | `{source_type}` | `{corpus_role}` | `{availability_status}` | "
            "`{clone_mode}` | `{commit}` | `{license_status}` |".format(**source)
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("registry"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    registry = build_source_registry(args.root / "clone_status.tsv")
    write_json(args.out_dir / "source_registry.json", registry)
    (args.out_dir / "source_registry.md").write_text(render_markdown(registry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run source registry tests**

Run:

```bash
pytest -q tests/test_workload_registry_tools.py
```

Expected result:

```text
3 passed
```

- [ ] **Step 5: Generate source registry artifacts**

Run:

```bash
python scripts/generate_source_registry.py
```

Expected outputs:

```text
registry/source_registry.json
registry/source_registry.md
```

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add scripts/generate_source_registry.py tests/test_workload_registry_tools.py registry/.gitkeep registry/source_registry.json registry/source_registry.md
git commit -m "feat: add workload source registry generator"
```

## Task 2: Workload Registry Draft Tool

**Files:**
- Create: `scripts/generate_workload_registry.py`
- Modify: `tests/test_workload_registry_tools.py`

- [ ] **Step 1: Write failing tests for workload candidate generation**

Append to `tests/test_workload_registry_tools.py`:

```python
from scripts.generate_workload_registry import discover_workloads_for_source


def test_discover_workloads_for_gpu_rodinia_cuda_dirs(tmp_path):
    root = tmp_path / "gpu-rodinia"
    (root / "cuda" / "bfs").mkdir(parents=True)
    (root / "cuda" / "hotspot").mkdir(parents=True)

    workloads = discover_workloads_for_source("gpu-rodinia", root)
    ids = {item["workload_id"] for item in workloads}

    assert "gpu-rodinia_bfs" in ids
    assert "gpu-rodinia_hotspot" in ids
    assert all(item["source_id"] == "gpu-rodinia" for item in workloads)


def test_discover_workloads_for_full_network_source_uses_curated_candidates(tmp_path):
    root = tmp_path / "mlperf-inference"
    root.mkdir()

    workloads = discover_workloads_for_source("mlperf-inference", root)
    ids = {item["workload_id"] for item in workloads}

    assert "mlperf-inference_bert" in ids
    assert "mlperf-inference_resnet50" in ids
    assert all(item["workload_family"] == "full_network" for item in workloads)
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest -q tests/test_workload_registry_tools.py
```

Expected result:

```text
ModuleNotFoundError: No module named 'scripts.generate_workload_registry'
```

- [ ] **Step 3: Implement `scripts/generate_workload_registry.py`**

Create `scripts/generate_workload_registry.py` with:

```python
#!/usr/bin/env python3
"""Generate a draft workload registry from corpus source registry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CURATED = {
    "mlperf-inference": [
        ("bert", "full_network", "high", "medium", "medium"),
        ("resnet50", "full_network", "high", "large", "low"),
        ("dlrm-v2", "full_network", "high", "medium", "high"),
        ("retinanet", "full_network", "high", "large", "medium"),
        ("3d-unet", "full_network", "high", "large", "medium"),
        ("stable-diffusion", "full_network", "high", "large", "medium"),
    ],
    "deepbench": [
        ("gemm", "dnn_primitive", "medium", "large", "low"),
        ("rnn", "dnn_primitive", "medium", "medium", "medium"),
        ("convolution", "dnn_primitive", "medium", "large", "low"),
    ],
    "cutlass": [
        ("gemm", "kernel_generator", "medium", "large", "low"),
        ("conv", "kernel_generator", "medium", "large", "low"),
        ("attention", "kernel_generator", "high", "large", "medium"),
    ],
    "gunrock": [
        ("bfs", "irregular_graph", "medium", "medium", "high"),
        ("sssp", "irregular_graph", "medium", "medium", "high"),
        ("pagerank", "irregular_graph", "medium", "medium", "high"),
        ("connected-components", "irregular_graph", "medium", "medium", "high"),
    ],
    "pannotia": [
        ("bfs", "irregular_graph", "medium", "medium", "high"),
        ("coloring", "irregular_graph", "medium", "medium", "high"),
        ("pagerank", "irregular_graph", "medium", "medium", "high"),
    ],
    "lammps": [
        ("lj-small-step", "hpc_full_application", "high", "large", "medium"),
        ("eam-small-step", "hpc_full_application", "high", "large", "medium"),
    ],
    "gromacs": [
        ("water-small-step", "hpc_full_application", "high", "large", "medium"),
        ("protein-small-step", "hpc_full_application", "high", "large", "medium"),
    ],
}


def pressure_for(source_id: str, name: str) -> tuple[str, str, str, str]:
    lowered = name.lower()
    if any(token in lowered for token in ["bfs", "spmv", "histo", "streamcluster", "particle"]):
        return ("irregular_or_sparse", "medium", "medium", "high")
    if any(token in lowered for token in ["sgemm", "gemm", "lbm", "stencil", "cfd", "lava", "hotspot"]):
        return ("benchmark_kernel", "medium", "large", "low")
    if source_id in {"shoc", "altis"}:
        return ("benchmark_suite_candidate", "medium", "medium", "medium")
    return ("benchmark_kernel", "low", "medium", "medium")


def make_record(source_id: str, name: str, family: str, kernel_count: str, large_kernel: str, irregularity: str, relative_path: str) -> dict:
    return {
        "workload_id": f"{source_id}_{name}".replace("/", "_"),
        "source_id": source_id,
        "workload_name": name,
        "workload_family": family,
        "claim_role": "candidate",
        "relative_path": relative_path,
        "expected_kernel_count_class": kernel_count,
        "expected_large_kernel_class": large_kernel,
        "expected_irregularity_class": irregularity,
        "build_status": "pending",
        "run_status": "pending",
        "input_status": "pending",
        "license_status": "needs_review",
    }


def discover_workloads_for_source(source_id: str, root: Path) -> list[dict]:
    if source_id in CURATED:
        return [
            make_record(source_id, name, family, kernel_count, large_kernel, irregularity, ".")
            for name, family, kernel_count, large_kernel, irregularity in CURATED[source_id]
        ]

    candidates: list[dict] = []
    scan_roots = [root / "cuda", root / "CUDA", root / "src", root / "test"]
    for scan_root in scan_roots:
        if not scan_root.exists() or not scan_root.is_dir():
            continue
        for child in sorted(scan_root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            family, kernel_count, large_kernel, irregularity = pressure_for(source_id, child.name)
            candidates.append(
                make_record(source_id, child.name, family, kernel_count, large_kernel, irregularity, str(child.relative_to(root)))
            )
    return candidates


def build_workload_registry(source_registry_path: Path) -> dict:
    source_registry = json.loads(source_registry_path.read_text())
    workloads = []
    for source in source_registry["sources"]:
        if not source["availability_status"].startswith("source"):
            continue
        workloads.extend(discover_workloads_for_source(source["source_id"], Path(source["local_path"])))
    return {
        "schema_version": "workload_registry_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workloads": workloads,
    }


def render_markdown(registry: dict) -> str:
    lines = [
        "# GPU Workload Trace Corpus Workload Registry Draft",
        "",
        f"Generated at: `{registry['generated_at']}`",
        "",
        "| Workload | Source | Family | Kernel Count | Large Kernel | Irregularity | Path |",
        "|----------|--------|--------|--------------|--------------|--------------|------|",
    ]
    for item in registry["workloads"]:
        lines.append(
            "| `{workload_id}` | `{source_id}` | `{workload_family}` | `{expected_kernel_count_class}` | "
            "`{expected_large_kernel_class}` | `{expected_irregularity_class}` | `{relative_path}` |".format(**item)
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-registry", type=Path, default=Path("registry/source_registry.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("registry"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    registry = build_workload_registry(args.source_registry)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "workload_registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    (args.out_dir / "workload_registry.md").write_text(render_markdown(registry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run workload registry tests**

Run:

```bash
pytest -q tests/test_workload_registry_tools.py
```

Expected result:

```text
5 passed
```

- [ ] **Step 5: Generate workload registry artifacts**

Run:

```bash
python scripts/generate_source_registry.py
python scripts/generate_workload_registry.py
```

Expected outputs:

```text
registry/workload_registry.json
registry/workload_registry.md
```

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add scripts/generate_workload_registry.py tests/test_workload_registry_tools.py registry/workload_registry.json registry/workload_registry.md
git commit -m "feat: add workload registry draft generator"
```

## Task 3: Documentation and Verification

**Files:**
- Modify: `docs/workload_download_status.md`
- Modify: `progress.md`
- Modify: `task_plan.md`

- [ ] **Step 1: Update download status document**

Add this section to `docs/workload_download_status.md`:

```markdown
## Registry Artifacts

- `registry/source_registry.json`
- `registry/source_registry.md`
- `registry/workload_registry.json`
- `registry/workload_registry.md`

`workload_registry.*` is a draft. Build/run/input/license statuses remain `pending` or `needs_review` until Gate C2/C3.
```

- [ ] **Step 2: Run all registry tests and smoke generation**

Run:

```bash
pytest -q tests/test_workload_registry_tools.py
python scripts/generate_source_registry.py
python scripts/generate_workload_registry.py
```

Expected:

```text
5 passed
```

The two Python commands should exit with code 0.

- [ ] **Step 3: Run existing baseline tests**

Run:

```bash
pytest -q tests/test_build_kernel_cards.py tests/test_build_middle_layer.py tests/test_check_analysis_cards.py
```

Expected:

```text
22 passed
```

- [ ] **Step 4: Update progress tracking**

In `progress.md`, record:

```markdown
### 阶段 5：Gate C0/C1 Registry
- **状态：** complete
- 执行的操作：
  - 生成 source registry。
  - 生成 workload registry draft。
  - 运行 registry tests 和 baseline tests。
- 创建/修改的文件：
  - `scripts/generate_source_registry.py`
  - `scripts/generate_workload_registry.py`
  - `tests/test_workload_registry_tools.py`
  - `registry/source_registry.json`
  - `registry/source_registry.md`
  - `registry/workload_registry.json`
  - `registry/workload_registry.md`
```

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add docs/workload_download_status.md progress.md task_plan.md
git commit -m "docs: record workload registry generation status"
```

## Self-Review

Spec coverage:

- Gate C0 is covered by Task 1 and generated `source_registry.*`.
- Gate C1 is covered by Task 2 and generated `workload_registry.*`.
- Gate C2-C5 are intentionally outside this plan. They require input asset planning, trace acquisition, measured artifacts, and dataset split policy.

No large downloaded third-party source, model weight, dataset, trace, or run artifact should be added to git.

