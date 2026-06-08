# A Line GCL ResNet-50 Full Trace Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and validate the complete real ResNet-50 full-trace GCL reproduction path from the existing formal Gate0 NVBit trace root, without bounded invocation slices or synthetic fixtures as final evidence.

**Architecture:** Keep the existing Gate1-Gate7/Gate9 pipeline implementation, but add a formal full-trace runner, full-trace audit manifest, and acceptance tests that prove the final run used `invocation_limit=None` and no `invocation_ids` filter. Unit tests may use small fixtures and monkeypatching, but the final RLCR evidence must include one real full-trace command against `artifacts/gcl_resnet50_gate0_formal_trace/traces`.

**Tech Stack:** Python, pytest, existing `experiments.gcl_phase_b.resnet50_gate_pipeline`, existing real ResNet-50 Gate0 artifacts, JSON artifacts.

---

## Current Context

The previous RLCR completed formal bounded-slice validation. It proved that the real Gate0 root can drive Gate1-Gate7 and that a selected non-degenerate real invocation slice reaches `selected_k = 2`.

That is not the same as a full-network run.

Existing formal root:

```text
artifacts/gcl_resnet50_gate0_formal_trace/traces
```

Known scale:

```text
trace root size: 2.4G
threadblock protobuf files: 124876
scheduler metadata kernel invocations: 265
scheduler metadata CTA records: 124876
```

Existing pipeline entrypoint:

```python
experiments.gcl_phase_b.resnet50_gate_pipeline.run_resnet50_gate1_to_gate7(
    root,
    out_dir,
    seed=20260606,
    baseline_artifacts_path=None,
    invocation_limit=None,
    invocation_ids=None,
)
```

The full-trace RLCR must not claim success from:

- `invocation_limit=1`;
- `invocation_limit=2`;
- `invocation_ids=["d_0_s_0_k_267", "d_0_s_0_k_276", "d_0_s_0_k_291"]`;
- synthetic fixture roots;
- artifact-shape helper roots;
- mini-transformer traces.

## Acceptance Criteria

- AC-1: A dedicated full-trace runner exists and always calls the existing pipeline with `invocation_limit=None` and `invocation_ids=None`.
- AC-2: The full-trace runner writes a `resnet50_full_trace_reproduction_manifest.json` with formal root path, source Gate0 manifest hash, invocation count, CTA count, output artifact hashes, elapsed time, and resource status.
- AC-3: The runner rejects or marks non-final any run whose adapter bundle records `formal_replay_invocation_limit` or whose source uses an invocation id filter.
- AC-4: The runner can be tested without running the 2.4G trace by monkeypatching the pipeline, while still proving the exact full-run call contract.
- AC-5: The actual full-trace command is documented and writes artifacts to a stable output root under `artifacts/gcl_resnet50_full_trace_reproduction/`.
- AC-6: The full-trace run reaches at least Gate7 with `final_gate = gate9_report_only` when no baseline is supplied, or `final_gate = gate9_evaluated` when valid baseline artifacts are supplied.
- AC-7: Full-trace success evidence includes the generated `kernel_embedding_table.json`, `selector_artifacts.json`, `gate7_cluster_correctness_manifest.json`, and `resnet50_full_trace_reproduction_manifest.json`.
- AC-8: If the full run cannot complete because of resource or runtime limits, the runner writes `resnet50_full_trace_reproduction_blocker_report.json` and must not emit a success manifest.

## File Structure

- Create: `scripts/run_resnet50_full_trace_gcl.py`
  - Dedicated CLI for the real full-trace reproduction run.
  - Calls `run_resnet50_gate1_to_gate7()` with no invocation slicing.
  - Writes full-trace audit manifest or blocker report.

- Create: `tests/gcl_resnet50/test_full_trace_reproduction_runner.py`
  - Unit tests for the runner contract.
  - Uses monkeypatching to avoid running the real 2.4G trace in unit tests.

- Modify: `experiments/gcl_phase_b/resnet50_gate_pipeline.py`
  - Add enough manifest metadata to distinguish full run from bounded run if the current manifest is insufficient.
  - Keep existing bounded-slice tests working.

- Modify: `tests/gcl_phase_b/test_resnet50_gate_pipeline.py`
  - Add focused tests for the new manifest metadata if pipeline-level metadata is added.

- Optional Create: `artifacts/gcl_resnet50_full_trace_reproduction/`
  - Created by the full-run command, not by unit tests.

## Task 1: Add Full-Trace Runner Contract Tests

**Files:**
- Create: `tests/gcl_resnet50/test_full_trace_reproduction_runner.py`
- Create later: `scripts/run_resnet50_full_trace_gcl.py`

- [ ] **Step 1: Write the failing test for the no-slicing call contract**

Create `tests/gcl_resnet50/test_full_trace_reproduction_runner.py` with:

```python
import json
from pathlib import Path

from scripts import run_resnet50_full_trace_gcl


def test_full_trace_runner_calls_pipeline_without_invocation_slicing(tmp_path, monkeypatch):
    calls = {}

    def fake_pipeline(root, out_dir, seed, baseline_artifacts_path=None, invocation_limit=None, invocation_ids=None):
        calls["root"] = root
        calls["out_dir"] = out_dir
        calls["seed"] = seed
        calls["baseline_artifacts_path"] = baseline_artifacts_path
        calls["invocation_limit"] = invocation_limit
        calls["invocation_ids"] = invocation_ids
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "kernel_embedding_table.json").write_text(json.dumps({"embeddings": [1, 2]}))
        (out_dir / "selector_artifacts.json").write_text(json.dumps({"k_selection_report": {"selected_k": 2}}))
        (out_dir / "gate7_cluster_correctness_manifest.json").write_text(
            json.dumps({"gate7_cluster_correctness_manifest_hash": "gate7-hash"})
        )
        return {
            "artifact_type": "gcl_resnet50_gate1_7_pipeline_manifest",
            "final_gate": "gate9_report_only",
            "hashes": {
                "embedding_table_hash": "embedding-hash",
                "selector_manifest_hash": "selector-hash",
                "gate7_correctness_manifest_hash": "gate7-hash",
                "gate8_tuning_vector_proposal_hash": "gate8-hash",
                "gate9_sampled_vs_full_evaluation_hash": "gate9-hash",
            },
            "pipeline_manifest_hash": "pipeline-hash",
        }

    monkeypatch.setattr(run_resnet50_full_trace_gcl, "run_resnet50_gate1_to_gate7", fake_pipeline)
    root = tmp_path / "formal_root"
    root.mkdir()
    (root / "gate0_trace_acquisition_manifest.json").write_text(
        json.dumps(
            {
                "artifact_status": "formal",
                "formal_input_eligible": True,
                "input_scope": "full_resnet50_inference_trace",
                "kernel_invocation_count": 265,
                "cta_record_count": 124876,
                "gate0_trace_acquisition_manifest_hash": "gate0-hash",
            }
        )
    )

    result = run_resnet50_full_trace_gcl.run_full_trace_reproduction(
        input_root=root,
        out_dir=tmp_path / "out",
        seed=20260608,
        baseline_artifacts=None,
    )

    assert calls["invocation_limit"] is None
    assert calls["invocation_ids"] is None
    assert result["run_scope"] == "real_resnet50_full_trace"
    assert result["formal_full_trace_run"] is True
    assert result["input_kernel_invocation_count"] == 265
    assert result["input_cta_record_count"] == 124876
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest -q tests/gcl_resnet50/test_full_trace_reproduction_runner.py::test_full_trace_runner_calls_pipeline_without_invocation_slicing
```

Expected: FAIL because `scripts/run_resnet50_full_trace_gcl.py` does not exist.

## Task 2: Implement the Minimal Full-Trace Runner

**Files:**
- Create: `scripts/run_resnet50_full_trace_gcl.py`

- [ ] **Step 1: Add the runner implementation**

Create `scripts/run_resnet50_full_trace_gcl.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate7
from experiments.gcl_phase_b.utils import stable_hash


FULL_TRACE_MANIFEST = "resnet50_full_trace_reproduction_manifest.json"
FULL_TRACE_BLOCKER = "resnet50_full_trace_reproduction_blocker_report.json"
GATE0_MANIFEST = "gate0_trace_acquisition_manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_gate0_manifest(input_root: Path) -> dict[str, Any]:
    manifest_path = input_root / GATE0_MANIFEST
    if not manifest_path.exists():
        raise ValueError(f"missing Gate0 formal manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_status") != "formal":
        raise ValueError("Gate0 manifest is not formal")
    if manifest.get("formal_input_eligible") is not True:
        raise ValueError("Gate0 manifest is not formal-input eligible")
    if manifest.get("input_scope") != "full_resnet50_inference_trace":
        raise ValueError(f"Gate0 input_scope is not full ResNet50: {manifest.get('input_scope')}")
    return manifest


def _artifact_presence(out_dir: Path) -> dict[str, bool]:
    filenames = [
        "kernel_embedding_table.json",
        "selector_artifacts.json",
        "gate7_cluster_correctness_manifest.json",
        "gate1_7_pipeline_manifest.json",
    ]
    return {filename: (out_dir / filename).exists() for filename in filenames}


def run_full_trace_reproduction(
    *,
    input_root: Path,
    out_dir: Path,
    seed: int,
    baseline_artifacts: Path | None,
) -> dict[str, Any]:
    gate0_manifest = _load_gate0_manifest(input_root)
    started = time.monotonic()
    try:
        pipeline_manifest = run_resnet50_gate1_to_gate7(
            input_root,
            out_dir,
            seed=seed,
            baseline_artifacts_path=baseline_artifacts,
            invocation_limit=None,
            invocation_ids=None,
        )
    except Exception as exc:
        blocker = {
            "artifact_type": "gcl_resnet50_full_trace_reproduction_blocker_report",
            "run_scope": "real_resnet50_full_trace",
            "formal_full_trace_run": False,
            "seed": seed,
            "input_root": str(input_root),
            "blocker_reason": type(exc).__name__,
            "blocker_message": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        blocker["blocker_report_hash"] = stable_hash(blocker)
        _write_json(out_dir / FULL_TRACE_BLOCKER, blocker)
        raise

    manifest = {
        "artifact_type": "gcl_resnet50_full_trace_reproduction_manifest",
        "artifact_version": "full_trace_reproduction_manifest_v1",
        "run_scope": "real_resnet50_full_trace",
        "formal_full_trace_run": True,
        "seed": seed,
        "input_root": str(input_root),
        "source_gate0_manifest_hash": gate0_manifest.get("gate0_trace_acquisition_manifest_hash"),
        "input_kernel_invocation_count": gate0_manifest.get("kernel_invocation_count"),
        "input_cta_record_count": gate0_manifest.get("cta_record_count"),
        "invocation_limit": None,
        "invocation_ids": None,
        "baseline_artifacts_path": str(baseline_artifacts) if baseline_artifacts else None,
        "final_gate": pipeline_manifest["final_gate"],
        "pipeline_manifest_hash": pipeline_manifest["pipeline_manifest_hash"],
        "pipeline_hashes": pipeline_manifest["hashes"],
        "artifact_presence": _artifact_presence(out_dir),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "resource_status": "completed",
    }
    manifest["full_trace_reproduction_manifest_hash"] = stable_hash(manifest)
    _write_json(out_dir / FULL_TRACE_MANIFEST, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--baseline-artifacts", type=Path)
    args = parser.parse_args()
    manifest = run_full_trace_reproduction(
        input_root=args.input_root,
        out_dir=args.out,
        seed=args.seed,
        baseline_artifacts=args.baseline_artifacts,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the contract test**

Run:

```bash
pytest -q tests/gcl_resnet50/test_full_trace_reproduction_runner.py::test_full_trace_runner_calls_pipeline_without_invocation_slicing
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_resnet50_full_trace_gcl.py tests/gcl_resnet50/test_full_trace_reproduction_runner.py
git commit -m "feat: add resnet50 full trace gcl runner"
```

## Task 3: Reject Non-Full or Non-Formal Inputs

**Files:**
- Modify: `tests/gcl_resnet50/test_full_trace_reproduction_runner.py`
- Modify: `scripts/run_resnet50_full_trace_gcl.py`

- [ ] **Step 1: Add failing tests for invalid Gate0 manifests**

Append to `tests/gcl_resnet50/test_full_trace_reproduction_runner.py`:

```python
import pytest


def test_full_trace_runner_rejects_non_full_gate0_scope(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "gate0_trace_acquisition_manifest.json").write_text(
        json.dumps(
            {
                "artifact_status": "formal",
                "formal_input_eligible": True,
                "input_scope": "bounded_invocation_slice",
            }
        )
    )

    with pytest.raises(ValueError, match="full ResNet50"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=tmp_path / "out",
            seed=20260608,
            baseline_artifacts=None,
        )


def test_full_trace_runner_rejects_debug_gate0_manifest(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "gate0_trace_acquisition_manifest.json").write_text(
        json.dumps(
            {
                "artifact_status": "debug_not_formal",
                "formal_input_eligible": False,
                "input_scope": "full_resnet50_inference_trace",
            }
        )
    )

    with pytest.raises(ValueError, match="not formal"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=tmp_path / "out",
            seed=20260608,
            baseline_artifacts=None,
        )
```

- [ ] **Step 2: Run the tests**

Run:

```bash
pytest -q tests/gcl_resnet50/test_full_trace_reproduction_runner.py
```

Expected: PASS if Task 2 implementation already checks these fields. If the tests fail, tighten `_load_gate0_manifest()`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_resnet50_full_trace_gcl.py tests/gcl_resnet50/test_full_trace_reproduction_runner.py
git commit -m "test: reject non-full resnet50 trace inputs"
```

## Task 4: Add Full-Trace Pipeline Metadata Guard

**Files:**
- Modify: `experiments/gcl_phase_b/resnet50_gate_pipeline.py`
- Modify: `tests/gcl_phase_b/test_resnet50_gate_pipeline.py`

- [ ] **Step 1: Write failing pipeline metadata test**

Add this test to `tests/gcl_phase_b/test_resnet50_gate_pipeline.py`:

```python
def test_resnet50_gate_pipeline_manifest_records_full_trace_scope(tmp_path):
    out_dir = tmp_path / "full_scope"

    manifest = run_resnet50_gate1_to_gate7(
        FORMAL_ROOT,
        out_dir,
        seed=20260607,
        invocation_limit=None,
        invocation_ids=None,
    )

    assert manifest["run_scope"] == "real_resnet50_full_trace"
    assert manifest["invocation_limit"] is None
    assert manifest["invocation_ids"] is None
    assert manifest["input_kernel_invocation_count"] == 265
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate_pipeline_manifest_records_full_trace_scope
```

Expected: FAIL because the manifest does not yet record these fields.

- [ ] **Step 3: Add metadata to pipeline manifest**

In `experiments/gcl_phase_b/resnet50_gate_pipeline.py`, update the returned manifest in `run_resnet50_gate1_to_gate7()` to include:

```python
"run_scope": "real_resnet50_full_trace"
if invocation_limit is None and invocation_ids is None
else "real_resnet50_bounded_slice",
"invocation_limit": invocation_limit,
"invocation_ids": invocation_ids,
"input_kernel_invocation_count": len(adapter_bundle["kernel_invocation_table"]),
```

Apply the same fields to `_gate5_pipeline_manifest(...)` by passing `invocation_limit`, `invocation_ids`, and `adapter_bundle` through its call site.

- [ ] **Step 4: Run the test**

Run:

```bash
pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate_pipeline_manifest_records_full_trace_scope
```

Expected: PASS.

- [ ] **Step 5: Run bounded-slice regression**

Run:

```bash
pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate_pipeline_real_root_reaches_gate9_with_baseline_artifacts tests/gcl_resnet50/test_gate6_selector.py::test_gate6_runs_silhouette_k_and_deterministic_kmeans_on_real_root
```

Expected: PASS. Existing bounded real-root tests must remain valid and explicitly report bounded scope.

- [ ] **Step 6: Commit**

```bash
git add experiments/gcl_phase_b/resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py
git commit -m "feat: record resnet50 full trace pipeline scope"
```

## Task 5: Add a Fast Dry-Run CLI Test

**Files:**
- Modify: `tests/gcl_resnet50/test_full_trace_reproduction_runner.py`

- [ ] **Step 1: Add CLI test with monkeypatched runner**

Add:

```python
def test_full_trace_runner_cli_writes_manifest(tmp_path, monkeypatch, capsys):
    def fake_pipeline(root, out_dir, seed, baseline_artifacts_path=None, invocation_limit=None, invocation_ids=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "kernel_embedding_table.json").write_text(json.dumps({"embeddings": [1]}))
        (out_dir / "selector_artifacts.json").write_text(json.dumps({"selector_manifest_hash": "selector"}))
        (out_dir / "gate7_cluster_correctness_manifest.json").write_text(
            json.dumps({"gate7_cluster_correctness_manifest_hash": "gate7"})
        )
        return {
            "final_gate": "gate9_report_only",
            "hashes": {
                "embedding_table_hash": "embedding",
                "selector_manifest_hash": "selector",
                "gate7_correctness_manifest_hash": "gate7",
                "gate8_tuning_vector_proposal_hash": "gate8",
                "gate9_sampled_vs_full_evaluation_hash": "gate9",
            },
            "pipeline_manifest_hash": "pipeline",
        }

    monkeypatch.setattr(run_resnet50_full_trace_gcl, "run_resnet50_gate1_to_gate7", fake_pipeline)
    root = tmp_path / "root"
    root.mkdir()
    (root / "gate0_trace_acquisition_manifest.json").write_text(
        json.dumps(
            {
                "artifact_status": "formal",
                "formal_input_eligible": True,
                "input_scope": "full_resnet50_inference_trace",
                "kernel_invocation_count": 265,
                "cta_record_count": 124876,
                "gate0_trace_acquisition_manifest_hash": "gate0",
            }
        )
    )
    out_dir = tmp_path / "out"

    run_resnet50_full_trace_gcl.main_args(
        [
            "--input-root",
            str(root),
            "--out",
            str(out_dir),
            "--seed",
            "20260608",
        ]
    )

    manifest = json.loads((out_dir / "resnet50_full_trace_reproduction_manifest.json").read_text())
    assert manifest["formal_full_trace_run"] is True
    assert manifest["artifact_presence"]["kernel_embedding_table.json"] is True
    assert "real_resnet50_full_trace" in capsys.readouterr().out
```

- [ ] **Step 2: Update runner to expose `main_args()`**

Modify `scripts/run_resnet50_full_trace_gcl.py`:

```python
def main_args(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--baseline-artifacts", type=Path)
    args = parser.parse_args(argv)
    manifest = run_full_trace_reproduction(
        input_root=args.input_root,
        out_dir=args.out,
        seed=args.seed,
        baseline_artifacts=args.baseline_artifacts,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    main_args()
```

- [ ] **Step 3: Run the runner tests**

Run:

```bash
pytest -q tests/gcl_resnet50/test_full_trace_reproduction_runner.py
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_resnet50_full_trace_gcl.py tests/gcl_resnet50/test_full_trace_reproduction_runner.py
git commit -m "test: cover resnet50 full trace runner cli"
```

## Task 6: Execute the Real Full-Trace Run

**Files:**
- Runtime output: `artifacts/gcl_resnet50_full_trace_reproduction/`
- No source file changes required unless the run exposes a defect.

- [ ] **Step 1: Run full-trace reproduction without Gate9 baseline**

Run:

```bash
python3 scripts/run_resnet50_full_trace_gcl.py \
  --input-root artifacts/gcl_resnet50_gate0_formal_trace/traces \
  --out artifacts/gcl_resnet50_full_trace_reproduction \
  --seed 20260608
```

Expected:

```text
resnet50_full_trace_reproduction_manifest.json is written
final_gate is gate9_report_only
formal_full_trace_run is true
run_scope is real_resnet50_full_trace
invocation_limit is null
invocation_ids is null
```

- [ ] **Step 2: Inspect full-trace output artifacts**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/gcl_resnet50_full_trace_reproduction")
manifest = json.loads((root / "resnet50_full_trace_reproduction_manifest.json").read_text())
embedding = json.loads((root / "kernel_embedding_table.json").read_text())
selector = json.loads((root / "selector_artifacts.json").read_text())
gate7 = json.loads((root / "gate7_cluster_correctness_manifest.json").read_text())

print("run_scope", manifest["run_scope"])
print("formal_full_trace_run", manifest["formal_full_trace_run"])
print("final_gate", manifest["final_gate"])
print("input_kernel_invocation_count", manifest["input_kernel_invocation_count"])
print("embedding_rows", len(embedding["embeddings"]))
print("selected_k", selector["k_selection_report"]["selected_k"])
print("gate7_claim_status", gate7["claim_status"])
PY
```

Expected:

```text
run_scope real_resnet50_full_trace
formal_full_trace_run True
input_kernel_invocation_count 265
embedding_rows 265
```

`selected_k` must be at least `1`. If it is `1`, the run is complete but clustering is degenerate and must be reported as such. If it is greater than `1`, report the cluster count and representative anchors.

- [ ] **Step 3: Commit source changes, not large generated artifacts**

Check artifact size before deciding whether to commit generated output:

```bash
du -sh artifacts/gcl_resnet50_full_trace_reproduction
git status --short
```

Commit only source/test/docs changes unless the generated manifest files are small and intentionally tracked:

```bash
git add scripts/run_resnet50_full_trace_gcl.py tests/gcl_resnet50/test_full_trace_reproduction_runner.py experiments/gcl_phase_b/resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py
git commit -m "feat: run resnet50 full trace gcl reproduction"
```

## Task 7: Handle Resource Failure Correctly

**Files:**
- Modify: `tests/gcl_resnet50/test_full_trace_reproduction_runner.py`
- Modify: `scripts/run_resnet50_full_trace_gcl.py`

- [ ] **Step 1: Add failing blocker-report test**

Add:

```python
def test_full_trace_runner_writes_blocker_report_on_resource_failure(tmp_path, monkeypatch):
    def fake_pipeline(*args, **kwargs):
        raise RuntimeError("out of memory while tensorizing full trace")

    monkeypatch.setattr(run_resnet50_full_trace_gcl, "run_resnet50_gate1_to_gate7", fake_pipeline)
    root = tmp_path / "root"
    root.mkdir()
    (root / "gate0_trace_acquisition_manifest.json").write_text(
        json.dumps(
            {
                "artifact_status": "formal",
                "formal_input_eligible": True,
                "input_scope": "full_resnet50_inference_trace",
                "kernel_invocation_count": 265,
                "cta_record_count": 124876,
                "gate0_trace_acquisition_manifest_hash": "gate0",
            }
        )
    )
    out_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match="out of memory"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=out_dir,
            seed=20260608,
            baseline_artifacts=None,
        )

    blocker = json.loads((out_dir / "resnet50_full_trace_reproduction_blocker_report.json").read_text())
    assert blocker["run_scope"] == "real_resnet50_full_trace"
    assert blocker["formal_full_trace_run"] is False
    assert blocker["blocker_reason"] == "RuntimeError"
    assert not (out_dir / "resnet50_full_trace_reproduction_manifest.json").exists()
```

- [ ] **Step 2: Run blocker test**

Run:

```bash
pytest -q tests/gcl_resnet50/test_full_trace_reproduction_runner.py::test_full_trace_runner_writes_blocker_report_on_resource_failure
```

Expected: PASS if Task 2 blocker path is implemented. If not, implement the blocker write path exactly as shown in Task 2.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_resnet50_full_trace_gcl.py tests/gcl_resnet50/test_full_trace_reproduction_runner.py
git commit -m "test: record full trace resource blockers"
```

## Task 8: Final Verification For RLCR

**Files:**
- No planned source changes.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest -q tests/gcl_resnet50/test_full_trace_reproduction_runner.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate_pipeline_manifest_records_full_trace_scope
```

Expected: PASS.

- [ ] **Step 2: Run existing formal ResNet50 regression suite**

Run:

```bash
pytest -q tests/gcl_resnet50 tests/gcl_phase_b/test_resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_adapter.py tests/gcl_phase_b/test_resnet50_manifest.py tests/gcl_phase_b/test_resnet50_gate_replay.py
```

Expected: PASS.

- [ ] **Step 3: Verify full-trace artifact contract**

Run:

```bash
test -f artifacts/gcl_resnet50_full_trace_reproduction/resnet50_full_trace_reproduction_manifest.json
test -f artifacts/gcl_resnet50_full_trace_reproduction/kernel_embedding_table.json
test -f artifacts/gcl_resnet50_full_trace_reproduction/selector_artifacts.json
test -f artifacts/gcl_resnet50_full_trace_reproduction/gate7_cluster_correctness_manifest.json
```

Expected: all commands exit `0`.

- [ ] **Step 4: Record final RLCR summary**

The round summary must state one of these exact outcomes:

```text
FULL_TRACE_COMPLETE: real ResNet50 full-trace GCL reproduction reached <final_gate>, embedding_rows=<N>, selected_k=<K>.
```

or:

```text
FULL_TRACE_BLOCKED: real ResNet50 full-trace GCL reproduction did not complete; blocker report written at artifacts/gcl_resnet50_full_trace_reproduction/resnet50_full_trace_reproduction_blocker_report.json.
```

Do not write `FULL_TRACE_COMPLETE` unless `resnet50_full_trace_reproduction_manifest.json` exists and records:

```text
formal_full_trace_run = true
run_scope = real_resnet50_full_trace
invocation_limit = null
invocation_ids = null
input_kernel_invocation_count = 265
```

## Self-Review

- Spec coverage: The plan covers dedicated full-trace execution, no-slice enforcement, formal Gate0 input validation, artifact audit, full-run command, resource failure handling, and final RLCR summary language.
- Self-review scan: No unresolved filler markers or unspecified test steps are present.
- Type consistency: The plan consistently uses `run_full_trace_reproduction`, `resnet50_full_trace_reproduction_manifest.json`, `real_resnet50_full_trace`, `invocation_limit`, and `invocation_ids`.
