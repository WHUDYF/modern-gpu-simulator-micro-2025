# A-line GCL-M0 Offline Embedding Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first GCL path that consumes fixture/offline kernel embeddings and emits GCL cluster, anchor, and structural compression artifacts comparable to PKA-M0.

**Architecture:** Add a GCL-specific selector core that validates embedding rows, normalizes embeddings, runs deterministic farthest-first K-Means, and builds anchor/evaluation rows. Add a thin `gcl_m0_pipeline.py` wrapper that reads a fixture embedding table and writes artifacts under the requested output directory. PKA code remains unchanged except for importing shared deterministic K-Means helpers.

**Tech Stack:** Python 3, NumPy, pytest, existing `experiments/baseline_diagnosis` artifact helpers.

---

## Scope Check

This plan implements only GCL-M0 from the overall GCL architecture spec:

- fixture/offline embedding table input
- embedding validation
- deterministic fixed-K clustering
- representative anchor export
- structural compression evaluation

This plan does not implement trace acquisition, graph construction, RGCN training, graph augmentation, silhouette-K, simulator execution, or cross-architecture evaluation. Those belong to later GCL-M1/M2/M3 plans.

## File Structure

Create:

- `experiments/baseline_diagnosis/fixtures/gcl_m0_embedding_table_l1.json`
  - Fixture embedding input with four deterministic records.
- `experiments/baseline_diagnosis/gcl_selector_core.py`
  - Pure GCL-M0 embedding selector logic.
- `experiments/baseline_diagnosis/gcl_m0_pipeline.py`
  - Thin pipeline wrapper that loads fixtures and writes formal artifacts.
- `experiments/baseline_diagnosis/tests/test_gcl_selector_core.py`
  - Unit tests for validation, normalization, forbidden-field rejection, and output semantics.
- `experiments/baseline_diagnosis/tests/test_gcl_m0_pipeline.py`
  - Wrapper/artifact tests.

Modify:

- No existing production file needs modification for M0.
- Existing PKA tests must continue to pass unchanged.

## Artifact Names

GCL-M0 writes:

- `gcl_embedding_table_l1.json`
- `gcl_kmeans_clusters_l1.json`
- `gcl_representative_anchor_table_l1.json`
- `gcl_compression_evaluation_l1.json`

The fixture input uses:

- `representation_mode = "gcl_m0_embedding_fixture"`

---

### Task 1: Add GCL-M0 Fixture Embedding Table

**Files:**
- Create: `experiments/baseline_diagnosis/fixtures/gcl_m0_embedding_table_l1.json`
- Test: `experiments/baseline_diagnosis/tests/test_gcl_selector_core.py`

- [ ] **Step 1: Write the failing fixture validation test**

Create `experiments/baseline_diagnosis/tests/test_gcl_selector_core.py` with this initial content:

```python
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gcl_selector_core as gcl


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "gcl_m0_embedding_table_l1.json"
)


def test_fixture_embedding_rows_are_valid():
    records = json.loads(FIXTURE_PATH.read_text())

    gcl.validate_embedding_records(
        records,
        expected_representation_mode="gcl_m0_embedding_fixture",
    )

    assert len(records) == 4
    assert {len(row["embedding"]) for row in records} == {4}
    assert all(math.isfinite(value) for row in records for value in row["embedding"])
    assert all(row["source_graph_hash"] for row in records)
    assert all(row["encoder_manifest_hash"] for row in records)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_gcl_selector_core.py::test_fixture_embedding_rows_are_valid
```

Expected:

```text
ModuleNotFoundError: No module named 'gcl_selector_core'
```

- [ ] **Step 3: Add the fixture embedding table**

Create `experiments/baseline_diagnosis/fixtures/gcl_m0_embedding_table_l1.json`:

```json
[
  {
    "record_id": "gcl_m0_l1_bw",
    "kernel_invocation_id": "l1_bw_32f#1",
    "representation_mode": "gcl_m0_embedding_fixture",
    "embedding_dim": 4,
    "embedding": [0.10, 0.12, 0.90, 0.88],
    "source_graph_hash": "fixture-source-gcl-m0-l1-bw",
    "encoder_manifest_hash": "fixture-encoder-gcl-m0-v1",
    "embedding_hash": "fixture-embedding-gcl-m0-l1-bw",
    "weight_input": {"weight_mode": "member_count_fallback", "value": 1.0}
  },
  {
    "record_id": "gcl_m0_mem_bw",
    "kernel_invocation_id": "mem_bw#1",
    "representation_mode": "gcl_m0_embedding_fixture",
    "embedding_dim": 4,
    "embedding": [0.12, 0.10, 0.87, 0.91],
    "source_graph_hash": "fixture-source-gcl-m0-mem-bw",
    "encoder_manifest_hash": "fixture-encoder-gcl-m0-v1",
    "embedding_hash": "fixture-embedding-gcl-m0-mem-bw",
    "weight_input": {"weight_mode": "member_count_fallback", "value": 1.0}
  },
  {
    "record_id": "gcl_m0_shared_bw",
    "kernel_invocation_id": "shared_bw#1",
    "representation_mode": "gcl_m0_embedding_fixture",
    "embedding_dim": 4,
    "embedding": [0.88, 0.86, 0.15, 0.12],
    "source_graph_hash": "fixture-source-gcl-m0-shared-bw",
    "encoder_manifest_hash": "fixture-encoder-gcl-m0-v1",
    "embedding_hash": "fixture-embedding-gcl-m0-shared-bw",
    "weight_input": {"weight_mode": "member_count_fallback", "value": 1.0}
  },
  {
    "record_id": "gcl_m0_maxflops",
    "kernel_invocation_id": "MaxFlops#1",
    "representation_mode": "gcl_m0_embedding_fixture",
    "embedding_dim": 4,
    "embedding": [0.91, 0.89, 0.10, 0.14],
    "source_graph_hash": "fixture-source-gcl-m0-maxflops",
    "encoder_manifest_hash": "fixture-encoder-gcl-m0-v1",
    "embedding_hash": "fixture-embedding-gcl-m0-maxflops",
    "weight_input": {"weight_mode": "member_count_fallback", "value": 1.0}
  }
]
```

- [ ] **Step 4: Add the minimal validation module**

Create `experiments/baseline_diagnosis/gcl_selector_core.py`:

```python
"""Deterministic selector core for GCL-M0 embedding inputs."""

from __future__ import annotations

import math
from typing import Any


def _record_id(rec: dict[str, Any]) -> str:
    return str(rec.get("record_id") or rec.get("kernel_invocation_id"))


def validate_embedding_records(
    records: list[dict[str, Any]],
    expected_representation_mode: str | None = None,
) -> None:
    if len(records) < 2:
        raise ValueError("GCL selector requires at least two records")

    seen: set[str] = set()
    expected_dim: int | None = None
    for rec in records:
        record_id = _record_id(rec)
        if not record_id:
            raise ValueError("GCL selector record missing record_id/kernel_invocation_id")
        if record_id in seen:
            raise ValueError(f"duplicate GCL selector record_id: {record_id}")
        seen.add(record_id)

        if expected_representation_mode and rec.get("representation_mode") != expected_representation_mode:
            raise ValueError(
                f"{record_id}: representation_mode {rec.get('representation_mode')} "
                f"!= {expected_representation_mode}"
            )

        embedding = rec.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"{record_id}: embedding must be a non-empty list")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in embedding):
            raise ValueError(f"{record_id}: embedding values must be finite numbers")

        declared_dim = rec.get("embedding_dim")
        if declared_dim != len(embedding):
            raise ValueError(f"{record_id}: embedding_dim {declared_dim} != {len(embedding)}")
        if expected_dim is None:
            expected_dim = len(embedding)
        elif expected_dim != len(embedding):
            raise ValueError(f"{record_id}: embedding dimension {len(embedding)} != {expected_dim}")

        if not rec.get("source_graph_hash"):
            raise ValueError(f"{record_id}: missing source_graph_hash")
        if not rec.get("encoder_manifest_hash"):
            raise ValueError(f"{record_id}: missing encoder_manifest_hash")
        if not rec.get("embedding_hash"):
            raise ValueError(f"{record_id}: missing embedding_hash")
```

- [ ] **Step 5: Run the fixture validation test**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_gcl_selector_core.py::test_fixture_embedding_rows_are_valid
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit**

```bash
git add \
  experiments/baseline_diagnosis/fixtures/gcl_m0_embedding_table_l1.json \
  experiments/baseline_diagnosis/gcl_selector_core.py \
  experiments/baseline_diagnosis/tests/test_gcl_selector_core.py
git commit -m "Add GCL M0 embedding fixture validation"
```

---

### Task 2: Implement Embedding Matrix and Normalization

**Files:**
- Modify: `experiments/baseline_diagnosis/gcl_selector_core.py`
- Modify: `experiments/baseline_diagnosis/tests/test_gcl_selector_core.py`

- [ ] **Step 1: Add failing tests for matrix extraction, z-score normalization, and validation failures**

Append these tests to `experiments/baseline_diagnosis/tests/test_gcl_selector_core.py`:

```python
import pytest


def test_build_embedding_matrix_sorts_by_record_id_and_normalizes():
    records = [
        {
            "record_id": "b",
            "kernel_invocation_id": "kernel_b#1",
            "representation_mode": "gcl_m0_embedding_fixture",
            "embedding_dim": 3,
            "embedding": [3.0, 10.0, 5.0],
            "source_graph_hash": "source-b",
            "encoder_manifest_hash": "encoder",
            "embedding_hash": "embedding-b",
        },
        {
            "record_id": "a",
            "kernel_invocation_id": "kernel_a#1",
            "representation_mode": "gcl_m0_embedding_fixture",
            "embedding_dim": 3,
            "embedding": [1.0, 10.0, 9.0],
            "source_graph_hash": "source-a",
            "encoder_manifest_hash": "encoder",
            "embedding_hash": "embedding-a",
        },
    ]

    sorted_records, record_ids, matrix = gcl.build_embedding_matrix(records)
    normalized, metadata = gcl.preprocess_embeddings(matrix)

    assert [row["record_id"] for row in sorted_records] == ["a", "b"]
    assert record_ids == ["a", "b"]
    assert matrix.tolist() == [[1.0, 10.0, 9.0], [3.0, 10.0, 5.0]]
    assert normalized.tolist() == [[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]
    assert metadata["embedding_dim"] == 3
    assert metadata["normalization"] == "z_score"
    assert metadata["zero_std_dimensions"] == [1]


def test_validate_embedding_records_rejects_forbidden_fields():
    records = json.loads(FIXTURE_PATH.read_text())
    records[0]["kernel_name"] = "must_not_enter_selector"

    with pytest.raises(ValueError, match="forbidden selector fields"):
        gcl.validate_embedding_records(
            records,
            expected_representation_mode="gcl_m0_embedding_fixture",
        )


def test_validate_embedding_records_rejects_mixed_dimensions():
    records = json.loads(FIXTURE_PATH.read_text())
    records[0]["embedding"] = [0.1, 0.2]
    records[0]["embedding_dim"] = 2

    with pytest.raises(ValueError, match="embedding dimension"):
        gcl.validate_embedding_records(
            records,
            expected_representation_mode="gcl_m0_embedding_fixture",
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_gcl_selector_core.py
```

Expected:

```text
FAILED ... AttributeError: module 'gcl_selector_core' has no attribute 'build_embedding_matrix'
```

- [ ] **Step 3: Implement matrix extraction and normalization**

Replace `experiments/baseline_diagnosis/gcl_selector_core.py` with:

```python
"""Deterministic selector core for GCL-M0 embedding inputs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


FORBIDDEN_SELECTOR_FIELDS = {
    "kernel_name",
    "source_path",
    "expected_behavior_axis",
    "family",
    "regime",
    "shape_hint",
    "trace_order",
    "grid_dim",
    "block_dim",
    "simulator_cycles",
    "full_workload_cycles",
}


def _record_id(rec: dict[str, Any]) -> str:
    return str(rec.get("record_id") or rec.get("kernel_invocation_id") or "")


def validate_embedding_records(
    records: list[dict[str, Any]],
    expected_representation_mode: str | None = None,
) -> None:
    if len(records) < 2:
        raise ValueError("GCL selector requires at least two records")

    seen: set[str] = set()
    expected_dim: int | None = None
    for rec in records:
        forbidden_hits = sorted(FORBIDDEN_SELECTOR_FIELDS & set(rec))
        if forbidden_hits:
            raise ValueError(f"forbidden selector fields in GCL record: {forbidden_hits}")

        record_id = _record_id(rec)
        if not record_id:
            raise ValueError("GCL selector record missing record_id/kernel_invocation_id")
        if record_id in seen:
            raise ValueError(f"duplicate GCL selector record_id: {record_id}")
        seen.add(record_id)

        if expected_representation_mode and rec.get("representation_mode") != expected_representation_mode:
            raise ValueError(
                f"{record_id}: representation_mode {rec.get('representation_mode')} "
                f"!= {expected_representation_mode}"
            )

        embedding = rec.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"{record_id}: embedding must be a non-empty list")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in embedding):
            raise ValueError(f"{record_id}: embedding values must be finite numbers")

        declared_dim = rec.get("embedding_dim")
        if declared_dim != len(embedding):
            raise ValueError(f"{record_id}: embedding_dim {declared_dim} != {len(embedding)}")
        if expected_dim is None:
            expected_dim = len(embedding)
        elif expected_dim != len(embedding):
            raise ValueError(f"{record_id}: embedding dimension {len(embedding)} != {expected_dim}")

        if not rec.get("source_graph_hash"):
            raise ValueError(f"{record_id}: missing source_graph_hash")
        if not rec.get("encoder_manifest_hash"):
            raise ValueError(f"{record_id}: missing encoder_manifest_hash")
        if not rec.get("embedding_hash"):
            raise ValueError(f"{record_id}: missing embedding_hash")


def build_embedding_matrix(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], np.ndarray]:
    sorted_records = sorted(records, key=_record_id)
    record_ids = [_record_id(rec) for rec in sorted_records]
    matrix = np.array([rec["embedding"] for rec in sorted_records], dtype=float)
    return sorted_records, record_ids, matrix


def preprocess_embeddings(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    zero_std_indices = [idx for idx, std in enumerate(stds) if std == 0]
    safe_stds = np.where(stds == 0, 1.0, stds)
    normalized = (matrix - means) / safe_stds
    return normalized, {
        "representation": "gcl_embedding",
        "normalization": "z_score",
        "embedding_dim": int(matrix.shape[1]),
        "mean": means.tolist(),
        "std_deviation": safe_stds.tolist(),
        "zero_std_dimensions": zero_std_indices,
    }
```

- [ ] **Step 4: Run the core tests**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_gcl_selector_core.py
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline_diagnosis/gcl_selector_core.py experiments/baseline_diagnosis/tests/test_gcl_selector_core.py
git commit -m "Add GCL embedding normalization"
```

---

### Task 3: Build GCL Selector Outputs in Core

**Files:**
- Modify: `experiments/baseline_diagnosis/gcl_selector_core.py`
- Modify: `experiments/baseline_diagnosis/tests/test_gcl_selector_core.py`

- [ ] **Step 1: Add failing output tests**

Append this test to `experiments/baseline_diagnosis/tests/test_gcl_selector_core.py`:

```python
def test_build_gcl_outputs_uses_embeddings_and_writes_comparable_semantics():
    records = json.loads(FIXTURE_PATH.read_text())

    outputs = gcl.build_gcl_outputs(
        records,
        mode="gcl_m0_embedding_fixture",
        representation_mode="gcl_m0_embedding_fixture",
    )

    assert outputs["embedding_table"]["representation_mode"] == "gcl_m0_embedding_fixture"
    assert outputs["clusters"]["method"] == "deterministic_farthest_first_kmeans"
    assert outputs["clusters"]["k_selection"]["mode"] == "deterministic_fixed_k"
    assert outputs["clusters"]["k"] == 2
    assert outputs["anchors"]["selector_name"] == "gcl_m0_embedding_selector"
    assert outputs["anchors"]["forbidden_field_audit"]["status"] == "passed"
    assert len(outputs["anchors"]["anchors"]) == 2
    assert outputs["evaluation"]["metric_scope"] == "structural_only_not_simulator_accuracy"
    assert outputs["evaluation"]["compression_ratio"] == 2.0
    assert outputs["evaluation"]["weighted_coverage"] == 1.0
    assert outputs["evaluation"]["top_k_coverage"]["1"] == 0.5
    assert outputs["evaluation"]["top_k_coverage"]["2"] == 1.0
    assert outputs["deterministic_replay_hash"]
```

- [ ] **Step 2: Run the output test to verify it fails**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_gcl_selector_core.py::test_build_gcl_outputs_uses_embeddings_and_writes_comparable_semantics
```

Expected:

```text
AttributeError: module 'gcl_selector_core' has no attribute 'build_gcl_outputs'
```

- [ ] **Step 3: Implement GCL output builder**

Append this code to `experiments/baseline_diagnosis/gcl_selector_core.py`:

```python
from pka_selector_core import _dist2, farthest_first_kmeans
from shared_acquisition import stable_hash


def _weights(records: list[dict[str, Any]], weight_mode: str) -> list[float]:
    if weight_mode == "timing_weight":
        return [float((rec.get("weight_input") or {}).get("value", 1.0)) for rec in records]
    return [1.0 for _ in records]


def build_gcl_outputs(
    records: list[dict[str, Any]],
    mode: str,
    representation_mode: str,
    weight_mode: str = "member_count_fallback",
    timing_unit: str | None = None,
) -> dict[str, Any]:
    validate_embedding_records(records, expected_representation_mode=representation_mode)
    sorted_records, record_ids, matrix = build_embedding_matrix(records)
    normalized, preprocessing = preprocess_embeddings(matrix)
    assignments, centers, kmeans_meta = farthest_first_kmeans(normalized, record_ids)
    kmeans_meta["centroids"] = centers
    kmeans_meta["distance"] = "squared_euclidean_in_normalized_embedding_space"

    weights = _weights(sorted_records, weight_mode)
    total_weight = sum(weights) or 1.0
    clusters_by_index: dict[int, list[int]] = {}
    for idx, cluster_idx in enumerate(assignments):
        clusters_by_index.setdefault(cluster_idx, []).append(idx)

    replay_hash = stable_hash({
        "mode": mode,
        "representation_mode": representation_mode,
        "record_ids": record_ids,
        "normalized_embeddings": normalized.tolist(),
        "assignments": assignments,
    })

    embedding_rows = []
    for idx, rec in enumerate(sorted_records):
        embedding_rows.append({
            "mode": mode,
            "record_id": record_ids[idx],
            "kernel_invocation_id": rec.get("kernel_invocation_id"),
            "representation_mode": representation_mode,
            "embedding_dim": rec["embedding_dim"],
            "embedding": rec["embedding"],
            "normalized_embedding": normalized[idx].tolist(),
            "source_graph_hash": rec["source_graph_hash"],
            "encoder_manifest_hash": rec["encoder_manifest_hash"],
            "embedding_hash": rec["embedding_hash"],
            "weight_input": rec.get("weight_input", {"weight_mode": "member_count_fallback", "value": 1.0}),
        })

    cluster_assignments: dict[str, str] = {}
    members_by_cluster: dict[str, list[str]] = {}
    distance_to_centroid: dict[str, float] = {}
    anchor_rows = []

    for ordinal, cluster_idx in enumerate(sorted(clusters_by_index), 1):
        member_indices = clusters_by_index[cluster_idx]
        center = np.array(centers[cluster_idx])
        cluster_id = f"{mode}-cluster-{ordinal}"
        representative_idx = min(
            member_indices,
            key=lambda idx: (_dist2(normalized[idx], center), record_ids[idx]),
        )
        member_ids = [record_ids[idx] for idx in member_indices]
        cluster_weight = sum(weights[idx] for idx in member_indices)
        members_by_cluster[cluster_id] = member_ids
        for idx in member_indices:
            distance = _dist2(normalized[idx], center)
            cluster_assignments[record_ids[idx]] = cluster_id
            distance_to_centroid[record_ids[idx]] = distance
        anchor_rows.append({
            "anchor_id": f"gcl_m0_anchor_{ordinal - 1:03d}",
            "cluster_id": cluster_id,
            "representative_record_id": record_ids[representative_idx],
            "members": member_ids,
            "coverage_count": len(member_indices),
            "coverage_weight": cluster_weight / total_weight,
            "weight": cluster_weight,
            "representative_distance_to_centroid": _dist2(normalized[representative_idx], center),
            "cluster_label": cluster_id,
        })

    sorted_anchor_weights = sorted((row["coverage_weight"] for row in anchor_rows), reverse=True)
    k_selection = {
        "mode": "deterministic_fixed_k",
        "rule": "ceil(sqrt(n_records)), clamped to [2, n_records]",
        "n_records": len(record_ids),
    }
    forbidden_field_audit = {
        "status": "passed",
        "forbidden_fields": sorted(FORBIDDEN_SELECTOR_FIELDS),
        "actual_read_fields": [
            "record_id",
            "kernel_invocation_id",
            "representation_mode",
            "embedding_dim",
            "embedding",
            "source_graph_hash",
            "encoder_manifest_hash",
            "embedding_hash",
            "weight_input",
        ],
        "violations": [],
    }

    embedding_table = {
        "artifact_name": "gcl_embedding_table_l1",
        "mode": mode,
        "representation_mode": representation_mode,
        "normalization_config": preprocessing,
        "records": embedding_rows,
        "deterministic_replay_hash": replay_hash,
    }
    clusters = {
        "artifact_name": "gcl_kmeans_clusters_l1",
        "mode": mode,
        "representation_mode": representation_mode,
        "method": "deterministic_farthest_first_kmeans",
        "k_selection": k_selection,
        "k": kmeans_meta["k"],
        "kmeans_config": kmeans_meta,
        "cluster_assignments": cluster_assignments,
        "members_by_cluster": members_by_cluster,
        "distance_to_centroid": distance_to_centroid,
        "centroids": centers,
        "inertia": sum(distance_to_centroid.values()),
        "deterministic_replay_hash": replay_hash,
    }
    anchors = {
        "artifact_name": "gcl_representative_anchor_table_l1",
        "mode": mode,
        "representation_mode": representation_mode,
        "selector_name": "gcl_m0_embedding_selector",
        "embedding_dim": int(matrix.shape[1]),
        "clustering_config": kmeans_meta,
        "selection_rule": "nearest_centroid_real_record",
        "forbidden_field_audit": forbidden_field_audit,
        "anchors": anchor_rows,
        "deterministic_replay_hash": replay_hash,
    }
    evaluation = {
        "artifact_name": "gcl_compression_evaluation_l1",
        "mode": mode,
        "representation_mode": representation_mode,
        "metric_scope": "structural_only_not_simulator_accuracy",
        "input_records": len(record_ids),
        "anchor_count": len(anchor_rows),
        "compression_ratio": len(record_ids) / max(1, len(anchor_rows)),
        "coverage_count": len(record_ids),
        "weighted_coverage": sum(row["coverage_weight"] for row in anchor_rows),
        "weight_mode": weight_mode,
        "timing_unit": timing_unit,
        "top_k_coverage": {
            str(k): sum(sorted_anchor_weights[:k])
            for k in range(1, len(sorted_anchor_weights) + 1)
        },
        "anchor_balance": max(sorted_anchor_weights) if sorted_anchor_weights else 0.0,
        "cluster_size_distribution": {
            row["cluster_id"]: row["coverage_count"]
            for row in anchor_rows
        },
        "deterministic_replay_hash": replay_hash,
    }
    return {
        "embedding_table": embedding_table,
        "clusters": clusters,
        "anchors": anchors,
        "evaluation": evaluation,
        "deterministic_replay_hash": replay_hash,
    }
```

- [ ] **Step 4: Run the core tests**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_gcl_selector_core.py
```

Expected:

```text
5 passed
```

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline_diagnosis/gcl_selector_core.py experiments/baseline_diagnosis/tests/test_gcl_selector_core.py
git commit -m "Build GCL M0 selector outputs"
```

---

### Task 4: Add GCL-M0 Pipeline Wrapper and Artifact Tests

**Files:**
- Create: `experiments/baseline_diagnosis/gcl_m0_pipeline.py`
- Create: `experiments/baseline_diagnosis/tests/test_gcl_m0_pipeline.py`

- [ ] **Step 1: Write the failing pipeline test**

Create `experiments/baseline_diagnosis/tests/test_gcl_m0_pipeline.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gcl_m0_pipeline


def test_gcl_m0_pipeline_writes_deterministic_artifacts(tmp_path):
    outputs = gcl_m0_pipeline.run(output_dir=tmp_path)

    assert outputs["embedding_table"]["artifact_name"] == "gcl_embedding_table_l1"
    assert outputs["clusters"]["method"] == "deterministic_farthest_first_kmeans"
    assert outputs["anchors"]["selector_name"] == "gcl_m0_embedding_selector"
    assert outputs["evaluation"]["metric_scope"] == "structural_only_not_simulator_accuracy"
    assert outputs["evaluation"]["compression_ratio"] == 2.0

    assert (tmp_path / "gcl_embedding_table_l1.json").exists()
    assert (tmp_path / "gcl_kmeans_clusters_l1.json").exists()
    assert (tmp_path / "gcl_representative_anchor_table_l1.json").exists()
    assert (tmp_path / "gcl_compression_evaluation_l1.json").exists()

    anchors = json.loads((tmp_path / "gcl_representative_anchor_table_l1.json").read_text())
    assert anchors["representation_mode"] == "gcl_m0_embedding_fixture"
    assert anchors["forbidden_field_audit"]["status"] == "passed"
    assert len(anchors["anchors"]) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_gcl_m0_pipeline.py
```

Expected:

```text
ModuleNotFoundError: No module named 'gcl_m0_pipeline'
```

- [ ] **Step 3: Implement the pipeline wrapper**

Create `experiments/baseline_diagnosis/gcl_m0_pipeline.py`:

```python
"""M0 wrapper for the GCL offline embedding selector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gcl_selector_core import build_gcl_outputs
from shared_acquisition import ARTIFACT_DIR, REPO_ROOT, stable_hash, write_json

FIXTURE_PATH = (
    REPO_ROOT
    / "experiments"
    / "baseline_diagnosis"
    / "fixtures"
    / "gcl_m0_embedding_table_l1.json"
)


def run(input_path: Path = FIXTURE_PATH, output_dir: Path = ARTIFACT_DIR) -> dict:
    records = json.loads(input_path.read_text())
    outputs = build_gcl_outputs(
        records,
        mode="gcl_m0_embedding_fixture",
        representation_mode="gcl_m0_embedding_fixture",
    )
    input_hash = stable_hash(records)
    outputs["embedding_table"]["input_embedding_fixture_hash"] = input_hash
    outputs["clusters"]["input_embedding_table_hash"] = input_hash
    outputs["anchors"]["input_embedding_table_hash"] = input_hash
    outputs["evaluation"]["input_embedding_table_hash"] = input_hash

    write_json(output_dir / "gcl_embedding_table_l1.json", outputs["embedding_table"])
    write_json(output_dir / "gcl_kmeans_clusters_l1.json", outputs["clusters"])
    write_json(output_dir / "gcl_representative_anchor_table_l1.json", outputs["anchors"])
    write_json(output_dir / "gcl_compression_evaluation_l1.json", outputs["evaluation"])
    return outputs


def main() -> int:
    run()
    print("GCL M0 embedding selector pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the pipeline test**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_gcl_m0_pipeline.py
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add experiments/baseline_diagnosis/gcl_m0_pipeline.py experiments/baseline_diagnosis/tests/test_gcl_m0_pipeline.py
git commit -m "Add GCL M0 pipeline wrapper"
```

---

### Task 5: Add Timing Weight Coverage Test

**Files:**
- Modify: `experiments/baseline_diagnosis/tests/test_gcl_selector_core.py`

- [ ] **Step 1: Add failing timing-weight test**

Append this test to `experiments/baseline_diagnosis/tests/test_gcl_selector_core.py`:

```python
def test_build_gcl_outputs_honors_timing_weight():
    records = json.loads(FIXTURE_PATH.read_text())
    weights = [10.0, 30.0, 20.0, 40.0]
    for row, weight in zip(records, weights):
        row["weight_input"] = {
            "weight_mode": "timing_weight",
            "timing_unit": "duration_ns",
            "value": weight,
        }

    outputs = gcl.build_gcl_outputs(
        records,
        mode="gcl_m0_embedding_fixture",
        representation_mode="gcl_m0_embedding_fixture",
        weight_mode="timing_weight",
        timing_unit="duration_ns",
    )

    assert outputs["evaluation"]["weight_mode"] == "timing_weight"
    assert outputs["evaluation"]["timing_unit"] == "duration_ns"
    assert outputs["evaluation"]["weighted_coverage"] == 1.0
    assert outputs["evaluation"]["top_k_coverage"]["1"] == 0.6
    assert outputs["evaluation"]["top_k_coverage"]["2"] == 1.0
```

- [ ] **Step 2: Run the timing-weight test**

Run:

```bash
pytest -q experiments/baseline_diagnosis/tests/test_gcl_selector_core.py::test_build_gcl_outputs_honors_timing_weight
```

Expected:

```text
1 passed
```

If this fails because cluster membership changes, inspect `outputs["anchors"]["anchors"]` and update the fixture embeddings so the first two records remain one cluster and the last two records remain another cluster. Do not change production logic to satisfy this test.

- [ ] **Step 3: Run all GCL tests**

Run:

```bash
pytest -q \
  experiments/baseline_diagnosis/tests/test_gcl_selector_core.py \
  experiments/baseline_diagnosis/tests/test_gcl_m0_pipeline.py
```

Expected:

```text
7 passed
```

- [ ] **Step 4: Commit**

```bash
git add experiments/baseline_diagnosis/tests/test_gcl_selector_core.py
git commit -m "Cover GCL M0 timing weights"
```

---

### Task 6: Run PKA Regression and Wrapper Smoke

**Files:**
- No code changes expected.
- Verification only.

- [ ] **Step 1: Run GCL and PKA focused tests**

Run:

```bash
pytest -q \
  experiments/baseline_diagnosis/tests/test_gcl_selector_core.py \
  experiments/baseline_diagnosis/tests/test_gcl_m0_pipeline.py \
  experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py \
  experiments/baseline_diagnosis/tests/test_m1_selector.py
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 2: Run the GCL-M0 wrapper smoke into a temporary directory**

Run:

```bash
tmpdir="$(mktemp -d)"
PYTHONPATH=experiments/baseline_diagnosis python - <<PY
from pathlib import Path
import gcl_m0_pipeline

out = Path("$tmpdir")
outputs = gcl_m0_pipeline.run(output_dir=out)
assert outputs["evaluation"]["compression_ratio"] == 2.0
assert (out / "gcl_embedding_table_l1.json").exists()
assert (out / "gcl_kmeans_clusters_l1.json").exists()
assert (out / "gcl_representative_anchor_table_l1.json").exists()
assert (out / "gcl_compression_evaluation_l1.json").exists()
print("GCL M0 wrapper smoke complete")
PY
rm -rf "$tmpdir"
```

Expected:

```text
GCL M0 wrapper smoke complete
```

The command writes to a temporary output directory, so no runtime artifacts should appear in the repository.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git status --short
```

Expected:

```text
Only intentional GCL-M0 source, fixture, and test files are modified or untracked.
Generated artifacts under artifacts/a_line/l1 are not staged.
```

- [ ] **Step 4: Commit final verification metadata only if source files changed**

If Step 1 or Step 2 required source/test fixes, commit those fixes:

```bash
git add experiments/baseline_diagnosis
git commit -m "Stabilize GCL M0 selector verification"
```

If no source/test files changed, do not create an empty commit.

---

## Self-Review Checklist

Spec coverage:

- GCL-M0 fixture embedding input is covered by Task 1.
- Embedding validation and finite numeric checks are covered by Tasks 1 and 2.
- Clustering on embeddings rather than PKA 12D features is covered by Task 3.
- Anchor artifact and structural compression evaluation are covered by Tasks 3 and 4.
- `representation_mode = "gcl_m0_embedding_fixture"` is covered by Tasks 1, 3, and 4.
- Forbidden-field audit is covered by Tasks 2 and 3.
- Timing/member-count weights are covered by Tasks 3 and 5.
- PKA baseline non-regression is covered by Task 6.

Implementation boundaries:

- No trace acquisition, graph construction, RGCN training, or simulator accuracy is included.
- No existing PKA production file is modified.
- K selection is deterministic fixed-K only.

Verification commands:

- `pytest -q experiments/baseline_diagnosis/tests/test_gcl_selector_core.py`
- `pytest -q experiments/baseline_diagnosis/tests/test_gcl_m0_pipeline.py`
- `pytest -q experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py experiments/baseline_diagnosis/tests/test_m1_selector.py`
