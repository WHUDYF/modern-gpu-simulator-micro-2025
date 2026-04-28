# Trace Compression Microbench Necessity Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal evidence probe that tests whether compression-derived trace signatures add behavior information beyond flat profile features for microbench matching.

**Architecture:** Add a small Python experiment package under `experiments/trace_compression_behavior/`. It reads a curated catalog of target/candidate kernels, extracts flat profile signatures and compression signatures from existing JSON artifacts, computes distance matrices, detects profile-similar/compression-different conflict pairs, and emits JSON/Markdown reports. The first implementation uses local artifacts and synthetic fixtures only; it does not implement online trace compression or AI generation.

**Tech Stack:** Python 3.10+ stdlib, `pytest`, JSON/Markdown outputs.

---

## Scope

This plan implements the first gate of the independent academic line:

```text
target / candidate traces
  -> flat profile signature
  -> compression signature
  -> distance matrices
  -> conflict pairs
  -> necessity report
```

It intentionally does not implement:

- CUDA microbench generation;
- LLM / agent code generation;
- online trace compression;
- simulator validation;
- changes to L1 / PKA selectors.

Success means the repo can produce a report that answers:

> Do compression-derived signatures expose behavior differences that flat profile summaries miss?

## File Structure

Create:

- `experiments/trace_compression_behavior/__init__.py`  
  Marks the probe package.

- `experiments/trace_compression_behavior/models.py`  
  Dataclasses and validation helpers for catalog entries, signatures, distances, and conflict pairs.

- `experiments/trace_compression_behavior/catalog.py`  
  Loads and validates a JSON catalog of target/candidate kernels.

- `experiments/trace_compression_behavior/signatures.py`  
  Extracts flat profile signatures and compression signatures from catalog-referenced JSON records.

- `experiments/trace_compression_behavior/distance.py`  
  Normalizes numeric fields and computes pairwise distances.

- `experiments/trace_compression_behavior/conflicts.py`  
  Detects profile-similar/compression-different and runtime-similar/compression-different pairs.

- `experiments/trace_compression_behavior/report.py`  
  Emits machine-readable JSON and human-readable Markdown reports.

- `experiments/trace_compression_behavior/run_probe.py`  
  CLI entry point that wires catalog loading, extraction, distance, conflict detection, and report generation.

- `experiments/trace_compression_behavior/fixtures/synthetic_catalog.json`  
  Small synthetic catalog for deterministic unit tests.

- `experiments/trace_compression_behavior/fixtures/synthetic_records.json`  
  Small records with controlled profile/compression features.

- `experiments/trace_compression_behavior/catalogs/initial_probe_catalog.json`  
  First real/local catalog referencing existing repository artifacts.

- `experiments/trace_compression_behavior/tests/test_models.py`

- `experiments/trace_compression_behavior/tests/test_catalog.py`

- `experiments/trace_compression_behavior/tests/test_signatures.py`

- `experiments/trace_compression_behavior/tests/test_distance.py`

- `experiments/trace_compression_behavior/tests/test_conflicts.py`

- `experiments/trace_compression_behavior/tests/test_report.py`

- `experiments/trace_compression_behavior/tests/test_run_probe.py`

Output paths created by the CLI:

- `experiments/trace_compression_behavior/results/necessity_probe/signature_table.json`
- `experiments/trace_compression_behavior/results/necessity_probe/profile_distance_matrix.json`
- `experiments/trace_compression_behavior/results/necessity_probe/compression_distance_matrix.json`
- `experiments/trace_compression_behavior/results/necessity_probe/conflict_pairs.json`
- `experiments/trace_compression_behavior/results/necessity_probe/necessity_report.md`
- `experiments/trace_compression_behavior/results/necessity_probe/necessity_report.json`

Do not modify:

- `experiments/baseline_diagnosis/pka_baseline_selector.py`
- `experiments/baseline_diagnosis/pka_feature_extractor.py`
- `experiments/baseline_diagnosis/b_line_consumer_l1.py`
- any L1 schema or selector tests

## Data Contracts

### Catalog Entry

Each catalog item describes one target or candidate record.

```json
{
  "id": "regular_memory_fixture",
  "label": "regular-memory",
  "role": "candidate",
  "source_path": "experiments/trace_compression_behavior/fixtures/synthetic_records.json",
  "record_pointer": "/records/regular_memory",
  "profile_fields": {
    "runtime": "hardware_metrics.duration_ns",
    "num_instructions": "dynamic_stats.total_dynamic_insts",
    "global_loads": "dynamic_stats.global_loads",
    "global_stores": "dynamic_stats.global_stores",
    "branch_ops": "dynamic_stats.branch_ops",
    "thread_blocks": "dynamic_stats.num_blocks"
  },
  "compression_fields": {
    "instruction_run_coverage": "compression_features.instruction_run_coverage.mean",
    "shared_pc_sequence_coverage": "compression_features.shared_pc_sequence_coverage.mean",
    "warp_pc_override_density": "compression_features.warp_pc_override_density.mean",
    "cross_tb_delta_coverage": "compression_features.cross_tb_delta_coverage.mean",
    "global_address_offset_coverage": "compression_features.global_address_offset_coverage.mean",
    "address_override_density": "compression_features.address_override_density.mean",
    "full_encoding_fallback_rate": "compression_features.full_encoding_fallback_rate.mean"
  }
}
```

Allowed `role` values:

- `target`
- `candidate`
- `control`

### Signature Record

The extractor emits one signature record per catalog entry.

```json
{
  "id": "regular_memory_fixture",
  "label": "regular-memory",
  "role": "candidate",
  "source_path": "experiments/trace_compression_behavior/fixtures/synthetic_records.json",
  "profile_signature": {
    "runtime": 1000.0,
    "num_instructions": 1000000.0,
    "global_loads": 500000.0,
    "global_stores": 500000.0,
    "branch_ops": 1000.0,
    "thread_blocks": 128.0
  },
  "compression_signature": {
    "instruction_run_coverage": 0.92,
    "shared_pc_sequence_coverage": 0.97,
    "warp_pc_override_density": 0.01,
    "cross_tb_delta_coverage": 0.94,
    "global_address_offset_coverage": 0.95,
    "address_override_density": 0.02,
    "full_encoding_fallback_rate": 0.0
  },
  "missing_profile_fields": [],
  "missing_compression_fields": [],
  "confidence": "high"
}
```

Confidence rules:

- `high`: no missing compression fields and no missing profile fields.
- `medium`: one or two compression fields missing, with all profile fields present.
- `low`: more than two compression fields missing or any profile field missing.

### Conflict Pair

Conflict detection emits pairs like:

```json
{
  "left_id": "regular_memory_fixture",
  "right_id": "irregular_gather_fixture",
  "conflict_type": "profile_similar_compression_different",
  "profile_distance": 0.05,
  "compression_distance": 0.72,
  "explanation": "Flat profile counts are close, but memory-structure signature differs: cross_tb_delta_coverage 0.94 vs 0.18 and address_override_density 0.02 vs 0.68."
}
```

## Thresholds

Initial deterministic thresholds:

- `profile_similar_threshold = 0.15`
- `runtime_similar_threshold = 0.10`
- `compression_different_threshold = 0.35`
- `compression_similar_threshold = 0.15`

The report must print these thresholds and label them as first-pass analysis parameters, not final research constants.

---

### Task 1: Create Package Skeleton and Fixtures

**Files:**
- Create: `experiments/trace_compression_behavior/__init__.py`
- Create: `experiments/trace_compression_behavior/fixtures/synthetic_records.json`
- Create: `experiments/trace_compression_behavior/fixtures/synthetic_catalog.json`
- Create: `experiments/trace_compression_behavior/tests/test_catalog.py`

- [ ] **Step 1: Create package marker**

Create `experiments/trace_compression_behavior/__init__.py`:

```python
"""Trace-compression behavior signature probe package."""
```

- [ ] **Step 2: Create synthetic records fixture**

Create `experiments/trace_compression_behavior/fixtures/synthetic_records.json` with three controlled records:

```json
{
  "records": {
    "regular_memory": {
      "hardware_metrics": {"duration_ns": 1000.0},
      "dynamic_stats": {
        "total_dynamic_insts": 1000000.0,
        "global_loads": 500000.0,
        "global_stores": 500000.0,
        "branch_ops": 1000.0,
        "num_blocks": 128.0
      },
      "compression_features": {
        "instruction_run_coverage": {"mean": 0.92},
        "shared_pc_sequence_coverage": {"mean": 0.97},
        "warp_pc_override_density": {"mean": 0.01},
        "cross_tb_delta_coverage": {"mean": 0.94},
        "global_address_offset_coverage": {"mean": 0.95},
        "address_override_density": {"mean": 0.02},
        "full_encoding_fallback_rate": {"mean": 0.0}
      }
    },
    "irregular_gather": {
      "hardware_metrics": {"duration_ns": 1030.0},
      "dynamic_stats": {
        "total_dynamic_insts": 1010000.0,
        "global_loads": 505000.0,
        "global_stores": 495000.0,
        "branch_ops": 1200.0,
        "num_blocks": 128.0
      },
      "compression_features": {
        "instruction_run_coverage": {"mean": 0.55},
        "shared_pc_sequence_coverage": {"mean": 0.62},
        "warp_pc_override_density": {"mean": 0.12},
        "cross_tb_delta_coverage": {"mean": 0.18},
        "global_address_offset_coverage": {"mean": 0.22},
        "address_override_density": {"mean": 0.68},
        "full_encoding_fallback_rate": {"mean": 0.31}
      }
    },
    "branch_heavy": {
      "hardware_metrics": {"duration_ns": 980.0},
      "dynamic_stats": {
        "total_dynamic_insts": 980000.0,
        "global_loads": 210000.0,
        "global_stores": 205000.0,
        "branch_ops": 120000.0,
        "num_blocks": 128.0
      },
      "compression_features": {
        "instruction_run_coverage": {"mean": 0.34},
        "shared_pc_sequence_coverage": {"mean": 0.41},
        "warp_pc_override_density": {"mean": 0.46},
        "cross_tb_delta_coverage": {"mean": 0.70},
        "global_address_offset_coverage": {"mean": 0.73},
        "address_override_density": {"mean": 0.10},
        "full_encoding_fallback_rate": {"mean": 0.12}
      }
    }
  }
}
```

- [ ] **Step 3: Create synthetic catalog**

Create `experiments/trace_compression_behavior/fixtures/synthetic_catalog.json`:

```json
{
  "catalog_id": "synthetic_fixture_probe",
  "description": "Deterministic fixture catalog for necessity probe tests.",
  "entries": [
    {
      "id": "regular_memory_fixture",
      "label": "regular-memory",
      "role": "target",
      "source_path": "experiments/trace_compression_behavior/fixtures/synthetic_records.json",
      "record_pointer": "/records/regular_memory"
    },
    {
      "id": "irregular_gather_fixture",
      "label": "irregular-gather",
      "role": "candidate",
      "source_path": "experiments/trace_compression_behavior/fixtures/synthetic_records.json",
      "record_pointer": "/records/irregular_gather"
    },
    {
      "id": "branch_heavy_fixture",
      "label": "branch-heavy",
      "role": "candidate",
      "source_path": "experiments/trace_compression_behavior/fixtures/synthetic_records.json",
      "record_pointer": "/records/branch_heavy"
    }
  ]
}
```

- [ ] **Step 4: Write failing catalog test**

Create `experiments/trace_compression_behavior/tests/test_catalog.py`:

```python
from pathlib import Path

from experiments.trace_compression_behavior.catalog import load_catalog


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "experiments/trace_compression_behavior/fixtures/synthetic_catalog.json"


def test_load_catalog_reads_entries_with_stable_ids():
    catalog = load_catalog(FIXTURE)

    assert catalog.catalog_id == "synthetic_fixture_probe"
    assert [entry.id for entry in catalog.entries] == [
        "regular_memory_fixture",
        "irregular_gather_fixture",
        "branch_heavy_fixture",
    ]
    assert catalog.entries[0].role == "target"
```

- [ ] **Step 5: Run the failing test**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_catalog.py -q
```

Expected: FAIL with `ModuleNotFoundError` or missing `load_catalog`.

---

### Task 2: Implement Catalog and Models

**Files:**
- Create: `experiments/trace_compression_behavior/models.py`
- Create: `experiments/trace_compression_behavior/catalog.py`
- Modify: `experiments/trace_compression_behavior/tests/test_catalog.py`

- [ ] **Step 1: Add model dataclasses**

Create `experiments/trace_compression_behavior/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Role = Literal["target", "candidate", "control"]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    label: str
    role: Role
    source_path: Path
    record_pointer: str


@dataclass(frozen=True)
class Catalog:
    catalog_id: str
    description: str
    entries: list[CatalogEntry]


@dataclass(frozen=True)
class SignatureRecord:
    id: str
    label: str
    role: Role
    source_path: str
    profile_signature: dict[str, float]
    compression_signature: dict[str, float]
    missing_profile_fields: list[str]
    missing_compression_fields: list[str]
    confidence: Confidence


@dataclass(frozen=True)
class DistancePair:
    left_id: str
    right_id: str
    distance: float


@dataclass(frozen=True)
class ConflictPair:
    left_id: str
    right_id: str
    conflict_type: str
    profile_distance: float
    compression_distance: float
    explanation: str
```

- [ ] **Step 2: Implement catalog loader**

Create `experiments/trace_compression_behavior/catalog.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.trace_compression_behavior.models import Catalog, CatalogEntry


VALID_ROLES = {"target", "candidate", "control"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_catalog(path: str | Path) -> Catalog:
    catalog_path = Path(path)
    data = _load_json(catalog_path)
    entries = []
    seen_ids = set()
    for raw in data.get("entries", []):
        entry_id = raw.get("id")
        if not entry_id:
            raise ValueError("catalog entry missing id")
        if entry_id in seen_ids:
            raise ValueError(f"duplicate catalog entry id: {entry_id}")
        seen_ids.add(entry_id)

        role = raw.get("role")
        if role not in VALID_ROLES:
            raise ValueError(f"catalog entry {entry_id} has invalid role: {role}")

        source_path = _repo_root() / raw["source_path"]
        if not source_path.exists():
            raise FileNotFoundError(f"catalog entry {entry_id} source_path does not exist: {source_path}")

        entries.append(
            CatalogEntry(
                id=entry_id,
                label=raw["label"],
                role=role,
                source_path=source_path,
                record_pointer=raw["record_pointer"],
            )
        )
    if not entries:
        raise ValueError("catalog contains no entries")
    return Catalog(
        catalog_id=data["catalog_id"],
        description=data.get("description", ""),
        entries=entries,
    )
```

- [ ] **Step 3: Extend catalog tests for failures**

Append to `experiments/trace_compression_behavior/tests/test_catalog.py`:

```python
import json


def test_load_catalog_rejects_duplicate_ids(tmp_path):
    source = ROOT / "experiments/trace_compression_behavior/fixtures/synthetic_records.json"
    payload = {
        "catalog_id": "bad",
        "entries": [
            {"id": "dup", "label": "a", "role": "target", "source_path": str(source.relative_to(ROOT)), "record_pointer": "/records/regular_memory"},
            {"id": "dup", "label": "b", "role": "candidate", "source_path": str(source.relative_to(ROOT)), "record_pointer": "/records/irregular_gather"},
        ],
    }
    path = tmp_path / "bad_catalog.json"
    path.write_text(json.dumps(payload))

    try:
        load_catalog(path)
    except ValueError as exc:
        assert "duplicate catalog entry id" in str(exc)
    else:
        raise AssertionError("duplicate ids should fail")
```

- [ ] **Step 4: Run catalog tests**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_catalog.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/trace_compression_behavior
git commit -m "Add trace compression behavior probe catalog"
```

---

### Task 3: Extract Profile and Compression Signatures

**Files:**
- Create: `experiments/trace_compression_behavior/signatures.py`
- Create: `experiments/trace_compression_behavior/tests/test_signatures.py`

- [ ] **Step 1: Write failing signature tests**

Create `experiments/trace_compression_behavior/tests/test_signatures.py`:

```python
from pathlib import Path

from experiments.trace_compression_behavior.catalog import load_catalog
from experiments.trace_compression_behavior.signatures import build_signature_records


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "experiments/trace_compression_behavior/fixtures/synthetic_catalog.json"


def test_build_signature_records_extracts_flat_and_compression_signatures():
    catalog = load_catalog(FIXTURE)
    records = build_signature_records(catalog)

    regular = next(record for record in records if record.id == "regular_memory_fixture")
    irregular = next(record for record in records if record.id == "irregular_gather_fixture")

    assert regular.profile_signature["num_instructions"] == 1000000.0
    assert regular.compression_signature["cross_tb_delta_coverage"] == 0.94
    assert regular.confidence == "high"
    assert irregular.compression_signature["address_override_density"] == 0.68


def test_build_signature_records_keeps_missing_fields_explicit(tmp_path):
    catalog = load_catalog(FIXTURE)
    records = build_signature_records(catalog, compression_fields=["cross_tb_delta_coverage", "not_present"])

    regular = next(record for record in records if record.id == "regular_memory_fixture")

    assert regular.missing_compression_fields == ["not_present"]
    assert regular.confidence == "medium"
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_signatures.py -q
```

Expected: FAIL with missing `signatures` module.

- [ ] **Step 3: Implement signature extractor**

Create `experiments/trace_compression_behavior/signatures.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.trace_compression_behavior.models import Catalog, CatalogEntry, SignatureRecord


DEFAULT_PROFILE_FIELDS = {
    "runtime": "hardware_metrics.duration_ns",
    "num_instructions": "dynamic_stats.total_dynamic_insts",
    "global_loads": "dynamic_stats.global_loads",
    "global_stores": "dynamic_stats.global_stores",
    "branch_ops": "dynamic_stats.branch_ops",
    "thread_blocks": "dynamic_stats.num_blocks",
}

DEFAULT_COMPRESSION_FIELDS = {
    "instruction_run_coverage": "compression_features.instruction_run_coverage.mean",
    "shared_pc_sequence_coverage": "compression_features.shared_pc_sequence_coverage.mean",
    "warp_pc_override_density": "compression_features.warp_pc_override_density.mean",
    "cross_tb_delta_coverage": "compression_features.cross_tb_delta_coverage.mean",
    "global_address_offset_coverage": "compression_features.global_address_offset_coverage.mean",
    "address_override_density": "compression_features.address_override_density.mean",
    "full_encoding_fallback_rate": "compression_features.full_encoding_fallback_rate.mean",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_pointer(payload: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError(f"record_pointer must start with '/': {pointer}")
    current = payload
    for part in pointer.strip("/").split("/"):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"record_pointer {pointer} could not resolve segment {part}")
        current = current[part]
    return current


def _get_path(payload: dict[str, Any], path: str) -> float | None:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if current is None:
        return None
    return float(current)


def _confidence(missing_profile: list[str], missing_compression: list[str]) -> str:
    if missing_profile:
        return "low"
    if not missing_compression:
        return "high"
    if len(missing_compression) <= 2:
        return "medium"
    return "low"


def _extract(entry: CatalogEntry, profile_fields: dict[str, str], compression_fields: dict[str, str]) -> SignatureRecord:
    payload = _load_json(entry.source_path)
    record_payload = _resolve_pointer(payload, entry.record_pointer)
    profile_signature: dict[str, float] = {}
    compression_signature: dict[str, float] = {}
    missing_profile_fields: list[str] = []
    missing_compression_fields: list[str] = []

    for name, source_path in profile_fields.items():
        value = _get_path(record_payload, source_path)
        if value is None:
            missing_profile_fields.append(name)
        else:
            profile_signature[name] = value

    for name, source_path in compression_fields.items():
        value = _get_path(record_payload, source_path)
        if value is None:
            missing_compression_fields.append(name)
        else:
            compression_signature[name] = value

    return SignatureRecord(
        id=entry.id,
        label=entry.label,
        role=entry.role,
        source_path=str(entry.source_path),
        profile_signature=profile_signature,
        compression_signature=compression_signature,
        missing_profile_fields=missing_profile_fields,
        missing_compression_fields=missing_compression_fields,
        confidence=_confidence(missing_profile_fields, missing_compression_fields),
    )


def build_signature_records(
    catalog: Catalog,
    *,
    profile_fields: list[str] | None = None,
    compression_fields: list[str] | None = None,
) -> list[SignatureRecord]:
    selected_profile = {
        name: DEFAULT_PROFILE_FIELDS[name]
        for name in (profile_fields or list(DEFAULT_PROFILE_FIELDS))
    }
    selected_compression = {
        name: DEFAULT_COMPRESSION_FIELDS[name]
        for name in (compression_fields or list(DEFAULT_COMPRESSION_FIELDS))
        if name in DEFAULT_COMPRESSION_FIELDS
    }
    for name in compression_fields or []:
        if name not in DEFAULT_COMPRESSION_FIELDS:
            selected_compression[name] = f"compression_features.{name}.mean"
    return [_extract(entry, selected_profile, selected_compression) for entry in catalog.entries]
```

- [ ] **Step 4: Run signature tests**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_signatures.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/trace_compression_behavior/signatures.py experiments/trace_compression_behavior/tests/test_signatures.py
git commit -m "Extract trace compression behavior signatures"
```

---

### Task 4: Compute Distance Matrices

**Files:**
- Create: `experiments/trace_compression_behavior/distance.py`
- Create: `experiments/trace_compression_behavior/tests/test_distance.py`

- [ ] **Step 1: Write failing distance tests**

Create `experiments/trace_compression_behavior/tests/test_distance.py`:

```python
from experiments.trace_compression_behavior.distance import pairwise_distances
from experiments.trace_compression_behavior.models import SignatureRecord


def _record(record_id: str, profile: dict[str, float], compression: dict[str, float]) -> SignatureRecord:
    return SignatureRecord(
        id=record_id,
        label=record_id,
        role="candidate",
        source_path="fixture",
        profile_signature=profile,
        compression_signature=compression,
        missing_profile_fields=[],
        missing_compression_fields=[],
        confidence="high",
    )


def test_pairwise_distances_uses_shared_fields_and_normalizes_scale():
    records = [
        _record("a", {"num_instructions": 100.0, "runtime": 10.0}, {"cross_tb_delta_coverage": 0.9}),
        _record("b", {"num_instructions": 110.0, "runtime": 11.0}, {"cross_tb_delta_coverage": 0.1}),
        _record("c", {"num_instructions": 300.0, "runtime": 30.0}, {"cross_tb_delta_coverage": 0.85}),
    ]

    profile_pairs = pairwise_distances(records, "profile")
    compression_pairs = pairwise_distances(records, "compression")

    ab_profile = next(pair for pair in profile_pairs if {pair.left_id, pair.right_id} == {"a", "b"})
    ab_compression = next(pair for pair in compression_pairs if {pair.left_id, pair.right_id} == {"a", "b"})

    assert ab_profile.distance < 0.10
    assert ab_compression.distance > 0.70
```

- [ ] **Step 2: Run failing distance test**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_distance.py -q
```

Expected: FAIL with missing `distance` module.

- [ ] **Step 3: Implement normalized distance**

Create `experiments/trace_compression_behavior/distance.py`:

```python
from __future__ import annotations

from itertools import combinations

from experiments.trace_compression_behavior.models import DistancePair, SignatureRecord


SignatureKind = str


def _signature(record: SignatureRecord, kind: SignatureKind) -> dict[str, float]:
    if kind == "profile":
        return record.profile_signature
    if kind == "compression":
        return record.compression_signature
    raise ValueError(f"unknown signature kind: {kind}")


def _field_ranges(records: list[SignatureRecord], kind: SignatureKind) -> dict[str, tuple[float, float]]:
    fields = sorted(set().union(*(_signature(record, kind).keys() for record in records)))
    result = {}
    for field in fields:
        values = [_signature(record, kind)[field] for record in records if field in _signature(record, kind)]
        if values:
            result[field] = (min(values), max(values))
    return result


def _normalized_distance(
    left: SignatureRecord,
    right: SignatureRecord,
    kind: SignatureKind,
    ranges: dict[str, tuple[float, float]],
) -> float:
    left_sig = _signature(left, kind)
    right_sig = _signature(right, kind)
    shared_fields = sorted(set(left_sig) & set(right_sig))
    if not shared_fields:
        return 1.0
    diffs = []
    for field in shared_fields:
        low, high = ranges[field]
        span = high - low
        if span == 0:
            diffs.append(0.0)
        else:
            diffs.append(abs(left_sig[field] - right_sig[field]) / span)
    return sum(diffs) / len(diffs)


def pairwise_distances(records: list[SignatureRecord], kind: SignatureKind) -> list[DistancePair]:
    ranges = _field_ranges(records, kind)
    pairs = []
    for left, right in combinations(records, 2):
        pairs.append(
            DistancePair(
                left_id=left.id,
                right_id=right.id,
                distance=_normalized_distance(left, right, kind, ranges),
            )
        )
    return pairs
```

- [ ] **Step 4: Run distance tests**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_distance.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/trace_compression_behavior/distance.py experiments/trace_compression_behavior/tests/test_distance.py
git commit -m "Compute behavior signature distance matrices"
```

---

### Task 5: Detect Conflict Pairs

**Files:**
- Create: `experiments/trace_compression_behavior/conflicts.py`
- Create: `experiments/trace_compression_behavior/tests/test_conflicts.py`

- [ ] **Step 1: Write failing conflict test**

Create `experiments/trace_compression_behavior/tests/test_conflicts.py`:

```python
from pathlib import Path

from experiments.trace_compression_behavior.catalog import load_catalog
from experiments.trace_compression_behavior.conflicts import detect_conflicts
from experiments.trace_compression_behavior.distance import pairwise_distances
from experiments.trace_compression_behavior.signatures import build_signature_records


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "experiments/trace_compression_behavior/fixtures/synthetic_catalog.json"


def test_detect_conflicts_finds_profile_similar_compression_different_pair():
    records = build_signature_records(load_catalog(FIXTURE))
    profile = pairwise_distances(records, "profile")
    compression = pairwise_distances(records, "compression")

    conflicts = detect_conflicts(records, profile, compression)

    pair = next(conflict for conflict in conflicts if conflict.left_id == "regular_memory_fixture" and conflict.right_id == "irregular_gather_fixture")
    assert pair.conflict_type == "profile_similar_compression_different"
    assert "cross_tb_delta_coverage" in pair.explanation
    assert "address_override_density" in pair.explanation
```

- [ ] **Step 2: Run failing conflict test**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_conflicts.py -q
```

Expected: FAIL with missing `conflicts` module.

- [ ] **Step 3: Implement conflict detection**

Create `experiments/trace_compression_behavior/conflicts.py`:

```python
from __future__ import annotations

from experiments.trace_compression_behavior.models import ConflictPair, DistancePair, SignatureRecord


PROFILE_SIMILAR_THRESHOLD = 0.15
COMPRESSION_DIFFERENT_THRESHOLD = 0.35
RUNTIME_SIMILAR_THRESHOLD = 0.10


def _distance_map(pairs: list[DistancePair]) -> dict[tuple[str, str], float]:
    result = {}
    for pair in pairs:
        key = tuple(sorted((pair.left_id, pair.right_id)))
        result[key] = pair.distance
    return result


def _record_map(records: list[SignatureRecord]) -> dict[str, SignatureRecord]:
    return {record.id: record for record in records}


def _field_delta(left: SignatureRecord, right: SignatureRecord, field: str) -> str:
    left_value = left.compression_signature.get(field)
    right_value = right.compression_signature.get(field)
    return f"{field} {left_value} vs {right_value}"


def _explain(left: SignatureRecord, right: SignatureRecord) -> str:
    parts = [
        "Flat profile counts are close, but memory-structure signature differs:",
        _field_delta(left, right, "cross_tb_delta_coverage"),
        _field_delta(left, right, "address_override_density"),
    ]
    return " ".join(parts)


def detect_conflicts(
    records: list[SignatureRecord],
    profile_pairs: list[DistancePair],
    compression_pairs: list[DistancePair],
    *,
    profile_similar_threshold: float = PROFILE_SIMILAR_THRESHOLD,
    compression_different_threshold: float = COMPRESSION_DIFFERENT_THRESHOLD,
) -> list[ConflictPair]:
    profile_by_pair = _distance_map(profile_pairs)
    compression_by_pair = _distance_map(compression_pairs)
    records_by_id = _record_map(records)
    conflicts = []
    for key, profile_distance in sorted(profile_by_pair.items()):
        compression_distance = compression_by_pair.get(key)
        if compression_distance is None:
            continue
        if profile_distance <= profile_similar_threshold and compression_distance >= compression_different_threshold:
            left = records_by_id[key[0]]
            right = records_by_id[key[1]]
            conflicts.append(
                ConflictPair(
                    left_id=key[0],
                    right_id=key[1],
                    conflict_type="profile_similar_compression_different",
                    profile_distance=profile_distance,
                    compression_distance=compression_distance,
                    explanation=_explain(left, right),
                )
            )
    return conflicts
```

- [ ] **Step 4: Run conflict tests**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_conflicts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/trace_compression_behavior/conflicts.py experiments/trace_compression_behavior/tests/test_conflicts.py
git commit -m "Detect profile-compression conflict pairs"
```

---

### Task 6: Emit JSON and Markdown Reports

**Files:**
- Create: `experiments/trace_compression_behavior/report.py`
- Create: `experiments/trace_compression_behavior/tests/test_report.py`

- [ ] **Step 1: Write failing report test**

Create `experiments/trace_compression_behavior/tests/test_report.py`:

```python
import json
from pathlib import Path

from experiments.trace_compression_behavior.catalog import load_catalog
from experiments.trace_compression_behavior.conflicts import detect_conflicts
from experiments.trace_compression_behavior.distance import pairwise_distances
from experiments.trace_compression_behavior.report import write_reports
from experiments.trace_compression_behavior.signatures import build_signature_records


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "experiments/trace_compression_behavior/fixtures/synthetic_catalog.json"


def test_write_reports_creates_json_and_markdown(tmp_path):
    records = build_signature_records(load_catalog(FIXTURE))
    profile = pairwise_distances(records, "profile")
    compression = pairwise_distances(records, "compression")
    conflicts = detect_conflicts(records, profile, compression)

    write_reports(tmp_path, records, profile, compression, conflicts)

    report = json.loads((tmp_path / "necessity_report.json").read_text())
    markdown = (tmp_path / "necessity_report.md").read_text()

    assert report["summary"]["signature_records"] == 3
    assert report["summary"]["conflict_pairs"] >= 1
    assert "profile-similar / compression-different" in markdown
```

- [ ] **Step 2: Run failing report test**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_report.py -q
```

Expected: FAIL with missing `report` module.

- [ ] **Step 3: Implement report writer**

Create `experiments/trace_compression_behavior/report.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from experiments.trace_compression_behavior.models import ConflictPair, DistancePair, SignatureRecord


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _markdown(records: list[SignatureRecord], conflicts: list[ConflictPair]) -> str:
    lines = [
        "# Trace Compression Behavior Necessity Probe",
        "",
        "## Summary",
        "",
        f"- Signature records: {len(records)}",
        f"- Conflict pairs: {len(conflicts)}",
        "",
        "## Interpretation",
        "",
        "A profile-similar / compression-different pair supports the claim that flat profile summaries can miss trace-level execution structure relevant to microbench generation.",
        "",
        "## Conflict Pairs",
        "",
    ]
    if not conflicts:
        lines.append("No conflict pairs were detected with the current thresholds.")
    for conflict in conflicts:
        lines.extend(
            [
                f"### {conflict.left_id} vs {conflict.right_id}",
                "",
                f"- Type: `{conflict.conflict_type}`",
                f"- Profile distance: `{conflict.profile_distance:.4f}`",
                f"- Compression distance: `{conflict.compression_distance:.4f}`",
                f"- Explanation: {conflict.explanation}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_reports(
    output_dir: Path,
    records: list[SignatureRecord],
    profile_pairs: list[DistancePair],
    compression_pairs: list[DistancePair],
    conflicts: list[ConflictPair],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "signature_table.json", [asdict(record) for record in records])
    _write_json(output_dir / "profile_distance_matrix.json", [asdict(pair) for pair in profile_pairs])
    _write_json(output_dir / "compression_distance_matrix.json", [asdict(pair) for pair in compression_pairs])
    _write_json(output_dir / "conflict_pairs.json", [asdict(conflict) for conflict in conflicts])
    _write_json(
        output_dir / "necessity_report.json",
        {
            "summary": {
                "signature_records": len(records),
                "profile_distance_pairs": len(profile_pairs),
                "compression_distance_pairs": len(compression_pairs),
                "conflict_pairs": len(conflicts),
            },
            "conflicts": [asdict(conflict) for conflict in conflicts],
        },
    )
    (output_dir / "necessity_report.md").write_text(_markdown(records, conflicts))
```

- [ ] **Step 4: Run report tests**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_report.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/trace_compression_behavior/report.py experiments/trace_compression_behavior/tests/test_report.py
git commit -m "Write trace compression behavior reports"
```

---

### Task 7: Add CLI Runner

**Files:**
- Create: `experiments/trace_compression_behavior/run_probe.py`
- Create: `experiments/trace_compression_behavior/tests/test_run_probe.py`

- [ ] **Step 1: Write failing CLI test**

Create `experiments/trace_compression_behavior/tests/test_run_probe.py`:

```python
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "experiments/trace_compression_behavior/fixtures/synthetic_catalog.json"


def test_run_probe_cli_writes_report(tmp_path):
    result = subprocess.run(
        [
            "python3",
            "-m",
            "experiments.trace_compression_behavior.run_probe",
            "--catalog",
            str(FIXTURE),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "necessity_report.md").exists()
    assert "wrote necessity probe report" in result.stdout
```

- [ ] **Step 2: Run failing CLI test**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_run_probe.py -q
```

Expected: FAIL with missing `run_probe` module.

- [ ] **Step 3: Implement CLI**

Create `experiments/trace_compression_behavior/run_probe.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.trace_compression_behavior.catalog import load_catalog
from experiments.trace_compression_behavior.conflicts import detect_conflicts
from experiments.trace_compression_behavior.distance import pairwise_distances
from experiments.trace_compression_behavior.report import write_reports
from experiments.trace_compression_behavior.signatures import build_signature_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trace-compression behavior necessity probe.")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_catalog(args.catalog)
    records = build_signature_records(catalog)
    profile_pairs = pairwise_distances(records, "profile")
    compression_pairs = pairwise_distances(records, "compression")
    conflicts = detect_conflicts(records, profile_pairs, compression_pairs)
    write_reports(args.output_dir, records, profile_pairs, compression_pairs, conflicts)
    print(f"wrote necessity probe report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_run_probe.py -q
```

Expected: PASS.

- [ ] **Step 5: Run full probe test suite**

Run:

```bash
pytest experiments/trace_compression_behavior/tests -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/trace_compression_behavior/run_probe.py experiments/trace_compression_behavior/tests/test_run_probe.py
git commit -m "Add trace compression behavior probe CLI"
```

---

### Task 8: Add Initial Real-Artifact Catalog

**Files:**
- Create: `experiments/trace_compression_behavior/catalogs/initial_probe_catalog.json`
- Create: `experiments/trace_compression_behavior/tests/test_initial_catalog.py`

- [ ] **Step 1: Inspect existing local records**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
path = Path("experiments/mini_transformer/mini_transformer_v4_full.json")
data = json.loads(path.read_text())
records = data.get("per_kernel", {})
print(path)
print("per_kernel_count", len(records))
print("first_keys", list(records)[:5])
PY
```

Expected: prints a nonzero `per_kernel_count`.

- [ ] **Step 2: Create real-artifact catalog**

Create `experiments/trace_compression_behavior/catalogs/initial_probe_catalog.json` using at least five records from `experiments/mini_transformer/mini_transformer_v4_full.json`.

The entries must follow this shape:

```json
{
  "catalog_id": "initial_local_probe",
  "description": "Initial probe over existing mini_transformer local artifacts.",
  "entries": [
    {
      "id": "mini_transformer_kernel_1",
      "label": "mini-transformer-local",
      "role": "target",
      "source_path": "experiments/mini_transformer/mini_transformer_v4_full.json",
      "record_pointer": "/per_kernel/kernel_1"
    }
  ]
}
```

Use actual `per_kernel` keys from the inspection output. Do not invent keys that are absent.

- [ ] **Step 3: Write catalog smoke test**

Create `experiments/trace_compression_behavior/tests/test_initial_catalog.py`:

```python
from pathlib import Path

from experiments.trace_compression_behavior.catalog import load_catalog
from experiments.trace_compression_behavior.signatures import build_signature_records


ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "experiments/trace_compression_behavior/catalogs/initial_probe_catalog.json"


def test_initial_probe_catalog_loads_and_extracts_records():
    catalog = load_catalog(CATALOG)
    records = build_signature_records(catalog)

    assert len(records) >= 5
    assert any(record.compression_signature for record in records)
```

- [ ] **Step 4: Run initial catalog test**

Run:

```bash
pytest experiments/trace_compression_behavior/tests/test_initial_catalog.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/trace_compression_behavior/catalogs/initial_probe_catalog.json experiments/trace_compression_behavior/tests/test_initial_catalog.py
git commit -m "Add initial trace compression behavior probe catalog"
```

---

### Task 9: Run the Probe and Save Baseline Results

**Files:**
- Generated: `experiments/trace_compression_behavior/results/necessity_probe/*`

- [ ] **Step 1: Run synthetic fixture probe**

Run:

```bash
python3 -m experiments.trace_compression_behavior.run_probe \
  --catalog experiments/trace_compression_behavior/fixtures/synthetic_catalog.json \
  --output-dir experiments/trace_compression_behavior/results/synthetic_probe
```

Expected: prints `wrote necessity probe report`.

- [ ] **Step 2: Inspect synthetic report**

Run:

```bash
sed -n '1,160p' experiments/trace_compression_behavior/results/synthetic_probe/necessity_report.md
```

Expected: report includes at least one `profile-similar / compression-different` pair.

- [ ] **Step 3: Run initial real-artifact probe**

Run:

```bash
python3 -m experiments.trace_compression_behavior.run_probe \
  --catalog experiments/trace_compression_behavior/catalogs/initial_probe_catalog.json \
  --output-dir experiments/trace_compression_behavior/results/necessity_probe
```

Expected: prints `wrote necessity probe report`.

- [ ] **Step 4: Inspect real-artifact report**

Run:

```bash
sed -n '1,200p' experiments/trace_compression_behavior/results/necessity_probe/necessity_report.md
```

Expected: report prints signature record count and conflict pair count. Zero conflict pairs is allowed for this first local catalog, but the report must state that no conflict pairs were detected under current thresholds.

- [ ] **Step 5: Commit generated baseline results**

Only commit small JSON/Markdown result files. If any generated file exceeds 1 MB, leave it untracked and add a note to the Markdown report with the path and file size.

```bash
git add experiments/trace_compression_behavior/results/synthetic_probe experiments/trace_compression_behavior/results/necessity_probe
git commit -m "Record initial trace compression behavior probe results"
```

---

### Task 10: Write Research Interpretation Note

**Files:**
- Create: `docs/trace-compression-behavior-necessity-probe-2026-04-29.md`

- [ ] **Step 1: Generate interpretation note from probe results**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

synthetic_path = Path("experiments/trace_compression_behavior/results/synthetic_probe/necessity_report.json")
real_path = Path("experiments/trace_compression_behavior/results/necessity_probe/necessity_report.json")
out_path = Path("docs/trace-compression-behavior-necessity-probe-2026-04-29.md")

synthetic = json.loads(synthetic_path.read_text())
real = json.loads(real_path.read_text())

synthetic_conflicts = synthetic["summary"]["conflict_pairs"]
real_conflicts = real["summary"]["conflict_pairs"]

real_decision = (
    "当前真实本地 catalog 已经出现 compression-signature conflict，下一步可以扩展 target 数量并进入 microbench matching。"
    if real_conflicts
    else "当前真实本地 catalog 暂未出现 conflict pair。这个结果不否定学术线，但说明第一批真实样本的行为多样性不足；下一步应补充 regular-memory、irregular-gather、branch-heavy、reduction 和 atomic-contention targets。"
)

top_conflicts = real.get("conflicts", [])[:3]
conflict_lines = []
if top_conflicts:
    for conflict in top_conflicts:
        conflict_lines.extend(
            [
                f"- `{conflict['left_id']}` vs `{conflict['right_id']}`",
                f"  - profile distance: `{conflict['profile_distance']:.4f}`",
                f"  - compression distance: `{conflict['compression_distance']:.4f}`",
                f"  - explanation: {conflict['explanation']}",
            ]
        )
else:
    conflict_lines.append("- 本轮真实本地 catalog 没有检测到 profile-similar / compression-different pair。")

out_path.write_text(
    "\n".join(
        [
            "# Trace Compression Behavior Necessity Probe",
            "",
            "日期：2026-04-29",
            "",
            "## 1. 问题",
            "",
            "这次 probe 检查 flat profile signatures 是否足以作为 microbench generation target，以及 compression-derived signatures 是否提供额外的 trace-level behavior signal。",
            "",
            "## 2. 判定逻辑",
            "",
            "支持 compression-guided microbench generation 的最强证据是 profile-similar、compression-different，并且这种差异能被人工解释为真实执行结构差异。",
            "",
            "这说明只用 runtime、instruction count、memory op count 等 flat features 会混淆不同执行结构。",
            "",
            "## 3. 本轮输入",
            "",
            "- Synthetic fixture catalog: `experiments/trace_compression_behavior/fixtures/synthetic_catalog.json`",
            "- Initial local catalog: `experiments/trace_compression_behavior/catalogs/initial_probe_catalog.json`",
            "",
            "## 4. 本轮结果",
            "",
            f"- Synthetic signature records: `{synthetic['summary']['signature_records']}`",
            f"- Synthetic conflict pairs: `{synthetic_conflicts}`",
            f"- Local signature records: `{real['summary']['signature_records']}`",
            f"- Local conflict pairs: `{real_conflicts}`",
            "",
            "## 5. 真实本地样本解释",
            "",
            *conflict_lines,
            "",
            "## 6. 决策",
            "",
            real_decision,
            "",
            "进入 microbench matching 的最低条件是：synthetic fixture 展示预期 conflict pattern，并且真实本地 catalog 的行为覆盖被扩展到至少五类 target。",
            "",
        ]
    )
)
print(out_path)
PY
```

Expected: prints `docs/trace-compression-behavior-necessity-probe-2026-04-29.md`.

- [ ] **Step 2: Check note has no template remnants**

Run:

```bash
rg -n "Summarize|replace|template" docs/trace-compression-behavior-necessity-probe-2026-04-29.md
```

Expected: no output.

- [ ] **Step 3: Commit interpretation note**

```bash
git add docs/trace-compression-behavior-necessity-probe-2026-04-29.md
git commit -m "Document trace compression behavior necessity probe"
```

---

## Final Verification

Run:

```bash
pytest experiments/trace_compression_behavior/tests -q
python3 -m experiments.trace_compression_behavior.run_probe \
  --catalog experiments/trace_compression_behavior/fixtures/synthetic_catalog.json \
  --output-dir /tmp/trace_compression_behavior_synthetic_probe
git diff --check
```

Expected:

- all `experiments/trace_compression_behavior/tests` pass;
- CLI prints `wrote necessity probe report`;
- `/tmp/trace_compression_behavior_synthetic_probe/necessity_report.md` exists;
- `git diff --check` exits with code 0.

## Handoff Criteria

This plan is complete when:

1. The probe package can run on deterministic fixtures.
2. The synthetic fixture report demonstrates at least one profile-similar/compression-different pair.
3. The initial local catalog can be loaded and analyzed.
4. The interpretation note states whether current real artifacts support, weaken, or require expanding the catalog for the academic line.
5. No L1 selector, schema, or B-line code was modified as part of this probe.
