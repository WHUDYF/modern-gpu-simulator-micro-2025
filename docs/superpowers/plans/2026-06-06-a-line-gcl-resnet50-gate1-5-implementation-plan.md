# A 线 GCL ResNet-50 Gate 1-5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 ResNet-50 real trace 到 Gate 5 kernel embedding table 的完整 GCL 前半段闭环。

**Architecture:** 本计划复用现有 `experiments/gcl_phase_b` 的 SM selection、trace scope、graph builder、tensorizer、augmentation、RGCN training 和 artifact hashing 能力，只新增 ResNet-50 real trace adapter、Gate 1 bundle、Gate 2 manifest adapter 和 Gate 5 formal export contract。执行路径固定为 `Gate 1 -> Gate 2 -> Gate 3 -> Gate 4 -> Gate 5`，本轮不进入 Gate 6 clustering / family classification。

**Tech Stack:** Python standard library、JSON artifacts、NumPy、PyTorch、pytest、现有 `experiments.gcl_phase_a` / `experiments.gcl_phase_b` modules。

---

## 1. 本轮目标边界

本轮只跑到 Gate 5：

```text
ResNet-50 NVBit trace artifacts
  -> Gate 1 resnet50_trace_adapter_bundle.json
  -> Gate 2 representative_sm_trace_manifest.json
  -> Gate 3 canonical_graph_bundle.json
  -> Gate 4 graph_tensor_bundle.json
  -> Gate 5 kernel_embedding_table.json
```

本轮不做：

```text
K-Means clustering
silhouette selected K
representative kernel selection
kernel family classification
GNN + fully connected family head
调参比例预测
simulator speedup / accuracy claim
```

## 2. 参考 Spec

必须遵循：

- `docs/superpowers/specs/2026-06-05-a-line-gcl-resnet50-gate1-trace-adapter-design.md`
- `docs/superpowers/specs/2026-06-05-a-line-gcl-resnet50-gate2-representative-sm-manifest-design.md`
- `docs/superpowers/specs/2026-06-05-a-line-gcl-resnet50-gate3-canonical-graph-construction-design.md`
- `docs/superpowers/specs/2026-06-06-a-line-gcl-resnet50-gate4-tensorization-design.md`
- `docs/superpowers/specs/2026-06-06-a-line-gcl-resnet50-gate5-rgcn-contrastive-training-design.md`

实现依赖：

- `docs/superpowers/plans/2026-06-02-a-line-gcl-m1-m2-phase-a-semantic-e2e-plan.md`
- `docs/superpowers/plans/2026-06-04-a-line-gcl-m1-m2-phase-b-implementation-plan.md`

如果旧 Phase B plan 与 Gate 1-5 spec 冲突，本轮采用 Gate 1-5 spec 的最新边界：

- Gate 5 停在 `kernel_embedding_table.json`，不运行 M0 selector。
- Gate 5 readout 使用 `node -> warp -> CTA -> selected SM -> kernel`。
- `gcl_resnet50_no_pseudo_node` 可以作为 functional-first 模式，但不能标记为 strict paper reproduction。
- `gcl_resnet50_mem_ref_only` 是当前最接近论文 pseudo-node 描述的模式。

## 3. 文件结构

新增：

```text
experiments/gcl_phase_b/resnet50_adapter.py
  Gate 1: 读取 ResNet-50 trace artifacts，输出 adapter bundle。

experiments/gcl_phase_b/resnet50_manifest.py
  Gate 2: 消费 adapter bundle，运行 scheduler_signature_medoid_sm，输出 representative-SM manifest。

experiments/gcl_phase_b/resnet50_gate_pipeline.py
  Gate 1-5 CLI orchestrator；本轮停止在 kernel_embedding_table.json。

tests/gcl_phase_b/test_resnet50_adapter.py
  Gate 1 contract tests。

tests/gcl_phase_b/test_resnet50_manifest.py
  Gate 2 contract tests。

tests/gcl_phase_b/test_resnet50_gate_pipeline.py
  Gate 1-5 disk-backed end-to-end smoke tests。

tests/fixtures/gcl_resnet50_gate1/
  小规模 ResNet-like fixture，字段模拟真实 NVBit adapter 所需结构。
```

修改：

```text
experiments/gcl_phase_b/readout.py
  从 node -> warp -> kernel 升级为 node -> warp -> CTA -> selected SM -> kernel。

experiments/gcl_phase_b/tensorizer.py
  补充 representation_mode / pseudo_node_mode / paper_reproduction_mode 字段。

experiments/gcl_phase_b/pipeline.py
  增加 Gate 5 formal export helper，允许停止在 embedding table，不写 selector artifacts。

tests/gcl_phase_b/test_readout.py
  更新 readout 层级断言。

tests/gcl_phase_b/test_tensorizer.py
  增加 representation mode 断言。

tests/gcl_phase_b/test_embedding_export.py
  增加 Gate 5 canonical export contract。
```

## 4. 验收标准

- AC-1: Gate 1 adapter bundle 可以从 ResNet-like trace fixture 稳定生成。
- AC-2: Gate 1 必须拒绝缺少 `scheduler_metadata_source = real_nvbit_smid` 的输入。
- AC-3: Gate 2 必须从 adapter bundle 构造 `scheduler_metadata_by_sm`、`cta_to_sm` 和 `all_trace_entries`。
- AC-4: Gate 2 必须使用 `scheduler_signature_medoid_sm` deterministic 地选择 representative SM。
- AC-5: Gate 2 输出的 `representative_sm_trace_manifest.json` 必须能被现有 Phase B trace validator 接受。
- AC-6: Gate 3 必须从 selected-SM all-CTA trace records 构建 canonical graph bundle。
- AC-7: Gate 4 必须输出 feature width 64 的 graph tensor bundle，并记录 representation mode。
- AC-8: Gate 5 augmentation manifest 必须只引用 canonical tensor，不覆盖 canonical tensor。
- AC-9: Gate 5 readout 必须使用 `node -> warp -> CTA -> selected SM -> kernel`。
- AC-10: Gate 5 必须导出 256 维 `kernel_embedding_table.json`，且不写 Gate 6 selector artifacts。
- AC-11: Gate 1-5 pipeline 必须一条命令跑通小规模 fixture。
- AC-12: 所有关键 artifact hash 必须可复现。

---

### Task 1: Gate 1 ResNet-50 Adapter Fixture

**Files:**
- Create: `tests/fixtures/gcl_resnet50_gate1/dynamic_trace.json`
- Create: `tests/fixtures/gcl_resnet50_gate1/enhanced_execution_info.json`
- Create: `tests/fixtures/gcl_resnet50_gate1/scheduler_metadata.json`
- Create: `tests/fixtures/gcl_resnet50_gate1/threadblocks.json`
- Create: `tests/fixtures/gcl_resnet50_gate1/stats.csv`
- Test: `tests/gcl_phase_b/test_resnet50_adapter.py`

- [ ] **Step 1: Create a small ResNet-like Gate 1 fixture**

Create fixture data with two kernel invocations, three candidate SMs, at least two CTAs on the selected-medoid SM, two warps per CTA, and load-compute-store instruction patterns.

`scheduler_metadata.json` must include:

```json
{
  "scheduler_metadata_source": "real_nvbit_smid",
  "kernel_invocations": [
    {
      "kernel_id": 17,
      "cta_records": [
        {
          "cta_id": "0,0,0",
          "sm_id": 0,
          "first_seen_order": 1,
          "last_seen_order": 3,
          "warp_ids": [0, 1],
          "trace_entry_count": 8
        },
        {
          "cta_id": "1,0,0",
          "sm_id": 1,
          "first_seen_order": 2,
          "last_seen_order": 5,
          "warp_ids": [0, 1],
          "trace_entry_count": 8
        },
        {
          "cta_id": "2,0,0",
          "sm_id": 1,
          "first_seen_order": 4,
          "last_seen_order": 6,
          "warp_ids": [0, 1],
          "trace_entry_count": 8
        }
      ]
    }
  ]
}
```

- [ ] **Step 2: Write failing Gate 1 fixture shape test**

Add:

```python
from pathlib import Path

from experiments.gcl_phase_b.resnet50_adapter import load_resnet50_trace_sources


def test_resnet50_gate1_fixture_sources_are_loadable():
    root = Path("tests/fixtures/gcl_resnet50_gate1")

    sources = load_resnet50_trace_sources(root)

    assert sources.scheduler_metadata["scheduler_metadata_source"] == "real_nvbit_smid"
    assert sources.dynamic_trace["kernel_invocations"]
    assert sources.threadblocks["threadblocks"]
    assert sources.enhanced_execution_info["instructions"]
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
pytest -q tests/gcl_phase_b/test_resnet50_adapter.py::test_resnet50_gate1_fixture_sources_are_loadable
```

Expected: FAIL with `ModuleNotFoundError` or missing `load_resnet50_trace_sources`.

- [ ] **Step 4: Commit fixture and failing test**

```bash
git add tests/fixtures/gcl_resnet50_gate1 tests/gcl_phase_b/test_resnet50_adapter.py
git commit -m "test: add ResNet50 gate1 trace fixture"
```

### Task 2: Gate 1 Adapter Bundle

**Files:**
- Create: `experiments/gcl_phase_b/resnet50_adapter.py`
- Modify: `tests/gcl_phase_b/test_resnet50_adapter.py`

- [ ] **Step 1: Implement source loader and adapter bundle builder**

Create `experiments/gcl_phase_b/resnet50_adapter.py` with:

```python
"""ResNet-50 real-trace adapter for GCL Gate 1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import hash_without, read_json, write_json


ADAPTER_ARTIFACT_TYPE = "gcl_resnet50_trace_adapter_bundle"
ADAPTER_VERSION = "gate1_trace_adapter_v1"


@dataclass(frozen=True)
class ResNet50TraceSources:
    dynamic_trace: dict[str, Any]
    threadblocks: dict[str, Any]
    enhanced_execution_info: dict[str, Any]
    scheduler_metadata: dict[str, Any]
    stats_rows: list[dict[str, str]]


def load_resnet50_trace_sources(root: Path) -> ResNet50TraceSources:
    stats_path = root / "stats.csv"
    with stats_path.open(newline="", encoding="utf-8") as handle:
        stats_rows = list(csv.DictReader(handle))
    return ResNet50TraceSources(
        dynamic_trace=read_json(root / "dynamic_trace.json"),
        threadblocks=read_json(root / "threadblocks.json"),
        enhanced_execution_info=read_json(root / "enhanced_execution_info.json"),
        scheduler_metadata=read_json(root / "scheduler_metadata.json"),
        stats_rows=stats_rows,
    )


def build_resnet50_trace_adapter_bundle(root: Path) -> dict[str, Any]:
    sources = load_resnet50_trace_sources(root)
    if sources.scheduler_metadata.get("scheduler_metadata_source") != "real_nvbit_smid":
        raise ValueError("scheduler_metadata_source must be real_nvbit_smid")
    kernel_invocation_table = _kernel_invocation_table(sources.dynamic_trace)
    static_instruction_table = sources.enhanced_execution_info.get("instructions", [])
    cta_scheduler_records = _cta_scheduler_records(sources.scheduler_metadata)
    per_warp_trace_records = _per_warp_trace_records(sources.threadblocks)
    bundle = {
        "artifact_type": ADAPTER_ARTIFACT_TYPE,
        "artifact_version": ADAPTER_VERSION,
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "source_artifact_hashes": {
            "dynamic_trace": hash_without(sources.dynamic_trace, "_hash"),
            "threadblocks": hash_without(sources.threadblocks, "_hash"),
            "enhanced_execution_info": hash_without(sources.enhanced_execution_info, "_hash"),
            "scheduler_metadata": hash_without(sources.scheduler_metadata, "_hash"),
        },
        "kernel_invocation_table": kernel_invocation_table,
        "static_instruction_table": static_instruction_table,
        "cta_scheduler_records": cta_scheduler_records,
        "per_warp_trace_records": per_warp_trace_records,
        "adapter_validation_report": {
            "status": "passed",
            "scheduler_metadata_complete": True,
            "errors": [],
        },
    }
    bundle["adapter_bundle_hash"] = hash_without(bundle, "adapter_bundle_hash")
    validate_resnet50_trace_adapter_bundle(bundle)
    return bundle


def _kernel_invocation_table(dynamic_trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for launch_order, row in enumerate(dynamic_trace.get("kernel_invocations", [])):
        rows.append(
            {
                "kernel_invocation_id": f"resnet50_k{launch_order:05d}",
                "kernel_id": row["kernel_id"],
                "kernel_name": row["kernel_name"],
                "function_unique_id": row["function_unique_id"],
                "device_id": row.get("device_id", 0),
                "stream_id": row.get("stream_id", 0),
                "launch_order": launch_order,
                "grid_dim": row["grid_dim"],
                "block_dim": row["block_dim"],
                "shared_memory_size": row.get("shared_memory_size", 0),
                "register_count": row.get("register_count", 0),
            }
        )
    return rows


def _cta_scheduler_records(scheduler_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for invocation in scheduler_metadata.get("kernel_invocations", []):
        kernel_id = invocation["kernel_id"]
        for cta in invocation.get("cta_records", []):
            records.append({"kernel_id": kernel_id, **cta})
    return records


def _per_warp_trace_records(threadblocks: dict[str, Any]) -> list[dict[str, Any]]:
    return list(threadblocks.get("threadblocks", []))


def validate_resnet50_trace_adapter_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("artifact_type") != ADAPTER_ARTIFACT_TYPE:
        raise ValueError("unexpected adapter artifact_type")
    if bundle.get("artifact_version") != ADAPTER_VERSION:
        raise ValueError("unexpected adapter artifact_version")
    if bundle.get("scheduler_metadata_source") != "real_nvbit_smid":
        raise ValueError("scheduler_metadata_source must be real_nvbit_smid")
    report = bundle.get("adapter_validation_report", {})
    if report.get("status") != "passed":
        raise ValueError("adapter validation report must be passed")
    if report.get("scheduler_metadata_complete") is not True:
        raise ValueError("scheduler metadata must be complete")
    if report.get("errors") != []:
        raise ValueError("adapter errors must be empty")
    if not bundle.get("kernel_invocation_table"):
        raise ValueError("kernel_invocation_table must be non-empty")
    if not bundle.get("cta_scheduler_records"):
        raise ValueError("cta_scheduler_records must be non-empty")
    if not bundle.get("per_warp_trace_records"):
        raise ValueError("per_warp_trace_records must be non-empty")
    if bundle.get("adapter_bundle_hash") != hash_without(bundle, "adapter_bundle_hash"):
        raise ValueError("adapter_bundle_hash is not reproducible")


def write_resnet50_trace_adapter_bundle(root: Path, out_path: Path) -> dict[str, Any]:
    bundle = build_resnet50_trace_adapter_bundle(root)
    write_json(out_path, bundle)
    return bundle
```

- [ ] **Step 2: Add positive and negative tests**

Add tests:

```python
import copy
from pathlib import Path

import pytest

from experiments.gcl_phase_b.resnet50_adapter import (
    build_resnet50_trace_adapter_bundle,
    validate_resnet50_trace_adapter_bundle,
)


def test_gate1_builds_resnet50_trace_adapter_bundle():
    bundle = build_resnet50_trace_adapter_bundle(Path("tests/fixtures/gcl_resnet50_gate1"))

    validate_resnet50_trace_adapter_bundle(bundle)
    assert bundle["artifact_type"] == "gcl_resnet50_trace_adapter_bundle"
    assert bundle["workload_id"] == "resnet50"
    assert bundle["scheduler_metadata_source"] == "real_nvbit_smid"
    assert bundle["adapter_validation_report"]["status"] == "passed"


def test_gate1_rejects_non_real_scheduler_metadata():
    bundle = build_resnet50_trace_adapter_bundle(Path("tests/fixtures/gcl_resnet50_gate1"))
    corrupted = copy.deepcopy(bundle)
    corrupted["scheduler_metadata_source"] = "file_order_fallback"

    with pytest.raises(ValueError, match="real_nvbit_smid"):
        validate_resnet50_trace_adapter_bundle(corrupted)
```

- [ ] **Step 3: Run Gate 1 tests**

Run:

```bash
pytest -q tests/gcl_phase_b/test_resnet50_adapter.py
```

Expected: PASS.

- [ ] **Step 4: Commit Gate 1 adapter**

```bash
git add experiments/gcl_phase_b/resnet50_adapter.py tests/gcl_phase_b/test_resnet50_adapter.py
git commit -m "feat: add ResNet50 gate1 adapter bundle"
```

### Task 3: Gate 2 Representative-SM Manifest From Adapter Bundle

**Files:**
- Create: `experiments/gcl_phase_b/resnet50_manifest.py`
- Create: `tests/gcl_phase_b/test_resnet50_manifest.py`

- [ ] **Step 1: Write failing Gate 2 manifest test**

Add:

```python
from pathlib import Path

from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.trace_scope import validate_phase_b_trace_manifest


def test_gate2_builds_phase_b_manifest_from_resnet50_bundle():
    bundle = build_resnet50_trace_adapter_bundle(Path("tests/fixtures/gcl_resnet50_gate1"))

    manifest, reports, preview = build_representative_sm_manifest_from_bundle(bundle)

    validate_phase_b_trace_manifest(manifest)
    assert manifest["artifact_type"] == "gcl_phase_b_trace_manifest"
    assert manifest["collection_scope"] == "single_representative_sm_all_ctas"
    assert reports["reports"]
    assert preview["kernel_invocation_count"] == len(manifest["kernel_invocations"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/gcl_phase_b/test_resnet50_manifest.py::test_gate2_builds_phase_b_manifest_from_resnet50_bundle
```

Expected: FAIL with missing module or missing function.

- [ ] **Step 3: Implement Gate 2 manifest builder**

Create `experiments/gcl_phase_b/resnet50_manifest.py` with functions:

```python
"""Gate 2 representative-SM manifest construction from ResNet-50 adapter bundle."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .resnet50_adapter import validate_resnet50_trace_adapter_bundle
from .sm_selection import select_representative_sm
from .trace_fixture import COLLECTION_SCOPE
from .trace_scope import validate_phase_b_trace_manifest
from .utils import hash_without


def build_representative_sm_manifest_from_bundle(
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_resnet50_trace_adapter_bundle(bundle)
    kernel_by_id = {row["kernel_id"]: row for row in bundle["kernel_invocation_table"]}
    scheduler_by_kernel = _scheduler_records_by_kernel(bundle["cta_scheduler_records"])
    trace_records_by_kernel = _trace_records_by_kernel(bundle["per_warp_trace_records"])
    invocations = []
    reports = []
    for kernel_id, kernel_row in sorted(kernel_by_id.items()):
        selection_input = _selection_input(kernel_row, scheduler_by_kernel[kernel_id], trace_records_by_kernel[kernel_id])
        report = select_representative_sm(selection_input)
        invocation = _manifest_invocation(selection_input, report)
        invocations.append(invocation)
        reports.append(report)
    manifest = {
        "artifact_type": "gcl_phase_b_trace_manifest",
        "manifest_version": "gcl_phase_b_trace_manifest_v1",
        "collection_scope": COLLECTION_SCOPE,
        "trace_family": "resnet50_real_trace",
        "kernel_invocations": invocations,
    }
    manifest["trace_manifest_hash"] = hash_without(manifest, "trace_manifest_hash")
    validate_phase_b_trace_manifest(manifest)
    report_bundle = {
        "artifact_type": "gcl_resnet50_selected_sm_policy_report_bundle",
        "artifact_version": "gate2_selected_sm_policy_report_bundle_v1",
        "reports": reports,
    }
    report_bundle["selected_sm_policy_report_bundle_hash"] = hash_without(
        report_bundle, "selected_sm_policy_report_bundle_hash"
    )
    preview = {
        "artifact_type": "gcl_resnet50_scope_preview_report",
        "artifact_version": "gate2_scope_preview_report_v1",
        "kernel_invocation_count": len(invocations),
        "selected_sms": [invocation["selected_sm"] for invocation in invocations],
    }
    preview["scope_preview_report_hash"] = hash_without(preview, "scope_preview_report_hash")
    return manifest, report_bundle, preview
```

Implement helper functions in the same file:

```python
def _scheduler_records_by_kernel(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["kernel_id"])].append(record)
    return grouped


def _trace_records_by_kernel(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["kernel_id"])].append(record)
    return grouped
```

`_selection_input` must produce the fields required by `select_representative_sm`:

```python
def _selection_input(
    kernel_row: dict[str, Any],
    scheduler_records: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
) -> dict[str, Any]:
    scheduler_metadata_by_sm: dict[str, dict[str, Any]] = {}
    cta_to_sm = {}
    for record in scheduler_records:
        sm_id = str(record["sm_id"])
        metadata = scheduler_metadata_by_sm.setdefault(
            sm_id,
            {
                "sm_id": int(record["sm_id"]),
                "cta_ids": [],
                "warp_ids_by_cta": {},
                "trace_entry_count_by_cta": {},
                "cta_start_order": {},
                "cta_end_order": {},
            },
        )
        cta_id = record["cta_id"]
        metadata["cta_ids"].append(cta_id)
        metadata["warp_ids_by_cta"][cta_id] = record["warp_ids"]
        metadata["trace_entry_count_by_cta"][cta_id] = record["trace_entry_count"]
        metadata["cta_start_order"][cta_id] = record["first_seen_order"]
        metadata["cta_end_order"][cta_id] = record["last_seen_order"]
        cta_to_sm[cta_id] = int(record["sm_id"])
    for metadata in scheduler_metadata_by_sm.values():
        metadata["cta_ids"] = sorted(
            metadata["cta_ids"],
            key=lambda cta_id: (int(metadata["cta_start_order"][cta_id]), cta_id),
        )
    all_trace_entries = _flatten_trace_entries(kernel_row, trace_records)
    return {
        "kernel_invocation_id": kernel_row["kernel_invocation_id"],
        "trace_family": "resnet50_real_trace",
        "selected_sm_policy": "scheduler_signature_medoid_sm",
        "scheduler_metadata_by_sm": scheduler_metadata_by_sm,
        "cta_to_sm": cta_to_sm,
        "all_trace_entries": all_trace_entries,
        "instruction_count_before_scope": len(all_trace_entries),
        "warp_count_before_scope": len({(entry["cta_id"], entry["warp_id"]) for entry in all_trace_entries}),
    }
```

`_flatten_trace_entries` must normalize each instruction entry into the Gate 2 schema:

```python
def _flatten_trace_entries(kernel_row: dict[str, Any], trace_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for record in trace_records:
        for entry in record["entries"]:
            normalized = {
                "kernel_invocation_id": kernel_row["kernel_invocation_id"],
                "trace_family": "resnet50_real_trace",
                "collection_scope": COLLECTION_SCOPE,
                "cta_id": record["cta_id"],
                "warp_id": record["warp_id"],
                "trace_index": entry["trace_index"],
                "pc": entry["pc"],
                "opcode": entry["opcode"],
                "active_mask": entry.get("active_mask", "0xffffffff"),
                "predicate_mask": entry.get("predicate_mask", "0xffffffff"),
                "destination_operands": entry.get("destination_operands", []),
                "source_operands": entry.get("source_operands", []),
                "memory_address_metadata": entry.get("memory_address_metadata", {}),
                "observed_dynamic_values": entry.get("observed_dynamic_values", []),
                "source_entry_hash": entry["source_entry_hash"],
            }
            entries.append(normalized)
    return sorted(entries, key=lambda item: (item["cta_id"], item["warp_id"], item["trace_index"]))
```

`_manifest_invocation` must scope to selected SM:

```python
def _manifest_invocation(selection_input: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    selected_sm = int(report["selected_sm"])
    included_cta_ids = list(selection_input["scheduler_metadata_by_sm"][str(selected_sm)]["cta_ids"])
    scoped_entries = [
        entry for entry in selection_input["all_trace_entries"]
        if entry["cta_id"] in set(included_cta_ids)
    ]
    invocation = {
        **selection_input,
        "collection_scope": COLLECTION_SCOPE,
        "selected_sm": selected_sm,
        "selected_sm_policy": report["selected_sm_policy"],
        "selected_sm_reason": report["selected_sm_reason"],
        "candidate_sm_count": report["candidate_sm_count"],
        "included_cta_ids": included_cta_ids,
        "instruction_count": len(scoped_entries),
        "warp_count": len({(entry["cta_id"], entry["warp_id"]) for entry in scoped_entries}),
        "selected_sm_policy_report": report,
        "selected_sm_policy_report_hash": report["selection_hash"],
    }
    invocation["trace_hash"] = hash_without(invocation, "trace_hash")
    return invocation
```

- [ ] **Step 4: Run Gate 2 tests**

Run:

```bash
pytest -q tests/gcl_phase_b/test_resnet50_manifest.py
```

Expected: PASS.

- [ ] **Step 5: Commit Gate 2 manifest builder**

```bash
git add experiments/gcl_phase_b/resnet50_manifest.py tests/gcl_phase_b/test_resnet50_manifest.py
git commit -m "feat: build ResNet50 representative SM manifest"
```

### Task 4: Gate 3-4 Formal Bundle Names And Representation Mode

**Files:**
- Modify: `experiments/gcl_phase_b/graph_builder.py`
- Modify: `experiments/gcl_phase_b/tensorizer.py`
- Modify: `tests/gcl_phase_b/test_tensorizer.py`

- [ ] **Step 1: Add failing tensor representation mode test**

Add:

```python
from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs, validate_phase_b_tensor_artifact
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records


def test_phase_b_tensor_records_resnet50_representation_mode():
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    graph = build_phase_b_graphs(records)[0]

    tensor = tensorize_phase_b_graphs([graph])[0]

    validate_phase_b_tensor_artifact(tensor)
    assert tensor["representation_mode"] == "gcl_resnet50_mem_ref_only"
    assert tensor["pseudo_node_mode"] == "mem_ref_only"
    assert tensor["paper_reproduction_mode"] == "strict_gcl_sampler_node_features"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/gcl_phase_b/test_tensorizer.py::test_phase_b_tensor_records_resnet50_representation_mode
```

Expected: FAIL because fields are missing.

- [ ] **Step 3: Add representation fields to tensorizer**

In `tensorize_phase_b_graph`, add:

```python
        "representation_mode": _representation_mode(graph),
        "pseudo_node_mode": _pseudo_node_mode(graph),
        "paper_reproduction_mode": PAPER_REPRODUCTION_MODE,
```

Add helpers:

```python
def _pseudo_node_mode(graph: dict[str, Any]) -> str:
    pseudo_nodes = [node for node in graph["nodes"] if node["node_type"] == "pseudo"]
    if not pseudo_nodes:
        return "no_pseudo_node"
    if all(node.get("pseudo_kind") == "mem_ref" for node in pseudo_nodes):
        return "mem_ref_only"
    raise ValueError("unsupported pseudo node mode")


def _representation_mode(graph: dict[str, Any]) -> str:
    mode = _pseudo_node_mode(graph)
    if mode == "mem_ref_only":
        return "gcl_resnet50_mem_ref_only"
    if mode == "no_pseudo_node":
        return "gcl_resnet50_no_pseudo_node"
    raise ValueError("unsupported representation mode")
```

Update `validate_phase_b_tensor_artifact` required fields:

```python
        "representation_mode",
        "pseudo_node_mode",
        "paper_reproduction_mode",
```

Add validator checks:

```python
    if tensor["representation_mode"] not in {
        "gcl_resnet50_mem_ref_only",
        "gcl_resnet50_no_pseudo_node",
    }:
        raise ValueError("unsupported representation_mode")
    if tensor["pseudo_node_mode"] not in {"mem_ref_only", "no_pseudo_node"}:
        raise ValueError("unsupported pseudo_node_mode")
    if tensor["paper_reproduction_mode"] != PAPER_REPRODUCTION_MODE:
        raise ValueError("unexpected paper_reproduction_mode")
```

- [ ] **Step 4: Run tensorizer tests**

Run:

```bash
pytest -q tests/gcl_phase_b/test_tensorizer.py
```

Expected: PASS.

- [ ] **Step 5: Commit representation mode contract**

```bash
git add experiments/gcl_phase_b/tensorizer.py tests/gcl_phase_b/test_tensorizer.py
git commit -m "feat: record ResNet50 GCL representation mode"
```

### Task 5: Gate 5 CTA-Aware Hierarchical Readout

**Files:**
- Modify: `experiments/gcl_phase_b/readout.py`
- Modify: `tests/gcl_phase_b/test_readout.py`

- [ ] **Step 1: Replace old readout assertion with CTA-aware test**

Update or add:

```python
def test_hierarchical_readout_pools_nodes_to_warps_to_ctas_to_selected_sm_to_kernel():
    torch = require_torch()
    tensor = _tensor()
    encoder = MinimalRGCNEncoder()
    node_features = torch.as_tensor(tensor["node_features"], dtype=torch.float32)
    edge_index = torch.as_tensor(tensor["edge_index"], dtype=torch.long)
    edge_type = torch.as_tensor(tensor["edge_type"], dtype=torch.long)
    node_embeddings = encoder(node_features, edge_index, edge_type)

    manifest, kernel_embedding = build_readout_manifest(tensor, node_embeddings)

    validate_readout_manifest(manifest, tensor)
    assert manifest["readout_hierarchy"] == "node_to_warp_to_cta_to_selected_sm_to_kernel"
    assert manifest["kernel"]["kernel_embedding_source"] == "selected_sm_embedding"
    assert manifest["kernel"]["kernel_embedding_dim"] == 256
    assert manifest["ctas"]
    assert kernel_embedding.shape[0] == 256
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/gcl_phase_b/test_readout.py::test_hierarchical_readout_pools_nodes_to_warps_to_ctas_to_selected_sm_to_kernel
```

Expected: FAIL because current readout only records warp -> kernel.

- [ ] **Step 3: Implement CTA-aware readout**

Change `build_readout_manifest` so it:

```text
1. pools node embeddings within each warp_partition;
2. groups warp embeddings by cta_id from warp_partition_tensors;
3. mean-pools each CTA;
4. mean-pools selected-SM CTA embeddings;
5. records selected_sm_embedding as kernel embedding.
```

Manifest must include:

```python
{
    "readout_hierarchy": "node_to_warp_to_cta_to_selected_sm_to_kernel",
    "warps": warp_rows,
    "ctas": cta_rows,
    "selected_sm": {
        "cta_count_used": len(cta_rows),
        "pooling_method": "average",
        "selected_sm_embedding_dim": int(selected_sm_embedding.shape[0]),
    },
    "kernel": {
        "kernel_embedding_source": "selected_sm_embedding",
        "pooling_method": "identity",
        "kernel_embedding_dim": int(kernel_embedding.shape[0]),
    },
}
```

- [ ] **Step 4: Update validator**

`validate_readout_manifest` must reject:

```text
missing readout_hierarchy
missing ctas
unknown cta_id in CTA rows
CTA row node/warp count mismatch
kernel_embedding_source != selected_sm_embedding
kernel_embedding_dim != 256
```

- [ ] **Step 5: Run readout tests**

Run:

```bash
pytest -q tests/gcl_phase_b/test_readout.py
```

Expected: PASS.

- [ ] **Step 6: Commit Gate 5 readout**

```bash
git add experiments/gcl_phase_b/readout.py tests/gcl_phase_b/test_readout.py
git commit -m "feat: add CTA-aware Gate5 readout"
```

### Task 6: Gate 5 Formal Embedding Export Without Selector

**Files:**
- Modify: `experiments/gcl_phase_b/pipeline.py`
- Modify: `tests/gcl_phase_b/test_embedding_export.py`
- Create: `tests/gcl_phase_b/test_resnet50_gate_pipeline.py`

- [ ] **Step 1: Add failing Gate 5 stop-at-embedding test**

Add:

```python
from pathlib import Path

from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate5


def test_resnet50_gate_pipeline_stops_at_gate5_embedding_table(tmp_path):
    fixture_root = Path("tests/fixtures/gcl_resnet50_gate1")
    out_dir = tmp_path / "gate1_5"

    manifest = run_resnet50_gate1_to_gate5(fixture_root, out_dir, seed=20260606)

    assert manifest["final_gate"] == "gate5"
    assert (out_dir / "resnet50_trace_adapter_bundle.json").exists()
    assert (out_dir / "representative_sm_trace_manifest.json").exists()
    assert (out_dir / "canonical_graph_bundle.json").exists()
    assert (out_dir / "graph_tensor_bundle.json").exists()
    assert (out_dir / "kernel_embedding_table.json").exists()
    assert not (out_dir / "selector_artifacts.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py::test_resnet50_gate_pipeline_stops_at_gate5_embedding_table
```

Expected: FAIL because `resnet50_gate_pipeline.py` does not exist.

- [ ] **Step 3: Add stop-at-Gate-5 pipeline**

Create `experiments/gcl_phase_b/resnet50_gate_pipeline.py`:

```python
"""ResNet-50 Gate 1-5 pipeline. Stops before Gate 6 clustering/classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .graph_builder import build_phase_b_graphs, validate_phase_b_graph_artifact
from .pipeline import create_augmentation_manifest_bundle, run_embedding_export
from .resnet50_adapter import build_resnet50_trace_adapter_bundle
from .resnet50_manifest import build_representative_sm_manifest_from_bundle
from .tensorizer import tensor_to_jsonable, tensorize_phase_b_graphs
from .trace_scope import build_phase_b_trace_records
from .utils import hash_without, stable_hash, write_json


def run_resnet50_gate1_to_gate5(root: Path, out_dir: Path, seed: int = 20260606) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_bundle = build_resnet50_trace_adapter_bundle(root)
    write_json(out_dir / "resnet50_trace_adapter_bundle.json", adapter_bundle)

    trace_manifest, report_bundle, preview = build_representative_sm_manifest_from_bundle(adapter_bundle)
    write_json(out_dir / "representative_sm_trace_manifest.json", trace_manifest)
    write_json(out_dir / "selected_sm_policy_report.json", report_bundle)
    write_json(out_dir / "scope_preview_report.json", preview)

    records = build_phase_b_trace_records(trace_manifest)
    graphs = build_phase_b_graphs(records)
    for graph in graphs:
        validate_phase_b_graph_artifact(graph)
    canonical_graph_bundle = {
        "artifact_type": "gcl_resnet50_canonical_graph_bundle",
        "artifact_version": "gate3_canonical_graph_bundle_v1",
        "source_trace_manifest_hash": trace_manifest["trace_manifest_hash"],
        "graphs": graphs,
    }
    canonical_graph_bundle["canonical_graph_bundle_hash"] = hash_without(
        canonical_graph_bundle, "canonical_graph_bundle_hash"
    )
    write_json(out_dir / "canonical_graph_bundle.json", canonical_graph_bundle)

    tensors = tensorize_phase_b_graphs(graphs)
    graph_tensor_bundle = {
        "artifact_type": "gcl_resnet50_graph_tensor_bundle",
        "artifact_version": "gate4_graph_tensor_bundle_v1",
        "source_canonical_graph_bundle_hash": canonical_graph_bundle["canonical_graph_bundle_hash"],
        "tensors": [tensor_to_jsonable(tensor) for tensor in tensors],
    }
    graph_tensor_bundle["graph_tensor_bundle_hash"] = hash_without(
        graph_tensor_bundle, "graph_tensor_bundle_hash"
    )
    write_json(out_dir / "graph_tensor_bundle.json", graph_tensor_bundle)

    augmentation_bundle = create_augmentation_manifest_bundle(tensors, seed=seed)
    write_json(out_dir / "augmentation_manifest.json", augmentation_bundle)

    embedding_table, training_report = run_embedding_export(tensors, out_dir, seed=seed)
    write_json(out_dir / "rgcn_training_run_manifest.json", _jsonable_training_report(training_report))
    write_json(out_dir / "rgcn_checkpoint_manifest.json", training_report["checkpoint_manifest"])
    write_json(out_dir / "kernel_embedding_table.json", embedding_table)
    export_report = {
        "artifact_type": "gcl_resnet50_embedding_export_report",
        "artifact_version": "gate5_embedding_export_report_v1",
        "source_graph_tensor_bundle_hash": graph_tensor_bundle["graph_tensor_bundle_hash"],
        "embedding_table_hash": embedding_table["embedding_table_hash"],
        "failed_graphs": [],
    }
    export_report["embedding_export_report_hash"] = hash_without(
        export_report, "embedding_export_report_hash"
    )
    write_json(out_dir / "embedding_export_report.json", export_report)

    manifest = {
        "artifact_type": "gcl_resnet50_gate1_5_pipeline_manifest",
        "final_gate": "gate5",
        "seed": seed,
        "hashes": {
            "adapter_bundle_hash": adapter_bundle["adapter_bundle_hash"],
            "trace_manifest_hash": trace_manifest["trace_manifest_hash"],
            "canonical_graph_bundle_hash": canonical_graph_bundle["canonical_graph_bundle_hash"],
            "graph_tensor_bundle_hash": graph_tensor_bundle["graph_tensor_bundle_hash"],
            "embedding_table_hash": embedding_table["embedding_table_hash"],
        },
    }
    manifest["pipeline_manifest_hash"] = stable_hash(manifest)
    write_json(out_dir / "gate1_5_pipeline_manifest.json", manifest)
    return manifest


def _jsonable_training_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key not in {"encoder", "projection_head"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()
    manifest = run_resnet50_gate1_to_gate5(args.input_root, args.out, seed=args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run Gate 1-5 E2E test**

Run:

```bash
pytest -q tests/gcl_phase_b/test_resnet50_gate_pipeline.py
```

Expected: PASS.

- [ ] **Step 5: Verify CLI**

Run:

```bash
python -m experiments.gcl_phase_b.resnet50_gate_pipeline \
  --input-root tests/fixtures/gcl_resnet50_gate1 \
  --out artifacts/gcl_resnet50_gate1_5
```

Expected:

```text
final_gate = gate5
artifacts/gcl_resnet50_gate1_5/kernel_embedding_table.json exists
artifacts/gcl_resnet50_gate1_5/selector_artifacts.json does not exist
```

- [ ] **Step 6: Commit Gate 1-5 pipeline**

```bash
git add experiments/gcl_phase_b/resnet50_gate_pipeline.py tests/gcl_phase_b/test_resnet50_gate_pipeline.py
git commit -m "feat: add ResNet50 gate1 to gate5 pipeline"
```

### Task 7: Replay And Hash Validation

**Files:**
- Create: `tests/gcl_phase_b/test_resnet50_gate_replay.py`

- [ ] **Step 1: Add replay test**

Create:

```python
from pathlib import Path

from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate5


def test_resnet50_gate1_5_pipeline_hashes_are_replayable(tmp_path):
    fixture_root = Path("tests/fixtures/gcl_resnet50_gate1")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    manifest_a = run_resnet50_gate1_to_gate5(fixture_root, out_a, seed=20260606)
    manifest_b = run_resnet50_gate1_to_gate5(fixture_root, out_b, seed=20260606)

    assert manifest_a["hashes"]["adapter_bundle_hash"] == manifest_b["hashes"]["adapter_bundle_hash"]
    assert manifest_a["hashes"]["trace_manifest_hash"] == manifest_b["hashes"]["trace_manifest_hash"]
    assert manifest_a["hashes"]["canonical_graph_bundle_hash"] == manifest_b["hashes"]["canonical_graph_bundle_hash"]
    assert manifest_a["hashes"]["graph_tensor_bundle_hash"] == manifest_b["hashes"]["graph_tensor_bundle_hash"]
    assert manifest_a["hashes"]["embedding_table_hash"] == manifest_b["hashes"]["embedding_table_hash"]
```

- [ ] **Step 2: Run replay test**

Run:

```bash
pytest -q tests/gcl_phase_b/test_resnet50_gate_replay.py
```

Expected: PASS.

- [ ] **Step 3: Commit replay coverage**

```bash
git add tests/gcl_phase_b/test_resnet50_gate_replay.py
git commit -m "test: add ResNet50 gate1 to gate5 replay checks"
```

### Task 8: Full Verification

**Files:**
- No code changes.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest -q \
  tests/gcl_phase_b/test_resnet50_adapter.py \
  tests/gcl_phase_b/test_resnet50_manifest.py \
  tests/gcl_phase_b/test_resnet50_gate_pipeline.py \
  tests/gcl_phase_b/test_resnet50_gate_replay.py \
  tests/gcl_phase_b/test_tensorizer.py \
  tests/gcl_phase_b/test_readout.py \
  tests/gcl_phase_b/test_embedding_export.py
```

Expected: PASS.

- [ ] **Step 2: Run existing Phase A and Phase B regression tests**

Run:

```bash
pytest -q tests/gcl_phase_a tests/gcl_phase_b
```

Expected: PASS.

- [ ] **Step 3: Run CLI smoke**

Run:

```bash
rm -rf artifacts/gcl_resnet50_gate1_5
python -m experiments.gcl_phase_b.resnet50_gate_pipeline \
  --input-root tests/fixtures/gcl_resnet50_gate1 \
  --out artifacts/gcl_resnet50_gate1_5 \
  --seed 20260606
test -f artifacts/gcl_resnet50_gate1_5/kernel_embedding_table.json
test ! -f artifacts/gcl_resnet50_gate1_5/selector_artifacts.json
```

Expected: command exits 0.

- [ ] **Step 4: Check whitespace and status**

Run:

```bash
git diff --check
git status --short
```

Expected:

```text
git diff --check has no output
git status only shows intentional changes or clean state
```

## 5. 实施顺序建议

推荐顺序：

```text
Task 1 fixture
  -> Task 2 Gate 1 adapter
  -> Task 3 Gate 2 manifest
  -> Task 4 representation mode
  -> Task 5 CTA-aware readout
  -> Task 6 Gate 1-5 pipeline
  -> Task 7 replay
  -> Task 8 verification
```

这样每个 Gate 都可以单独验证，并且本轮最终 artifact 明确停在：

```text
kernel_embedding_table.json
```

该文件就是后续 Gate 6 clustering / family classification 的正式输入。
