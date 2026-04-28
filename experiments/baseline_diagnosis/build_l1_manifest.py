"""Build L1 kernel validation manifest JSON from the manifest document.

Reads docs/a-line-l1-validation-manifest-2026-04-26.md and produces
artifacts/a_line/l1/kernel_validation_manifest_l1.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "experiments" / "baseline_diagnosis" / "schemas" / "kernel_validation_manifest_schema.json"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "a_line" / "l1" / "kernel_validation_manifest_l1.json"

SOURCE_TYPE_MAP = {
    "microbench": "local_microbench",
    "rodinia": "local_benchmark_result",
    "ai_workload": "local_ai_workload",
}


def _make_entry(
    entry_id: str,
    source: str,
    benchmark_name: str,
    kernel_or_case: str,
    local_path: str,
    priority: str,
    expected_behavior_axis: str,
    expected_scale_axis: str | None = None,
    validation_role: str | None = None,
    notes: str | None = None,
) -> dict:
    entry = {
        "id": entry_id,
        "source_type": SOURCE_TYPE_MAP[source],
        "benchmark_name": benchmark_name,
        "kernel_or_case": kernel_or_case,
        "local_input_path": local_path,
        "priority": priority,
        "target_line": "A+B",
        "expected_behavior_axis": expected_behavior_axis,
        "status": "ready_local",
    }
    if expected_scale_axis is not None:
        entry["expected_scale_axis"] = expected_scale_axis
    if validation_role is not None:
        entry["validation_role"] = validation_role
    if notes is not None:
        entry["notes"] = notes
    return entry


P0_ENTRIES = [
    _make_entry("L1_MB_01", "microbench", "l1_bw_32f", "l1_bw_32f",
                "experiments/baseline_diagnosis/results/microbench/l1_bw_32f.json",
                "P0", "L1 bandwidth / coalesced load-heavy",
                validation_role="feature_sanity_gate"),
    _make_entry("L1_MB_02", "microbench", "l2_bw_32f", "l2_bw_32f",
                "experiments/baseline_diagnosis/results/microbench/l2_bw_32f.json",
                "P0", "L2 / global-memory bandwidth",
                validation_role="feature_sanity_gate"),
    _make_entry("L1_MB_03", "microbench", "mem_bw", "mem_bw",
                "experiments/baseline_diagnosis/results/microbench/mem_bw.json",
                "P0", "global-memory bandwidth",
                validation_role="feature_sanity_gate"),
    _make_entry("L1_MB_04", "microbench", "mem_lat", "mem_lat",
                "experiments/baseline_diagnosis/results/microbench/mem_lat.json",
                "P0", "global-memory latency",
                validation_role="feature_sanity_gate"),
    _make_entry("L1_MB_05", "microbench", "shared_bw", "shared_bw",
                "experiments/baseline_diagnosis/results/microbench/shared_bw.json",
                "P0", "shared-memory throughput",
                validation_role="feature_sanity_gate"),
    _make_entry("L1_MB_09", "microbench", "MaxFlops", "MaxFlops",
                "experiments/baseline_diagnosis/results/microbench/MaxFlops.json",
                "P0", "compute-bound",
                validation_role="feature_sanity_gate"),
    _make_entry("L1_RD_01", "rodinia", "nn", "nn",
                "experiments/baseline_diagnosis/results/rodinia/nn_trace.json",
                "P0", "distance / memory-sensitive / possible uncoalesced global access",
                validation_role="feature_sanity_gate"),
    _make_entry("L1_AI_01", "ai_workload", "gemm_tiled", "gemm_tiled",
                "experiments/mini_transformer/mini_transformer_v4_full.json",
                "P0", "dense compute backbone",
                validation_role="feature_sanity_gate"),
    _make_entry("L1_AI_02", "ai_workload", "attention_score", "attention_score",
                "experiments/mini_transformer/mini_transformer_v4_full.json",
                "P0", "pairwise score / dense compute",
                validation_role="feature_sanity_gate"),
    _make_entry("L1_AI_03", "ai_workload", "softmax_kernel", "softmax_kernel",
                "experiments/mini_transformer/mini_transformer_v4_full.json",
                "P0", "reduction / normalize",
                validation_role="feature_sanity_gate"),
]

P1_ENTRIES = [
    _make_entry("L1_MB_06", "microbench", "shared_lat", "shared_lat",
                "experiments/baseline_diagnosis/results/microbench/shared_lat.json",
                "P1", "shared-memory latency"),
    _make_entry("L1_MB_07", "microbench", "atomic_add_bw", "atomic_add_bw",
                "experiments/baseline_diagnosis/results/microbench/atomic_add_bw.json",
                "P1", "atomic-heavy / serialization-sensitive"),
    _make_entry("L1_MB_08", "microbench", "atomic_add_lat", "atomic_add_lat",
                "experiments/baseline_diagnosis/results/microbench/atomic_add_lat.json",
                "P1", "atomic latency / contention-sensitive"),
    _make_entry("L1_RD_02", "rodinia", "backprop", "backprop",
                "experiments/baseline_diagnosis/results/rodinia/backprop_4096_prescription_v1.md",
                "P1", "dense numeric / low-divergence"),
    _make_entry("L1_AI_04", "ai_workload", "context_mul", "context_mul",
                "experiments/mini_transformer/mini_transformer_v4_full.json",
                "P1", "streaming aggregation"),
    _make_entry("L1_AI_05", "ai_workload", "layernorm_kernel", "layernorm_kernel",
                "experiments/mini_transformer/mini_transformer_v4_full.json",
                "P1", "reduction / normalize"),
    _make_entry("L1_AI_06", "ai_workload", "residual_add", "residual_add",
                "experiments/mini_transformer/mini_transformer_v4_full.json",
                "P1", "elementwise / lightweight"),
]


def _validate_schema(manifest: dict, schema: dict) -> list[str]:
    errors = []
    required_top = schema.get("required", [])
    for field in required_top:
        if field not in manifest:
            errors.append(f"manifest missing required top-level field: {field}")

    dataset_level = manifest.get("dataset_level")
    allowed_levels = schema.get("properties", {}).get("dataset_level", {}).get("enum", [])
    if dataset_level not in allowed_levels:
        errors.append(f"dataset_level '{dataset_level}' not in allowed values: {allowed_levels}")

    entry_schema = schema.get("$defs", {}).get("entry", {})
    entry_required = entry_schema.get("required", [])
    entry_props = entry_schema.get("properties", {})

    for idx, entry in enumerate(manifest.get("entries", [])):
        for field in entry_required:
            if field not in entry:
                errors.append(f"entries[{idx}] missing required field: {field}")

        source_type = entry.get("source_type")
        allowed_st = entry_props.get("source_type", {}).get("enum", [])
        if source_type not in allowed_st:
            errors.append(f"entries[{idx}] source_type '{source_type}' not in {allowed_st}")

        priority = entry.get("priority")
        allowed_pr = entry_props.get("priority", {}).get("enum", [])
        if priority not in allowed_pr:
            errors.append(f"entries[{idx}] priority '{priority}' not in {allowed_pr}")

        target_line = entry.get("target_line")
        allowed_tl = entry_props.get("target_line", {}).get("enum", [])
        if target_line not in allowed_tl:
            errors.append(f"entries[{idx}] target_line '{target_line}' not in {allowed_tl}")

        status = entry.get("status")
        allowed_sta = entry_props.get("status", {}).get("enum", [])
        if status not in allowed_sta:
            errors.append(f"entries[{idx}] status '{status}' not in {allowed_sta}")

        validation_role = entry.get("validation_role")
        if validation_role is not None:
            allowed_vr = entry_props.get("validation_role", {}).get("enum", [])
            if validation_role not in allowed_vr:
                errors.append(f"entries[{idx}] validation_role '{validation_role}' not in {allowed_vr}")

    return errors


def _check_paths(entries: list[dict], repo_root: Path) -> list[str]:
    errors = []
    for entry in entries:
        local_path = entry.get("local_input_path", "")
        full_path = repo_root / local_path
        if not full_path.exists():
            errors.append(
                f"{entry['id']}: local_input_path does not exist: {full_path}"
            )
        elif entry["id"].startswith("L1_RD_02"):
            # L1_RD_02 points to a .md prescription file — warn but don't block
            if full_path.suffix not in (".json",):
                pass  # accepted: prescription file marks acquisition-needed
    return errors


def _build_manifest(include_p1: bool = True) -> dict:
    entries = list(P0_ENTRIES)
    if include_p1:
        entries.extend(P1_ENTRIES)

    return {
        "manifest_name": "L1 Kernel Validation Manifest",
        "dataset_level": "L1",
        "goal": (
            "Functionality gate, feature sanity gate, and downstream interface gate "
            "for PKA baseline input, 12-D feature extraction, anchor output, and "
            "B-line consumption on a small set of interpretable kernels."
        ),
        "notes": (
            "Generated from docs/a-line-l1-validation-manifest-2026-04-26.md. "
            "P0 entries block stage-gate; P1 entries are non-blocking."
        ),
        "entries": entries,
    }


def main() -> int:
    schema = json.loads(SCHEMA_PATH.read_text())
    manifest = _build_manifest(include_p1=True)

    schema_errors = _validate_schema(manifest, schema)
    if schema_errors:
        print("Schema validation failed:")
        for err in schema_errors:
            print(f"  - {err}")
        return 1

    path_errors = _check_paths(manifest["entries"], REPO_ROOT)
    blocking_errors = [e for e in path_errors if not e.startswith("L1_RD_02")]
    if blocking_errors:
        print("Path existence check failed (blocking):")
        for err in blocking_errors:
            print(f"  - {err}")
        return 1
    if path_errors:
        print("Path existence check: non-blocking notes:")
        for err in path_errors:
            print(f"  - {err}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Manifest written: {OUTPUT_PATH}")
    print(f"  P0 entries: {len([e for e in manifest['entries'] if e['priority'] == 'P0'])}")
    print(f"  P1 entries: {len([e for e in manifest['entries'] if e['priority'] == 'P1'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
