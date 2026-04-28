"""PKA feature extractor for L1.

Extracts the 12-dimensional PKA feature vector from heterogeneous source types.
Produces PkaFeatureTable (measured invocations) and PkaAcquisitionGap.
Enforces exactly-one-outcome per P0 invocation.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifacts" / "a_line" / "l1" / "kernel_validation_manifest_l1.json"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "a_line" / "l1"

PKA_FEATURES: dict[str, dict[str, str]] = {
    "coalesced_global_loads": {"canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "category": "coalesced_memory"},
    "coalesced_global_stores": {"canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "category": "coalesced_memory"},
    "coalesced_local_loads": {"canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", "category": "coalesced_memory"},
    "thread_global_loads": {"canonical_metric": "smsp__inst_executed_op_global_ld.sum", "category": "thread_instruction"},
    "thread_global_stores": {"canonical_metric": "smsp__inst_executed_op_global_st.sum", "category": "thread_instruction"},
    "thread_local_loads": {"canonical_metric": "smsp__inst_executed_op_local_ld.sum", "category": "thread_instruction"},
    "thread_shared_loads": {"canonical_metric": "smsp__inst_executed_op_shared_ld.sum", "category": "thread_instruction"},
    "thread_shared_stores": {"canonical_metric": "smsp__inst_executed_op_shared_st.sum", "category": "thread_instruction"},
    "thread_global_atomics": {"canonical_metric": "smsp__sass_inst_executed_op_global_atom.sum", "category": "thread_instruction"},
    "num_instructions": {"canonical_metric": "smsp__inst_executed.sum", "category": "scale_signal"},
    "divergence_efficiency": {"canonical_metric": "smsp__thread_inst_executed_per_inst_executed.ratio", "category": "efficiency_signal"},
    "num_thread_blocks": {"canonical_metric": "launch_grid_size", "category": "scale_signal"},
}


def _extract_pka_features(metric_map: dict[str, Any], source_path: str) -> dict[str, Any]:
    features = {}
    for pka_name, spec in PKA_FEATURES.items():
        canonical = spec["canonical_metric"]
        if canonical in metric_map:
            features[pka_name] = {
                "value": float(metric_map[canonical]), "status": "measured",
                "canonical_metric": canonical, "actual_source_metric": canonical,
                "source_artifact_path": source_path,
            }
        else:
            features[pka_name] = {
                "value": None, "status": "missing",
                "canonical_metric": canonical, "actual_source_metric": None,
                "source_artifact_path": source_path,
                "missing_reason": _missing_reason(canonical),
            }
    return features


def _missing_reason(canonical: str) -> str:
    if canonical == "launch_grid_size":
        return "launch_metadata_absent"
    return "canonical_metric_absent"


def _is_fully_measured(features: dict[str, Any]) -> bool:
    return all(f["status"] == "measured" for f in features.values())


def _collect_missing(features: dict[str, Any]) -> list[str]:
    return [name for name, f in features.items() if f["status"] != "measured"]


def _match_kernel_name(mangled: str, target: str) -> bool:
    """Match a demangled kernel_or_case against a mangled C++ kernel name."""
    target_lower = target.lower()
    mangled_lower = mangled.lower()
    if target_lower in mangled_lower:
        return True
    # Extract base name from mangled: _Z\d+base_name...
    import re
    m = re.match(r'_Z\d+(\w+)', mangled)
    if m:
        base = m.group(1).lower()
        if target_lower in base:
            return True
    return False


def _adapt_microbench(entry: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    """Adapt microbench JSON. One kernel per file, so we process all kernels found."""
    kernel_or_case = entry["kernel_or_case"]
    try:
        data = json.loads(source_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{entry['id']}: source file is not valid JSON: {source_path}") from exc

    eei = data.get("enhanced_execution_info")
    if not eei or "kernels" not in eei:
        raise ValueError(f"{entry['id']}: source file missing enhanced_execution_info.kernels: {source_path}")

    results = []
    occurrence: dict[str, int] = {}
    for kernel in eei["kernels"]:
        kernel_name = kernel.get("kernel_name", "")
        occurrence.setdefault(kernel_or_case, 0)
        occurrence[kernel_or_case] += 1
        invocation_id = f"{kernel_or_case}#{occurrence[kernel_or_case]}"

        metric_map: dict[str, Any] = {}
        for k, v in kernel.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metric_map[k] = v
        tbf = data.get("threadblock_features", {})
        for tb_data in tbf.values():
            if isinstance(tb_data, dict):
                metric_map.update({k: v for k, v in tb_data.items() if isinstance(v, (int, float))})

        features = _extract_pka_features(metric_map, str(source_path))
        results.append(_make_record(entry, invocation_id, kernel_name, features))

    if not results:
        raise ValueError(f"{entry['id']}: microbench adapter produced zero outcomes for {source_path}")
    return results


def _adapt_rodinia(entry: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    """Adapt Rodinia trace/NCU artifact."""
    kernel_or_case = entry["kernel_or_case"]

    if source_path.suffix == ".md":
        features = {}
        for pka_name, spec in PKA_FEATURES.items():
            features[pka_name] = {
                "value": None, "status": "missing",
                "canonical_metric": spec["canonical_metric"],
                "actual_source_metric": None,
                "source_artifact_path": str(source_path),
                "missing_reason": "source_is_prescription_document",
            }
        return [_make_record(entry, f"{kernel_or_case}#1", kernel_or_case, features)]

    if source_path.suffix == ".csv":
        return _adapt_ncu_csv(entry, source_path)

    try:
        data = json.loads(source_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{entry['id']}: source file is not valid JSON: {source_path}") from exc

    eei = data.get("enhanced_execution_info")
    if not eei or "kernels" not in eei:
        raise ValueError(f"{entry['id']}: source file missing enhanced_execution_info.kernels: {source_path}")

    results = []
    occurrence: dict[str, int] = {}
    for kernel in eei["kernels"]:
        kernel_name = kernel.get("kernel_name", "")
        occurrence.setdefault(kernel_or_case, 0)
        occurrence[kernel_or_case] += 1
        invocation_id = f"{kernel_or_case}#{occurrence[kernel_or_case]}"

        metric_map: dict[str, Any] = {}
        for k, v in kernel.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metric_map[k] = v
        tbf = data.get("threadblock_features", {})
        for tb_data in tbf.values():
            if isinstance(tb_data, dict):
                metric_map.update({k: v for k, v in tb_data.items() if isinstance(v, (int, float))})

        features = _extract_pka_features(metric_map, str(source_path))
        results.append(_make_record(entry, invocation_id, kernel_name, features))

    if not results:
        raise ValueError(f"{entry['id']}: rodinia adapter produced zero outcomes for {source_path}")
    return results


def _adapt_ncu_csv(entry: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    """Parse Rodinia NCU CSV with metric_name, metric_value columns."""
    kernel_or_case = entry["kernel_or_case"]
    metric_map: dict[str, Any] = {}
    with open(source_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("metric_name", "").strip()
            val = row.get("metric_value", "").strip()
            if name and val:
                try:
                    metric_map[name] = float(val)
                except ValueError:
                    pass
    # Also check for launch grid from CSV metadata
    features = _extract_pka_features(metric_map, str(source_path))
    return [_make_record(entry, f"{kernel_or_case}#1", kernel_or_case, features)]


def _adapt_mini_transformer(entry: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    """Adapt mini-transformer full JSON."""
    kernel_or_case = entry["kernel_or_case"]
    try:
        data = json.loads(source_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{entry['id']}: source file is not valid JSON: {source_path}") from exc

    per_kernel = data.get("per_kernel")
    if not per_kernel:
        raise ValueError(f"{entry['id']}: source file missing per_kernel: {source_path}")

    results = []
    occurrence: dict[str, int] = {}
    # Sort by source key to get stable trace_order
    for source_key in sorted(per_kernel.keys()):
        item = per_kernel[source_key]
        kernel_name = item.get("kernel_name", "")
        if not _match_kernel_name(kernel_name, kernel_or_case):
            continue

        occurrence.setdefault(kernel_or_case, 0)
        occurrence[kernel_or_case] += 1
        invocation_id = f"{kernel_or_case}#{occurrence[kernel_or_case]}"

        metric_map: dict[str, Any] = {}
        for section in ("hardware_metrics", "dynamic_stats", "compression_features"):
            section_data = item.get(section, {})
            if isinstance(section_data, dict):
                for k, v in section_data.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        metric_map[k] = v
                    elif isinstance(v, dict) and "mean" in v:
                        metric_map[k] = v["mean"]

        features = _extract_pka_features(metric_map, str(source_path))
        results.append(_make_record(entry, invocation_id, kernel_name, features))

    if not results:
        raise ValueError(f"{entry['id']}: mini-transformer adapter produced zero outcomes for kernel_or_case={kernel_or_case} in {source_path}")
    return results


ADAPTERS = {
    "local_microbench": _adapt_microbench,
    "local_benchmark_result": _adapt_rodinia,
    "local_ai_workload": _adapt_mini_transformer,
}


def _make_record(entry: dict[str, Any], invocation_id: str, kernel_name: str, features: dict[str, Any]) -> dict[str, Any]:
    rec = {
        "manifest_id": entry["id"],
        "priority": entry["priority"],
        "source_type": entry["source_type"],
        "kernel_or_case": entry["kernel_or_case"],
        "kernel_invocation_id": invocation_id,
        "kernel_name": kernel_name,
        "source_path": features.get(list(features.keys())[0], {}).get("source_artifact_path", ""),
        "features": features,
    }
    if _is_fully_measured(features):
        rec["outcome"] = "measured"
    else:
        rec["outcome"] = "acquisition_gap"
        rec["missing_metrics"] = _collect_missing(features)
    return rec


def _validate_outcomes(records: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    """Post-adapter validation: exactly one outcome per P0 invocation, no duplicates."""
    errors = []
    p0_entries = {e["id"]: e for e in manifest["entries"] if e["priority"] == "P0"}
    p0_outcomes: dict[str, list[dict]] = {}

    for rec in records:
        mid = rec.get("manifest_id", "")
        if mid in p0_entries:
            p0_outcomes.setdefault(mid, []).append(rec)

    # Check each P0 entry has at least one outcome
    for mid in p0_entries:
        if mid not in p0_outcomes:
            errors.append(f"{mid}: P0 manifest entry has zero outcomes")
        else:
            outcomes = p0_outcomes[mid]
            # Check for duplicate invocation_ids
            seen = set()
            for rec in outcomes:
                kid = rec.get("kernel_invocation_id", "")
                if kid in seen:
                    errors.append(f"{mid}: duplicate kernel_invocation_id: {kid}")
                seen.add(kid)
                # Check for ambiguous outcome
                if rec.get("outcome") not in ("measured", "acquisition_gap"):
                    errors.append(f"{mid}: invalid outcome type: {rec.get('outcome')}")

    return errors


def _build_availability_matrix(records: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {}
    for rec in records:
        kid = rec.get("kernel_invocation_id", "unknown")
        features = rec.get("features", {})
        matrix[kid] = {}
        for pka_name in PKA_FEATURES:
            f = features.get(pka_name, {})
            matrix[kid][pka_name] = f.get("status", "unknown")
    return {
        "pka_features": list(PKA_FEATURES.keys()),
        "availability": matrix,
        "total_invocations": len(matrix),
        "fully_measured_count": sum(1 for row in matrix.values() if all(v == "measured" for v in row.values())),
    }


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    entries = manifest["entries"]

    all_records = []
    adapter_errors = []

    for entry in entries:
        source_type = entry["source_type"]
        source_path = REPO_ROOT / entry.get("local_input_path", "")
        adapter = ADAPTERS.get(source_type)
        if adapter is None:
            print(f"Warning: no adapter for source_type={source_type}, entry={entry['id']}")
            continue
        try:
            records = adapter(entry, source_path)
            all_records.extend(records)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            adapter_errors.append(str(exc))

    if adapter_errors:
        print("Adapter errors:")
        for err in adapter_errors:
            print(f"  - {err}")
        return 1

    outcome_errors = _validate_outcomes(all_records, manifest)
    if outcome_errors:
        print("Outcome validation failed:")
        for err in outcome_errors:
            print(f"  - {err}")
        return 1

    measured = [r for r in all_records if r["outcome"] == "measured"]
    gaps = [r for r in all_records if r["outcome"] == "acquisition_gap"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "pka_feature_table_l1.json").write_text(json.dumps(measured, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT_DIR / "pka_acquisition_gap_l1.json").write_text(json.dumps(gaps, indent=2, ensure_ascii=False) + "\n")

    matrix = _build_availability_matrix(all_records)
    (OUTPUT_DIR / "pka_metric_availability_matrix_l1.json").write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n")

    p0_gaps = [r for r in gaps if r["priority"] == "P0"]

    print(f"Feature extraction complete:")
    print(f"  Measured invocations: {len(measured)}")
    print(f"  Acquisition gap invocations: {len(gaps)}")
    print(f"  P0 gap invocations: {len(p0_gaps)}")
    print(f"  Metric availability: {matrix['fully_measured_count']}/{matrix['total_invocations']} fully measured")

    if p0_gaps:
        print("\nBLOCKED: P0 acquisition gaps detected.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
