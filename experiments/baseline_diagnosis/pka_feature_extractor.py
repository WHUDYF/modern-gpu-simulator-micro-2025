"""PKA feature extractor for L1.

Extracts the 12-dimensional PKA feature vector from heterogeneous source types
(microbench JSON, Rodinia trace/NCU artifacts, mini-transformer full JSON).

Produces PkaFeatureTable (measured invocations) and PkaAcquisitionGap
(incomplete invocations) following the measured-only policy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifacts" / "a_line" / "l1" / "kernel_validation_manifest_l1.json"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "a_line" / "l1"

PKA_FEATURES: dict[str, dict[str, str]] = {
    "coalesced_global_loads": {
        "canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "category": "coalesced_memory",
    },
    "coalesced_global_stores": {
        "canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "category": "coalesced_memory",
    },
    "coalesced_local_loads": {
        "canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
        "category": "coalesced_memory",
    },
    "thread_global_loads": {
        "canonical_metric": "smsp__inst_executed_op_global_ld.sum",
        "category": "thread_instruction",
    },
    "thread_global_stores": {
        "canonical_metric": "smsp__inst_executed_op_global_st.sum",
        "category": "thread_instruction",
    },
    "thread_local_loads": {
        "canonical_metric": "smsp__inst_executed_op_local_ld.sum",
        "category": "thread_instruction",
    },
    "thread_shared_loads": {
        "canonical_metric": "smsp__inst_executed_op_shared_ld.sum",
        "category": "thread_instruction",
    },
    "thread_shared_stores": {
        "canonical_metric": "smsp__inst_executed_op_shared_st.sum",
        "category": "thread_instruction",
    },
    "thread_global_atomics": {
        "canonical_metric": "smsp__sass_inst_executed_op_global_atom.sum",
        "category": "thread_instruction",
    },
    "num_instructions": {
        "canonical_metric": "smsp__inst_executed.sum",
        "category": "scale_signal",
    },
    "divergence_efficiency": {
        "canonical_metric": "smsp__thread_inst_executed_per_inst_executed.ratio",
        "category": "efficiency_signal",
    },
    "num_thread_blocks": {
        "canonical_metric": "launch_grid_size",
        "category": "scale_signal",
    },
}


def _extract_pka_features(
    feature_map: dict[str, Any],
    source_artifact_path: str,
) -> dict[str, Any]:
    """Extract PKA features from a flat feature name -> value map.

    For each of the 12 PKA features, checks if the canonical metric name
    is present as a key in feature_map. Returns per-feature {value, status,
    canonical_metric, actual_source_metric, source_artifact_path}.
    """
    features: dict[str, Any] = {}
    for pka_name, spec in PKA_FEATURES.items():
        canonical = spec["canonical_metric"]
        if canonical in feature_map:
            features[pka_name] = {
                "value": float(feature_map[canonical]),
                "status": "measured",
                "canonical_metric": canonical,
                "actual_source_metric": canonical,
                "source_artifact_path": source_artifact_path,
            }
        else:
            features[pka_name] = {
                "value": None,
                "status": "missing",
                "canonical_metric": canonical,
                "actual_source_metric": None,
                "source_artifact_path": source_artifact_path,
            }
    return features


def _is_fully_measured(features: dict[str, Any]) -> bool:
    return all(f["status"] == "measured" for f in features.values())


def _collect_missing(features: dict[str, Any]) -> list[str]:
    return [name for name, f in features.items() if f["status"] != "measured"]


def _adapt_microbench(source_path: Path, kernel_or_case: str) -> list[dict[str, Any]]:
    """Adapt microbench JSON to PKA feature records.

    Each microbench file contains exactly one kernel invocation. The kernel
    name in the file is C++ mangled (e.g., _Z5l1_bwPjS_PfS0____0), so we
    do NOT filter by kernel_or_case — the manifest entry's local_input_path
    already identifies the correct file. We process all kernels found.
    """
    data = json.loads(source_path.read_text())
    eei = data.get("enhanced_execution_info", {})
    kernels = eei.get("kernels", [])

    results: list[dict[str, Any]] = []
    occurrence: dict[str, int] = {}

    for idx, kernel in enumerate(kernels):
        kernel_name = kernel.get("kernel_name", "")

        occurrence.setdefault(kernel_or_case, 0)
        occurrence[kernel_or_case] += 1
        invocation_id = f"{kernel_or_case}#{occurrence[kernel_or_case]}"

        # Collect all available numeric data into a flat metric map
        metric_map: dict[str, Any] = {}
        # enhanced_execution_info kernel fields that are numeric
        for k, v in kernel.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metric_map[k] = v
        # threadblock_features per-TB data
        tbf = data.get("threadblock_features", {})
        for tb_key, tb_data in tbf.items():
            if isinstance(tb_data, dict):
                metric_map.update(
                    {k: v for k, v in tb_data.items() if isinstance(v, (int, float))}
                )

        features = _extract_pka_features(metric_map, str(source_path))
        all_measured = _is_fully_measured(features)

        if all_measured:
            results.append({
                "record_id": f"L1_MB_{kernel_or_case}",
                "kernel_invocation_id": invocation_id,
                "kernel_name": kernel_name,
                "feature_mode": "pka_l1_measured_only",
                "features": features,
                "metadata": {
                    "source_path": str(source_path),
                    "source_type": "local_microbench",
                },
            })
        else:
            results.append({
                "record_id": f"L1_MB_{kernel_or_case}",
                "kernel_invocation_id": invocation_id,
                "kernel_name": kernel_name,
                "outcome": "acquisition_gap",
                "missing_metrics": _collect_missing(features),
                "feature_details": features,
                "source_path": str(source_path),
            })

    return results


def _adapt_rodinia(source_path: Path, kernel_or_case: str) -> list[dict[str, Any]]:
    """Adapt Rodinia trace/NCU artifact to PKA feature records.

    Rodinia trace JSON has enhanced_execution_info + threadblock_features.
    Kernel names are C++ mangled (e.g., _Z6euclidPcffPfiii___0 for nn).
    We process all kernels in the file without strict name filtering since
    each trace file corresponds to one benchmark.
    """
    if source_path.suffix == ".md":
        return [{
            "record_id": f"L1_RD_{kernel_or_case}",
            "kernel_invocation_id": f"{kernel_or_case}#1",
            "kernel_name": kernel_or_case,
            "outcome": "acquisition_gap",
            "missing_metrics": list(PKA_FEATURES.keys()),
            "feature_details": {},
            "source_path": str(source_path),
            "gap_reason": "source is a prescription document, not measurement data",
        }]

    data = json.loads(source_path.read_text())
    eei = data.get("enhanced_execution_info", {})
    kernels = eei.get("kernels", [])

    results: list[dict[str, Any]] = []
    occurrence: dict[str, int] = {}

    for kernel in kernels:
        kernel_name = kernel.get("kernel_name", "")

        occurrence.setdefault(kernel_or_case, 0)
        occurrence[kernel_or_case] += 1
        invocation_id = f"{kernel_or_case}#{occurrence[kernel_or_case]}"

        # Collect all available data
        metric_map: dict[str, Any] = {}
        for k, v in kernel.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metric_map[k] = v
        tbf = data.get("threadblock_features", {})
        for tb_key, tb_data in tbf.items():
            if isinstance(tb_data, dict):
                metric_map.update(
                    {k: v for k, v in tb_data.items() if isinstance(v, (int, float))}
                )

        features = _extract_pka_features(metric_map, str(source_path))
        all_measured = _is_fully_measured(features)

        if all_measured:
            results.append({
                "record_id": f"L1_RD_{kernel_or_case}",
                "kernel_invocation_id": invocation_id,
                "kernel_name": kernel_name,
                "feature_mode": "pka_l1_measured_only",
                "features": features,
                "metadata": {"source_path": str(source_path), "source_type": "local_benchmark_result"},
            })
        else:
            results.append({
                "record_id": f"L1_RD_{kernel_or_case}",
                "kernel_invocation_id": invocation_id,
                "kernel_name": kernel_name,
                "outcome": "acquisition_gap",
                "missing_metrics": _collect_missing(features),
                "feature_details": features,
                "source_path": str(source_path),
            })

    return results


def _adapt_mini_transformer(source_path: Path, kernel_or_case: str) -> list[dict[str, Any]]:
    """Adapt mini-transformer full JSON to PKA feature records.

    Mini-transformer JSON has per_kernel format with hardware_metrics
    and dynamic_stats. The hardware_metrics use a different vocabulary
    (achieved_occupancy_pct, compute_throughput_pct, etc.) than the
    canonical 12 PKA Nsight metric names.
    """
    data = json.loads(source_path.read_text())
    per_kernel = data.get("per_kernel", {})

    results: list[dict[str, Any]] = []
    occurrence: dict[str, int] = {}

    for source_key, item in per_kernel.items():
        kernel_name = item.get("kernel_name", "")
        if kernel_or_case not in kernel_name:
            continue

        occurrence.setdefault(kernel_or_case, 0)
        occurrence[kernel_or_case] += 1
        invocation_id = f"{kernel_or_case}#{occurrence[kernel_or_case]}"

        # Merge all available numeric fields into a flat metric map
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

        # Special handling for num_thread_blocks: check dynamic_stats.num_blocks
        # but only accept if actual launch_grid_size metric exists
        if features["num_thread_blocks"]["status"] != "measured":
            dyn_blocks = (
                item.get("dynamic_stats", {}).get("num_blocks")
            )
            if dyn_blocks is not None and isinstance(dyn_blocks, (int, float)):
                # num_blocks != launch_grid_size in general;
                # marking as missing to follow the strict contract
                pass

        all_measured = _is_fully_measured(features)

        if all_measured:
            results.append({
                "record_id": f"L1_AI_{kernel_or_case}",
                "kernel_invocation_id": invocation_id,
                "kernel_name": kernel_name,
                "feature_mode": "pka_l1_measured_only",
                "features": features,
                "metadata": {"source_path": str(source_path), "source_type": "local_ai_workload"},
            })
        else:
            results.append({
                "record_id": f"L1_AI_{kernel_or_case}",
                "kernel_invocation_id": invocation_id,
                "kernel_name": kernel_name,
                "outcome": "acquisition_gap",
                "missing_metrics": _collect_missing(features),
                "feature_details": features,
                "source_path": str(source_path),
            })

    return results


ADAPTERS = {
    "local_microbench": _adapt_microbench,
    "local_benchmark_result": _adapt_rodinia,
    "local_ai_workload": _adapt_mini_transformer,
}


def _build_metric_availability_matrix(
    all_records: list[dict[str, Any]],
    manifest_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a 10x12 availability matrix for P0 entries."""
    p0_ids = [e["id"] for e in manifest_entries if e["priority"] == "P0"]
    matrix: dict[str, dict[str, str]] = {}
    for record in all_records:
        rec_id = record.get("record_id", "")
        if rec_id not in [f"L1_MB_{e['kernel_or_case']}" for e in manifest_entries if e["priority"] == "P0"]:
            # Map record_id to manifest id
            pass
        # Use kernel_invocation_id as a more stable key
        kid = record.get("kernel_invocation_id", record.get("record_id", "unknown"))
        features = record.get("features", record.get("feature_details", {}))
        matrix[kid] = {}
        for pka_name in PKA_FEATURES:
            f = features.get(pka_name, {})
            matrix[kid][pka_name] = f.get("status", "unknown")
    return {
        "pka_features": list(PKA_FEATURES.keys()),
        "availability": matrix,
        "total_invocations": len(matrix),
        "fully_measured_count": sum(
            1 for row in matrix.values()
            if all(v == "measured" for v in row.values())
        ),
    }


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    entries = manifest["entries"]

    all_feature_records: list[dict[str, Any]] = []
    all_gap_records: list[dict[str, Any]] = []

    for entry in entries:
        source_type = entry["source_type"]
        source_path = REPO_ROOT / entry["local_input_path"]
        kernel_or_case = entry["kernel_or_case"]

        adapter = ADAPTERS.get(source_type)
        if adapter is None:
            print(f"Warning: no adapter for source_type={source_type}, entry={entry['id']}")
            continue

        records = adapter(source_path, kernel_or_case)

        for rec in records:
            if rec.get("outcome") == "acquisition_gap":
                all_gap_records.append(rec)
            else:
                all_feature_records.append(rec)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write PkaFeatureTable (measured invocations only)
    feature_table_path = OUTPUT_DIR / "pka_feature_table_l1.json"
    feature_table_path.write_text(
        json.dumps(all_feature_records, indent=2, ensure_ascii=False) + "\n"
    )

    # Write PkaAcquisitionGap
    gap_path = OUTPUT_DIR / "pka_acquisition_gap_l1.json"
    gap_path.write_text(
        json.dumps(all_gap_records, indent=2, ensure_ascii=False) + "\n"
    )

    # Write metric availability matrix
    matrix = _build_metric_availability_matrix(
        all_feature_records + all_gap_records, entries
    )
    matrix_path = OUTPUT_DIR / "pka_metric_availability_matrix_l1.json"
    matrix_path.write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False) + "\n"
    )

    # Determine P0 gap status
    p0_entries = [e for e in entries if e["priority"] == "P0"]
    p0_gap_records = [
        r for r in all_gap_records
        if any(r.get("record_id", "").endswith(e["kernel_or_case"]) for e in p0_entries)
    ]

    print(f"Feature extraction complete:")
    print(f"  Measured invocations: {len(all_feature_records)}")
    print(f"  Acquisition gap invocations: {len(all_gap_records)}")
    print(f"  P0 gap invocations: {len(p0_gap_records)}")
    print(f"  Metric availability: {matrix['fully_measured_count']}/{matrix['total_invocations']} fully measured")

    if p0_gap_records:
        print()
        print("BLOCKED: P0 acquisition gaps detected. Selector (T5) and B-line (T6) must not run.")
        print("Gap invocations:")
        for r in p0_gap_records:
            print(f"  - {r.get('kernel_invocation_id', r.get('record_id'))}: "
                  f"missing {len(r.get('missing_metrics', []))} metrics")
        print()
        print("Outputs written:")
        print(f"  {feature_table_path}")
        print(f"  {gap_path}")
        print(f"  {matrix_path}")
        return 2  # signal: blocked on acquisition

    print(f"Outputs written:")
    print(f"  {feature_table_path}")
    print(f"  {gap_path}")
    print(f"  {matrix_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
