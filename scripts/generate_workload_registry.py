#!/usr/bin/env python3
"""Generate a draft workload registry from the source registry."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_REGISTRY = Path("registry/source_registry.json")
DEFAULT_OUTPUT_DIR = Path("registry")

CURATED_WORKLOADS = {
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
CURATED_WORKLOAD_PATHS = {
    "mlperf-inference": {
        "bert": "language",
        "resnet50": "vision/classification_and_detection",
        "dlrm-v2": "recommendation",
        "retinanet": "vision",
        "3d-unet": "vision/medical_imaging",
        "stable-diffusion": "text_to_image",
    },
    "deepbench": {
        "gemm": "code/nvidia/gemm_bench.cu",
        "rnn": "code/nvidia/rnn_bench.cu",
        "convolution": "code/nvidia/conv_bench.cu",
    },
    "cutlass": {
        "gemm": "examples",
        "conv": "examples",
        "attention": "examples",
    },
    "gunrock": {
        "bfs": "examples/algorithms/bfs",
        "sssp": "examples/algorithms/sssp",
        "pagerank": "examples/algorithms/pr",
        "connected-components": "examples/algorithms/cc",
    },
    "pannotia": {
        "bfs": "graph_app/bc",
        "coloring": "graph_app/color",
        "pagerank": "graph_app/prk",
    },
    "lammps": {
        "lj-small-step": "examples/melt",
        "eam-small-step": "examples/HEAT",
    },
    "gromacs": {
        "water-small-step": "share/top",
        "protein-small-step": "share/top",
    },
}
SPARSE_DISCOVERABLE_SOURCES = {"hecbench"}

SCAN_DIRECTORIES = ("cuda", "CUDA", "src", "test")
SUPPORT_DIRECTORY_NAMES = {
    "common",
    "cuda",
    "include",
    "mpi",
    "opencl",
    "stability",
    "util",
    "utils",
}
HECBENCH_CUDA_CANDIDATES = (
    "bfs",
    "sgemm",
    "spmv",
    "backprop",
    "hotspot",
    "lud",
    "nw",
    "attention",
    "streamcluster",
    "particlefilter",
    "b+tree",
    "cfd",
    "lavaMD",
)


def default_generated_at() -> str:
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        return datetime.fromtimestamp(int(source_date_epoch), timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def pressure_for(source_id: str, name: str) -> tuple[str, str, str, str]:
    lowered = name.lower()
    if any(token in lowered for token in ("bfs", "spmv", "histo", "streamcluster", "particle")):
        return ("irregular_or_sparse", "medium", "medium", "high")
    if any(token in lowered for token in ("sgemm", "gemm", "lbm", "stencil", "cfd", "lava", "hotspot")):
        return ("benchmark_kernel", "medium", "large", "low")
    if source_id in {"shoc", "altis"}:
        return ("benchmark_suite_candidate", "medium", "medium", "medium")
    return ("benchmark_kernel", "low", "medium", "medium")


def slugify_workload_part(value: str) -> str:
    """Normalize workload ID parts to lowercase [a-z0-9._-] slugs."""
    lowered = value.lower()
    portable = re.sub(r"[^a-z0-9._-]+", "-", lowered)
    collapsed = re.sub(r"-{2,}", "-", portable)
    return collapsed.strip("-")


def make_record(
    source_id: str,
    name: str,
    family: str,
    kernel_count: str,
    large_kernel: str,
    irregularity: str,
    relative_path: str,
) -> dict[str, str]:
    normalized_source_id = slugify_workload_part(source_id)
    normalized_name = slugify_workload_part(name)
    return {
        "workload_id": f"{normalized_source_id}_{normalized_name}",
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


def curated_workload_path(source_id: str, name: str) -> str:
    return CURATED_WORKLOAD_PATHS.get(source_id, {}).get(name, ".")


def discover_gpu_parboil_workloads(source_id: str, root: Path) -> list[dict[str, str]]:
    benchmarks_root = root / "benchmarks"
    if not benchmarks_root.is_dir():
        return []

    workloads = []
    for benchmark in sorted(benchmarks_root.iterdir(), key=lambda path: path.name):
        source_dir = benchmark / "src"
        if not benchmark.is_dir() or benchmark.name.startswith(".") or not source_dir.is_dir():
            continue
        family, kernel_count, large_kernel, irregularity = pressure_for(source_id, benchmark.name)
        workloads.append(
            make_record(
                source_id,
                benchmark.name,
                family,
                kernel_count,
                large_kernel,
                irregularity,
                str(source_dir.relative_to(root)),
            )
        )
    return workloads


def discover_hecbench_workloads(source_id: str, root: Path) -> list[dict[str, str]]:
    workloads = []
    for name in HECBENCH_CUDA_CANDIDATES:
        cuda_path = root / "src" / f"{name}-cuda"
        if not cuda_path.is_dir():
            continue
        family, kernel_count, large_kernel, irregularity = pressure_for(source_id, name)
        workloads.append(
            make_record(
                source_id,
                name,
                family,
                kernel_count,
                large_kernel,
                irregularity,
                str(cuda_path.relative_to(root)),
            )
        )
    return workloads


def discover_nested_cuda_workloads(source_id: str, root: Path) -> list[dict[str, str]]:
    cuda_root = root / "src" / "cuda"
    if not cuda_root.is_dir():
        return []

    workloads = []
    for level_dir in sorted(cuda_root.iterdir(), key=lambda path: path.name):
        if not level_dir.is_dir() or level_dir.name.startswith("."):
            continue
        if not level_dir.name.lower().startswith("level"):
            continue
        for child in sorted(level_dir.iterdir(), key=lambda path: path.name):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if child.name.lower() in SUPPORT_DIRECTORY_NAMES:
                continue
            family, kernel_count, large_kernel, irregularity = pressure_for(source_id, child.name)
            workloads.append(
                make_record(
                    source_id,
                    child.name,
                    family,
                    kernel_count,
                    large_kernel,
                    irregularity,
                    str(child.relative_to(root)),
                )
            )
    return workloads


def discover_workloads_for_source(source_id: str, root: Path) -> list[dict[str, str]]:
    if source_id in CURATED_WORKLOADS:
        return [
            make_record(
                source_id,
                name,
                family,
                kernel_count,
                large_kernel,
                irregularity,
                curated_workload_path(source_id, name),
            )
            for name, family, kernel_count, large_kernel, irregularity in CURATED_WORKLOADS[source_id]
        ]
    if source_id == "gpu-parboil":
        return discover_gpu_parboil_workloads(source_id, root)
    if source_id == "hecbench":
        return discover_hecbench_workloads(source_id, root)
    if source_id in {"shoc", "altis"}:
        return discover_nested_cuda_workloads(source_id, root)

    workloads: list[dict[str, str]] = []
    for directory_name in SCAN_DIRECTORIES:
        scan_root = root / directory_name
        if not scan_root.is_dir():
            continue
        for child in sorted(scan_root.iterdir(), key=lambda path: path.name):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if child.name.lower() in SUPPORT_DIRECTORY_NAMES:
                continue
            family, kernel_count, large_kernel, irregularity = pressure_for(source_id, child.name)
            record = make_record(
                source_id,
                child.name,
                family,
                kernel_count,
                large_kernel,
                irregularity,
                str(child.relative_to(root)),
            )
            workloads.append(record)
    return workloads


def append_unique_workload(workloads: list[dict[str, str]], workload: dict[str, str], seen_ids: set[str]) -> None:
    workload_id = workload["workload_id"]
    if workload_id in seen_ids:
        raise ValueError(f"Duplicate workload_id after slug normalization: {workload_id}")
    seen_ids.add(workload_id)
    workloads.append(workload)


def workload_path_exists(source_root: Path, workload: dict[str, str]) -> bool:
    relative_path = workload.get("relative_path", ".")
    return (source_root / relative_path).exists()


def build_workload_registry(source_registry_path: Path, generated_at: str | None = None) -> dict[str, Any]:
    source_registry = json.loads(source_registry_path.read_text())
    workloads = []
    seen_ids: set[str] = set()
    for source in source_registry["sources"]:
        source_id = source["source_id"]
        availability_status = source.get("availability_status")
        if availability_status == "source_unavailable":
            continue
        if (
            availability_status == "source_sparse_available"
            and source_id not in CURATED_WORKLOADS
            and source_id not in SPARSE_DISCOVERABLE_SOURCES
        ):
            continue
        source_root = Path(source["local_path"])
        for workload in discover_workloads_for_source(source_id, source_root):
            should_validate_path = (
                source_id in CURATED_WORKLOADS or availability_status == "source_sparse_available"
            )
            if should_validate_path and not workload_path_exists(source_root, workload):
                continue
            append_unique_workload(workloads, workload, seen_ids)
    return {
        "schema_version": "workload_registry_v1",
        "generated_at": generated_at or default_generated_at(),
        "workloads": workloads,
    }


def render_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# GPU Workload Trace Corpus Workload Registry Draft",
        "",
        f"Generated at: `{registry['generated_at']}`",
        "",
        "| Workload | Source | Family | Kernel Count | Large Kernel | Irregularity | Path |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in registry["workloads"]:
        lines.append(
            "| `{workload_id}` | `{source_id}` | `{workload_family}` | `{expected_kernel_count_class}` | "
            "`{expected_large_kernel_class}` | `{expected_irregularity_class}` | `{relative_path}` |".format(**item)
        )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(registry))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=DEFAULT_SOURCE_REGISTRY,
        help="Source registry JSON to read.",
    )
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for workload_registry.json and workload_registry.md.",
    )
    parser.add_argument(
        "--generated-at",
        help="Timestamp to write into generated artifacts for deterministic output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = build_workload_registry(args.source_registry, generated_at=args.generated_at)
    write_json(args.output_dir / "workload_registry.json", registry)
    write_markdown(args.output_dir / "workload_registry.md", registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
