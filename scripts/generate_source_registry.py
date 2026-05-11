#!/usr/bin/env python3
"""Generate GPU workload corpus source registry artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/home/dyf/workloads/trace-compressions-industrial-codex-workload")
DEFAULT_OUTPUT_DIR = Path("registry")

SOURCE_METADATA = {
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
        return subprocess.check_output(
            ["git", "-C", str(path), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_clone_status(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def is_git_checkout(path: Path) -> bool:
    return path.exists() and (path / ".git").exists()


def git_config_true(path: Path, key: str) -> bool:
    value = run_git(path, ["config", "--get", key])
    return value is not None and value.lower() == "true"


def infer_clone_mode(path: Path) -> str:
    if not is_git_checkout(path):
        return "unavailable"
    if git_config_true(path, "core.sparseCheckout") or git_config_true(path, "remote.origin.promisor"):
        return "sparse_partial"
    return "shallow_or_full"


def infer_availability(path: Path) -> str:
    if not is_git_checkout(path):
        return "source_unavailable"
    if infer_clone_mode(path) == "sparse_partial":
        return "source_sparse_available"
    return "source_available"


def build_source_registry(status_path: Path) -> dict[str, Any]:
    sources = []
    for row in parse_clone_status(status_path):
        source_id = row["name"]
        local_path = Path(row["path"])
        source_type, corpus_role = SOURCE_METADATA.get(source_id, ("unknown", "candidate"))
        commit = run_git(local_path, ["rev-parse", "--short", "HEAD"]) or row["commit"]

        sources.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "corpus_role": corpus_role,
                "url": row["url"],
                "local_path": str(local_path),
                "commit": commit,
                "clone_status": row["status"],
                "clone_mode": infer_clone_mode(local_path),
                "availability_status": infer_availability(local_path),
                "license_status": "needs_review",
            }
        )

    return {
        "schema_version": "source_registry_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def render_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# GPU Workload Trace Corpus Source Registry",
        "",
        f"Generated at: `{registry['generated_at']}`",
        "",
        "| Source | Type | Role | Availability | Clone Mode | Commit | License |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source in registry["sources"]:
        lines.append(
            "| `{source_id}` | `{source_type}` | `{corpus_role}` | `{availability_status}` | "
            "`{clone_mode}` | `{commit}` | `{license_status}` |".format(**source)
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(registry))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="External workload root containing clone_status.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for source_registry.json and source_registry.md.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = build_source_registry(args.root / "clone_status.tsv")
    write_json(args.output_dir / "source_registry.json", registry)
    write_markdown(args.output_dir / "source_registry.md", registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
