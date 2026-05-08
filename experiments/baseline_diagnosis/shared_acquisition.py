"""Shared acquisition helpers for the A-line PKA measured loops."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "a_line" / "l1"

FEATURE_ORDER = [
    "coalesced_global_loads",
    "coalesced_global_stores",
    "coalesced_local_loads",
    "thread_global_loads",
    "thread_global_stores",
    "thread_local_loads",
    "thread_shared_loads",
    "thread_shared_stores",
    "thread_global_atomics",
    "num_instructions",
    "divergence_efficiency",
    "num_thread_blocks",
]

COUNT_FEATURES = {
    "coalesced_global_loads",
    "coalesced_global_stores",
    "coalesced_local_loads",
    "thread_global_loads",
    "thread_global_stores",
    "thread_local_loads",
    "thread_shared_loads",
    "thread_shared_stores",
    "thread_global_atomics",
    "num_instructions",
    "num_thread_blocks",
}

RATIO_FEATURES = {"divergence_efficiency"}

FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "coalesced_global_loads": {
        "canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "source_candidates": ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"],
    },
    "coalesced_global_stores": {
        "canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "source_candidates": ["l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"],
    },
    "coalesced_local_loads": {
        "canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
        "source_candidates": ["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"],
    },
    "thread_global_loads": {
        "canonical_metric": "smsp__inst_executed_op_global_ld.sum",
        "source_candidates": ["smsp__inst_executed_op_global_ld.sum"],
    },
    "thread_global_stores": {
        "canonical_metric": "smsp__inst_executed_op_global_st.sum",
        "source_candidates": ["smsp__inst_executed_op_global_st.sum"],
    },
    "thread_local_loads": {
        "canonical_metric": "smsp__inst_executed_op_local_ld.sum",
        "source_candidates": ["smsp__inst_executed_op_local_ld.sum"],
    },
    "thread_shared_loads": {
        "canonical_metric": "smsp__inst_executed_op_shared_ld.sum",
        "source_candidates": ["smsp__inst_executed_op_shared_ld.sum"],
    },
    "thread_shared_stores": {
        "canonical_metric": "smsp__inst_executed_op_shared_st.sum",
        "source_candidates": ["smsp__inst_executed_op_shared_st.sum"],
    },
    "thread_global_atomics": {
        "canonical_metric": "smsp__sass_inst_executed_op_global_atom.sum",
        "source_candidates": ["smsp__sass_inst_executed_op_global_atom.sum"],
    },
    "num_instructions": {
        "canonical_metric": "smsp__inst_executed.sum",
        "source_candidates": ["smsp__inst_executed.sum"],
    },
    "divergence_efficiency": {
        "canonical_metric": "smsp__thread_inst_executed_per_inst_executed.ratio",
        "source_candidates": ["smsp__thread_inst_executed_per_inst_executed.ratio"],
    },
    "num_thread_blocks": {
        "canonical_metric": "launch_grid_size",
        "source_candidates": [],
        "source_kind": "launch_metadata",
    },
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def artifact_ref(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def stable_hash(data: Any) -> str:
    return hashlib.sha256(stable_json(data).encode()).hexdigest()


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_hash(command: list[str]) -> str:
    return stable_hash(command)[:12]


def sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return token or "workload"


def parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_grid_size(raw: str | None) -> tuple[float | None, dict[str, Any]]:
    if not raw:
        return None, {"raw_grid_size": raw, "normalization_rule": "missing"}
    nums = [int(x) for x in re.findall(r"\d+", raw)]
    if not nums:
        return None, {"raw_grid_size": raw, "normalization_rule": "parse_failed"}
    product = 1
    for num in nums[:3]:
        product *= num
    return float(product), {
        "raw_grid_size": raw,
        "normalized_value": product,
        "normalization_rule": "product_of_grid_dimensions",
    }


def environment_signature() -> dict[str, Any]:
    def run_tail(command: list[str], timeout: int = 5) -> dict[str, Any]:
        try:
            proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
            return {
                "command": command,
                "exit_code": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"command": command, "exit_code": None, "stdout_tail": "", "stderr_tail": str(exc)}

    nvidia = run_tail(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"])
    nvcc = run_tail(["nvcc", "--version"]) if shutil.which("nvcc") else {"command": ["nvcc", "--version"], "exit_code": None, "stdout_tail": "", "stderr_tail": "nvcc not found"}
    ncu_version = run_tail(["ncu", "--version"]) if shutil.which("ncu") else {"command": ["ncu", "--version"], "exit_code": None, "stdout_tail": "", "stderr_tail": "ncu not found"}
    gpu_name = None
    compute_capability = None
    driver_version = None
    if nvidia.get("stdout_tail"):
        parts = [part.strip() for part in nvidia["stdout_tail"].splitlines()[0].split(",")]
        if len(parts) >= 3:
            gpu_name, compute_capability, driver_version = parts[:3]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "PATH": os.environ.get("PATH", ""),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "ncu_path": shutil.which("ncu"),
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "driver_version": driver_version,
        "cuda_version": nvcc.get("stdout_tail"),
        "nsight_compute_version": ncu_version.get("stdout_tail"),
        "probe_results": {
            "nvidia_smi": nvidia,
            "nvcc": nvcc,
            "ncu_version": ncu_version,
        },
    }


def valid_environment_manifest(data: dict[str, Any]) -> bool:
    required = {
        "gpu_name",
        "compute_capability",
        "driver_version",
        "cuda_version",
        "nsight_compute_version",
        "environment_signature",
        "capture_timestamp",
        "target_run_command",
        "ncu_capture_command",
        "selected_metrics",
        "output_csv_path",
    }
    return required.issubset(data)


def metric_available_in_query(metric: str, query_text: str) -> bool:
    for line in query_text.splitlines():
        tokens = [token.strip() for token in re.split(r"[\s,]+", line.strip()) if token.strip()]
        if metric in tokens:
            return True
    return False


def selected_metric_records(query_text: str = "", query_status: str = "static_fixture") -> list[dict[str, Any]]:
    records = []
    for feature_name in FEATURE_ORDER:
        spec = FEATURE_SPECS[feature_name]
        canonical = spec["canonical_metric"]
        if feature_name == "num_thread_blocks":
            records.append({
                "feature_name": feature_name,
                "canonical_metric": canonical,
                "actual_source_metric": "Grid Size",
                "resolution_status": "launch_metadata",
                "selected_for_ncu_metrics": False,
            })
        else:
            actual = spec["source_candidates"][0]
            if query_status == "static_fixture":
                resolution_status = "available"
                selected = True
            elif query_status != "completed":
                resolution_status = "query_unavailable"
                selected = False
            elif metric_available_in_query(actual, query_text):
                resolution_status = "available"
                selected = True
            else:
                resolution_status = "unsupported"
                selected = False
            records.append({
                "feature_name": feature_name,
                "canonical_metric": canonical,
                "actual_source_metric": actual,
                "resolution_status": resolution_status,
                "selected_for_ncu_metrics": selected,
            })
    return records


def selected_ncu_metrics(metric_records: list[dict[str, Any]] | None = None) -> list[str]:
    metric_records = selected_metric_records() if metric_records is None else metric_records
    return [
        row["actual_source_metric"]
        for row in metric_records
        if row["selected_for_ncu_metrics"]
    ]


def has_ncu_csv_header(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    with path.open(newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].strip() == "ID" or any(col.strip() == "Metric Name" for col in row):
                return True
    return False


def parse_ncu_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        header = None
        rows: list[dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            if header is None:
                if row[0].strip() == "ID":
                    header = [col.strip() for col in row]
                continue
            if len(row) >= len(header):
                if len(row) > len(header) and header == ["ID", "Kernel Name", "Grid Size", "Metric Name", "Metric Value"]:
                    row = [row[0], row[1], ",".join(row[2:-2]), row[-2], row[-1]]
                rows.append({key: value for key, value in zip(header, row)})
    if header is None:
        raise ValueError(f"NCU CSV has no ID header: {path}")

    invocations: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        inv_id = row.get("ID", "").strip() or str(len(order))
        if inv_id not in invocations:
            order.append(inv_id)
            invocations[inv_id] = {
                "csv_invocation_id": inv_id,
                "kernel_name": row.get("Kernel Name", "").strip(),
                "metric_map": {},
                "grid_size_raw": row.get("Grid Size", "").strip(),
                "duration": None,
                "elapsed_cycles": None,
            }
        item = invocations[inv_id]
        if not item["kernel_name"]:
            item["kernel_name"] = row.get("Kernel Name", "").strip()
        if not item["grid_size_raw"]:
            item["grid_size_raw"] = row.get("Grid Size", "").strip()
        metric_name = row.get("Metric Name", "").strip()
        metric_value = parse_numeric(row.get("Metric Value", ""))
        if metric_name and metric_value is not None:
            item["metric_map"][metric_name] = metric_value
        if metric_name in {"gpu__time_duration.sum", "gpu__time_duration.avg"}:
            item["duration"] = metric_value
        if metric_name in {"sm__cycles_elapsed.sum", "sm__cycles_elapsed.avg"}:
            item["elapsed_cycles"] = metric_value

    parsed = []
    for index, inv_id in enumerate(order):
        item = invocations[inv_id]
        grid_value, grid_prov = normalize_grid_size(item.get("grid_size_raw"))
        item["occurrence_index"] = index
        item["grid_size_normalized"] = grid_value
        item["grid_size_provenance"] = grid_prov
        parsed.append(item)
    return parsed


def kernel_name_matches(kernel_name: str, target: str) -> bool:
    kernel_lower = kernel_name.lower()
    target_lower = target.lower()
    if target_lower in kernel_lower:
        return True
    stripped_kernel = re.sub(r"[^a-z0-9]+", "", kernel_lower)
    stripped_target = re.sub(r"[^a-z0-9]+", "", target_lower)
    return bool(stripped_target and stripped_target in stripped_kernel)


def feature_record(
    feature_name: str,
    value: float,
    actual_source_metric: str,
    source_artifact_path: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "value": value,
        "status": "measured",
        "canonical_metric": FEATURE_SPECS[feature_name]["canonical_metric"],
        "actual_source_metric": actual_source_metric,
        "source_artifact_path": source_artifact_path,
        "provenance": provenance,
    }


def missing_feature_record(
    feature_name: str,
    source_artifact_path: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "value": None,
        "status": "missing",
        "canonical_metric": FEATURE_SPECS[feature_name]["canonical_metric"],
        "actual_source_metric": None,
        "source_artifact_path": source_artifact_path,
        "provenance": {"missing_reason": reason},
    }
