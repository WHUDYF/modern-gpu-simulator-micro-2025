from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PROFILE_FIELDS = {
    "workload_id",
    "simulator_binary",
    "working_directory",
    "trace_path",
    "gpgpusim_config",
    "trace_config",
    "environment",
    "extra_cli_args",
    "setup_script",
    "parser",
}


def _resolve_path(path_str: str) -> str:
    if path_str == "":
        return ""
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


DEFAULT_WORKLOAD_PROFILES: dict[str, dict[str, Any]] = {
    "mini_transformer_v4": {
        "workload_id": "mini_transformer_v4",
        "working_directory": "simulator-remodeled",
        "simulator_binary": "simulator-remodeled/gpu-simulator/bin/release/accel-sim.out",
        "setup_script": "simulator-remodeled/gpu-simulator/setup_environment_no_git.sh",
        "trace_path": "experiments/mini_transformer/traces/dynamic_trace.pb",
        "gpgpusim_config": "simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080_TI/gpgpusim.config",
        "trace_config": "simulator-remodeled/gpu-simulator/configs/tested-cfgs/SM86_RTX3080_TI/trace.config",
        "environment": {
            "CUDA_INSTALL_PATH": "/usr/local/cuda-12.8",
            "OMP_NUM_THREADS": "4",
        },
        # The repository-local mini_transformer trace is missing extra_info metadata.
        # Disable extra traces so the first smoke execution can still exercise the bridge.
        "extra_cli_args": ["-is_extra_traces_enabled", "0"],
        "parser": {
            "sim_cycles_patterns": [
                r"gpu_tot_sim_cycle\s*=\s*([0-9]+)",
                r"gpu_sim_cycle\s*=\s*([0-9]+)",
            ],
            "simulation_time_patterns": [
                r"gpgpu_simulation_time\s*=\s*([0-9]+(?:\.[0-9]+)?)",
            ],
        },
    }
}


def _validate_profile(profile: dict[str, Any]) -> None:
    missing = REQUIRED_PROFILE_FIELDS - set(profile.keys())
    if missing:
        raise ValueError(f"workload profile is missing required fields: {sorted(missing)}")
    if not isinstance(profile["environment"], dict):
        raise ValueError("workload profile field 'environment' must be a dict")
    if not isinstance(profile["extra_cli_args"], list):
        raise ValueError("workload profile field 'extra_cli_args' must be a list")
    if not isinstance(profile["parser"], dict):
        raise ValueError("workload profile field 'parser' must be a dict")


def _normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(profile)
    for field in ("working_directory", "simulator_binary", "setup_script", "trace_path", "gpgpusim_config", "trace_config"):
        normalized[field] = _resolve_path(normalized[field])
    normalized["environment"] = {str(k): str(v) for k, v in normalized["environment"].items()}
    normalized["extra_cli_args"] = [str(item) for item in normalized["extra_cli_args"]]
    return normalized


def load_workload_profile(workload_id: str, profile_path: Path | None = None) -> dict[str, Any]:
    if profile_path is not None:
        raw_profile = json.loads(profile_path.read_text())
    else:
        if workload_id not in DEFAULT_WORKLOAD_PROFILES:
            raise KeyError(f"Unknown workload profile: {workload_id}")
        raw_profile = DEFAULT_WORKLOAD_PROFILES[workload_id]
    _validate_profile(raw_profile)
    normalized = _normalize_profile(raw_profile)
    if normalized["workload_id"] != workload_id:
        raise ValueError(
            f"workload profile mismatch: requested {workload_id}, profile defines {normalized['workload_id']}"
        )
    for field in ("simulator_binary", "trace_path", "gpgpusim_config", "trace_config"):
        if not Path(normalized[field]).exists():
            raise FileNotFoundError(f"Workload profile path does not exist for {field}: {normalized[field]}")
    if normalized["setup_script"] and not Path(normalized["setup_script"]).exists():
        raise FileNotFoundError(f"Workload profile setup script does not exist: {normalized['setup_script']}")
    if not Path(normalized["working_directory"]).exists():
        raise FileNotFoundError(f"Workload profile working directory does not exist: {normalized['working_directory']}")
    return normalized
