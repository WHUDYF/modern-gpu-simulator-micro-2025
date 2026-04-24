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
    "scenario_overrides",
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
            "reference_metrics": {
                "full_features_path": "experiments/mini_transformer/mini_transformer_v4_full.json",
                "writeback_map_path": "experiments/backend_pipeline/results/mini_transformer_v4/backend_writeback_map_v1.json",
            },
        },
        "scenario_overrides": {
            "S1_register_pressure": {
                "description": "Reduce available registers per shader core for register-pressure probing.",
                "config_edits": [
                    {
                        "target": "gpgpusim_config",
                        "pattern": r"^-gpgpu_shader_registers\s+.*$",
                        "replacement": "-gpgpu_shader_registers                32768 # scenario override: register pressure",
                    }
                ],
            },
            "S2_occupancy_balance": {
                "description": "Reduce CTA concurrency to probe occupancy-sensitive behavior.",
                "config_edits": [
                    {
                        "target": "gpgpusim_config",
                        "pattern": r"^-gpgpu_shader_cta\s+.*$",
                        "replacement": "-gpgpu_shader_cta                      16 # scenario override: occupancy balance",
                    }
                ],
            },
            "S3_cache_capacity": {
                "description": "Increase L2 associativity for cache-capacity probing.",
                "config_edits": [
                    {
                        "target": "gpgpusim_config",
                        "pattern": r"^-gpgpu_cache:dl2\s+.*$",
                        "replacement": "-gpgpu_cache:dl2     S:64:128:32,L:B:m:L:P,A:192:96,32:0,32 # scenario override: cache capacity",
                    }
                ],
            },
            "S4_reduction_path": {
                "description": "Increase memory unit ports for reduction-path probing.",
                "config_edits": [
                    {
                        "target": "gpgpusim_config",
                        "pattern": r"^-gpgpu_mem_unit_ports\s+.*$",
                        "replacement": "-gpgpu_mem_unit_ports                    2 # scenario override: reduction path",
                    }
                ],
            },
            "S5_shared_memory_coupling": {
                "description": "Reduce shared memory size to stress shmem-coupled dense behavior.",
                "config_edits": [
                    {
                        "target": "gpgpusim_config",
                        "pattern": r"^-gpgpu_shmem_size\s+.*$",
                        "replacement": "-gpgpu_shmem_size                   65536 # scenario override: shared memory coupling",
                    }
                ],
            },
            "S6_locality_path": {
                "description": "Reduce L1D associativity to probe locality-sensitive behavior.",
                "config_edits": [
                    {
                        "target": "gpgpusim_config",
                        "pattern": r"^-gpgpu_cache:dl1\s+.*$",
                        "replacement": "-gpgpu_cache:dl1     S:4:128:64,L:T:m:L:L,A:384:48,32:0,32 # scenario override: locality path",
                    }
                ],
            },
            "S7_constraint_regression": {
                "description": "Tighten runtime stat interval for lightweight regression-check runs.",
                "config_edits": [
                    {
                        "target": "gpgpusim_config",
                        "pattern": r"^-gpgpu_runtime_stat\s+.*$",
                        "replacement": "-gpgpu_runtime_stat                  1000 # scenario override: constraint regression",
                    }
                ],
            },
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
    if not isinstance(profile["scenario_overrides"], dict):
        raise ValueError("workload profile field 'scenario_overrides' must be a dict")


def _normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(profile)
    for field in ("working_directory", "simulator_binary", "setup_script", "trace_path", "gpgpusim_config", "trace_config"):
        normalized[field] = _resolve_path(normalized[field])
    normalized["environment"] = {str(k): str(v) for k, v in normalized["environment"].items()}
    normalized["extra_cli_args"] = [str(item) for item in normalized["extra_cli_args"]]
    normalized["scenario_overrides"] = {
        str(scenario_id): {
            "description": str(payload.get("description", "")),
            "config_edits": [
                {
                    "target": str(edit["target"]),
                    "pattern": str(edit["pattern"]),
                    "replacement": str(edit["replacement"]),
                }
                for edit in payload.get("config_edits", [])
            ],
        }
        for scenario_id, payload in normalized["scenario_overrides"].items()
    }
    if "reference_metrics" in normalized["parser"]:
        ref = dict(normalized["parser"]["reference_metrics"])
        for key in ("full_features_path", "writeback_map_path"):
            if key in ref:
                ref[key] = _resolve_path(ref[key])
        normalized["parser"]["reference_metrics"] = ref
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
