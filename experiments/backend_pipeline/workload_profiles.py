from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PROFILE_FIELDS = {
    "workload_id",
    "execution_mode",
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


def _resolve_path(path_str: str, base_dir: Path) -> str:
    if path_str == "":
        return ""
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


DEFAULT_WORKLOAD_PROFILES: dict[str, dict[str, Any]] = {
    "mini_transformer_v4": {
        "workload_id": "mini_transformer_v4",
        "execution_mode": "validation",
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
        "extra_cli_args": ["-is_extra_traces_enabled", "0"],
        "parser": {
            "sim_cycles_patterns": [
                r"gpu_tot_sim_cycle\s*=\s*([0-9]+)",
                r"gpu_sim_cycle\s*=\s*([0-9]+)",
            ],
            "simulation_time_patterns": [
                r"gpgpu_simulation_time\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                r"gpgpu_simulation_time\s*=\s*.*\((\d+)\s+sec\)",
            ],
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
                "description": "Increase trace-side miscellaneous queue latency for reduction-path probing.",
                "config_edits": [
                    {
                        "target": "trace_config",
                        "pattern": r"^-trace_opcode_latency_initiation_miscellaneous_queue\s+.*$",
                        "replacement": "-trace_opcode_latency_initiation_miscellaneous_queue 4,4 # scenario override: reduction path",
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


SMOKE_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "mini_transformer_v4": {
        "execution_mode": "smoke",
        "extra_cli_args": ["-gpgpu_max_cycle", "10"],
        "smoke_trace_builder": {
            "mode": "trimmed_dummy_extra_info",
            "kernel_launches": {
                "R1_qkv_projection_dense": {
                    "kernel_id": 1,
                    "function_unique_id": 1,
                    "kernel_name": "_Z10gemm_tiledPKfS0_Pfiii___0",
                    "threadblock_file": "d_0_s_0_k_1_0,0,0.pb",
                },
                "R2_attention_score_dense": {
                    "kernel_id": 5,
                    "function_unique_id": 2,
                    "kernel_name": "_Z15attention_scorePKfS0_Pfiii___0",
                    "threadblock_file": "d_0_s_0_k_5_0,0,0.pb",
                },
                "R3_output_projection_dense": {
                    "kernel_id": 8,
                    "function_unique_id": 1,
                    "kernel_name": "_Z10gemm_tiledPKfS0_Pfiii___0",
                    "threadblock_file": "d_0_s_0_k_8_0,0,0.pb",
                },
                "R4_ffn_expand_dense": {
                    "kernel_id": 11,
                    "function_unique_id": 1,
                    "kernel_name": "_Z10gemm_tiledPKfS0_Pfiii___0",
                    "threadblock_file": "d_0_s_0_k_11_0,0,0.pb",
                },
                "R5_ffn_contract_dense": {
                    "kernel_id": 12,
                    "function_unique_id": 1,
                    "kernel_name": "_Z10gemm_tiledPKfS0_Pfiii___0",
                    "threadblock_file": "d_0_s_0_k_12_0,0,0.pb",
                },
                "R6_softmax_reduction": {
                    "kernel_id": 6,
                    "function_unique_id": 3,
                    "kernel_name": "_Z14softmax_kernelPfii___0",
                    "threadblock_file": "d_0_s_0_k_6_0,0,0.pb",
                },
                "R7_layernorm_reduction": {
                    "kernel_id": 10,
                    "function_unique_id": 6,
                    "kernel_name": "_Z16layernorm_kernelPfii___0",
                    "threadblock_file": "d_0_s_0_k_10_0,0,0.pb",
                },
                "R8_context_streaming": {
                    "kernel_id": 7,
                    "function_unique_id": 4,
                    "kernel_name": "_Z11context_mulPKfS0_Pfiii___0",
                    "threadblock_file": "d_0_s_0_k_7_0,0,0.pb",
                },
                "R9_residual_elementwise": {
                    "kernel_id": 9,
                    "function_unique_id": 5,
                    "kernel_name": "_Z12residual_addPfPKfi___0",
                    "threadblock_file": "d_0_s_0_k_9_0,0,0.pb",
                },
            },
        },
    }
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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
    if "smoke_trace_builder" in profile and not isinstance(profile["smoke_trace_builder"], dict):
        raise ValueError("workload profile field 'smoke_trace_builder' must be a dict when present")


def _normalize_profile(profile: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    normalized = dict(profile)
    for field in ("working_directory", "simulator_binary", "setup_script", "trace_path", "gpgpusim_config", "trace_config"):
        normalized[field] = _resolve_path(normalized[field], base_dir)
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
    return normalized


def load_workload_profile(workload_id: str, profile_path: Path | None = None, *, smoke_mode: bool = False) -> dict[str, Any]:
    if profile_path is not None:
        raw_profile = json.loads(profile_path.read_text())
        base_dir = profile_path.resolve().parent
    else:
        if workload_id not in DEFAULT_WORKLOAD_PROFILES:
            raise KeyError(f"Unknown workload profile: {workload_id}")
        raw_profile = DEFAULT_WORKLOAD_PROFILES[workload_id]
        base_dir = REPO_ROOT
    if smoke_mode:
        if workload_id not in SMOKE_PROFILE_OVERRIDES:
            raise KeyError(f"Unknown smoke profile override for workload: {workload_id}")
        if profile_path is not None:
            raw_profile = _deep_merge(SMOKE_PROFILE_OVERRIDES[workload_id], raw_profile)
        else:
            raw_profile = _deep_merge(raw_profile, SMOKE_PROFILE_OVERRIDES[workload_id])
        raw_profile["execution_mode"] = "smoke"
    _validate_profile(raw_profile)
    normalized = _normalize_profile(raw_profile, base_dir=base_dir)
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
