import csv
import json
from pathlib import Path
from typing import Any

from experiments.baseline_diagnosis.proto_gen import trace_pb2, threadblock_pb2


def write_minimal_artifact_shape_resnet50_root(
    root: Path,
    *,
    evidence_scope: str = "synthetic_artifact_shape_unit_test_only",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_dynamic_trace(root / "dynamic_trace.pb")
    _write_threadblocks(root / "threadblocks")
    _write_json(root / "enhanced_execution_info.json", _enhanced_execution_info())
    _write_json(root / "scheduler_metadata.json", _scheduler_metadata())
    _write_json(
        root / "nvbit_collection_evidence.json",
        _nvbit_collection_evidence(evidence_scope=evidence_scope),
    )
    _write_stats(root / "stats.csv")
    return root


def _write_dynamic_trace(path: Path) -> None:
    trace = trace_pb2.Trace()
    trace.name = "resnet50_formal_unit_trace"
    trace.binary_version = 1
    trace.nvbit_version = "unit-nvbit"
    trace.accelsim_version = 1
    device = trace.gpu_device[0]
    device.id = 0
    stream = device.streams[0]
    stream.id = 0
    for kernel_id, name, function_unique_id, registers in [
        (17, "resnet50_conv2d_fprop_tile", 1701, 64),
        (17, "resnet50_conv2d_fprop_tile", 1702, 66),
    ]:
        kernel = stream.kernels.add()
        kernel.id = kernel_id
        kernel.name = name
        kernel.function_unique_id = function_unique_id
        kernel.size_shared_memory = 0
        kernel.number_of_registers = registers
        kernel.grid_dim.x = 2
        kernel.grid_dim.y = 1
        kernel.grid_dim.z = 1
        kernel.block_dim.x = 64
        kernel.block_dim.y = 1
        kernel.block_dim.z = 1
    path.write_bytes(trace.SerializeToString())


def _write_threadblocks(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for launch_order, function_unique_id in [(0, 1701), (1, 1702)]:
        for cta_id in ["0,0,0", "1,0,0"]:
            cta_dir = root / "device_0" / "stream_0" / f"kernel_{launch_order}"
            cta_dir.mkdir(parents=True, exist_ok=True)
            block = threadblock_pb2.threadblock()
            x, y, z = [int(part) for part in cta_id.split(",")]
            block.block_id.x = x
            block.block_id.y = y
            block.block_id.z = z
            for warp_id in [0, 1]:
                warp = block.warps[warp_id]
                warp.id = warp_id
                for offset in range(4):
                    instruction = warp.instructions.add()
                    instruction.pc = 4096 + launch_order * 1024 + offset * 4
                    instruction.function_unique_id = function_unique_id
                    instruction.active_mask = 0xFFFFFFFF
                    instruction.predicate_mask = 0xFFFFFFFF
            filename = f"d_0_s_0_k_{launch_order}_{cta_id}.pb"
            (cta_dir / filename).write_bytes(block.SerializeToString())


def _enhanced_execution_info() -> dict[str, Any]:
    rows = []
    for launch_order, function_unique_id, base_pc in [(0, 1701, 4096), (1, 1702, 5120)]:
        for offset, opcode, operands in [
            (0, "MOV", ["R4", "input:base"]),
            (4, "LDG.E.64.SYS", ["R8", "R4"]),
            (8, "FADD", ["R9", "R8", "input:bias"]),
            (12, "STG.E.64.SYS", ["R4", "R9"]),
        ]:
            rows.append(
                {
                    "function_unique_id": function_unique_id,
                    "pc": base_pc + offset,
                    "opcode": opcode,
                    "operands": operands,
                    "control_bits": "0x0",
                    "launch_order": launch_order,
                }
            )
    return {
        "artifact_type": "resnet50_enhanced_execution_info_nvbit",
        "instructions": rows,
    }


def _scheduler_metadata() -> dict[str, Any]:
    return {
        "artifact_type": "resnet50_scheduler_metadata_nvbit",
        "scheduler_metadata_source": "real_nvbit_smid",
        "kernel_invocations": [
            {
                "kernel_id": 17,
                "launch_order": launch_order,
                "cta_records": [
                    {
                        "cta_id": "0,0,0",
                        "sm_id": 1,
                        "first_seen_order": 1,
                        "last_seen_order": 2,
                        "warp_ids": [0, 1],
                        "trace_entry_count": 8,
                        "threadblock_pb": f"device_0/stream_0/kernel_{launch_order}/d_0_s_0_k_{launch_order}_0,0,0.pb",
                    },
                    {
                        "cta_id": "1,0,0",
                        "sm_id": 2,
                        "first_seen_order": 3,
                        "last_seen_order": 4,
                        "warp_ids": [0, 1],
                        "trace_entry_count": 8,
                        "threadblock_pb": f"device_0/stream_0/kernel_{launch_order}/d_0_s_0_k_{launch_order}_1,0,0.pb",
                    },
                ],
            }
            for launch_order in [0, 1]
        ],
    }


def _nvbit_collection_evidence(*, evidence_scope: str) -> dict[str, Any]:
    evidence = {
        "artifact_status": "formal_collection_evidence",
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "collection_status": "completed",
        "fixture_backed": False,
        "nvbit_loaded": True,
        "runner_invocation": ["python", "run_resnet50.py"],
    }
    if evidence_scope:
        evidence["evidence_scope"] = evidence_scope
    return evidence


def _write_stats(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kernel_id", "kernel_name", "launch_order"])
        writer.writeheader()
        writer.writerow(
            {
                "kernel_id": "17",
                "kernel_name": "resnet50_conv2d_fprop_tile",
                "launch_order": "0",
            }
        )
        writer.writerow(
            {
                "kernel_id": "17",
                "kernel_name": "resnet50_conv2d_fprop_tile",
                "launch_order": "1",
            }
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
