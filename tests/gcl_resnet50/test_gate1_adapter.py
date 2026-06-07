import json
import shutil
from pathlib import Path

import pytest

from experiments.gcl_phase_b.resnet50_adapter import (
    build_resnet50_artifact_shape_trace_adapter_bundle,
    build_resnet50_trace_adapter_bundle,
    mark_resnet50_fixture_debug_not_formal,
    validate_resnet50_trace_adapter_bundle,
)
from tests.gcl_resnet50.formal_fixture import write_minimal_artifact_shape_resnet50_root

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")
FORMAL_ROOT = Path("artifacts/gcl_resnet50_gate0_formal_trace/traces")


def _fixture_backed_root(tmp_path):
    root = tmp_path / "fixture_backed"
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "dynamic_trace.pb").write_bytes(b"formal-protobuf-placeholder")
    threadblocks_dir = root / "threadblocks"
    threadblocks_dir.mkdir()
    shutil.copy(root / "threadblocks.json", threadblocks_dir / "threadblocks.json")
    return root


def test_gate1_requires_real_gate0_manifest_before_formal_adapter(tmp_path):
    root = _fixture_backed_root(tmp_path)

    with pytest.raises(ValueError, match="Gate0 formal acquisition manifest"):
        build_resnet50_trace_adapter_bundle(root)


def test_gate1_rejects_fixture_as_formal_input():
    debug_report = mark_resnet50_fixture_debug_not_formal(FIXTURE_ROOT)

    assert debug_report["artifact_status"] == "debug_not_formal"
    assert debug_report["formal_input_eligible"] is False
    with pytest.raises(ValueError, match="Gate0 formal acquisition manifest"):
        build_resnet50_trace_adapter_bundle(FIXTURE_ROOT)


def test_gate1_reports_missing_static_instruction_metadata(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "artifact_shape_trace")
    info_path = root / "enhanced_execution_info.json"
    info = json.loads(info_path.read_text())
    info["instructions"] = []
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="static instruction metadata"):
        build_resnet50_artifact_shape_trace_adapter_bundle(root)


def test_gate1_artifact_shape_adapter_reads_dynamic_trace_pb_and_threadblock_directory(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "artifact_shape_trace")
    (root / "dynamic_trace.json").write_text("{}", encoding="utf-8")
    (root / "threadblocks.json").write_text("{}", encoding="utf-8")

    bundle = build_resnet50_artifact_shape_trace_adapter_bundle(root)

    assert bundle["artifact_status"] == "debug_not_formal"
    assert bundle["formal_input_eligible"] is False
    assert set(bundle["source_artifact_hashes"]) == {
        "dynamic_trace.pb",
        "threadblocks/",
        "enhanced_execution_info.json",
        "scheduler_metadata.json",
    }
    invocation_ids = [row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]]
    assert invocation_ids == ["resnet50_k00000", "resnet50_k00001"]
    assert [row["launch_order"] for row in bundle["kernel_invocation_table"]] == [0, 1]
    assert {row["kernel_id"] for row in bundle["kernel_invocation_table"]} == {17}
    assert len(bundle["per_warp_trace_records"]) == 8
    first = bundle["per_warp_trace_records"][0]
    assert first["kernel_invocation_id"] == "resnet50_k00000"
    assert first["cta_id"] == "0,0,0"
    assert first["warp_id"] == 0
    assert [entry["opcode"] for entry in first["entries"]] == [
        "MOV",
        "LDG.E.64.SYS",
        "FADD",
        "STG.E.64.SYS",
    ]


def test_gate1_artifact_shape_adapter_rejects_missing_threadblock_pb_from_scheduler_metadata(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "artifact_shape_trace")
    missing = (
        root
        / "threadblocks"
        / "device_0"
        / "stream_0"
        / "kernel_0"
        / "d_0_s_0_k_0_0,0,0.pb"
    )
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="threadblock protobuf"):
        build_resnet50_artifact_shape_trace_adapter_bundle(root)


def test_gate1_builds_formal_adapter_from_real_resnet50_trace_root():
    bundle = build_resnet50_trace_adapter_bundle(FORMAL_ROOT)

    validate_resnet50_trace_adapter_bundle(bundle)
    assert bundle["artifact_status"] == "formal"
    assert bundle["formal_input_eligible"] is True
    assert bundle["trace_source"] == "nvbit"
    assert bundle["scheduler_metadata_source"] == "real_nvbit_smid"
    assert bundle["kernel_invocation_table"]
    assert bundle["static_instruction_table"]
    assert bundle["cta_scheduler_records"]
    assert bundle["per_warp_trace_records"]
    invocation_ids = {row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]}
    assert "d_0_s_0_k_267" in invocation_ids
    assert all(
        record["kernel_invocation_id"] in invocation_ids
        for record in bundle["cta_scheduler_records"]
    )
    assert all(
        record["kernel_invocation_id"] in invocation_ids
        for record in bundle["per_warp_trace_records"]
    )
    assert all("entries" in record for record in bundle["per_warp_trace_records"])
