import json
import shutil
from pathlib import Path

import pytest

from experiments.gcl_phase_b.resnet50_adapter import (
    build_resnet50_trace_adapter_bundle,
    mark_resnet50_fixture_debug_not_formal,
    validate_resnet50_trace_adapter_bundle,
)
from experiments.gcl_phase_b.resnet50_gate0 import record_resnet50_gate0_trace_acquisition

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def _formal_gate0_root(tmp_path):
    root = tmp_path / "formal_gate0"
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "dynamic_trace.pb").write_bytes(b"formal-protobuf-placeholder")
    threadblocks_dir = root / "threadblocks"
    threadblocks_dir.mkdir()
    shutil.copy(root / "threadblocks.json", threadblocks_dir / "threadblocks.json")
    record_resnet50_gate0_trace_acquisition(root)
    return root


def test_gate1_builds_formal_adapter_from_real_resnet50_trace(tmp_path):
    root = _formal_gate0_root(tmp_path)

    bundle = build_resnet50_trace_adapter_bundle(root)

    validate_resnet50_trace_adapter_bundle(bundle)
    assert bundle["artifact_status"] == "formal"
    assert bundle["formal_input_eligible"] is True
    assert bundle["source_gate0_manifest_hash"]
    assert bundle["trace_source"] == "nvbit"
    assert bundle["input_scope"] == "full_resnet50_inference_trace"
    assert bundle["adapter_validation_report"]["status"] == "passed"


def test_gate1_rejects_fixture_as_formal_input():
    debug_report = mark_resnet50_fixture_debug_not_formal(FIXTURE_ROOT)

    assert debug_report["artifact_status"] == "debug_not_formal"
    assert debug_report["formal_input_eligible"] is False
    with pytest.raises(ValueError, match="Gate0 formal acquisition manifest"):
        build_resnet50_trace_adapter_bundle(FIXTURE_ROOT)


def test_gate1_emits_kernel_cta_warp_trace_records(tmp_path):
    root = _formal_gate0_root(tmp_path)

    bundle = build_resnet50_trace_adapter_bundle(root)

    assert bundle["kernel_invocation_table"]
    assert bundle["cta_scheduler_records"]
    assert bundle["per_warp_trace_records"]
    assert bundle["static_instruction_table"]
    assert all("kernel_invocation_id" in row for row in bundle["cta_scheduler_records"])
    assert all("kernel_invocation_id" in row for row in bundle["per_warp_trace_records"])


def test_gate1_reports_missing_static_instruction_metadata(tmp_path):
    root = _formal_gate0_root(tmp_path)
    info_path = root / "enhanced_execution_info.json"
    info = json.loads(info_path.read_text())
    info["instructions"] = []
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="static instruction metadata"):
        build_resnet50_trace_adapter_bundle(root)
