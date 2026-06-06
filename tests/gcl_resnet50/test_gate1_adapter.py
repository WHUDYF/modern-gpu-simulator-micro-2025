import json
import shutil
from pathlib import Path

import pytest

from experiments.gcl_phase_b.resnet50_adapter import (
    build_resnet50_trace_adapter_bundle,
    mark_resnet50_fixture_debug_not_formal,
    validate_resnet50_trace_adapter_bundle,
)

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


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
    root = _fixture_backed_root(tmp_path)
    info_path = root / "enhanced_execution_info.json"
    info = json.loads(info_path.read_text())
    info["instructions"] = []
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="Gate0 formal acquisition manifest"):
        build_resnet50_trace_adapter_bundle(root)
