import json
import shutil
from pathlib import Path

import pytest

from experiments.gcl_phase_b.resnet50_gate0 import (
    record_resnet50_gate0_trace_acquisition,
    validate_gate0_trace_acquisition_manifest,
)

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def _formal_gate0_root(tmp_path):
    root = tmp_path / "formal_gate0"
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "dynamic_trace.pb").write_bytes(b"formal-protobuf-placeholder")
    threadblocks_dir = root / "threadblocks"
    threadblocks_dir.mkdir()
    shutil.copy(root / "threadblocks.json", threadblocks_dir / "threadblocks.json")
    return root


def test_gate0_records_real_resnet50_nvbit_trace_provenance(tmp_path):
    root = _formal_gate0_root(tmp_path)

    manifest = record_resnet50_gate0_trace_acquisition(root)

    validate_gate0_trace_acquisition_manifest(manifest)
    assert manifest["workload_id"] == "resnet50"
    assert manifest["execution_mode"] == "real_trace"
    assert manifest["trace_source"] == "nvbit"
    assert manifest["input_scope"] == "full_resnet50_inference_trace"
    assert manifest["scheduler_metadata_source"] == "real_nvbit_smid"
    assert manifest["artifact_status"] == "formal"
    assert manifest["formal_input_eligible"] is True
    assert set(manifest["source_artifact_hashes"]) == {
        "dynamic_trace.pb",
        "threadblocks/",
        "enhanced_execution_info.json",
        "scheduler_metadata.json",
        "stats.csv",
    }


def test_gate0_rejects_missing_real_smid_metadata(tmp_path):
    root = _formal_gate0_root(tmp_path)
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    scheduler["scheduler_metadata_source"] = "file_order_fallback"
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")

    with pytest.raises(ValueError, match="real_nvbit_smid"):
        record_resnet50_gate0_trace_acquisition(root)
