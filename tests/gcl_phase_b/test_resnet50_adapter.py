import copy
import json
import shutil
from pathlib import Path

import pytest

from experiments.gcl_phase_b.resnet50_adapter import (
    build_resnet50_trace_adapter_bundle,
    load_resnet50_trace_sources,
    mark_resnet50_fixture_debug_not_formal,
    validate_resnet50_trace_adapter_bundle,
)
from experiments.gcl_phase_b.resnet50_gate0 import record_resnet50_gate0_trace_acquisition
from experiments.gcl_phase_b.utils import hash_without

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


def test_resnet50_gate1_fixture_sources_are_debug_not_formal():
    report = mark_resnet50_fixture_debug_not_formal(FIXTURE_ROOT)

    assert report["artifact_status"] == "debug_not_formal"
    assert report["formal_input_eligible"] is False


def test_resnet50_gate1_formal_sources_are_loadable(tmp_path):
    sources = load_resnet50_trace_sources(_formal_gate0_root(tmp_path))

    assert sources.scheduler_metadata["scheduler_metadata_source"] == "real_nvbit_smid"
    assert sources.gate0_manifest["artifact_status"] == "formal"
    assert sources.dynamic_trace["kernel_invocations"]
    assert sources.threadblocks["threadblocks"]
    assert sources.enhanced_execution_info["instructions"]
    assert sources.stats_rows


def test_gate1_builds_resnet50_trace_adapter_bundle(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))

    validate_resnet50_trace_adapter_bundle(bundle)
    assert bundle["artifact_type"] == "gcl_resnet50_trace_adapter_bundle"
    assert bundle["workload_id"] == "resnet50"
    assert bundle["scheduler_metadata_source"] == "real_nvbit_smid"
    assert bundle["adapter_validation_report"]["status"] == "passed"
    assert len(bundle["kernel_invocation_table"]) == 2
    assert all("kernel_invocation_id" in row for row in bundle["cta_scheduler_records"])
    assert all("kernel_invocation_id" in row for row in bundle["per_warp_trace_records"])


def test_gate1_rejects_non_real_scheduler_metadata(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))
    corrupted = copy.deepcopy(bundle)
    corrupted["scheduler_metadata_source"] = "file_order_fallback"

    with pytest.raises(ValueError, match="real_nvbit_smid"):
        validate_resnet50_trace_adapter_bundle(corrupted)


def test_gate1_rejects_non_reproducible_adapter_hash(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))
    bundle["kernel_invocation_table"][0]["kernel_name"] = "changed"

    with pytest.raises(ValueError, match="adapter_bundle_hash"):
        validate_resnet50_trace_adapter_bundle(bundle)


def test_gate1_rejects_duplicate_cta_scheduler_records_for_same_invocation(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))
    bundle["cta_scheduler_records"].append(copy.deepcopy(bundle["cta_scheduler_records"][0]))
    bundle["adapter_bundle_hash"] = "stale"

    with pytest.raises(ValueError, match="duplicate cta scheduler record"):
        validate_resnet50_trace_adapter_bundle(bundle)


def test_gate1_rejects_scheduler_trace_count_mismatch_with_trace_records(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))
    bundle["cta_scheduler_records"][0]["trace_entry_count"] += 5
    bundle["adapter_bundle_hash"] = hash_without(bundle, "adapter_bundle_hash")

    with pytest.raises(ValueError, match="trace_entry_count"):
        validate_resnet50_trace_adapter_bundle(bundle)


def test_gate1_rejects_invalid_scheduler_order(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))
    record = bundle["cta_scheduler_records"][0]
    record["first_seen_order"] = record["last_seen_order"] + 1
    bundle["adapter_bundle_hash"] = hash_without(bundle, "adapter_bundle_hash")

    with pytest.raises(ValueError, match="first_seen_order"):
        validate_resnet50_trace_adapter_bundle(bundle)


def test_gate1_rejects_scheduler_warp_id_mismatch(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))
    bundle["cta_scheduler_records"][0]["warp_ids"] = [99]
    bundle["adapter_bundle_hash"] = hash_without(bundle, "adapter_bundle_hash")

    with pytest.raises(ValueError, match="warp_ids"):
        validate_resnet50_trace_adapter_bundle(bundle)


def test_gate1_rejects_stray_warp_trace_cta(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))
    bundle["per_warp_trace_records"][0]["cta_id"] = "stray,0,0"
    bundle["adapter_bundle_hash"] = hash_without(bundle, "adapter_bundle_hash")

    with pytest.raises(ValueError, match="CTA set"):
        validate_resnet50_trace_adapter_bundle(bundle)


def test_gate1_preserves_repeated_kernel_id_when_raw_sources_have_launch_order(tmp_path):
    root = tmp_path / "repeated_kernel_id"
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "dynamic_trace.pb").write_bytes(b"formal-protobuf-placeholder")
    threadblocks_dir = root / "threadblocks"
    threadblocks_dir.mkdir()
    shutil.copy(root / "threadblocks.json", threadblocks_dir / "threadblocks.json")
    dynamic_path = root / "dynamic_trace.json"
    scheduler_path = root / "scheduler_metadata.json"
    threadblocks_path = root / "threadblocks.json"
    dynamic = json.loads(dynamic_path.read_text())
    scheduler = json.loads(scheduler_path.read_text())
    threadblocks = json.loads(threadblocks_path.read_text())
    repeated_kernel_id = dynamic["kernel_invocations"][0]["kernel_id"]
    dynamic["kernel_invocations"][1]["kernel_id"] = repeated_kernel_id
    for launch_order, invocation in enumerate(scheduler["kernel_invocations"]):
        invocation["kernel_id"] = repeated_kernel_id
        invocation["launch_order"] = launch_order
    for record in threadblocks["threadblocks"]:
        if record["kernel_id"] == 18:
            record["kernel_id"] = repeated_kernel_id
            record["launch_order"] = 1
        else:
            record["launch_order"] = 0
    dynamic_path.write_text(json.dumps(dynamic), encoding="utf-8")
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    threadblocks_path.write_text(json.dumps(threadblocks), encoding="utf-8")
    shutil.copy(threadblocks_path, threadblocks_dir / "threadblocks.json")
    record_resnet50_gate0_trace_acquisition(root)

    bundle = build_resnet50_trace_adapter_bundle(root)

    invocation_ids = {row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]}
    scheduler_ids = {row["kernel_invocation_id"] for row in bundle["cta_scheduler_records"]}
    trace_ids = {row["kernel_invocation_id"] for row in bundle["per_warp_trace_records"]}
    assert invocation_ids == {"resnet50_k00000", "resnet50_k00001"}
    assert scheduler_ids == invocation_ids
    assert trace_ids == invocation_ids


def test_gate1_rejects_ambiguous_repeated_kernel_id_without_launch_order(tmp_path):
    root = tmp_path / "ambiguous_repeated_kernel_id"
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "dynamic_trace.pb").write_bytes(b"formal-protobuf-placeholder")
    threadblocks_dir = root / "threadblocks"
    threadblocks_dir.mkdir()
    shutil.copy(root / "threadblocks.json", threadblocks_dir / "threadblocks.json")
    dynamic_path = root / "dynamic_trace.json"
    dynamic = json.loads(dynamic_path.read_text())
    dynamic["kernel_invocations"][1]["kernel_id"] = dynamic["kernel_invocations"][0]["kernel_id"]
    dynamic_path.write_text(json.dumps(dynamic), encoding="utf-8")
    record_resnet50_gate0_trace_acquisition(root)

    with pytest.raises(ValueError, match="requires kernel_invocation_id or launch_order"):
        build_resnet50_trace_adapter_bundle(root)


def test_gate1_rejects_raw_record_with_contradictory_kernel_id_and_launch_order(tmp_path):
    root = tmp_path / "contradictory_kernel_launch_order"
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "dynamic_trace.pb").write_bytes(b"formal-protobuf-placeholder")
    threadblocks_dir = root / "threadblocks"
    threadblocks_dir.mkdir()
    shutil.copy(root / "threadblocks.json", threadblocks_dir / "threadblocks.json")
    threadblocks_path = root / "threadblocks.json"
    threadblocks = json.loads(threadblocks_path.read_text())
    threadblocks["threadblocks"][0]["launch_order"] = 1
    threadblocks_path.write_text(json.dumps(threadblocks), encoding="utf-8")
    shutil.copy(threadblocks_path, threadblocks_dir / "threadblocks.json")
    record_resnet50_gate0_trace_acquisition(root)

    with pytest.raises(ValueError, match="kernel_id does not match"):
        build_resnet50_trace_adapter_bundle(root)
