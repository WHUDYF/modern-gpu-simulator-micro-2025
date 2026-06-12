import json
import shutil
from pathlib import Path

import pytest

from experiments.baseline_diagnosis.proto_gen import trace_pb2
from experiments.gcl_phase_b.resnet50_adapter import (
    build_resnet50_artifact_shape_trace_adapter_bundle,
    build_resnet50_trace_adapter_bundle,
    mark_resnet50_fixture_debug_not_formal,
    validate_resnet50_trace_adapter_bundle,
)
from experiments.gcl_phase_b.resnet50_gate0 import record_resnet50_gate0_trace_acquisition
from experiments.gcl_phase_b.resnet50_gate0 import _source_artifact_hashes
from experiments.gcl_phase_b.utils import hash_without
from gcl_resnet50.test_gate0_trace_acquisition import (
    _write_collector_bound_gate0_evidence,
    _write_minimal_real_dynamic_trace_pb,
    _write_real_gate0_contract_artifacts,
)
from gcl_resnet50.formal_fixture import write_minimal_artifact_shape_resnet50_root
from gcl_resnet50.real_chain import FORMAL_ROOT, require_formal_root

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def _fixture_backed_root(tmp_path):
    root = tmp_path / "fixture_backed"
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "dynamic_trace.pb").write_bytes(b"formal-protobuf-placeholder")
    threadblocks_dir = root / "threadblocks"
    threadblocks_dir.mkdir()
    shutil.copy(root / "threadblocks.json", threadblocks_dir / "threadblocks.json")
    return root


def _write_formal_gate0_manifest(root):
    manifest = {
        "artifact_type": "gcl_resnet50_gate0_trace_acquisition_manifest",
        "artifact_version": "gate0_trace_acquisition_manifest_v1",
        "artifact_status": "formal",
        "formal_input_eligible": True,
        "workload_id": "resnet50",
        "execution_mode": "real_trace",
        "trace_source": "nvbit",
        "input_scope": "full_resnet50_inference_trace",
        "scheduler_metadata_source": "real_nvbit_smid",
        "nvbit_collection_evidence_hash": "unit-test-evidence-hash",
        "source_artifact_hashes": _source_artifact_hashes(root),
    }
    manifest["gate0_manifest_hash"] = hash_without(manifest, "gate0_manifest_hash")
    (root / "gate0_trace_acquisition_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_gate1_requires_real_gate0_manifest_before_formal_adapter(tmp_path):
    root = _fixture_backed_root(tmp_path)

    with pytest.raises(ValueError, match="Gate0 formal acquisition manifest"):
        build_resnet50_trace_adapter_bundle(root)


def test_gate1_rejects_formal_root_when_source_hashes_changed_after_gate0(tmp_path):
    root = tmp_path / "mutated_formal_root"
    _write_real_gate0_contract_artifacts(root)
    _write_collector_bound_gate0_evidence(root, "external-collector-session")
    record_resnet50_gate0_trace_acquisition(root)
    _write_minimal_real_dynamic_trace_pb(
        root / "dynamic_trace.pb",
        trace_name="mutated_after_gate0",
    )

    with pytest.raises(ValueError, match="Gate0 source artifact hashes mismatch"):
        build_resnet50_trace_adapter_bundle(root)


def test_real_chain_skips_when_formal_root_artifacts_are_missing(tmp_path):
    from gcl_resnet50.real_chain import require_formal_root

    with pytest.raises(pytest.skip.Exception, match="real ResNet-50 Gate0 trace"):
        require_formal_root(tmp_path / "missing_formal_root")


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


def test_gate1_rejects_unordered_multi_stream_pb(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "multi_stream_trace")
    trace = trace_pb2.Trace()
    trace.name = "multi_stream_without_global_order"
    device = trace.gpu_device[0]
    device.id = 0
    for launch_order, (stream_id, kernel_id, function_unique_id) in enumerate(
        [(0, 101, 1701), (1, 202, 1702)]
    ):
        stream = device.streams[stream_id]
        stream.id = stream_id
        kernel = stream.kernels.add()
        kernel.id = kernel_id
        kernel.name = f"stream_{stream_id}_kernel"
        kernel.function_unique_id = function_unique_id
        kernel.grid_dim.x = 1
        kernel.grid_dim.y = 1
        kernel.grid_dim.z = 1
        kernel.block_dim.x = 32
        kernel.block_dim.y = 1
        kernel.block_dim.z = 1
    (root / "dynamic_trace.pb").write_bytes(trace.SerializeToString())
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    for launch_order, invocation in enumerate(scheduler["kernel_invocations"]):
        invocation["kernel_id"] = 101 if launch_order == 0 else 202
        invocation["stream_id"] = 0 if launch_order == 0 else 1
        invocation["device_id"] = 0
        invocation["launch_order"] = launch_order
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    _write_formal_gate0_manifest(root)

    with pytest.raises(ValueError, match="multi-stream dynamic_trace.pb lacks global launch order"):
        build_resnet50_trace_adapter_bundle(root, invocation_ids=["resnet50_k00001"])


def test_gate1_formal_pb_uses_launch_order_invocation_ids_for_reused_kernel_id(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "formal_reused_kernel_id")
    _write_formal_gate0_manifest(root)

    bundle = build_resnet50_trace_adapter_bundle(root, invocation_ids=["resnet50_k00001"])

    invocation_ids = [row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]]
    scheduler_ids = {row["kernel_invocation_id"] for row in bundle["cta_scheduler_records"]}
    trace_ids = {row["kernel_invocation_id"] for row in bundle["per_warp_trace_records"]}
    assert invocation_ids == ["resnet50_k00001"]
    assert scheduler_ids == {"resnet50_k00001"}
    assert trace_ids == {"resnet50_k00001"}
    assert {row["kernel_id"] for row in bundle["kernel_invocation_table"]} == {17}
    assert bundle["kernel_invocation_table"][0]["launch_order"] == 1


def test_gate1_canonical_invocation_ids_filter_legacy_scheduler_repeated_kernel(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "formal_canonical_id_legacy_scheduler"
    )
    _write_formal_gate0_manifest(root)
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    for invocation in scheduler["kernel_invocations"]:
        invocation.pop("launch_order", None)
        invocation.pop("kernel_invocation_id", None)
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    _write_formal_gate0_manifest(root)

    bundle = build_resnet50_trace_adapter_bundle(root, invocation_ids=["resnet50_k00001"])

    invocation_ids = [row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]]
    scheduler_ids = {row["kernel_invocation_id"] for row in bundle["cta_scheduler_records"]}
    trace_ids = {row["kernel_invocation_id"] for row in bundle["per_warp_trace_records"]}
    assert invocation_ids == ["resnet50_k00001"]
    assert scheduler_ids == {"resnet50_k00001"}
    assert trace_ids == {"resnet50_k00001"}
    assert {record["cta_id"] for record in bundle["cta_scheduler_records"]} == {
        "0,0,0",
        "1,0,0",
    }


def test_gate1_invocation_limit_does_not_widen_repeated_kernel_fallback_scheduler(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "formal_limit_fallback")
    _write_formal_gate0_manifest(root)
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    for invocation in scheduler["kernel_invocations"]:
        invocation.pop("launch_order", None)
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    _write_formal_gate0_manifest(root)

    bundle = build_resnet50_trace_adapter_bundle(root, invocation_limit=1)

    invocation_ids = {row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]}
    scheduler_ids = {row["kernel_invocation_id"] for row in bundle["cta_scheduler_records"]}
    trace_ids = {row["kernel_invocation_id"] for row in bundle["per_warp_trace_records"]}
    assert invocation_ids == {"resnet50_k00000"}
    assert scheduler_ids == {"resnet50_k00000"}
    assert trace_ids == {"resnet50_k00000"}
    assert len(bundle["cta_scheduler_records"]) == 2
    assert len(bundle["per_warp_trace_records"]) == 2


def test_gate1_invocation_ids_preserve_explicit_scheduler_ids_without_launch_order(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "formal_explicit_scheduler_without_launch_order"
    )
    _write_formal_gate0_manifest(root)
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    for index, invocation in enumerate(scheduler["kernel_invocations"]):
        invocation.pop("launch_order", None)
        invocation["kernel_invocation_id"] = f"scheduler-explicit-{index}"
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    _write_formal_gate0_manifest(root)

    bundle = build_resnet50_trace_adapter_bundle(root, invocation_ids=["resnet50_k00001"])

    assert [row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]] == [
        "resnet50_k00001"
    ]
    assert {row["kernel_invocation_id"] for row in bundle["cta_scheduler_records"]} == {
        "resnet50_k00001"
    }
    assert {row["kernel_invocation_id"] for row in bundle["per_warp_trace_records"]} == {
        "resnet50_k00001"
    }


def test_gate1_invocation_limit_filters_reordered_scheduler_by_selected_invocation_id(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "formal_limit_reordered_scheduler")
    _write_formal_gate0_manifest(root)
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    scheduler["kernel_invocations"] = list(reversed(scheduler["kernel_invocations"]))
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    _write_formal_gate0_manifest(root)

    bundle = build_resnet50_trace_adapter_bundle(root, invocation_limit=1)

    invocation_ids = {row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]}
    scheduler_ids = {row["kernel_invocation_id"] for row in bundle["cta_scheduler_records"]}
    trace_ids = {row["kernel_invocation_id"] for row in bundle["per_warp_trace_records"]}
    assert invocation_ids == {"resnet50_k00000"}
    assert scheduler_ids == {"resnet50_k00000"}
    assert trace_ids == {"resnet50_k00000"}
    assert len(bundle["cta_scheduler_records"]) == 2
    assert len(bundle["per_warp_trace_records"]) == 2


def test_gate1_invocation_limit_rejects_reordered_legacy_scheduler_without_identity(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "formal_limit_reordered_legacy_scheduler"
    )
    _write_formal_gate0_manifest(root)
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    for invocation in scheduler["kernel_invocations"]:
        invocation.pop("launch_order", None)
        invocation.pop("kernel_invocation_id", None)
    scheduler["kernel_invocations"] = list(reversed(scheduler["kernel_invocations"]))
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    _write_formal_gate0_manifest(root)

    with pytest.raises(ValueError, match="legacy scheduler metadata lacks stable invocation identity"):
        build_resnet50_trace_adapter_bundle(root, invocation_limit=1)


def test_gate1_full_legacy_scheduler_preserves_repeated_kernel_order(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "formal_full_legacy_scheduler"
    )
    _write_formal_gate0_manifest(root)
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    for invocation in scheduler["kernel_invocations"]:
        invocation.pop("launch_order", None)
        invocation.pop("kernel_invocation_id", None)
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    _write_formal_gate0_manifest(root)

    bundle = build_resnet50_trace_adapter_bundle(root)

    invocation_ids = [row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]]
    scheduler_ids = [row["kernel_invocation_id"] for row in bundle["cta_scheduler_records"]]
    assert invocation_ids == ["resnet50_k00000", "resnet50_k00001"]
    assert scheduler_ids == [
        "resnet50_k00000",
        "resnet50_k00000",
        "resnet50_k00001",
        "resnet50_k00001",
    ]


def test_gate1_legacy_repeated_kernel_sm_selection_is_per_launch(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "formal_legacy_scheduler_sm_selection"
    )
    _write_formal_gate0_manifest(root)
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    for invocation_index, invocation in enumerate(scheduler["kernel_invocations"]):
        invocation.pop("launch_order", None)
        invocation.pop("kernel_invocation_id", None)
        selected_sm = 1 if invocation_index == 0 else 2
        for cta in invocation["cta_records"]:
            cta["sm_id"] = selected_sm
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    _write_formal_gate0_manifest(root)

    bundle = build_resnet50_trace_adapter_bundle(root)

    trace_ctas_by_invocation = {}
    for record in bundle["per_warp_trace_records"]:
        trace_ctas_by_invocation.setdefault(record["kernel_invocation_id"], set()).add(
            record["cta_id"]
        )
    assert trace_ctas_by_invocation == {
        "resnet50_k00000": {"0,0,0", "1,0,0"},
        "resnet50_k00001": {"0,0,0", "1,0,0"},
    }


def test_gate1_legacy_invocation_alias_rejects_repeated_kernel_ambiguity(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "formal_ambiguous_legacy_alias"
    )
    _write_formal_gate0_manifest(root)

    with pytest.raises(ValueError, match="ambiguous legacy invocation_id"):
        build_resnet50_trace_adapter_bundle(root, invocation_ids=["d_0_s_0_k_17"])


def test_gate1_legacy_invocation_alias_selects_unique_launch(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "formal_legacy_alias")
    trace = trace_pb2.Trace()
    trace.ParseFromString((root / "dynamic_trace.pb").read_bytes())
    trace.gpu_device[0].streams[0].kernels[1].id = 18
    (root / "dynamic_trace.pb").write_bytes(trace.SerializeToString())
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    scheduler["kernel_invocations"][1]["kernel_id"] = 18
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    _write_formal_gate0_manifest(root)

    bundle = build_resnet50_trace_adapter_bundle(root, invocation_ids=["d_0_s_0_k_18"])

    invocation_ids = [row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]]
    scheduler_ids = {row["kernel_invocation_id"] for row in bundle["cta_scheduler_records"]}
    trace_ids = {row["kernel_invocation_id"] for row in bundle["per_warp_trace_records"]}
    assert invocation_ids == ["resnet50_k00001"]
    assert scheduler_ids == {"resnet50_k00001"}
    assert trace_ids == {"resnet50_k00001"}


def test_gate1_threadblock_fallback_path_uses_launch_order_not_kernel_id(tmp_path):
    root = write_minimal_artifact_shape_resnet50_root(tmp_path / "formal_threadblock_fallback")
    scheduler_path = root / "scheduler_metadata.json"
    scheduler = json.loads(scheduler_path.read_text())
    for invocation in scheduler["kernel_invocations"]:
        invocation["device_id"] = 0
        invocation["stream_id"] = 0
        for cta in invocation["cta_records"]:
            cta.pop("threadblock_pb", None)
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    _write_formal_gate0_manifest(root)

    bundle = build_resnet50_trace_adapter_bundle(root, invocation_ids=["resnet50_k00001"])

    assert {row["kernel_invocation_id"] for row in bundle["per_warp_trace_records"]} == {
        "resnet50_k00001"
    }
    assert {row["function_unique_id"] for row in bundle["kernel_invocation_table"]} == {1702}
    assert [entry["pc"] for entry in bundle["per_warp_trace_records"][0]["entries"]] == [
        5120,
        5124,
        5128,
        5132,
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
    require_formal_root()
    bundle = build_resnet50_trace_adapter_bundle(FORMAL_ROOT)

    validate_resnet50_trace_adapter_bundle(bundle)
    assert bundle["artifact_status"] == "formal"
    assert bundle["formal_input_eligible"] is True
    assert bundle["trace_source"] == "nvbit"
    assert bundle["input_scope"] == "full_resnet50_inference_trace"
    assert bundle["scheduler_metadata_source"] == "real_nvbit_smid"
    assert bundle["kernel_invocation_table"]
    assert bundle["static_instruction_table"]
    assert bundle["cta_scheduler_records"]
    assert bundle["per_warp_trace_records"]
    invocation_ids = {row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]}
    assert "resnet50_k00000" in invocation_ids
    assert all(
        record["kernel_invocation_id"] in invocation_ids
        for record in bundle["cta_scheduler_records"]
    )
    assert all(
        record["kernel_invocation_id"] in invocation_ids
        for record in bundle["per_warp_trace_records"]
    )
    assert all("entries" in record for record in bundle["per_warp_trace_records"])


def test_gate1_invocation_limit_bounds_real_root_materialization_before_threadblock_reads():
    require_formal_root()
    bundle = build_resnet50_trace_adapter_bundle(FORMAL_ROOT, invocation_limit=1)

    validate_resnet50_trace_adapter_bundle(bundle)
    assert bundle["input_scope"] == "bounded_resnet50_invocation_slice"
    assert len(bundle["kernel_invocation_table"]) == 1
    kept_id = bundle["kernel_invocation_table"][0]["kernel_invocation_id"]
    assert {row["kernel_invocation_id"] for row in bundle["cta_scheduler_records"]} == {kept_id}
    assert {row["kernel_invocation_id"] for row in bundle["per_warp_trace_records"]} == {kept_id}
    assert bundle["adapter_validation_report"]["formal_replay_invocation_limit"] == 1
    assert bundle["adapter_validation_report"]["trace_materialization_scope"] == (
        "representative_sm_all_ctas"
    )


def test_gate1_invocation_ids_mark_real_root_adapter_as_bounded_slice():
    require_formal_root()
    selected_ids = ["d_0_s_0_k_267", "d_0_s_0_k_272"]

    bundle = build_resnet50_trace_adapter_bundle(FORMAL_ROOT, invocation_ids=selected_ids)

    validate_resnet50_trace_adapter_bundle(bundle)
    assert bundle["input_scope"] == "bounded_resnet50_invocation_slice"
    assert [
        row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]
    ] == ["resnet50_k00000", "resnet50_k00005"]
    assert bundle["adapter_validation_report"]["formal_replay_invocation_ids"] == selected_ids
