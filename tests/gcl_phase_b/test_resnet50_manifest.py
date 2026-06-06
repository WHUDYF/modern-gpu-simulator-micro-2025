from pathlib import Path
import shutil

from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_gate0 import record_resnet50_gate0_trace_acquisition
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.trace_scope import validate_phase_b_trace_manifest
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


def test_gate2_builds_phase_b_manifest_from_resnet50_bundle(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))

    manifest, reports, preview = build_representative_sm_manifest_from_bundle(bundle)

    validate_phase_b_trace_manifest(manifest)
    assert manifest["artifact_type"] == "gcl_phase_b_trace_manifest"
    assert manifest["collection_scope"] == "single_representative_sm_all_ctas"
    assert reports["reports"]
    assert reports["source_adapter_bundle_hash"] == bundle["adapter_bundle_hash"]
    assert preview["source_adapter_bundle_hash"] == bundle["adapter_bundle_hash"]
    assert preview["kernel_invocation_count"] == len(manifest["kernel_invocations"])
    assert len(preview["invocations"]) == len(manifest["kernel_invocations"])


def test_gate2_manifest_includes_selected_sm_all_ctas_only(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))

    manifest, _reports, _preview = build_representative_sm_manifest_from_bundle(bundle)

    for invocation in manifest["kernel_invocations"]:
        selected_sm = invocation["selected_sm"]
        expected = set(invocation["scheduler_metadata_by_sm"][str(selected_sm)]["cta_ids"])
        assert set(invocation["included_cta_ids"]) == expected
        assert len(invocation["included_cta_ids"]) >= 2
        assert all(invocation["cta_to_sm"][cta_id] == selected_sm for cta_id in expected)
        scoped_entries = [
            entry
            for entry in invocation["all_trace_entries"]
            if entry["cta_id"] in invocation["included_cta_ids"]
        ]
        assert invocation["instruction_count"] == len(scoped_entries)


def test_gate2_uses_scheduler_signature_medoid_policy(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))

    manifest, reports, _preview = build_representative_sm_manifest_from_bundle(bundle)

    assert len(reports["reports"]) == len(manifest["kernel_invocations"])
    for invocation, report in zip(manifest["kernel_invocations"], reports["reports"]):
        assert report["selected_sm_policy"] == "scheduler_signature_medoid_sm"
        assert invocation["selected_sm_policy_report_hash"] == report["selection_hash"]
        assert invocation["selected_sm"] == report["selected_sm"]


def test_gate2_groups_repeated_kernel_id_by_kernel_invocation_id(tmp_path):
    bundle = build_resnet50_trace_adapter_bundle(_formal_gate0_root(tmp_path))
    repeated = dict(bundle["kernel_invocation_table"][1])
    repeated["kernel_id"] = bundle["kernel_invocation_table"][0]["kernel_id"]
    bundle["kernel_invocation_table"][1] = repeated
    for record in bundle["cta_scheduler_records"]:
        if record["kernel_invocation_id"] == repeated["kernel_invocation_id"]:
            record["kernel_id"] = repeated["kernel_id"]
    for record in bundle["per_warp_trace_records"]:
        if record["kernel_invocation_id"] == repeated["kernel_invocation_id"]:
            record["kernel_id"] = repeated["kernel_id"]
    bundle["adapter_bundle_hash"] = hash_without(bundle, "adapter_bundle_hash")

    manifest, reports, _preview = build_representative_sm_manifest_from_bundle(bundle)

    invocation_ids = [row["kernel_invocation_id"] for row in manifest["kernel_invocations"]]
    assert invocation_ids == ["resnet50_k00000", "resnet50_k00001"]
    assert len(reports["reports"]) == 2
