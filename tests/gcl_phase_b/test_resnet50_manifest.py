from pathlib import Path

from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_debug_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.trace_scope import validate_phase_b_trace_manifest
from experiments.gcl_phase_b.utils import hash_without

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def test_gate2_builds_debug_phase_b_manifest_from_resnet50_bundle():
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)

    manifest, reports, preview = build_representative_sm_manifest_from_bundle(bundle)

    validate_phase_b_trace_manifest(manifest)
    assert manifest["artifact_type"] == "gcl_phase_b_trace_manifest"
    assert manifest["collection_scope"] == "single_representative_sm_all_ctas"
    assert reports["reports"]
    assert reports["source_adapter_bundle_hash"] == bundle["adapter_bundle_hash"]
    assert preview["source_adapter_bundle_hash"] == bundle["adapter_bundle_hash"]
    assert preview["kernel_invocation_count"] == len(manifest["kernel_invocations"])
    assert len(preview["invocations"]) == len(manifest["kernel_invocations"])


def test_gate2_debug_manifest_includes_selected_sm_all_ctas_only():
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)

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


def test_gate2_debug_uses_scheduler_signature_medoid_policy():
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)

    manifest, reports, _preview = build_representative_sm_manifest_from_bundle(bundle)

    assert len(reports["reports"]) == len(manifest["kernel_invocations"])
    for invocation, report in zip(manifest["kernel_invocations"], reports["reports"]):
        assert report["selected_sm_policy"] == "scheduler_signature_medoid_sm"
        assert invocation["selected_sm_policy_report_hash"] == report["selection_hash"]
        assert invocation["selected_sm"] == report["selected_sm"]


def test_gate2_debug_groups_repeated_kernel_id_by_kernel_invocation_id():
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)
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


def test_gate2_preserves_adapter_launch_order_for_formal_invocation_ids():
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)
    original_ids = [
        row["kernel_invocation_id"] for row in bundle["kernel_invocation_table"]
    ]
    id_map = {
        original_ids[0]: "d_0_s_0_k_1",
        original_ids[1]: "d_0_s_0_k_2",
    }
    third = dict(bundle["kernel_invocation_table"][1])
    third["kernel_invocation_id"] = "d_0_s_0_k_10"
    third["kernel_id"] = 19
    third["launch_order"] = 10
    for row in bundle["kernel_invocation_table"]:
        row["kernel_invocation_id"] = id_map[row["kernel_invocation_id"]]
    bundle["kernel_invocation_table"].append(third)
    for record in bundle["cta_scheduler_records"]:
        record["kernel_invocation_id"] = id_map[record["kernel_invocation_id"]]
    for record in bundle["per_warp_trace_records"]:
        record["kernel_invocation_id"] = id_map[record["kernel_invocation_id"]]
    template_scheduler = [
        dict(record)
        for record in bundle["cta_scheduler_records"]
        if record["kernel_invocation_id"] == "d_0_s_0_k_2"
    ]
    template_traces = [
        dict(record)
        for record in bundle["per_warp_trace_records"]
        if record["kernel_invocation_id"] == "d_0_s_0_k_2"
    ]
    for record in template_scheduler:
        record["kernel_invocation_id"] = "d_0_s_0_k_10"
        record["kernel_id"] = 19
    for record in template_traces:
        record["kernel_invocation_id"] = "d_0_s_0_k_10"
        record["kernel_id"] = 19
    bundle["cta_scheduler_records"].extend(template_scheduler)
    bundle["per_warp_trace_records"].extend(template_traces)
    bundle["adapter_bundle_hash"] = hash_without(bundle, "adapter_bundle_hash")

    manifest, _reports, preview = build_representative_sm_manifest_from_bundle(bundle)

    invocation_ids = [row["kernel_invocation_id"] for row in manifest["kernel_invocations"]]
    preview_ids = [row["kernel_invocation_id"] for row in preview["invocations"]]
    assert invocation_ids == ["d_0_s_0_k_1", "d_0_s_0_k_2", "d_0_s_0_k_10"]
    assert preview_ids == invocation_ids
