from pathlib import Path

from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.trace_scope import validate_phase_b_trace_manifest

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def test_gate2_builds_phase_b_manifest_from_resnet50_bundle():
    bundle = build_resnet50_trace_adapter_bundle(FIXTURE_ROOT)

    manifest, reports, preview = build_representative_sm_manifest_from_bundle(bundle)

    validate_phase_b_trace_manifest(manifest)
    assert manifest["artifact_type"] == "gcl_phase_b_trace_manifest"
    assert manifest["collection_scope"] == "single_representative_sm_all_ctas"
    assert reports["reports"]
    assert preview["kernel_invocation_count"] == len(manifest["kernel_invocations"])


def test_gate2_manifest_includes_selected_sm_all_ctas_only():
    bundle = build_resnet50_trace_adapter_bundle(FIXTURE_ROOT)

    manifest, _reports, _preview = build_representative_sm_manifest_from_bundle(bundle)

    for invocation in manifest["kernel_invocations"]:
        selected_sm = invocation["selected_sm"]
        expected = set(invocation["scheduler_metadata_by_sm"][str(selected_sm)]["cta_ids"])
        assert set(invocation["included_cta_ids"]) == expected
        assert all(invocation["cta_to_sm"][cta_id] == selected_sm for cta_id in expected)
        scoped_entries = [
            entry
            for entry in invocation["all_trace_entries"]
            if entry["cta_id"] in invocation["included_cta_ids"]
        ]
        assert invocation["instruction_count"] == len(scoped_entries)


def test_gate2_uses_scheduler_signature_medoid_policy():
    bundle = build_resnet50_trace_adapter_bundle(FIXTURE_ROOT)

    manifest, reports, _preview = build_representative_sm_manifest_from_bundle(bundle)

    assert len(reports["reports"]) == len(manifest["kernel_invocations"])
    for invocation, report in zip(manifest["kernel_invocations"], reports["reports"]):
        assert report["selected_sm_policy"] == "scheduler_signature_medoid_sm"
        assert invocation["selected_sm_policy_report_hash"] == report["selection_hash"]
        assert invocation["selected_sm"] == report["selected_sm"]
