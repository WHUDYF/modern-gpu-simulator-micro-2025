import pytest
from pathlib import Path

from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_debug_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from tests.gcl_resnet50.formal_chain import build_artifact_shape_trace_manifest
from tests.gcl_resnet50.real_chain import build_real_trace_manifest

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def test_gate2_rejects_debug_adapter_bundle_as_formal_manifest_source():
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)

    manifest, _reports, _preview = build_representative_sm_manifest_from_bundle(bundle)

    assert manifest["artifact_status"] == "debug_not_formal"
    assert manifest["formal_input_eligible"] is False
    assert manifest["trace_source"] == "fixture"


def test_gate2_rejects_random_or_file_order_policy_in_formal_manifest():
    bundle = build_resnet50_debug_trace_adapter_bundle(FIXTURE_ROOT)
    bundle["artifact_status"] = "formal"
    bundle["formal_input_eligible"] = True
    bundle["execution_mode"] = "real_trace"
    bundle["trace_source"] = "nvbit"
    bundle["input_scope"] = "full_resnet50_inference_trace"
    bundle["source_gate0_manifest_hash"] = "unit-test-gate0"
    bundle["adapter_validation_report"]["status"] = "passed"
    bundle["adapter_bundle_hash"] = "stale"

    with pytest.raises(ValueError, match="adapter_bundle_hash"):
        build_representative_sm_manifest_from_bundle(bundle)


def test_gate2_builds_artifact_shape_representative_sm_manifest_without_formal_claim(tmp_path):
    _bundle, manifest, reports, preview = build_artifact_shape_trace_manifest(tmp_path)

    assert manifest["artifact_status"] == "debug_not_formal"
    assert manifest["formal_input_eligible"] is False
    assert manifest["trace_source"] == "synthetic_protobuf_artifact_shape"
    assert reports["reports"]
    assert preview["invocations"]
    for invocation in manifest["kernel_invocations"]:
        assert invocation["selected_sm_policy"] == "scheduler_signature_medoid_sm"
        assert invocation["included_cta_ids"] == [str(invocation["selected_sm"] - 1) + ",0,0"]


def test_gate2_selects_representative_sm_from_real_resnet50_scheduler_metadata():
    bundle, manifest, reports, preview = build_real_trace_manifest()

    assert bundle["artifact_status"] == "formal"
    assert manifest["artifact_status"] == "formal"
    assert manifest["formal_input_eligible"] is True
    assert manifest["trace_source"] == "nvbit"
    assert reports["reports"]
    assert preview["invocations"]
    for invocation, report in zip(manifest["kernel_invocations"], reports["reports"]):
        assert invocation["selected_sm_policy"] == "scheduler_signature_medoid_sm"
        assert invocation["selected_sm"] == report["selected_sm"]
        assert invocation["candidate_sm_count"] == report["candidate_sm_count"]
        selected_sm = invocation["selected_sm"]
        expected_ctas = set(
            invocation["scheduler_metadata_by_sm"][str(selected_sm)]["cta_ids"]
        )
        assert set(invocation["included_cta_ids"]) == expected_ctas
        assert all(invocation["cta_to_sm"][cta_id] == selected_sm for cta_id in expected_ctas)
