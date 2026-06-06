import pytest
from pathlib import Path

from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_debug_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from tests.gcl_resnet50.formal_chain import build_formal_trace_manifest

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


def test_gate2_builds_formal_representative_sm_manifest_from_gate1_bundle(tmp_path):
    _bundle, manifest, reports, preview = build_formal_trace_manifest(tmp_path)

    assert manifest["artifact_status"] == "formal"
    assert manifest["formal_input_eligible"] is True
    assert manifest["trace_source"] == "nvbit"
    assert reports["reports"]
    assert preview["invocations"]
    for invocation in manifest["kernel_invocations"]:
        assert invocation["selected_sm_policy"] == "scheduler_signature_medoid_sm"
        assert invocation["included_cta_ids"] == [str(invocation["selected_sm"] - 1) + ",0,0"]
