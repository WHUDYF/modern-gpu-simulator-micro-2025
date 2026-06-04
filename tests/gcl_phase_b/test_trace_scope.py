import copy

import pytest

from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.trace_scope import (
    build_phase_b_trace_records,
    build_scope_audit,
    validate_phase_b_trace_manifest,
    validate_scope_audit,
)


def test_trace_manifest_records_single_representative_sm_all_ctas():
    manifest = build_representative_sm_trace_manifest()

    validate_phase_b_trace_manifest(manifest)

    invocation = manifest["kernel_invocations"][0]
    assert invocation["collection_scope"] == "single_representative_sm_all_ctas"
    assert invocation["selected_sm_policy"] == "scheduler_signature_medoid_sm"
    assert invocation["selected_sm"] == 1
    assert set(invocation["included_cta_ids"]) == {"cta_2", "cta_3"}
    for cta_id in invocation["included_cta_ids"]:
        assert invocation["cta_to_sm"][cta_id] == invocation["selected_sm"]
    assert "selected_sm_policy_report_hash" in invocation


def test_trace_manifest_rejects_non_phase_b_scope_and_missing_report():
    manifest = build_representative_sm_trace_manifest()
    bad_scope = copy.deepcopy(manifest)
    bad_scope["kernel_invocations"][0]["collection_scope"] = "selected_warps_fixture"
    with pytest.raises(ValueError, match="collection_scope"):
        validate_phase_b_trace_manifest(bad_scope)

    missing_report = copy.deepcopy(manifest)
    del missing_report["kernel_invocations"][0]["selected_sm_policy_report_hash"]
    with pytest.raises(ValueError, match="selected_sm_policy_report_hash"):
        validate_phase_b_trace_manifest(missing_report)


def test_scope_audit_records_before_and_after_counts():
    manifest = build_representative_sm_trace_manifest()
    invocation = manifest["kernel_invocations"][0]

    audit = build_scope_audit(invocation)

    validate_scope_audit(audit, invocation)
    assert audit["scope_policy"] == "single_representative_sm_all_ctas"
    assert audit["instruction_count_after_scope"] == invocation["instruction_count"]
    assert audit["warp_count_after_scope"] == invocation["warp_count"]
    assert audit["before_scope_counts_available"] is True
    assert audit["trace_scope_hash"]


def test_scope_audit_rejects_fake_unavailable_before_counts():
    manifest = build_representative_sm_trace_manifest()
    invocation = manifest["kernel_invocations"][0]
    audit = build_scope_audit(invocation)
    audit["before_scope_counts_available"] = False
    audit["instruction_count_before_scope"] = 0
    audit["missing_before_scope_reason"] = ""

    with pytest.raises(ValueError, match="missing_before_scope_reason"):
        validate_scope_audit(audit, invocation)


def test_build_phase_b_trace_records_keeps_all_selected_sm_ctas():
    manifest = build_representative_sm_trace_manifest()

    records = build_phase_b_trace_records(manifest)

    assert len(records) == 1
    record = records[0]
    assert record["collection_scope"] == "single_representative_sm_all_ctas"
    assert {entry["cta_id"] for warp in record["warps"] for entry in warp["entries"]} == {
        "cta_2",
        "cta_3",
    }
