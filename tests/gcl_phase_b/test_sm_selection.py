import copy

import pytest

from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.sm_selection import (
    select_representative_sm,
    validate_selected_sm_policy_report,
)


def test_scheduler_signature_medoid_selects_nearest_sm():
    manifest = build_representative_sm_trace_manifest()
    invocation = manifest["kernel_invocations"][0]

    report = select_representative_sm(invocation, policy="scheduler_signature_medoid_sm")

    validate_selected_sm_policy_report(report)
    assert report["selected_sm_policy"] == "scheduler_signature_medoid_sm"
    assert report["selected_sm"] == 1
    assert report["tie_break_rule"] == "lowest_sm_id"
    assert report["distance_to_global_signature_by_sm"]["1"] == min(
        report["distance_to_global_signature_by_sm"].values()
    )


def test_scheduler_signature_medoid_ties_by_lowest_sm_id():
    manifest = build_representative_sm_trace_manifest()
    invocation = copy.deepcopy(manifest["kernel_invocations"][0])
    invocation["scheduler_metadata_by_sm"]["2"] = copy.deepcopy(
        invocation["scheduler_metadata_by_sm"]["1"]
    )
    invocation["scheduler_metadata_by_sm"]["2"]["sm_id"] = 2

    report = select_representative_sm(invocation, policy="scheduler_signature_medoid_sm")

    assert report["selected_sm"] == 1
    assert report["tie_break_rule"] == "lowest_sm_id"


def test_scheduler_signature_medoid_rejects_missing_order_metadata():
    manifest = build_representative_sm_trace_manifest()
    invocation = copy.deepcopy(manifest["kernel_invocations"][0])
    del invocation["scheduler_metadata_by_sm"]["1"]["cta_start_order"]

    with pytest.raises(ValueError, match="cta_start_order"):
        select_representative_sm(invocation, policy="scheduler_signature_medoid_sm")


def test_selected_sm_policy_report_is_hashable_and_complete():
    manifest = build_representative_sm_trace_manifest()
    invocation = manifest["kernel_invocations"][0]
    report = select_representative_sm(invocation, policy="scheduler_signature_medoid_sm")
    repeated = select_representative_sm(invocation, policy="scheduler_signature_medoid_sm")

    validate_selected_sm_policy_report(report)
    assert report["selection_hash"] == repeated["selection_hash"]
    assert {
        "selected_sm_policy",
        "selected_sm_policy_version",
        "candidate_sm_ids",
        "signature_fields",
        "signature_field_weights",
        "raw_signature_by_sm",
        "normalized_signature_by_sm",
        "global_sm_signature",
        "distance_metric",
        "distance_to_global_signature_by_sm",
        "tie_break_rule",
        "instruction_count_proxy_source",
        "selection_hash",
    }.issubset(report)


def test_selected_sm_policy_report_rejects_inconsistent_selection():
    manifest = build_representative_sm_trace_manifest()
    invocation = manifest["kernel_invocations"][0]
    report = select_representative_sm(invocation, policy="scheduler_signature_medoid_sm")
    report["selected_sm"] = 0

    with pytest.raises(ValueError, match="selected_sm"):
        validate_selected_sm_policy_report(report)
