import copy

import pytest

from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.utils import hash_without
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
    bad_scope["trace_manifest_hash"] = hash_without(bad_scope, "trace_manifest_hash")
    with pytest.raises(ValueError, match="collection_scope"):
        validate_phase_b_trace_manifest(bad_scope)

    missing_report = copy.deepcopy(manifest)
    del missing_report["kernel_invocations"][0]["selected_sm_policy_report_hash"]
    missing_report["trace_manifest_hash"] = hash_without(missing_report, "trace_manifest_hash")
    with pytest.raises(ValueError, match="selected_sm_policy_report_hash"):
        validate_phase_b_trace_manifest(missing_report)


def test_trace_manifest_rejects_missing_downstream_identity_fields():
    manifest = build_representative_sm_trace_manifest()
    missing_invocation_id = copy.deepcopy(manifest)
    del missing_invocation_id["kernel_invocations"][0]["kernel_invocation_id"]
    missing_invocation_id["kernel_invocations"][0]["trace_hash"] = hash_without(
        missing_invocation_id["kernel_invocations"][0], "trace_hash"
    )
    missing_invocation_id["trace_manifest_hash"] = hash_without(
        missing_invocation_id, "trace_manifest_hash"
    )
    with pytest.raises(ValueError, match="kernel_invocation_id"):
        validate_phase_b_trace_manifest(missing_invocation_id)

    missing_trace_family = copy.deepcopy(manifest)
    del missing_trace_family["kernel_invocations"][0]["trace_family"]
    missing_trace_family["kernel_invocations"][0]["trace_hash"] = hash_without(
        missing_trace_family["kernel_invocations"][0], "trace_hash"
    )
    missing_trace_family["trace_manifest_hash"] = hash_without(
        missing_trace_family, "trace_manifest_hash"
    )
    with pytest.raises(ValueError, match="trace_family"):
        validate_phase_b_trace_manifest(missing_trace_family)


def test_trace_manifest_rejects_missing_cta_mapping_metadata():
    manifest = build_representative_sm_trace_manifest()
    missing_cta_map = copy.deepcopy(manifest)
    del missing_cta_map["kernel_invocations"][0]["cta_to_sm"]
    missing_cta_map["kernel_invocations"][0]["trace_hash"] = hash_without(
        missing_cta_map["kernel_invocations"][0], "trace_hash"
    )
    missing_cta_map["trace_manifest_hash"] = hash_without(missing_cta_map, "trace_manifest_hash")
    with pytest.raises(ValueError, match="cta_to_sm"):
        validate_phase_b_trace_manifest(missing_cta_map)

    missing_scheduler = copy.deepcopy(manifest)
    del missing_scheduler["kernel_invocations"][0]["scheduler_metadata_by_sm"]
    missing_scheduler["kernel_invocations"][0]["trace_hash"] = hash_without(
        missing_scheduler["kernel_invocations"][0], "trace_hash"
    )
    missing_scheduler["trace_manifest_hash"] = hash_without(missing_scheduler, "trace_manifest_hash")
    with pytest.raises(ValueError, match="scheduler_metadata_by_sm"):
        validate_phase_b_trace_manifest(missing_scheduler)


def test_trace_manifest_rejects_inconsistent_scoped_counts_and_hash():
    manifest = build_representative_sm_trace_manifest()
    bad_count = copy.deepcopy(manifest)
    bad_count["kernel_invocations"][0]["instruction_count"] = 999
    bad_count["kernel_invocations"][0]["trace_hash"] = hash_without(
        bad_count["kernel_invocations"][0], "trace_hash"
    )
    bad_count["trace_manifest_hash"] = hash_without(bad_count, "trace_manifest_hash")

    with pytest.raises(ValueError, match="instruction_count"):
        validate_phase_b_trace_manifest(bad_count)

    bad_hash = copy.deepcopy(manifest)
    bad_hash["kernel_invocations"][0]["trace_hash"] = "stale"
    bad_hash["trace_manifest_hash"] = hash_without(bad_hash, "trace_manifest_hash")
    with pytest.raises(ValueError, match="trace_hash"):
        validate_phase_b_trace_manifest(bad_hash)


def test_trace_manifest_rejects_included_cta_without_trace_entries():
    manifest = build_representative_sm_trace_manifest()
    missing_cta_entries = copy.deepcopy(manifest)
    invocation = missing_cta_entries["kernel_invocations"][0]
    invocation["all_trace_entries"] = [
        entry for entry in invocation["all_trace_entries"] if entry["cta_id"] != "cta_3"
    ]
    scoped_entries = [
        entry for entry in invocation["all_trace_entries"] if entry["cta_id"] in invocation["included_cta_ids"]
    ]
    invocation["instruction_count"] = len(scoped_entries)
    invocation["warp_count"] = len({(entry["cta_id"], entry["warp_id"]) for entry in scoped_entries})
    invocation["instruction_count_before_scope"] = len(invocation["all_trace_entries"])
    invocation["warp_count_before_scope"] = len(
        {(entry["cta_id"], entry["warp_id"]) for entry in invocation["all_trace_entries"]}
    )
    invocation["trace_hash"] = hash_without(invocation, "trace_hash")
    missing_cta_entries["trace_manifest_hash"] = hash_without(
        missing_cta_entries, "trace_manifest_hash"
    )

    with pytest.raises(ValueError, match="CTA trace entries"):
        validate_phase_b_trace_manifest(missing_cta_entries)


def test_trace_manifest_rejects_included_cta_without_expected_warp_entries():
    manifest = build_representative_sm_trace_manifest()
    missing_warp_entries = copy.deepcopy(manifest)
    invocation = missing_warp_entries["kernel_invocations"][0]
    invocation["all_trace_entries"] = [
        entry
        for entry in invocation["all_trace_entries"]
        if not (entry["cta_id"] == "cta_3" and entry["warp_id"] == 1)
    ]
    scoped_entries = [
        entry for entry in invocation["all_trace_entries"] if entry["cta_id"] in invocation["included_cta_ids"]
    ]
    invocation["instruction_count"] = len(scoped_entries)
    invocation["warp_count"] = len({(entry["cta_id"], entry["warp_id"]) for entry in scoped_entries})
    invocation["instruction_count_before_scope"] = len(invocation["all_trace_entries"])
    invocation["warp_count_before_scope"] = len(
        {(entry["cta_id"], entry["warp_id"]) for entry in invocation["all_trace_entries"]}
    )
    invocation["trace_hash"] = hash_without(invocation, "trace_hash")
    missing_warp_entries["trace_manifest_hash"] = hash_without(
        missing_warp_entries, "trace_manifest_hash"
    )

    with pytest.raises(ValueError, match="warp trace entries"):
        validate_phase_b_trace_manifest(missing_warp_entries)


def test_trace_manifest_rejects_selected_cta_with_unexpected_warp_entries():
    manifest = build_representative_sm_trace_manifest()
    unexpected_warp_entries = copy.deepcopy(manifest)
    invocation = unexpected_warp_entries["kernel_invocations"][0]
    selected_sm = str(invocation["selected_sm"])
    invocation["scheduler_metadata_by_sm"][selected_sm]["warp_ids_by_cta"]["cta_3"] = [0]
    scoped_entries = [
        entry for entry in invocation["all_trace_entries"] if entry["cta_id"] in invocation["included_cta_ids"]
    ]
    invocation["instruction_count"] = len(scoped_entries)
    invocation["warp_count"] = len({(entry["cta_id"], entry["warp_id"]) for entry in scoped_entries})
    invocation["trace_hash"] = hash_without(invocation, "trace_hash")
    unexpected_warp_entries["trace_manifest_hash"] = hash_without(
        unexpected_warp_entries, "trace_manifest_hash"
    )

    with pytest.raises(ValueError, match="unexpected warp trace entries"):
        validate_phase_b_trace_manifest(unexpected_warp_entries)


def test_trace_manifest_rejects_stale_top_level_hash_after_invocation_rehash():
    manifest = build_representative_sm_trace_manifest()
    manifest["kernel_invocations"][0]["selected_sm_reason"] = "tampered_reason"
    manifest["kernel_invocations"][0]["trace_hash"] = hash_without(
        manifest["kernel_invocations"][0], "trace_hash"
    )

    with pytest.raises(ValueError, match="trace_manifest_hash"):
        validate_phase_b_trace_manifest(manifest)


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

    with pytest.raises(ValueError, match="instruction_count_before_scope"):
        validate_scope_audit(audit, invocation)


def test_scope_audit_rejects_after_count_mismatch_against_trace_entries():
    manifest = build_representative_sm_trace_manifest()
    invocation = copy.deepcopy(manifest["kernel_invocations"][0])
    invocation["instruction_count"] = 999
    audit = build_scope_audit(invocation)

    with pytest.raises(ValueError, match="instruction_count_after_scope"):
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


def test_build_phase_b_trace_records_orders_cta_ordinals_by_scheduler_start_order():
    manifest = build_representative_sm_trace_manifest()
    invocation = copy.deepcopy(manifest["kernel_invocations"][0])
    invocation["scheduler_metadata_by_sm"]["1"]["cta_ids"] = ["cta_10", "cta_2"]
    invocation["scheduler_metadata_by_sm"]["1"]["warp_ids_by_cta"] = {"cta_10": [0], "cta_2": [0]}
    invocation["scheduler_metadata_by_sm"]["1"]["trace_entry_count_by_cta"] = {"cta_10": 1, "cta_2": 1}
    invocation["scheduler_metadata_by_sm"]["1"]["cta_start_order"] = {"cta_10": 20, "cta_2": 10}
    invocation["scheduler_metadata_by_sm"]["1"]["cta_end_order"] = {"cta_10": 21, "cta_2": 11}
    invocation["cta_to_sm"] = {"cta_10": 1, "cta_2": 1}
    invocation["included_cta_ids"] = ["cta_10", "cta_2"]
    invocation["all_trace_entries"] = [
        {
            "kernel_invocation_id": invocation["kernel_invocation_id"],
            "trace_family": invocation["trace_family"],
            "collection_scope": invocation["collection_scope"],
            "cta_id": "cta_10",
            "warp_id": 0,
            "trace_index": 1,
            "pc": 0x2010,
            "opcode": "MOV",
            "active_mask": "0xffffffff",
            "destination_operands": ["R1"],
            "source_operands": ["input:a"],
            "observed_dynamic_values": [1.0],
        },
        {
            "kernel_invocation_id": invocation["kernel_invocation_id"],
            "trace_family": invocation["trace_family"],
            "collection_scope": invocation["collection_scope"],
            "cta_id": "cta_2",
            "warp_id": 0,
            "trace_index": 2,
            "pc": 0x2020,
            "opcode": "MOV",
            "active_mask": "0xffffffff",
            "destination_operands": ["R2"],
            "source_operands": ["input:b"],
            "observed_dynamic_values": [2.0],
        },
    ]
    invocation["instruction_count_before_scope"] = 2
    invocation["warp_count_before_scope"] = 2
    invocation["instruction_count"] = 2
    invocation["warp_count"] = 2
    invocation["trace_hash"] = hash_without(invocation, "trace_hash")
    manifest["kernel_invocations"] = [invocation]
    manifest["trace_manifest_hash"] = hash_without(manifest, "trace_manifest_hash")

    records = build_phase_b_trace_records(manifest)

    partitions_by_cta = {warp["cta_id"]: warp["warp_partition_id"] for warp in records[0]["warps"]}
    assert partitions_by_cta["cta_2"] == "1:0"
    assert partitions_by_cta["cta_10"] == "2:0"
