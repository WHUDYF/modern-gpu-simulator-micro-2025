import copy

import pytest

from experiments.gcl_phase_a.trace_fixture import (
    COLLECTION_SCOPE,
    REQUIRED_ENTRY_FIELDS,
    TRACE_FAMILIES,
    build_controlled_trace_fixture,
    iter_trace_entries,
    validate_trace_fixture,
)
from experiments.gcl_phase_a.utils import hash_without


def test_fixture_has_expected_size():
    fixture = build_controlled_trace_fixture()

    validate_trace_fixture(fixture)

    assert fixture["kernel_invocation_count"] == 12
    assert fixture["trace_family_count"] == 3
    assert len(fixture["records"]) == 12
    for record in fixture["records"]:
        assert record["collection_scope"] == COLLECTION_SCOPE
        assert len(record["warps"]) == 2
        for warp in record["warps"]:
            assert len(warp["entries"]) == 6
            for entry in warp["entries"]:
                assert REQUIRED_ENTRY_FIELDS.issubset(entry)

    assert len(list(iter_trace_entries(fixture))) == 144


def test_fixture_matches_phase_a_family_and_invocation_spec():
    fixture = build_controlled_trace_fixture()

    assert [record["kernel_invocation_id"] for record in fixture["records"]] == [
        f"gcl_pa_k{index:03d}" for index in range(12)
    ]
    assert tuple(record["trace_family"] for record in fixture["records"][0:4]) == (TRACE_FAMILIES[0],) * 4
    assert tuple(record["trace_family"] for record in fixture["records"][4:8]) == (TRACE_FAMILIES[1],) * 4
    assert tuple(record["trace_family"] for record in fixture["records"][8:12]) == (TRACE_FAMILIES[2],) * 4
    assert [entry["opcode"] for entry in fixture["records"][0]["warps"][0]["entries"]] == [
        "MOV",
        "IMAD.WIDE",
        "LDG.E.64.SYS",
        "FADD",
        "STG.E.64.SYS",
        "EXIT",
    ]


def test_fixture_variable_entries_have_minimum_dynamic_value_samples():
    fixture = build_controlled_trace_fixture()

    for _, _, entry in iter_trace_entries(fixture):
        if entry["source_operands"] or entry["destination_operands"]:
            assert len(entry["observed_dynamic_values"]) >= 4


def test_fixture_validator_rejects_missing_required_entry_field():
    fixture = build_controlled_trace_fixture()
    del fixture["records"][0]["warps"][0]["entries"][0]["opcode"]

    with pytest.raises(ValueError, match="missing required fields"):
        validate_trace_fixture(fixture)


def test_fixture_validator_rejects_underpopulated_dynamic_value_samples():
    fixture = copy.deepcopy(build_controlled_trace_fixture())
    entry = fixture["records"][0]["warps"][0]["entries"][0]
    entry["observed_dynamic_values"] = [1.0, 2.0, 3.0]
    entry["source_entry_hash"] = hash_without(entry, "source_entry_hash")
    fixture["fixture_hash"] = hash_without(fixture, "fixture_hash")

    with pytest.raises(ValueError, match="four observed dynamic values"):
        validate_trace_fixture(fixture)


def test_fixture_validator_rejects_wrong_collection_scope():
    fixture = copy.deepcopy(build_controlled_trace_fixture())
    fixture["records"][0]["warps"][0]["entries"][0]["collection_scope"] = "full_gpu_trace"

    with pytest.raises(ValueError, match="collection_scope"):
        validate_trace_fixture(fixture)
