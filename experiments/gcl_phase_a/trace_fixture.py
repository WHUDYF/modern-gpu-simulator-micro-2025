"""Controlled trace fixture for the GCL Phase A minimum semantic path."""

from __future__ import annotations

from typing import Any

from .utils import hash_without, stable_hash

FIXTURE_NAME = "gcl_phase_a_controlled_trace_fixture_v1"
COLLECTION_SCOPE = "selected_warps_fixture"
TRACE_FAMILIES = ("mem_load_fadd_store", "integer_imad_store", "load_branch_store")
REQUIRED_ENTRY_FIELDS = {
    "kernel_invocation_id",
    "trace_family",
    "collection_scope",
    "warp_id",
    "trace_index",
    "pc",
    "opcode",
    "active_mask",
    "destination_operands",
    "source_operands",
    "observed_dynamic_values",
    "source_entry_hash",
}


def _instruction_template(trace_family: str) -> list[tuple[str, list[str], list[str], list[float]]]:
    templates = {
        "mem_load_fadd_store": [
            ("MOV", ["Rbase.v1"], ["input:cta_id"], [1.0, 2.0, 3.0, 4.0]),
            ("IMAD.WIDE", ["Raddr.v1"], ["Rbase.v1", "input:stride"], [2.0, 3.0, 4.0, 5.0]),
            ("LDG.E.64.SYS", ["Rload.v1"], ["Raddr.v1"], [3.0, 5.0, 7.0, 9.0]),
            ("FADD", ["Rsum.v1"], ["Rload.v1", "input:bias"], [7.0, 8.0, 9.0, 10.0]),
            ("STG.E.64.SYS", [], ["Raddr.v1", "Rsum.v1"], [9.0, 10.0, 11.0, 12.0]),
            ("EXIT", [], [], [0.0]),
        ],
        "integer_imad_store": [
            ("MOV", ["Rbase.v1"], ["input:thread_id"], [1.0, 1.0, 1.0, 1.0]),
            ("IMAD", ["Rmul.v1"], ["Rbase.v1", "input:scale"], [2.0, 2.0, 3.0, 3.0]),
            ("IMAD", ["Raddr.v1"], ["Rmul.v1", "input:stride"], [4.0, 6.0, 8.0, 10.0]),
            ("IADD3", ["Rvalue.v1"], ["Rmul.v1", "unknown:carry"], [6.0, 8.0, 10.0, 12.0]),
            ("STG.E.64.SYS", [], ["Raddr.v1", "Rvalue.v1"], [12.0, 16.0, 20.0, 24.0]),
            ("EXIT", [], [], [0.0]),
        ],
        "load_branch_store": [
            ("LDG.E.64.SYS", ["Rload.v1"], ["Raddr.v1"], [2.0, 5.0, 8.0, 11.0]),
            ("ISETP", ["P0.v1"], ["Rload.v1", "input:limit"], [0.0, 1.0, 1.0, 0.0]),
            ("BRA", [], ["P0.v1", "unknown:branch_target"], [0.0, 1.0, 0.0, 1.0]),
            ("FADD", ["Rsum.v1"], ["Rload.v1", "input:bias"], [1.0, 6.0, 9.0, 12.0]),
            ("STG.E.64.SYS", [], ["Raddr.v1", "Rsum.v1"], [9.0, 10.0, 11.0, 12.0]),
            ("EXIT", [], [], [0.0]),
        ],
    }
    return templates[trace_family]


def _warp_token(token: str, warp_id: int) -> str:
    if token.startswith(("R", "P", "input:", "unknown:")):
        return f"{token}.w{warp_id}"
    return token


def _entry_hash(entry: dict[str, Any]) -> str:
    return hash_without(entry, "source_entry_hash")


def build_controlled_trace_fixture(seed: int = 20260602) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    invocation_index = 0
    for trace_family in TRACE_FAMILIES:
        for family_replica in range(4):
            kernel_invocation_id = f"gcl_pa_k{invocation_index:03d}"
            warps: list[dict[str, Any]] = []
            for warp_id in range(2):
                entries: list[dict[str, Any]] = []
                pc_base = 0x1000 + invocation_index * 0x100 + warp_id * 0x40
                for trace_index, (opcode, dests, srcs, values) in enumerate(
                    _instruction_template(trace_family)
                ):
                    shifted_values = [
                        round(value + family_replica * 0.25 + warp_id * 0.125, 6)
                        for value in values
                    ]
                    entry = {
                        "kernel_invocation_id": kernel_invocation_id,
                        "trace_family": trace_family,
                        "collection_scope": COLLECTION_SCOPE,
                        "warp_id": warp_id,
                        "trace_index": trace_index,
                        "pc": pc_base + trace_index * 8,
                        "opcode": opcode,
                        "active_mask": "0xffffffff",
                        "destination_operands": [_warp_token(token, warp_id) for token in dests],
                        "source_operands": [_warp_token(token, warp_id) for token in srcs],
                        "observed_dynamic_values": shifted_values,
                    }
                    entry["source_entry_hash"] = _entry_hash(entry)
                    entries.append(entry)
                warps.append({"warp_id": warp_id, "entries": entries})
            records.append(
                {
                    "kernel_invocation_id": kernel_invocation_id,
                    "trace_family": trace_family,
                    "collection_scope": COLLECTION_SCOPE,
                    "warps": warps,
                }
            )
            invocation_index += 1
    fixture = {
        "fixture_name": FIXTURE_NAME,
        "fixture_version": 1,
        "seed": seed,
        "collection_scope": COLLECTION_SCOPE,
        "kernel_invocation_count": len(records),
        "trace_family_count": len(TRACE_FAMILIES),
        "records": records,
    }
    fixture["fixture_hash"] = hash_without(fixture, "fixture_hash")
    return fixture


def iter_trace_entries(fixture: dict[str, Any]):
    for record in fixture.get("records", []):
        for warp in record.get("warps", []):
            for entry in warp.get("entries", []):
                yield record, warp, entry


def validate_trace_fixture(fixture: dict[str, Any]) -> None:
    if fixture.get("fixture_name") != FIXTURE_NAME:
        raise ValueError("unexpected fixture_name")
    if fixture.get("collection_scope") != COLLECTION_SCOPE:
        raise ValueError("unexpected fixture collection_scope")
    records = fixture.get("records")
    if not isinstance(records, list) or len(records) != 12:
        raise ValueError("fixture must contain 12 kernel invocation records")
    families = {record.get("trace_family") for record in records}
    if families != set(TRACE_FAMILIES):
        raise ValueError("fixture must contain exactly three trace families")
    for record in records:
        if record.get("collection_scope") != COLLECTION_SCOPE:
            raise ValueError("record collection_scope must be selected_warps_fixture")
        warps = record.get("warps")
        if not isinstance(warps, list) or len(warps) != 2:
            raise ValueError("each invocation must contain two warps")
        for warp in warps:
            entries = warp.get("entries")
            if not isinstance(entries, list) or len(entries) != 6:
                raise ValueError("each warp must contain six dynamic entries")
            for expected_trace_index, entry in enumerate(entries):
                missing = REQUIRED_ENTRY_FIELDS.difference(entry)
                if missing:
                    raise ValueError(f"trace entry missing required fields: {sorted(missing)}")
                if entry["collection_scope"] != COLLECTION_SCOPE:
                    raise ValueError("entry collection_scope must be selected_warps_fixture")
                if entry["kernel_invocation_id"] != record["kernel_invocation_id"]:
                    raise ValueError("entry kernel_invocation_id does not match record")
                if entry["trace_family"] != record["trace_family"]:
                    raise ValueError("entry trace_family does not match record")
                if entry["warp_id"] != warp["warp_id"]:
                    raise ValueError("entry warp_id does not match warp")
                if entry["trace_index"] != expected_trace_index:
                    raise ValueError("trace_index must be contiguous within each warp")
                if _entry_hash(entry) != entry["source_entry_hash"]:
                    raise ValueError("source_entry_hash is not reproducible")
                if (
                    entry["source_operands"] or entry["destination_operands"]
                ) and len(entry["observed_dynamic_values"]) < 4:
                    raise ValueError("variable trace entries require at least four observed dynamic values")
    if fixture.get("fixture_hash") != hash_without(fixture, "fixture_hash"):
        raise ValueError("fixture_hash is not reproducible")


def fixture_summary(fixture: dict[str, Any]) -> dict[str, Any]:
    entries = list(iter_trace_entries(fixture))
    return {
        "fixture_name": fixture["fixture_name"],
        "kernel_invocation_count": len(fixture["records"]),
        "trace_family_count": len({record["trace_family"] for record in fixture["records"]}),
        "dynamic_entry_count": len(entries),
        "fixture_hash": fixture["fixture_hash"],
    }
