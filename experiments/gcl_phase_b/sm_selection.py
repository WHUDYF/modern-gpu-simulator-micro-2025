"""Representative SM selection for GCL Phase B."""

from __future__ import annotations

from math import ceil, sqrt
from typing import Any

from .utils import hash_without

POLICY = "scheduler_signature_medoid_sm"
POLICY_VERSION = "v1"
REPORT_VERSION = "gcl_phase_b_selected_sm_policy_v1"
SIGNATURE_FIELDS = [
    "cta_count",
    "warp_count",
    "instruction_count_proxy",
    "first_cta_start_order",
    "last_cta_end_order",
    "cta_wave_coverage",
    "tail_cta_ratio",
]
SIGNATURE_WEIGHTS = {field: 1.0 for field in SIGNATURE_FIELDS}


def _require_mapping(metadata: dict[str, Any], field: str) -> dict[str, Any]:
    value = metadata.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"missing required scheduler metadata field: {field}")
    return value


def _raw_signature(metadata: dict[str, Any], global_first: int, global_last: int) -> dict[str, float]:
    cta_ids = metadata.get("cta_ids")
    if not cta_ids:
        raise ValueError("candidate SM must contain cta_ids")
    warp_ids_by_cta = _require_mapping(metadata, "warp_ids_by_cta")
    instruction_count_by_cta = metadata.get("instruction_count_by_cta")
    if not isinstance(instruction_count_by_cta, dict):
        instruction_count_by_cta = metadata.get("trace_entry_count_by_cta")
    if not isinstance(instruction_count_by_cta, dict):
        raise ValueError("missing required scheduler metadata field: instruction_count_by_cta")
    cta_start_order = _require_mapping(metadata, "cta_start_order")
    cta_end_order = _require_mapping(metadata, "cta_end_order")

    cta_count = len(set(cta_ids))
    warp_pairs = {
        (cta_id, warp_id)
        for cta_id in cta_ids
        for warp_id in warp_ids_by_cta.get(cta_id, [])
    }
    first_start = min(int(cta_start_order[cta_id]) for cta_id in cta_ids)
    last_end = max(int(cta_end_order[cta_id]) for cta_id in cta_ids)
    global_span = max(1, global_last - global_first + 1)
    tail_threshold = global_first + ceil(0.8 * global_span)
    tail_count = sum(1 for cta_id in cta_ids if int(cta_start_order[cta_id]) >= tail_threshold)
    return {
        "cta_count": float(cta_count),
        "warp_count": float(len(warp_pairs)),
        "instruction_count_proxy": float(
            sum(float(instruction_count_by_cta[cta_id]) for cta_id in cta_ids)
        ),
        "first_cta_start_order": float(first_start),
        "last_cta_end_order": float(last_end),
        "cta_wave_coverage": float((last_end - first_start + 1) / global_span),
        "tail_cta_ratio": float(tail_count / cta_count),
    }


def _candidate_metadata(invocation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = {
        str(sm_id): metadata
        for sm_id, metadata in invocation.get("scheduler_metadata_by_sm", {}).items()
        if metadata.get("cta_ids")
    }
    if not candidates:
        raise ValueError("no_candidate_sm_for_kernel_invocation")
    return candidates


def _global_orders(candidates: dict[str, dict[str, Any]]) -> tuple[int, int]:
    starts: list[int] = []
    ends: list[int] = []
    for metadata in candidates.values():
        start_order = _require_mapping(metadata, "cta_start_order")
        end_order = _require_mapping(metadata, "cta_end_order")
        for cta_id in metadata.get("cta_ids", []):
            starts.append(int(start_order[cta_id]))
            ends.append(int(end_order[cta_id]))
    return min(starts), max(ends)


def _normalize(raw_by_sm: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    normalized = {sm_id: {} for sm_id in raw_by_sm}
    for field in SIGNATURE_FIELDS:
        values = [signature[field] for signature in raw_by_sm.values()]
        minimum = min(values)
        maximum = max(values)
        for sm_id, signature in raw_by_sm.items():
            if maximum == minimum:
                normalized[sm_id][field] = 0.0
            else:
                normalized[sm_id][field] = (signature[field] - minimum) / (maximum - minimum)
    return normalized


def _mean_signature(normalized_by_sm: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        field: sum(signature[field] for signature in normalized_by_sm.values()) / len(normalized_by_sm)
        for field in SIGNATURE_FIELDS
    }


def _distances(
    normalized_by_sm: dict[str, dict[str, float]], global_signature: dict[str, float]
) -> dict[str, float]:
    return {
        sm_id: sqrt(
            sum(
                (signature[field] - global_signature[field]) ** 2 * SIGNATURE_WEIGHTS[field]
                for field in SIGNATURE_FIELDS
            )
        )
        for sm_id, signature in normalized_by_sm.items()
    }


def select_representative_sm(invocation: dict[str, Any], policy: str = POLICY) -> dict[str, Any]:
    if policy == "explicit_sm_id":
        selected = invocation.get("selected_sm")
        if selected is None:
            raise ValueError("explicit_sm_id requires selected_sm")
        return _explicit_report(invocation, int(selected))
    if policy != POLICY:
        raise ValueError(f"unsupported selected_sm_policy: {policy}")
    candidates = _candidate_metadata(invocation)
    global_first, global_last = _global_orders(candidates)
    raw_by_sm = {
        sm_id: _raw_signature(metadata, global_first, global_last)
        for sm_id, metadata in candidates.items()
    }
    normalized_by_sm = _normalize(raw_by_sm)
    global_signature = _mean_signature(normalized_by_sm)
    distances = _distances(normalized_by_sm, global_signature)
    selected_sm_id = min(distances, key=lambda sm_id: (distances[sm_id], int(sm_id)))
    report = {
        "artifact_name": "selected_sm_policy_report",
        "artifact_version": REPORT_VERSION,
        "kernel_invocation_id": invocation["kernel_invocation_id"],
        "selected_sm_policy": POLICY,
        "selected_sm_policy_version": POLICY_VERSION,
        "selected_sm": int(selected_sm_id),
        "selected_sm_reason": (
            f"SM {selected_sm_id} selected because it has the smallest equal-weight L2 "
            "distance to the global normalized scheduler signature."
        ),
        "candidate_sm_count": len(candidates),
        "candidate_sm_ids": [int(sm_id) for sm_id in sorted(candidates, key=int)],
        "signature_fields": SIGNATURE_FIELDS,
        "signature_field_weights": SIGNATURE_WEIGHTS,
        "normalization": {
            "method": "per_kernel_min_max",
            "zero_variance_policy": "set_normalized_value_to_zero_and_record_note",
        },
        "raw_signature_by_sm": raw_by_sm,
        "normalized_signature_by_sm": normalized_by_sm,
        "global_sm_signature": global_signature,
        "distance_metric": "equal_weight_l2",
        "distance_to_global_signature_by_sm": distances,
        "tie_break_rule": "lowest_sm_id",
        "instruction_count_proxy_source": "trace_entry_count",
    }
    report["selection_hash"] = hash_without(report, "selection_hash")
    validate_selected_sm_policy_report(report)
    return report


def _explicit_report(invocation: dict[str, Any], selected_sm: int) -> dict[str, Any]:
    candidates = _candidate_metadata(invocation)
    if str(selected_sm) not in candidates:
        raise ValueError("explicit selected_sm is not a candidate")
    report = {
        "artifact_name": "selected_sm_policy_report",
        "artifact_version": REPORT_VERSION,
        "kernel_invocation_id": invocation["kernel_invocation_id"],
        "selected_sm_policy": "explicit_sm_id",
        "selected_sm_policy_version": POLICY_VERSION,
        "selected_sm": selected_sm,
        "selected_sm_reason": f"SM {selected_sm} selected by explicit_sm_id for controlled replay.",
        "candidate_sm_count": len(candidates),
        "candidate_sm_ids": [int(sm_id) for sm_id in sorted(candidates, key=int)],
        "signature_fields": [],
        "signature_field_weights": {},
        "normalization": {"method": "not_applicable"},
        "raw_signature_by_sm": {},
        "normalized_signature_by_sm": {},
        "global_sm_signature": {},
        "distance_metric": "not_applicable",
        "distance_to_global_signature_by_sm": {str(selected_sm): 0.0},
        "tie_break_rule": "not_applicable",
        "instruction_count_proxy_source": "not_applicable",
    }
    report["selection_hash"] = hash_without(report, "selection_hash")
    return report


def validate_selected_sm_policy_report(report: dict[str, Any]) -> None:
    required = {
        "artifact_name",
        "artifact_version",
        "kernel_invocation_id",
        "selected_sm_policy",
        "selected_sm_policy_version",
        "selected_sm",
        "selected_sm_reason",
        "candidate_sm_count",
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
    }
    missing = required.difference(report)
    if missing:
        raise ValueError(f"selected SM policy report missing required fields: {sorted(missing)}")
    if report["selected_sm_policy"] == POLICY:
        _validate_signature_report_consistency(report)
        distances = report["distance_to_global_signature_by_sm"]
        expected = min(distances, key=lambda sm_id: (distances[sm_id], int(sm_id)))
        if int(expected) != int(report["selected_sm"]):
            raise ValueError("selected_sm does not match nearest scheduler signature medoid")
    if report["selection_hash"] != hash_without(report, "selection_hash"):
        raise ValueError("selection_hash is not reproducible")


def _validate_signature_report_consistency(report: dict[str, Any]) -> None:
    if report["signature_fields"] != SIGNATURE_FIELDS:
        raise ValueError("signature_fields do not match scheduler signature policy")
    if report["signature_field_weights"] != SIGNATURE_WEIGHTS:
        raise ValueError("signature weights do not match scheduler signature policy")
    candidate_ids = [str(sm_id) for sm_id in report["candidate_sm_ids"]]
    raw_by_sm = report["raw_signature_by_sm"]
    if set(raw_by_sm) != set(candidate_ids):
        raise ValueError("raw signature candidate set mismatch")
    for sm_id, signature in raw_by_sm.items():
        missing = set(SIGNATURE_FIELDS).difference(signature)
        if missing:
            raise ValueError(f"signature for SM {sm_id} missing fields: {sorted(missing)}")
        for count_field in ("cta_count", "warp_count", "instruction_count_proxy"):
            if signature[count_field] <= 0.0:
                raise ValueError("signature count fields must be positive")
        if signature["cta_wave_coverage"] <= 0.0:
            raise ValueError("signature cta_wave_coverage must be positive")
    expected_normalized = _normalize(raw_by_sm)
    if not _float_mapping_close(report["normalized_signature_by_sm"], expected_normalized):
        raise ValueError("normalized signature does not match raw signature")
    expected_global = _mean_signature(expected_normalized)
    if not _float_dict_close(report["global_sm_signature"], expected_global):
        raise ValueError("global signature does not match normalized signatures")
    expected_distances = _distances(expected_normalized, expected_global)
    if not _float_dict_close(report["distance_to_global_signature_by_sm"], expected_distances):
        raise ValueError("signature distances do not match normalized signatures")


def _float_mapping_close(actual: dict[str, dict[str, float]], expected: dict[str, dict[str, float]]) -> bool:
    if set(actual) != set(expected):
        return False
    return all(_float_dict_close(actual[key], expected[key]) for key in actual)


def _float_dict_close(actual: dict[str, float], expected: dict[str, float], tolerance: float = 1e-9) -> bool:
    if set(actual) != set(expected):
        return False
    return all(abs(float(actual[key]) - float(expected[key])) <= tolerance for key in actual)
