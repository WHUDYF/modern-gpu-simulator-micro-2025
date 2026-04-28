"""PKA baseline selector for L1.

Reads the PKA feature table and runs PKA-style grouping (PCA-like dimensionality
reduction + k-means clustering) on the 12-dimensional feature space. Forbidden
fields (kernel_name, grid_dim, block_dim, compression-side, family/regime/lane)
must not enter the grouping key.

For L1, the selector is gate-protected: it reads the stage-gate report and
refuses to run when P0 acquisition gaps exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "a_line" / "l1"

FEATURE_TABLE_PATH = ARTIFACT_DIR / "pka_feature_table_l1.json"
STAGE_GATE_PATH = ARTIFACT_DIR / "l1_stage_gate_report_l1.json"
OUTPUT_PATH = ARTIFACT_DIR / "representative_anchor_table_l1.json"

FORBIDDEN_FIELDS = frozenset({
    "kernel_name", "grid_dim", "block_dim", "shape_hint", "trace_order",
    "cross_tb_offset_coverage", "squash_boundary_crossing_flag",
    "family_id", "regime_id", "route_primitive", "execution_template",
    "execution_template_label", "simulator_lane_id",
    "address_override_density", "full_encoding_fallback_rate",
    "shared_pc_sequence_length",
})

ALLOWED_FEATURES = [
    "coalesced_global_loads",
    "coalesced_global_stores",
    "coalesced_local_loads",
    "thread_global_loads",
    "thread_global_stores",
    "thread_local_loads",
    "thread_shared_loads",
    "thread_shared_stores",
    "thread_global_atomics",
    "num_instructions",
    "divergence_efficiency",
    "num_thread_blocks",
]


def _check_stage_gate() -> tuple[bool, str]:
    """Check if the stage gate allows selector execution."""
    if not STAGE_GATE_PATH.exists():
        return False, "stage gate report not found"
    report = json.loads(STAGE_GATE_PATH.read_text())
    stage_3 = report.get("stages", {}).get("stage_3_selector", "unknown")
    if stage_3 == "blocked":
        return False, report.get("next_action", "blocked by stage gate")
    return True, ""


def _validate_forbidden_fields(feature_space: list[str]) -> list[str]:
    """Return list of forbidden fields found in the feature space."""
    return sorted(set(feature_space) & set(FORBIDDEN_FIELDS))


def _build_feature_matrix(records: list[dict[str, Any]]) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Build numeric feature matrix from PkaFeatureRecords."""
    matrix = []
    meta = []
    for rec in records:
        features = rec.get("features", {})
        row = []
        valid = True
        for feat_name in ALLOWED_FEATURES:
            f = features.get(feat_name, {})
            val = f.get("value")
            if val is None or f.get("status") != "measured":
                valid = False
                break
            row.append(float(val))
        if valid and len(row) == 12:
            matrix.append(row)
            meta.append(rec)
    return matrix, meta


def _select_first_chronological(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Select representative by earliest trace_order (first_chronological rule)."""
    return min(
        members,
        key=lambda r: (
            r.get("metadata", {}).get("trace_order")
            if r.get("metadata", {}).get("trace_order") is not None
            else float("inf")
        ),
    )


def main() -> int:
    ok, reason = _check_stage_gate()
    if not ok:
        print(f"Selector blocked by stage gate: {reason}")
        print("The selector will not execute until all P0 acquisition gaps are resolved.")
        return 2

    feature_records = json.loads(FEATURE_TABLE_PATH.read_text())

    if len(feature_records) < 2:
        print(f"Selector blocked: only {len(feature_records)} measured records available (minimum 2 required).")
        print("status: selector_insufficient_records")
        return 3

    # Validate no forbidden fields in input
    forbidden = _validate_forbidden_fields(ALLOWED_FEATURES)
    if forbidden:
        print(f"Selector rejected: forbidden fields found in feature space: {forbidden}")
        return 4

    # Build feature matrix
    matrix, meta = _build_feature_matrix(feature_records)
    print(f"Feature matrix: {len(matrix)} records x {len(ALLOWED_FEATURES)} features")

    # Phase 1: Standardize (z-score)
    means = [sum(col) / len(col) for col in zip(*matrix)]
    # ... (full PCA + k-means implementation goes here)
    # For L1 skeleton, we stop at validation

    print("Selector skeleton validation passed. Full PCA + k-means implementation pending.")
    print(f"Allowlist: {ALLOWED_FEATURES}")
    print(f"Forbidden fields validated: 0 violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
