"""Gate 5 formal M1 selector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from .pka_selector_core import build_outputs
    from .shared_acquisition import ARTIFACT_DIR, FEATURE_ORDER, artifact_ref, file_hash, read_json, write_json
except ImportError:
    from pka_selector_core import build_outputs
    from shared_acquisition import ARTIFACT_DIR, FEATURE_ORDER, artifact_ref, file_hash, read_json, write_json

ELIGIBILITY_PATH = ARTIFACT_DIR / "m1_selector_eligibility_l1.json"
SELECTOR_INPUT_PATH = ARTIFACT_DIR / "m1_selector_input_l1.json"

FORBIDDEN = {"kernel_name", "source_path", "expected_behavior_axis", "family", "regime", "shape_hint", "trace_order"}


def run() -> dict:
    eligibility = read_json(ELIGIBILITY_PATH, {})
    if not eligibility.get("gate5_allowed"):
        raise SystemExit(f"Gate5 blocked: {eligibility.get('selector_eligibility_state', 'missing_eligibility')}")
    if eligibility.get("selector_eligibility_state") not in {"selector_ready", "selector_ready_with_remaining_gaps"}:
        raise SystemExit("Gate5 blocked: invalid Gate4 eligibility state")
    weight_mode = eligibility.get("weight_mode")
    timing_unit = eligibility.get("timing_unit")
    if weight_mode not in {"member_count_fallback", "timing_weight"}:
        raise SystemExit("Gate5 blocked: missing or invalid weight_mode")
    records = read_json(SELECTOR_INPUT_PATH, [])
    if len(records) < 3:
        raise SystemExit("Gate5 blocked: fewer than 3 selector records")
    actual_fields = sorted({key for row in records for key in row})
    allowed_fields = ["record_id", "kernel_invocation_id", "features", "feature_mode", "weight_input"]
    forbidden_hits = sorted(FORBIDDEN & set(actual_fields))
    forbidden_audit = {
        "allowed_input_fields": allowed_fields,
        "forbidden_fields": sorted(FORBIDDEN),
        "actual_read_fields": actual_fields,
        "status": "passed" if not forbidden_hits else "failed",
        "violations": forbidden_hits,
    }
    if forbidden_hits:
        raise SystemExit(f"Gate5 blocked: forbidden selector fields {forbidden_hits}")
    for row in records:
        if row.get("feature_mode") != "pka_m1_measured":
            raise SystemExit("Gate5 blocked: non-M1 feature_mode")
        weight_input = row.get("weight_input") or {}
        if weight_mode == "timing_weight" and weight_input.get("timing_unit") != timing_unit:
            raise SystemExit("Gate5 blocked: selector input timing unit violates Gate4 contract")
    outputs = build_outputs(
        records,
        mode="pka_m1_measured",
        feature_mode="pka_m1_measured",
        weight_mode=weight_mode,
        timing_unit=timing_unit,
    )
    selector_hash = file_hash(SELECTOR_INPUT_PATH)
    eligibility_hash = file_hash(ELIGIBILITY_PATH)
    replay_hash = outputs["deterministic_replay_hash"]
    pca_path = ARTIFACT_DIR / "pka_pca_projection_l1.json"
    kmeans_path = ARTIFACT_DIR / "pka_kmeans_clusters_l1.json"
    anchor_path = ARTIFACT_DIR / "representative_anchor_table_l1.json"
    eval_path = ARTIFACT_DIR / "pka_compression_evaluation_l1.json"

    pca_artifact = {
        "artifact_name": "pka_pca_projection_l1",
        "mode": "pka_m1_measured",
        "method": "numpy_svd",
        "input_selector_projection_path": artifact_ref(SELECTOR_INPUT_PATH),
        "input_selector_projection_hash": selector_hash,
        "gate4_eligibility_path": artifact_ref(ELIGIBILITY_PATH),
        "gate4_eligibility_hash": eligibility_hash,
        "feature_order": FEATURE_ORDER,
        "normalization_config": outputs["projection"]["preprocessing"],
        "components": outputs["projection"]["pca"]["component_matrix"],
        "explained_variance": outputs["projection"]["pca"]["total_explained_variance"],
        "explained_variance_ratio": outputs["projection"]["pca"]["explained_variance_ratio"],
        "transformed_coordinates": outputs["projection"]["records"],
        "record_ids": [row["record_id"] for row in outputs["projection"]["records"]],
        "deterministic_replay_hash": replay_hash,
    }
    write_json(pca_path, pca_artifact)

    assignments: dict[str, str] = {}
    distance_to_centroid: dict[str, float] = {}
    members_by_cluster: dict[str, list[str]] = {}
    for row in outputs["clusters"]["records"]:
        assignments[row["record_id"]] = row["cluster_id"]
        distance_to_centroid[row["record_id"]] = row["distance_to_centroid"]
        members_by_cluster.setdefault(row["cluster_id"], []).append(row["record_id"])
    kmeans_meta = outputs["clusters"]["kmeans"]
    kmeans_artifact = {
        "artifact_name": "pka_kmeans_clusters_l1",
        "mode": "pka_m1_measured",
        "method": "deterministic_farthest_first_kmeans",
        "input_pca_artifact_path": artifact_ref(pca_path),
        "input_pca_artifact_hash": file_hash(pca_path),
        "kmeans_config": kmeans_meta,
        "k": kmeans_meta["k"],
        "initial_centroid_record_ids": kmeans_meta["initial_center_record_ids"],
        "initialization_trace": kmeans_meta["initial_center_record_ids"],
        "centroids": outputs["clusters"]["kmeans"].get("centroids", []),
        "assignments": assignments,
        "members_by_cluster": members_by_cluster,
        "distance_to_centroid": distance_to_centroid,
        "inertia": sum(distance_to_centroid.values()),
        "iterations_run": kmeans_meta["iterations"],
        "converged": kmeans_meta["iterations"] < kmeans_meta["max_iter"],
        "empty_cluster_events": [],
        "deterministic_replay_hash": replay_hash,
    }
    write_json(kmeans_path, kmeans_artifact)

    records_by_id = {str(row.get("record_id") or row.get("kernel_invocation_id")): row for row in records}
    anchor_table = []
    anchor_metadata_rows = []
    for index, row in enumerate(outputs["anchors"]):
        representative_record = records_by_id.get(row["rep_record_id"], {})
        representative_kernel = representative_record.get("kernel_invocation_id") or row["rep_kernel_id"]
        anchor_table.append({
            "rep_kernel_id": row["rep_kernel_id"],
            "kernel_name": str(representative_kernel),
            "cluster_id": row["cluster_id"],
            "member_invocations": row["member_invocations"],
            "coverage_count": row["coverage_count"],
            "coverage_weight": row["coverage_weight"],
            "time_weight": row["coverage_weight"],
        })
        anchor_metadata_rows.append({
            "anchor_id": f"m1_anchor_{index:03d}",
            "cluster_id": row["cluster_id"],
            "representative_record_id": row["rep_record_id"],
            "members": row["member_record_ids"],
            "weight": row["weight"],
            "representative_distance_to_centroid": row["representative_distance_to_centroid"],
            "cluster_label": row["cluster_id"],
        })
    anchor_artifact = {
        "artifact_name": "representative_anchor_table_l1",
        "mode": "pka_m1_measured",
        "feature_mode": "pka_m1_measured",
        "selector_name": "pka_m1_shared_core",
        "selected_features": FEATURE_ORDER,
        "normalization_config": outputs["projection"]["preprocessing"],
        "dimensionality_reduction_config": outputs["projection"]["pca"],
        "clustering_config": kmeans_meta,
        "selection_rule": "nearest_centroid_record",
        "forbidden_field_audit": forbidden_audit,
        "anchors": anchor_metadata_rows,
        "input_selector_projection_hash": selector_hash,
        "gate4_eligibility_hash": eligibility_hash,
        "deterministic_replay_hash": replay_hash,
    }
    write_json(anchor_path, anchor_table)

    evaluation = outputs["evaluation"]
    evaluation.update({
        "artifact_name": "pka_compression_evaluation_l1",
        "input_selector_projection_hash": selector_hash,
        "gate4_eligibility_hash": eligibility_hash,
        "pca_artifact_hash": file_hash(pca_path),
        "kmeans_artifact_hash": file_hash(kmeans_path),
        "anchor_table_hash": file_hash(anchor_path),
        "pca_diagnostics": outputs["projection"]["pca"],
        "kmeans_summary": kmeans_meta,
    })
    write_json(eval_path, evaluation)
    return {"projection": pca_artifact, "clusters": kmeans_artifact, "anchors": anchor_artifact, "evaluation": evaluation}


def main() -> int:
    run()
    print("Gate5 formal M1 selector complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
