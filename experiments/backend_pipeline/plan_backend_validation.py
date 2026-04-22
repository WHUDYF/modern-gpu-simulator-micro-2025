#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REVIEW_OBJECTS = {"R4_layernorm_reduction"}
CONSTRAINT_OBJECTS = {"R6_residual_elementwise"}
ORIGINAL_ORDER = ["R1_projection_dense", "R2_attention_score_dense", "R3_softmax_reduction", "R4_layernorm_reduction", "R5_context_streaming", "R6_residual_elementwise"]


def load_json(path: Path):
    return json.loads(path.read_text())


def _validation_role(regime_id: str) -> str:
    if regime_id in REVIEW_OBJECTS:
        return "review-object"
    if regime_id in CONSTRAINT_OBJECTS:
        return "constraint-object"
    return "main-object"


def _scenario_limit(regime_id: str, scenarios: list[str]) -> list[str]:
    role = _validation_role(regime_id)
    if role in {"review-object", "constraint-object"}:
        return scenarios[:1]
    return scenarios[:2]


def _top_three_families(priority_lane_table: list[dict]) -> list[str]:
    families = [row for row in priority_lane_table if row["object_level"] == "family" and row["priority_source"] == "importance-guided"]
    families.sort(key=lambda row: row["priority_rank"])
    return [row["family_id"] for row in families[:3]]


def _order_rows(rows: list[dict], source: str) -> list[dict]:
    if source == "importance-guided":
        return sorted(rows, key=lambda row: row["priority_rank"])
    if source == "time-only":
        return sorted(rows, key=lambda row: row["priority_rank"])
    if source == "name-based":
        return sorted(rows, key=lambda row: row["regime_id"])
    return sorted(rows, key=lambda row: ORIGINAL_ORDER.index(row["regime_id"]))


def build_run_manifest(priority_lane_table: list[dict]) -> list[dict]:
    top_families = _top_three_families(priority_lane_table)
    manifest = []
    regime_rows = [
        row for row in priority_lane_table
        if row["object_level"] == "regime" and (row["family_id"] in top_families or row["regime_id"] in CONSTRAINT_OBJECTS)
    ]
    for source in ["importance-guided", "time-only", "name-based", "no-priority"]:
        rows = _order_rows([row for row in regime_rows if row["priority_source"] == source], source)
        for row in rows:
            for scenario_id in _scenario_limit(row["regime_id"], row["parameter_scenario_ids"]):
                manifest.append(
                    {
                        "run_id": f"RUN_{source.replace('-', '_')}_{row['regime_id']}_{scenario_id}",
                        "object_level": "regime",
                        "object_id": row["regime_id"],
                        "family_id": row["family_id"],
                        "regime_id": row["regime_id"],
                        "priority_source": source,
                        "priority_rank": row["priority_rank"],
                        "simulator_lane_id": row["simulator_lane_id"],
                        "parameter_scenario_id": scenario_id,
                        "recommended_tuning_target": row["recommended_tuning_target"],
                        "canonical_status": row["canonical_status"],
                        "validation_role": _validation_role(row["regime_id"]),
                        "expected_signal": row["expected_signal"],
                        "run_status": "planned",
                    }
                )
    return manifest


def build_scenario_matrix(run_manifest: list[dict]) -> list[dict]:
    grouped = {}
    for row in run_manifest:
        key = (row["regime_id"], row["priority_source"])
        grouped.setdefault(key, {"object_id": row["regime_id"], "family_id": row["family_id"], "priority_source": row["priority_source"], "validation_role": row["validation_role"], "simulator_lane_id": row["simulator_lane_id"], "parameter_scenario_ids": []})
        grouped[key]["parameter_scenario_ids"].append(row["parameter_scenario_id"])
    return list(grouped.values())


def build_baseline_plan(run_manifest: list[dict], priority_lane_table: list[dict]) -> dict:
    top_families = _top_three_families(priority_lane_table)
    plan = {"comparison_scope": "family -> regime", "budget_policy": {"family_preselection": "Top-3 families (recommended target)", "main_object_scenarios": "up to 2", "review_object_scenarios": "1", "constraint_object_scenarios": "1, budget tail"}, "strategies": {}}
    for source in ["importance-guided", "time-only", "name-based", "no-priority"]:
        rows = [row for row in run_manifest if row["priority_source"] == source]
        plan["strategies"][source] = {"selected_families": top_families, "selected_regimes": [row["regime_id"] for row in rows], "run_count": len(rows)}
    return plan


def build_result_summary_template(run_manifest: list[dict]) -> list[dict]:
    if not run_manifest:
        return []
    sample = run_manifest[0]
    return [{
        "run_id": sample["run_id"],
        "object_id": sample["object_id"],
        "family_id": sample["family_id"],
        "regime_id": sample["regime_id"],
        "priority_source": sample["priority_source"],
        "parameter_scenario_id": sample["parameter_scenario_id"],
        "observed_metric_values": {},
        "baseline_delta": {},
        "sensitivity_score": None,
        "coverage_gain": None,
        "tuning_gain": None,
        "result_status": "inconclusive",
        "notes": "Template sample row. Duplicate and fill with real execution results.",
    }]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan backend validation.")
    parser.add_argument("--priority-lane-table", type=Path, required=True)
    parser.add_argument("--validation-worksheet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    priority_lane_table = load_json(args.priority_lane_table)
    _ = load_json(args.validation_worksheet)
    manifest = build_run_manifest(priority_lane_table)
    scenario_matrix = build_scenario_matrix(manifest)
    baseline_plan = build_baseline_plan(manifest, priority_lane_table)
    result_summary = build_result_summary_template(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "backend_run_manifest_v1.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "backend_scenario_matrix_v1.json").write_text(json.dumps(scenario_matrix, indent=2))
    (args.output_dir / "backend_baseline_plan_v1.json").write_text(json.dumps(baseline_plan, indent=2))
    (args.output_dir / "backend_result_summary_v1.json").write_text(json.dumps(result_summary, indent=2))
    print(f"[backend-planner] wrote execution plan files to {args.output_dir}")


if __name__ == "__main__":
    main()
