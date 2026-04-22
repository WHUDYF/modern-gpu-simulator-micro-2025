#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_REGIME_FIELDS = {
    "object_level",
    "object_id",
    "family_id",
    "regime_id",
    "priority_source",
    "priority_rank",
    "simulator_lane_id",
    "parameter_scenario_ids",
    "recommended_tuning_target",
    "canonical_status",
    "validation_role",
    "expected_signal",
    "original_order",
}
REQUIRED_FAMILY_FIELDS = {
    "object_level",
    "object_id",
    "family_id",
    "priority_source",
    "priority_rank",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def _required_strategies(worksheet: dict) -> list[str]:
    strategies = worksheet.get("budget_definition", {}).get("comparison_strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("validation worksheet must define non-empty budget_definition.comparison_strategies")
    return strategies


def _family_preselection_count(worksheet: dict) -> int:
    value = worksheet.get("budget_definition", {}).get("family_preselection_count")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("validation worksheet must define positive integer budget_definition.family_preselection_count")
    return value


def _max_scenarios(worksheet: dict, role: str) -> int:
    mapping = {
        "main-object": "main_object_max_scenarios",
        "review-object": "review_object_max_scenarios",
        "constraint-object": "constraint_object_max_scenarios",
    }
    value = worksheet.get("budget_definition", {}).get(mapping[role])
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"validation worksheet must define positive integer budget_definition.{mapping[role]}")
    return value


def _scenario_limit(row: dict, worksheet: dict) -> list[str]:
    role = row["validation_role"]
    if role in {"review-object", "constraint-object"}:
        return row["parameter_scenario_ids"][: _max_scenarios(worksheet, role)]
    return row["parameter_scenario_ids"][: _max_scenarios(worksheet, role)]


def _validate_priority_lane_table(priority_lane_table: list[dict], worksheet: dict) -> None:
    required_strategies = set(_required_strategies(worksheet))
    family_rows = [row for row in priority_lane_table if row.get("object_level") == "family"]
    regime_rows = [row for row in priority_lane_table if row.get("object_level") == "regime"]

    if not family_rows or not regime_rows:
        raise ValueError("priority lane table must contain both family and regime rows")

    for row in family_rows:
        missing = REQUIRED_FAMILY_FIELDS - set(row.keys())
        if missing:
            raise ValueError(f"family priority row is missing required fields: {sorted(missing)}")
    for row in regime_rows:
        missing = REQUIRED_REGIME_FIELDS - set(row.keys())
        if missing:
            raise ValueError(f"regime priority row is missing required fields: {sorted(missing)}")
        if not isinstance(row["parameter_scenario_ids"], list) or not row["parameter_scenario_ids"]:
            raise ValueError(f"regime row {row.get('regime_id')} must have non-empty parameter_scenario_ids")

    family_sources = {row["priority_source"] for row in family_rows}
    regime_sources = {row["priority_source"] for row in regime_rows}
    if family_sources != required_strategies or regime_sources != required_strategies:
        raise ValueError(
            "priority lane table must cover exactly the worksheet comparison strategies "
            f"(expected {sorted(required_strategies)}, got family={sorted(family_sources)}, regime={sorted(regime_sources)})"
        )


def _top_families_for_source(priority_lane_table: list[dict], source: str, limit: int) -> list[str]:
    families = [row for row in priority_lane_table if row["object_level"] == "family" and row["priority_source"] == source]
    families.sort(key=lambda row: row["priority_rank"])
    return [row["family_id"] for row in families[:limit]]


def _order_rows(rows: list[dict], source: str) -> list[dict]:
    if source == "importance-guided":
        return sorted(rows, key=lambda row: row["priority_rank"])
    if source == "time-only":
        return sorted(rows, key=lambda row: row["priority_rank"])
    if source == "name-based":
        return sorted(rows, key=lambda row: row["regime_id"])
    return sorted(rows, key=lambda row: row["original_order"])


def build_run_manifest(priority_lane_table: list[dict], worksheet: dict) -> list[dict]:
    _validate_priority_lane_table(priority_lane_table, worksheet)
    manifest = []
    strategies = _required_strategies(worksheet)
    family_limit = _family_preselection_count(worksheet)
    regime_rows = [row for row in priority_lane_table if row["object_level"] == "regime"]
    for source in strategies:
        top_families = _top_families_for_source(priority_lane_table, source, family_limit)
        selected_rows = [
            row for row in regime_rows
            if row["priority_source"] == source
            and (row["family_id"] in top_families or row["validation_role"] == "constraint-object")
        ]
        rows = _order_rows(selected_rows, source)
        for row in rows:
            for scenario_id in _scenario_limit(row, worksheet):
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
                        "validation_role": row["validation_role"],
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


def build_baseline_plan(run_manifest: list[dict], priority_lane_table: list[dict], worksheet: dict) -> dict:
    strategies = _required_strategies(worksheet)
    family_limit = _family_preselection_count(worksheet)
    budget = worksheet["budget_definition"]
    plan = {
        "comparison_scope": "family -> regime",
        "budget_policy": {
            "family_preselection_count": budget["family_preselection_count"],
            "main_object_max_scenarios": budget["main_object_max_scenarios"],
            "review_object_max_scenarios": budget["review_object_max_scenarios"],
            "constraint_object_max_scenarios": budget["constraint_object_max_scenarios"],
        },
        "strategies": {},
    }
    for source in strategies:
        rows = [row for row in run_manifest if row["priority_source"] == source]
        selected_families = []
        for row in rows:
            if row["family_id"] not in selected_families:
                selected_families.append(row["family_id"])
        selected_regimes = []
        for row in rows:
            if row["regime_id"] not in selected_regimes:
                selected_regimes.append(row["regime_id"])
        plan["strategies"][source] = {
            "selected_families": selected_families,
            "selected_regimes": selected_regimes,
            "run_count": len(rows),
        }
    return plan


def build_result_summary_template(run_manifest: list[dict]) -> list[dict]:
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan backend validation.")
    parser.add_argument("--priority-lane-table", type=Path, required=True)
    parser.add_argument("--validation-worksheet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    priority_lane_table = load_json(args.priority_lane_table)
    worksheet = load_json(args.validation_worksheet)
    manifest = build_run_manifest(priority_lane_table, worksheet)
    scenario_matrix = build_scenario_matrix(manifest)
    baseline_plan = build_baseline_plan(manifest, priority_lane_table, worksheet)
    result_summary = build_result_summary_template(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "backend_run_manifest_v1.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "backend_scenario_matrix_v1.json").write_text(json.dumps(scenario_matrix, indent=2))
    (args.output_dir / "backend_baseline_plan_v1.json").write_text(json.dumps(baseline_plan, indent=2))
    (args.output_dir / "backend_result_summary_v1.json").write_text(json.dumps(result_summary, indent=2))
    print(f"[backend-planner] wrote execution plan files to {args.output_dir}")


if __name__ == "__main__":
    main()
