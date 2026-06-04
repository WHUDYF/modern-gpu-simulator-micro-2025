#!/usr/bin/env python3
"""Apply backend result summaries into writeback update artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_json(path: Path):
    return json.loads(path.read_text())


def _validate_unique_rows(rows: list[dict], key_fields: tuple[str, ...], label: str) -> None:
    seen = set()
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            raise ValueError(f"duplicate {label} row for key {key}")
        seen.add(key)


def _derive_decision_update(result_status: str, sensitivity_score):
    if result_status not in {"success", "weak", "failed", "inconclusive", "parse-failed"}:
        raise ValueError(f"unsupported result_status for writeback: {result_status}")
    if result_status == "success":
        return "increase"
    if result_status == "weak":
        return "keep-with-note"
    if result_status == "failed":
        return "decrease"
    if sensitivity_score is not None:
        return "review"
    return "no-change"


def _derive_importance_update(result_status: str, tuning_gain):
    if result_status not in {"success", "weak", "failed", "inconclusive", "parse-failed"}:
        raise ValueError(f"unsupported result_status for writeback: {result_status}")
    if result_status == "success":
        return "promote"
    if result_status == "weak":
        return "hold"
    if result_status == "failed":
        return "demote"
    if tuning_gain is not None:
        return "review"
    return "no-change"


def _derive_validation_status(
    result_status: str,
    validation_role: str,
    execution_mode: str | None = None,
    parse_status: str | None = None,
) -> str:
    if result_status not in {"success", "weak", "failed", "inconclusive", "parse-failed"}:
        raise ValueError(f"unsupported result_status for writeback: {result_status}")
    if execution_mode == "smoke" or parse_status == "parsed-smoke":
        return "pending-review" if validation_role == "review-object" else "pending"
    if result_status == "success":
        return "validated"
    if result_status == "weak":
        return "selected"
    if result_status == "failed":
        return "failed"
    if result_status == "parse-failed":
        return "pending-review" if validation_role == "review-object" else "pending"
    if result_status == "inconclusive":
        return "pending-review" if validation_role == "review-object" else "pending"
    if validation_role == "review-object":
        return "pending-review"
    return "selected"


def _derive_review_status(current_review: str, result_status: str, validation_role: str) -> str:
    if result_status not in {"success", "weak", "failed", "inconclusive", "parse-failed"}:
        raise ValueError(f"unsupported result_status for writeback: {result_status}")
    if validation_role != "review-object":
        return current_review if current_review else "no-review"
    if result_status == "success":
        return "resolved-review"
    if result_status in {"weak", "inconclusive", "failed"}:
        return "keep-review"
    return current_review if current_review else "keep-review"


def _explanation_note(result_row: dict, validation_role: str, canonical_status: str) -> str:
    base = (
        f"{result_row['regime_id']} under {result_row['parameter_scenario_id']} "
        f"returned {result_row['result_status']}"
    )
    if validation_role == "constraint-object":
        return base + "; keep structural family while preserving memory-side/constraint interpretation."
    if validation_role == "review-object":
        return base + "; retain review semantics unless the result clearly resolves the family fit."
    if "weak-share" in canonical_status or canonical_status == "weak-share":
        return base + "; keep weak-share note while updating execution evidence."
    return base + "; propagate update into family and anchor explanation."


def build_writeback_updates(run_manifest: list[dict], result_summary: list[dict], writeback_map: list[dict]) -> list[dict]:
    _validate_unique_rows(run_manifest, ("run_id",), "run_manifest")
    _validate_unique_rows(result_summary, ("run_id",), "result_summary")
    _validate_unique_rows(writeback_map, ("regime_id", "parameter_scenario_id"), "writeback_map")

    manifest_by_run = {row["run_id"]: row for row in run_manifest}
    writeback_by_key = {
        (row["regime_id"], row["parameter_scenario_id"]): row for row in writeback_map
    }

    updates = []
    for result in result_summary:
        if result["run_id"] not in manifest_by_run:
            raise ValueError(f"result_summary row references unknown run_id: {result['run_id']}")
        manifest_row = manifest_by_run[result["run_id"]]
        for field in ("family_id", "regime_id", "priority_source", "parameter_scenario_id"):
            if result[field] != manifest_row[field]:
                raise ValueError(
                    f"result_summary row for {result['run_id']} mismatches manifest on {field}: "
                    f"{result[field]} != {manifest_row[field]}"
                )
        key = (result["regime_id"], result["parameter_scenario_id"])
        if key not in writeback_by_key:
            raise ValueError(f"writeback_map is missing key {key}")
        writeback_row = writeback_by_key[key]
        validation_role = manifest_row["validation_role"]
        result_status = result["result_status"]
        writeback_id = f"{writeback_row['writeback_id']}__{result['priority_source'].replace('-', '_')}"
        updates.append(
            {
                "writeback_id": writeback_id,
                "run_id": result["run_id"],
                "regime_id": result["regime_id"],
                "family_id": result["family_id"],
                "priority_source": result["priority_source"],
                "rep_kernel_ids": writeback_row["rep_kernel_ids"],
                "parameter_scenario_id": result["parameter_scenario_id"],
                "decision_update": _derive_decision_update(result_status, result.get("sensitivity_score")),
                "importance_update": _derive_importance_update(result_status, result.get("tuning_gain")),
                "validation_status_update": _derive_validation_status(
                    result_status,
                    validation_role,
                    result.get("execution_mode"),
                    result.get("parse_status"),
                ),
                "review_status_update": _derive_review_status(
                    writeback_row.get("review_status_update"),
                    result_status,
                    validation_role,
                ),
                "workload_explanation_note": _explanation_note(
                    result,
                    validation_role,
                    manifest_row["canonical_status"],
                ),
            }
        )
    return updates


def build_validation_status(run_manifest: list[dict], writeback_updates: list[dict], writeback_map: list[dict]) -> dict:
    manifest_by_regime = {}
    for row in run_manifest:
        manifest_by_regime.setdefault(
            row["regime_id"],
            {
                "family_id": row["family_id"],
                "validation_role": row["validation_role"],
                "canonical_status": row["canonical_status"],
                "priority_sources": [],
            },
        )
        if row["priority_source"] not in manifest_by_regime[row["regime_id"]]["priority_sources"]:
            manifest_by_regime[row["regime_id"]]["priority_sources"].append(row["priority_source"])

    default_review_by_regime = {}
    for row in writeback_map:
        default_review_by_regime.setdefault(row["regime_id"], row.get("review_status_update", "no-review"))

    updates_by_regime = {}
    for update in writeback_updates:
        updates_by_regime.setdefault(update["regime_id"], []).append(update)

    regime_status = []
    for regime_id, base in manifest_by_regime.items():
        updates = updates_by_regime.get(regime_id, [])
        validation_candidates = [item["validation_status_update"] for item in updates]
        review_candidates = [item["review_status_update"] for item in updates]
        if "validated" in validation_candidates:
            current_status = "validated"
        elif "failed" in validation_candidates:
            current_status = "failed"
        elif "selected" in validation_candidates:
            current_status = "selected"
        elif "pending-review" in validation_candidates:
            current_status = "pending-review"
        elif "pending" in validation_candidates:
            current_status = "pending"
        else:
            current_status = "pending-review" if base["validation_role"] == "review-object" else "pending"

        if "keep-review" in review_candidates:
            current_review = "keep-review"
        elif "resolved-review" in review_candidates:
            current_review = "resolved-review"
        else:
            current_review = default_review_by_regime.get(regime_id, "no-review")

        regime_status.append(
            {
                "regime_id": regime_id,
                "family_id": base["family_id"],
                "validation_role": base["validation_role"],
                "canonical_status": base["canonical_status"],
                "current_status": current_status,
                "review_status": current_review,
                "priority_sources": base["priority_sources"],
            }
        )

    family_status = {}
    for row in regime_status:
        current = family_status.setdefault(
            row["family_id"],
            {
                "family_id": row["family_id"],
                "regime_ids": [],
                "review_needed_regimes": [],
                "validated_regimes": [],
                "failed_regimes": [],
            },
        )
        current["regime_ids"].append(row["regime_id"])
        if row["review_status"] == "keep-review":
            current["review_needed_regimes"].append(row["regime_id"])
        if row["current_status"] == "validated":
            current["validated_regimes"].append(row["regime_id"])
        if row["current_status"] == "failed":
            current["failed_regimes"].append(row["regime_id"])

    return {
        "regime_status": regime_status,
        "family_status": list(family_status.values()),
    }


def write_outputs(output_dir: Path, writeback_updates: list[dict], validation_status: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "backend_writeback_updates_v1.json").write_text(json.dumps(writeback_updates, indent=2))
    (output_dir / "backend_validation_status_v1.json").write_text(json.dumps(validation_status, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply backend writeback updates from result summaries.")
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--result-summary", type=Path, required=True)
    parser.add_argument("--writeback-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_manifest = load_json(args.run_manifest)
    result_summary = load_json(args.result_summary)
    writeback_map = load_json(args.writeback_map)

    writeback_updates = build_writeback_updates(run_manifest, result_summary, writeback_map)
    validation_status = build_validation_status(run_manifest, writeback_updates, writeback_map)
    write_outputs(args.output_dir, writeback_updates, validation_status)
    print(f"[backend-writeback] wrote writeback outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
