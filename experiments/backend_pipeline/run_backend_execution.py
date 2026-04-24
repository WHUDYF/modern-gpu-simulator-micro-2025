#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.backend_pipeline.execution_bridge import (  # noqa: E402
    build_result_summary,
    build_run_specs,
    execute_run_specs,
    load_json,
    scenario_focus_map,
    select_manifest_rows,
    write_command_plan,
    write_execution_records,
    write_result_summary,
)
from experiments.backend_pipeline.workload_profiles import load_workload_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute backend manifest rows through a workload-specific execution bridge.")
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--validation-worksheet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "experiments" / "backend_pipeline" / "runs")
    parser.add_argument("--workload-id", default="mini_transformer_v4")
    parser.add_argument("--workload-profile", type=Path)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    manifest = load_json(args.run_manifest)
    worksheet = load_json(args.validation_worksheet)
    profile = load_workload_profile(args.workload_id, args.workload_profile)
    selected = select_manifest_rows(
        manifest,
        max_runs=args.max_runs,
        run_ids=set(args.run_ids) if args.run_ids else None,
    )
    if not selected:
        raise ValueError("No manifest rows selected for execution")
    run_specs = build_run_specs(selected, profile, scenario_focus_map(worksheet), args.runs_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command_plan_path = args.output_dir / "backend_command_plan_v1.json"
    execution_records_path = args.output_dir / "backend_execution_records_v1.json"
    result_summary_path = args.output_dir / "backend_result_summary_v1.json"

    write_command_plan(run_specs, command_plan_path)
    if args.plan_only:
        print(f"[backend-execution] wrote command plan to {command_plan_path}")
        return

    execution_records = execute_run_specs(run_specs, args.timeout_seconds)
    write_execution_records(execution_records, execution_records_path)
    result_summary = build_result_summary(run_specs, execution_records, profile["parser"])
    write_result_summary(result_summary, result_summary_path)
    print(f"[backend-execution] wrote command plan to {command_plan_path}")
    print(f"[backend-execution] wrote execution records to {execution_records_path}")
    print(f"[backend-execution] wrote result summary to {result_summary_path}")


if __name__ == "__main__":
    main()
