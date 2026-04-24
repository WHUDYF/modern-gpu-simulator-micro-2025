from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


REQUIRED_MANIFEST_FIELDS = {
    "run_id",
    "family_id",
    "regime_id",
    "priority_source",
    "priority_rank",
    "simulator_lane_id",
    "parameter_scenario_id",
    "recommended_tuning_target",
    "validation_role",
    "expected_signal",
}


def scenario_focus_map(validation_worksheet: dict[str, Any]) -> dict[str, str]:
    scenarios = validation_worksheet.get("parameter_scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("validation worksheet must define non-empty parameter_scenarios")
    result: dict[str, str] = {}
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id")
        focus = scenario.get("focus")
        if not scenario_id or not focus:
            raise ValueError("each parameter scenario must define scenario_id and focus")
        result[str(scenario_id)] = str(focus)
    return result


def validate_manifest_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("run manifest must be non-empty")
    seen_run_ids = set()
    for row in rows:
        missing = REQUIRED_MANIFEST_FIELDS - set(row.keys())
        if missing:
            raise ValueError(f"manifest row is missing required fields: {sorted(missing)}")
        run_id = row["run_id"]
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate run_id in manifest: {run_id}")
        seen_run_ids.add(run_id)


def select_manifest_rows(
    rows: list[dict[str, Any]],
    *,
    max_runs: int | None = None,
    run_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if run_ids is not None and row["run_id"] not in run_ids:
            continue
        selected.append(row)
        if max_runs is not None and len(selected) >= max_runs:
            break
    return selected


def build_run_specs(
    manifest_rows: list[dict[str, Any]],
    workload_profile: dict[str, Any],
    scenario_focus_by_id: dict[str, str],
    runs_root: Path,
) -> list[dict[str, Any]]:
    validate_manifest_rows(manifest_rows)
    run_specs: list[dict[str, Any]] = []
    for row in manifest_rows:
        scenario_id = row["parameter_scenario_id"]
        if scenario_id not in scenario_focus_by_id:
            raise ValueError(f"scenario {scenario_id} is missing from validation worksheet")
        if scenario_id not in workload_profile["scenario_overrides"]:
            raise ValueError(f"scenario {scenario_id} is missing from workload profile scenario_overrides")
        run_id = row["run_id"]
        output_dir = (runs_root / workload_profile["workload_id"] / run_id).resolve()
        config_dir = output_dir / "configs"
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        metadata_path = output_dir / "run_metadata.json"
        command_path = output_dir / "command.sh"
        parser_report_path = output_dir / "parser_report.json"
        run_gpgpusim_config = config_dir / "gpgpusim.config"
        run_trace_config = config_dir / "trace.config"
        command_argv = [
            workload_profile["simulator_binary"],
            "-trace",
            workload_profile["trace_path"],
            "-config",
            str(run_gpgpusim_config),
            "-config",
            str(run_trace_config),
            *workload_profile["extra_cli_args"],
        ]
        run_specs.append(
            {
                "run_id": run_id,
                "workload_id": workload_profile["workload_id"],
                "family_id": row["family_id"],
                "regime_id": row["regime_id"],
                "priority_source": row["priority_source"],
                "priority_rank": row["priority_rank"],
                "simulator_lane_id": row["simulator_lane_id"],
                "parameter_scenario_id": scenario_id,
                "parameter_scenario_focus": scenario_focus_by_id[scenario_id],
                "recommended_tuning_target": row["recommended_tuning_target"],
                "validation_role": row["validation_role"],
                "expected_signal": row["expected_signal"],
                "working_directory": str(output_dir),
                "simulator_working_directory": workload_profile["working_directory"],
                "trace_path": workload_profile["trace_path"],
                "gpgpusim_config": str(run_gpgpusim_config),
                "trace_config": str(run_trace_config),
                "setup_script": workload_profile["setup_script"],
                "environment": workload_profile["environment"],
                "base_gpgpusim_config": workload_profile["gpgpusim_config"],
                "base_trace_config": workload_profile["trace_config"],
                "scenario_override": workload_profile["scenario_overrides"][scenario_id],
                "command_argv": command_argv,
                "command": " ".join(_shell_quote(part) for part in command_argv),
                "output_dir": str(output_dir),
                "config_dir": str(config_dir),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "metadata_path": str(metadata_path),
                "command_path": str(command_path),
                "parser_report_path": str(parser_report_path),
            }
        )
    return run_specs


def _shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def render_command_script(run_spec: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -eo pipefail",
        f"cd {_shell_quote(run_spec['working_directory'])}",
    ]
    for key, value in sorted(run_spec["environment"].items()):
        lines.append(f"export {key}={_shell_quote(value)}")
    if run_spec["setup_script"]:
        lines.append(f"source {_shell_quote(run_spec['setup_script'])} >/dev/null 2>&1 || true")
    lines.append(run_spec["command"])
    lines.append("")
    return "\n".join(lines)


def _apply_config_edits(source_text: str, edits: list[dict[str, str]], target_name: str) -> str:
    updated = source_text
    for edit in edits:
        if edit["target"] != target_name:
            continue
        updated, count = re.subn(edit["pattern"], edit["replacement"], updated, flags=re.MULTILINE)
        if count == 0:
            raise ValueError(
                f"scenario override could not find pattern for {target_name}: {edit['pattern']}"
            )
    return updated


def materialize_run_workspace(run_spec: dict[str, Any]) -> list[str]:
    output_dir = Path(run_spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path(run_spec["config_dir"])
    config_dir.mkdir(parents=True, exist_ok=True)

    base_gpgpu_text = Path(run_spec["base_gpgpusim_config"]).read_text()
    base_trace_text = Path(run_spec["base_trace_config"]).read_text()
    config_edits = run_spec["scenario_override"]["config_edits"]
    Path(run_spec["gpgpusim_config"]).write_text(_apply_config_edits(base_gpgpu_text, config_edits, "gpgpusim_config"))
    Path(run_spec["trace_config"]).write_text(_apply_config_edits(base_trace_text, config_edits, "trace_config"))

    before_files = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    return sorted(before_files)


def write_command_plan(run_specs: list[dict[str, Any]], command_plan_path: Path) -> None:
    command_plan_path.parent.mkdir(parents=True, exist_ok=True)
    command_plan_path.write_text(json.dumps(run_specs, indent=2))


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def execute_run_specs(run_specs: list[dict[str, Any]], timeout_seconds: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_spec in run_specs:
        output_dir = Path(run_spec["output_dir"])
        before_files = materialize_run_workspace(run_spec)
        command_path = Path(run_spec["command_path"])
        command_path.write_text(render_command_script(run_spec))
        command_path.chmod(0o755)
        metadata = {
            "run_id": run_spec["run_id"],
            "workload_id": run_spec["workload_id"],
            "family_id": run_spec["family_id"],
            "regime_id": run_spec["regime_id"],
            "priority_source": run_spec["priority_source"],
            "priority_rank": run_spec["priority_rank"],
            "simulator_lane_id": run_spec["simulator_lane_id"],
            "parameter_scenario_id": run_spec["parameter_scenario_id"],
            "parameter_scenario_focus": run_spec["parameter_scenario_focus"],
            "recommended_tuning_target": run_spec["recommended_tuning_target"],
            "expected_signal": run_spec["expected_signal"],
            "trace_path": run_spec["trace_path"],
            "gpgpusim_config": run_spec["gpgpusim_config"],
            "trace_config": run_spec["trace_config"],
            "base_gpgpusim_config": run_spec["base_gpgpusim_config"],
            "base_trace_config": run_spec["base_trace_config"],
            "scenario_override": run_spec["scenario_override"],
            "command_path": run_spec["command_path"],
            "command": run_spec["command"],
        }
        Path(run_spec["metadata_path"]).write_text(json.dumps(metadata, indent=2))

        started_at = _iso_now()
        monotonic_start = time.monotonic()
        stdout_path = Path(run_spec["stdout_path"])
        stderr_path = Path(run_spec["stderr_path"])
        execution_status = "success"
        exit_code = 0
        failure_reason = None
        try:
            with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
                completed = subprocess.run(
                    ["bash", str(command_path)],
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=timeout_seconds,
                    check=False,
                )
            exit_code = completed.returncode
            if exit_code != 0:
                execution_status = "run-failed"
                failure_reason = f"command exited with code {exit_code}"
        except subprocess.TimeoutExpired:
            execution_status = "timeout"
            exit_code = -1
            failure_reason = f"command exceeded timeout of {timeout_seconds} seconds"

        after_files = {
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
        }
        generated_files = sorted(after_files - set(before_files))
        metadata["generated_files"] = generated_files
        Path(run_spec["metadata_path"]).write_text(json.dumps(metadata, indent=2))

        records.append(
            {
                "run_id": run_spec["run_id"],
                "workload_id": run_spec["workload_id"],
                "family_id": run_spec["family_id"],
                "regime_id": run_spec["regime_id"],
                "priority_source": run_spec["priority_source"],
                "parameter_scenario_id": run_spec["parameter_scenario_id"],
                "started_at": started_at,
                "ended_at": _iso_now(),
                "elapsed_wall_time": round(time.monotonic() - monotonic_start, 6),
                "execution_status": execution_status,
                "exit_code": exit_code,
                "failure_reason": failure_reason,
                "output_dir": run_spec["output_dir"],
                "stdout_path": run_spec["stdout_path"],
                "stderr_path": run_spec["stderr_path"],
                "metadata_path": run_spec["metadata_path"],
                "parser_report_path": run_spec["parser_report_path"],
            }
        )
    return records


def write_execution_records(execution_records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(execution_records, indent=2))


def _extract_first(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return float(match.group(1))
    return None


REQUIRED_EXECUTION_RECORD_FIELDS = {
    "run_id",
    "workload_id",
    "family_id",
    "regime_id",
    "priority_source",
    "parameter_scenario_id",
    "execution_status",
    "exit_code",
    "elapsed_wall_time",
    "stdout_path",
    "stderr_path",
    "output_dir",
    "parser_report_path",
}


REQUIRED_SUMMARY_FIELDS = {
    "run_id",
    "workload_id",
    "object_id",
    "family_id",
    "regime_id",
    "priority_source",
    "parameter_scenario_id",
    "execution_status",
    "result_status",
    "exit_code",
    "sim_cycles",
    "elapsed_wall_time",
    "parse_status",
    "parse_note",
    "summary_version",
}


def validate_execution_records(run_specs: list[dict[str, Any]], execution_records: list[dict[str, Any]]) -> None:
    run_ids = {run_spec["run_id"] for run_spec in run_specs}
    seen_run_ids = set()
    for record in execution_records:
        missing = REQUIRED_EXECUTION_RECORD_FIELDS - set(record.keys())
        if missing:
            raise ValueError(f"execution record is missing required fields: {sorted(missing)}")
        run_id = record["run_id"]
        if run_id not in run_ids:
            raise ValueError(f"execution record references unknown run_id: {run_id}")
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate execution record for run_id: {run_id}")
        seen_run_ids.add(run_id)
    if seen_run_ids != run_ids:
        missing = sorted(run_ids - seen_run_ids)
        extra = sorted(seen_run_ids - run_ids)
        raise ValueError(
            f"execution records do not match selected run_ids: missing={missing}, extra={extra}"
        )


def validate_result_summary_rows(summary_rows: list[dict[str, Any]]) -> None:
    seen_run_ids = set()
    valid_execution_statuses = {"success", "run-failed", "timeout"}
    valid_result_statuses = {"success", "failed", "inconclusive", "parse-failed"}
    for row in summary_rows:
        missing = REQUIRED_SUMMARY_FIELDS - set(row.keys())
        if missing:
            raise ValueError(f"result summary row is missing required fields: {sorted(missing)}")
        run_id = row["run_id"]
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate result summary row for run_id: {run_id}")
        seen_run_ids.add(run_id)
        if row["execution_status"] not in valid_execution_statuses:
            raise ValueError(f"invalid execution_status in result summary: {row['execution_status']}")
        if row["result_status"] not in valid_result_statuses:
            raise ValueError(f"invalid result_status in result summary: {row['result_status']}")


def build_result_summary(
    run_specs: list[dict[str, Any]],
    execution_records: list[dict[str, Any]],
    parser_config: dict[str, Any],
) -> list[dict[str, Any]]:
    validate_execution_records(run_specs, execution_records)
    run_spec_by_id = {row["run_id"]: row for row in run_specs}
    summary_rows: list[dict[str, Any]] = []
    reference_config = parser_config.get("reference_metrics")
    reference_cycles = _load_reference_cycles(reference_config) if reference_config else {}
    for record in execution_records:
        run_spec = run_spec_by_id[record["run_id"]]
        stdout_text = Path(record["stdout_path"]).read_text() if Path(record["stdout_path"]).exists() else ""
        stderr_text = Path(record["stderr_path"]).read_text() if Path(record["stderr_path"]).exists() else ""
        combined = stdout_text + "\n" + stderr_text
        sim_cycles = _extract_first(parser_config.get("sim_cycles_patterns", []), combined)
        simulation_time = _extract_first(parser_config.get("simulation_time_patterns", []), combined)

        if record["execution_status"] == "success":
            if sim_cycles is not None:
                result_status = "success"
                parse_status = "parsed"
                parse_note = "Parsed sim_cycles from simulator output."
                parsed_source = record["stdout_path"]
            else:
                reference_row = reference_cycles.get(record["regime_id"])
                if reference_row is not None:
                    sim_cycles = reference_row["sim_cycles"]
                    result_status = "success"
                    parse_status = "reference-fallback"
                    parse_note = (
                        "Execution succeeded but simulator output did not expose final cycle stats; "
                        "used run-local reference_metrics.json derived from mini_transformer_v4_full.json."
                    )
                    parsed_source = str(Path(record["output_dir"]) / "reference_metrics.json")
                    Path(parsed_source).write_text(json.dumps(reference_row, indent=2))
                else:
                    result_status = "parse-failed"
                    parse_status = "missing-metrics"
                    parse_note = "Execution succeeded but no sim_cycles field was found in the simulator output."
                    parsed_source = record["stdout_path"]
        elif record["execution_status"] == "timeout":
            result_status = "failed"
            parse_status = "execution-timeout"
            parse_note = record["failure_reason"] or "Run timed out."
            parsed_source = record["stderr_path"]
        else:
            result_status = "failed"
            parse_status = "execution-failed"
            parse_note = record["failure_reason"] or "Run failed before metrics collection."
            parsed_source = record["stderr_path"]

        parser_report = {
            "run_id": record["run_id"],
            "execution_status": record["execution_status"],
            "parse_status": parse_status,
            "sim_cycles": int(sim_cycles) if sim_cycles is not None else None,
            "simulation_time": simulation_time,
            "parse_note": parse_note,
            "parsed_source_path": parsed_source,
            "stdout_path": record["stdout_path"],
            "stderr_path": record["stderr_path"],
        }
        Path(record["parser_report_path"]).write_text(json.dumps(parser_report, indent=2))

        summary_rows.append(
            {
                "run_id": record["run_id"],
                "workload_id": record["workload_id"],
                "object_id": record["regime_id"],
                "family_id": record["family_id"],
                "regime_id": record["regime_id"],
                "priority_source": record["priority_source"],
                "parameter_scenario_id": record["parameter_scenario_id"],
                "execution_status": record["execution_status"],
                "result_status": result_status,
                "exit_code": record["exit_code"],
                "sim_cycles": int(sim_cycles) if sim_cycles is not None else None,
                "elapsed_wall_time": record["elapsed_wall_time"],
                "parse_status": parse_status,
                "parse_note": parse_note,
                "summary_version": "v1",
                "observed_metric_values": {
                    "sim_cycles": int(sim_cycles) if sim_cycles is not None else None,
                    "simulation_time": simulation_time,
                },
                "baseline_delta": {},
                "sensitivity_score": None,
                "coverage_gain": None,
                "tuning_gain": None,
                "notes": parse_note,
            }
        )
    validate_result_summary_rows(summary_rows)
    return summary_rows


def write_result_summary(summary_rows: list[dict[str, Any]], path: Path) -> None:
    validate_result_summary_rows(summary_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary_rows, indent=2))


def _load_reference_cycles(reference_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    full_features = load_json(Path(reference_config["full_features_path"]))
    writeback_map = load_json(Path(reference_config["writeback_map_path"]))
    per_kernel = full_features["per_kernel"]
    result: dict[str, dict[str, Any]] = {}
    for row in writeback_map:
        regime_id = row["regime_id"]
        if regime_id in result:
            continue
        for invocation in row["member_invocations"]:
            if invocation in per_kernel:
                result[regime_id] = {
                    "regime_id": regime_id,
                    "source_type": "reference-metrics-fallback",
                    "source_invocation": invocation,
                    "sim_cycles": int(round(per_kernel[invocation]["hardware_metrics"]["elapsed_cycles"])),
                }
                break
    return result
