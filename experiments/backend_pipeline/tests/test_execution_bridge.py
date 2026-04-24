import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from experiments.backend_pipeline.execution_bridge import (  # noqa: E402
    _load_trace_proto_modules,
    build_result_summary,
    build_run_specs,
    execute_run_specs,
    scenario_focus_map,
    select_manifest_rows,
    validate_result_summary_rows,
)
from experiments.backend_pipeline.workload_profiles import load_workload_profile  # noqa: E402


RUN_SCRIPT = ROOT / "run_backend_execution.py"
RESULTS_ROOT = ROOT / "results" / "mini_transformer_v4"


def _worksheet() -> dict:
    return json.loads((RESULTS_ROOT / "backend_validation_worksheet_v1.json").read_text())


def _manifest() -> list[dict]:
    return json.loads((RESULTS_ROOT / "backend_run_manifest_v1.json").read_text())


def test_builtin_profile_loads_existing_repo_assets():
    profile = load_workload_profile("mini_transformer_v4")
    assert Path(profile["simulator_binary"]).exists()
    assert Path(profile["trace_path"]).exists()
    assert Path(profile["gpgpusim_config"]).exists()
    assert Path(profile["trace_config"]).exists()
    assert profile["extra_cli_args"] == ["-gpgpu_max_cycle", "10"]
    assert "R1_projection_dense" in profile["smoke_trace_builder"]["kernel_launches"]


def test_run_specs_are_stable_for_actual_manifest(tmp_path):
    profile = load_workload_profile("mini_transformer_v4")
    rows = select_manifest_rows(_manifest(), max_runs=2)
    run_specs = build_run_specs(rows, profile, scenario_focus_map(_worksheet()), tmp_path / "runs")
    assert len(run_specs) == 2
    first = run_specs[0]
    second_run_specs = build_run_specs(rows, profile, scenario_focus_map(_worksheet()), tmp_path / "runs")
    assert run_specs == second_run_specs
    assert first["run_id"] in first["output_dir"]
    assert first["parameter_scenario_focus"]
    assert "accel-sim.out" in first["command"]
    assert run_specs[0]["scenario_override"] != run_specs[1]["scenario_override"]


def test_build_run_specs_fails_when_scenario_mapping_is_missing(tmp_path):
    profile = load_workload_profile("mini_transformer_v4")
    rows = select_manifest_rows(_manifest(), max_runs=1)
    bad_mapping = {}
    try:
        build_run_specs(rows, profile, bad_mapping, tmp_path / "runs")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing scenario mapping to fail")
    assert "missing from validation worksheet" in message


def test_execute_run_specs_records_success_and_parser_extracts_metrics(tmp_path):
    fake_sim = tmp_path / "fake_sim.sh"
    fake_sim.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'gpu_tot_sim_cycle = 42'\n"
        "echo 'gpgpu_simulation_time = 1.25'\n"
    )
    fake_sim.chmod(0o755)
    trace = tmp_path / "dynamic_trace.pb"
    trace.write_text("fake")
    gpgpu = tmp_path / "gpgpusim.config"
    gpgpu.write_text("cfg")
    trace_cfg = tmp_path / "trace.config"
    trace_cfg.write_text("cfg")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "workload_id": "mini_transformer_v4",
                "working_directory": str(tmp_path),
                "simulator_binary": str(fake_sim),
                "setup_script": "",
                "trace_path": str(trace),
                "gpgpusim_config": str(gpgpu),
                "trace_config": str(trace_cfg),
                    "environment": {},
                    "extra_cli_args": [],
                    "parser": {
                        "sim_cycles_patterns": [r"gpu_tot_sim_cycle\s*=\s*([0-9]+)"],
                        "simulation_time_patterns": [r"gpgpu_simulation_time\s*=\s*([0-9]+(?:\.[0-9]+)?)"],
                    },
                    "scenario_overrides": {"S1_register_pressure": {"description": "demo", "config_edits": []}},
                }
            )
        )
    profile = load_workload_profile("mini_transformer_v4", profile_path)
    rows = [
        {
            "run_id": "RUN_demo",
            "family_id": "F1_dense_tiled",
            "regime_id": "R1_projection_dense",
            "priority_source": "importance-guided",
            "priority_rank": 1,
            "simulator_lane_id": "L1_dense_projection",
            "parameter_scenario_id": "S1_register_pressure",
            "recommended_tuning_target": "register-sensitive",
            "validation_role": "main-object",
            "expected_signal": "demo",
        }
    ]
    run_specs = build_run_specs(
        rows,
        profile,
        {"S1_register_pressure": "register-sensitive"},
        tmp_path / "runs",
    )
    records = execute_run_specs(run_specs, timeout_seconds=5)
    summary = build_result_summary(run_specs, records, profile["parser"])
    assert records[0]["execution_status"] == "success"
    assert summary[0]["result_status"] == "success"
    assert summary[0]["sim_cycles"] == 42
    assert summary[0]["observed_metric_values"]["simulation_time"] == 1.25
    parser_report = json.loads(Path(records[0]["parser_report_path"]).read_text())
    assert parser_report["parse_status"] == "parsed"


def test_execute_run_specs_records_timeout(tmp_path):
    fake_sim = tmp_path / "slow_sim.sh"
    fake_sim.write_text("#!/usr/bin/env bash\nsleep 2\n")
    fake_sim.chmod(0o755)
    trace = tmp_path / "dynamic_trace.pb"
    trace.write_text("fake")
    gpgpu = tmp_path / "gpgpusim.config"
    gpgpu.write_text("cfg")
    trace_cfg = tmp_path / "trace.config"
    trace_cfg.write_text("cfg")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "workload_id": "mini_transformer_v4",
                "working_directory": str(tmp_path),
                "simulator_binary": str(fake_sim),
                "setup_script": "",
                "trace_path": str(trace),
                "gpgpusim_config": str(gpgpu),
                "trace_config": str(trace_cfg),
                    "environment": {},
                    "extra_cli_args": [],
                    "parser": {"sim_cycles_patterns": [], "simulation_time_patterns": []},
                    "scenario_overrides": {"S1_register_pressure": {"description": "demo", "config_edits": []}},
                }
            )
        )
    profile = load_workload_profile("mini_transformer_v4", profile_path)
    rows = [
        {
            "run_id": "RUN_timeout",
            "family_id": "F1_dense_tiled",
            "regime_id": "R1_projection_dense",
            "priority_source": "importance-guided",
            "priority_rank": 1,
            "simulator_lane_id": "L1_dense_projection",
            "parameter_scenario_id": "S1_register_pressure",
            "recommended_tuning_target": "register-sensitive",
            "validation_role": "main-object",
            "expected_signal": "demo",
        }
    ]
    run_specs = build_run_specs(
        rows,
        profile,
        {"S1_register_pressure": "register-sensitive"},
        tmp_path / "runs",
    )
    records = execute_run_specs(run_specs, timeout_seconds=1)
    summary = build_result_summary(run_specs, records, profile["parser"])
    assert records[0]["execution_status"] == "timeout"
    assert summary[0]["result_status"] == "failed"
    assert "timeout" in summary[0]["parse_note"] or "exceeded timeout" in summary[0]["parse_note"]


def test_result_summary_validation_rejects_duplicates():
    row = {
        "run_id": "RUN_dup",
        "workload_id": "mini_transformer_v4",
        "object_id": "R1_projection_dense",
        "family_id": "F1_dense_tiled",
        "regime_id": "R1_projection_dense",
        "priority_source": "importance-guided",
        "parameter_scenario_id": "S1_register_pressure",
        "execution_status": "success",
        "result_status": "success",
        "exit_code": 0,
        "sim_cycles": 1,
        "elapsed_wall_time": 0.1,
        "parse_status": "parsed",
        "parse_note": "ok",
        "summary_version": "v1",
    }
    try:
        validate_result_summary_rows([row, dict(row)])
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected duplicate run_id validation to fail")
    assert "duplicate result summary row" in message


def test_result_summary_validation_rejects_missing_required_fields():
    row = {
        "run_id": "RUN_missing",
        "workload_id": "mini_transformer_v4",
    }
    try:
        validate_result_summary_rows([row])
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing field validation to fail")
    assert "missing required fields" in message


def test_validate_execution_records_rejects_missing_selected_run():
    from experiments.backend_pipeline.execution_bridge import validate_execution_records

    run_specs = [{"run_id": "RUN_a"}, {"run_id": "RUN_b"}]
    execution_records = [
        {
            "run_id": "RUN_a",
            "workload_id": "mini_transformer_v4",
            "family_id": "F1_dense_tiled",
            "regime_id": "R1_projection_dense",
            "priority_source": "importance-guided",
            "parameter_scenario_id": "S1_register_pressure",
            "execution_status": "success",
            "exit_code": 0,
            "elapsed_wall_time": 0.1,
            "stdout_path": "stdout.log",
            "stderr_path": "stderr.log",
            "output_dir": "run_a",
            "parser_report_path": "parser_report.json",
        }
    ]
    try:
        validate_execution_records(run_specs, execution_records)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing selected run to fail validation")
    assert "execution records do not match selected run_ids" in message


def test_result_summary_validation_requires_summary_version():
    row = {
        "run_id": "RUN_missing_summary_version",
        "workload_id": "mini_transformer_v4",
        "object_id": "R1_projection_dense",
        "family_id": "F1_dense_tiled",
        "regime_id": "R1_projection_dense",
        "priority_source": "importance-guided",
        "parameter_scenario_id": "S1_register_pressure",
        "execution_status": "success",
        "result_status": "success",
        "exit_code": 0,
        "sim_cycles": 1,
        "elapsed_wall_time": 0.1,
        "parse_status": "parsed",
        "parse_note": "ok",
    }
    try:
        validate_result_summary_rows([row])
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing summary_version to fail validation")
    assert "missing required fields" in message


def test_cli_plan_only_writes_command_plan(tmp_path):
    output_dir = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--run-manifest",
            str(RESULTS_ROOT / "backend_run_manifest_v1.json"),
            "--validation-worksheet",
            str(RESULTS_ROOT / "backend_validation_worksheet_v1.json"),
            "--output-dir",
            str(output_dir),
            "--max-runs",
            "1",
            "--plan-only",
        ],
        check=True,
    )
    assert (output_dir / "backend_command_plan_v1.json").exists()
    assert not (output_dir / "backend_execution_records_v1.json").exists()


def test_builtin_profile_trimmed_trace_uses_selected_kernel_event(tmp_path):
    profile = load_workload_profile("mini_transformer_v4")
    rows = [
        {
            "run_id": "RUN_attention_smoke",
            "family_id": "F1_dense_tiled",
            "regime_id": "R2_attention_score_dense",
            "priority_source": "importance-guided",
            "priority_rank": 1,
            "simulator_lane_id": "L2_attention_score",
            "parameter_scenario_id": "S1_register_pressure",
            "recommended_tuning_target": "register-sensitive",
            "validation_role": "main-object",
            "expected_signal": "attention smoke",
        }
    ]
    run_specs = build_run_specs(rows, profile, {"S1_register_pressure": "register-sensitive"}, tmp_path / "runs")
    execute_run_specs(run_specs, timeout_seconds=30)
    trace_pb2, _, _, _ = _load_trace_proto_modules()
    trace = trace_pb2.Trace()
    trace.ParseFromString(Path(run_specs[0]["trace_path"]).read_bytes())
    stream = trace.gpu_device[0].streams[0]
    assert stream.ordered_cuda_events[-1] == "kernel-5.trace"
    assert stream.kernels[0].id == 5
