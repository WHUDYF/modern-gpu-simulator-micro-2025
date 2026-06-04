import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BUILD_SCRIPT = ROOT / "build_backend_outputs.py"
PLAN_SCRIPT = ROOT / "plan_backend_validation.py"
WRITEBACK_SCRIPT = ROOT / "apply_backend_writeback.py"
INPUT = REPO_ROOT / "experiments" / "mini_transformer" / "mini_transformer_v4_full.json"


def _prepare_environment(tmp_path: Path) -> Path:
    output_dir = tmp_path / "backend"
    subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--input", str(INPUT), "--output-dir", str(output_dir)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(PLAN_SCRIPT),
            "--priority-lane-table",
            str(output_dir / "backend_priority_lane_table_v1.json"),
            "--validation-worksheet",
            str(output_dir / "backend_validation_worksheet_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    return output_dir


def _write_result_summary(output_dir: Path) -> None:
    result_summary = [
        {
            "run_id": "RUN_importance_guided_R7_layernorm_reduction_S4_reduction_path",
            "object_id": "R7_layernorm_reduction",
            "family_id": "F2_reduction_normalize",
            "regime_id": "R7_layernorm_reduction",
            "priority_source": "importance-guided",
            "parameter_scenario_id": "S4_reduction_path",
            "observed_metric_values": {"demo": 1},
            "baseline_delta": {"demo": 0.1},
            "sensitivity_score": 0.2,
            "coverage_gain": 0.1,
            "tuning_gain": 0.0,
            "result_status": "inconclusive",
            "notes": "review object sample",
        },
        {
            "run_id": "RUN_importance_guided_R9_residual_elementwise_S7_constraint_regression",
            "object_id": "R9_residual_elementwise",
            "family_id": "F4_elementwise_residual",
            "regime_id": "R9_residual_elementwise",
            "priority_source": "importance-guided",
            "parameter_scenario_id": "S7_constraint_regression",
            "observed_metric_values": {"demo": 2},
            "baseline_delta": {"demo": -0.2},
            "sensitivity_score": 0.4,
            "coverage_gain": 0.0,
            "tuning_gain": 0.0,
            "result_status": "weak",
            "notes": "constraint object sample",
        },
        {
            "run_id": "RUN_importance_guided_R1_qkv_projection_dense_S1_register_pressure",
            "object_id": "R1_qkv_projection_dense",
            "family_id": "F1_dense_tiled_backbone",
            "regime_id": "R1_qkv_projection_dense",
            "priority_source": "importance-guided",
            "parameter_scenario_id": "S1_register_pressure",
            "observed_metric_values": {"demo": 3},
            "baseline_delta": {"demo": 0.5},
            "sensitivity_score": 0.9,
            "coverage_gain": 0.4,
            "tuning_gain": 0.6,
            "result_status": "success",
            "notes": "main object sample",
        },
    ]
    (output_dir / "backend_result_summary_v1.json").write_text(json.dumps(result_summary, indent=2))


def test_writeback_cli_writes_expected_files(tmp_path):
    output_dir = _prepare_environment(tmp_path)
    _write_result_summary(output_dir)
    subprocess.run(
        [
            sys.executable,
            str(WRITEBACK_SCRIPT),
            "--run-manifest",
            str(output_dir / "backend_run_manifest_v1.json"),
            "--result-summary",
            str(output_dir / "backend_result_summary_v1.json"),
            "--writeback-map",
            str(output_dir / "backend_writeback_map_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    assert (output_dir / "backend_writeback_updates_v1.json").exists()
    assert (output_dir / "backend_validation_status_v1.json").exists()


def test_writeback_preserves_review_object_status(tmp_path):
    output_dir = _prepare_environment(tmp_path)
    _write_result_summary(output_dir)
    subprocess.run(
        [
            sys.executable,
            str(WRITEBACK_SCRIPT),
            "--run-manifest",
            str(output_dir / "backend_run_manifest_v1.json"),
            "--result-summary",
            str(output_dir / "backend_result_summary_v1.json"),
            "--writeback-map",
            str(output_dir / "backend_writeback_map_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    updates = json.loads((output_dir / "backend_writeback_updates_v1.json").read_text())
    layernorm = next(row for row in updates if row["regime_id"] == "R7_layernorm_reduction")
    assert layernorm["review_status_update"] == "keep-review"
    assert layernorm["validation_status_update"] == "pending-review"


def test_writeback_preserves_constraint_object_bottleneck_note(tmp_path):
    output_dir = _prepare_environment(tmp_path)
    _write_result_summary(output_dir)
    subprocess.run(
        [
            sys.executable,
            str(WRITEBACK_SCRIPT),
            "--run-manifest",
            str(output_dir / "backend_run_manifest_v1.json"),
            "--result-summary",
            str(output_dir / "backend_result_summary_v1.json"),
            "--writeback-map",
            str(output_dir / "backend_writeback_map_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    updates = json.loads((output_dir / "backend_writeback_updates_v1.json").read_text())
    residual = next(row for row in updates if row["regime_id"] == "R9_residual_elementwise")
    assert "memory-side/constraint" in residual["workload_explanation_note"]
    validation = json.loads((output_dir / "backend_validation_status_v1.json").read_text())
    family_status = next(row for row in validation["family_status"] if row["family_id"] == "F4_elementwise_residual")
    assert "R9_residual_elementwise" in family_status["regime_ids"]


def test_writeback_fails_on_mismatched_run_and_regime(tmp_path):
    output_dir = _prepare_environment(tmp_path)
    bad = [
        {
            "run_id": "RUN_importance_guided_R1_qkv_projection_dense_S1_register_pressure",
            "object_id": "R2_attention_score_dense",
            "family_id": "F1_dense_tiled_backbone",
            "regime_id": "R2_attention_score_dense",
            "priority_source": "importance-guided",
            "parameter_scenario_id": "S1_register_pressure",
            "observed_metric_values": {},
            "baseline_delta": {},
            "sensitivity_score": 0.5,
            "coverage_gain": 0.0,
            "tuning_gain": 0.0,
            "result_status": "success",
            "notes": "intentional mismatch",
        }
    ]
    (output_dir / "backend_result_summary_v1.json").write_text(json.dumps(bad, indent=2))
    result = subprocess.run(
        [
            sys.executable,
            str(WRITEBACK_SCRIPT),
            "--run-manifest",
            str(output_dir / "backend_run_manifest_v1.json"),
            "--result-summary",
            str(output_dir / "backend_result_summary_v1.json"),
            "--writeback-map",
            str(output_dir / "backend_writeback_map_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "mismatches manifest" in result.stderr or "mismatches manifest" in result.stdout


def test_writeback_fails_on_duplicate_result_rows(tmp_path):
    output_dir = _prepare_environment(tmp_path)
    _write_result_summary(output_dir)
    result_summary = json.loads((output_dir / "backend_result_summary_v1.json").read_text())
    result_summary.append(result_summary[0])
    (output_dir / "backend_result_summary_v1.json").write_text(json.dumps(result_summary, indent=2))
    result = subprocess.run(
        [
            sys.executable,
            str(WRITEBACK_SCRIPT),
            "--run-manifest",
            str(output_dir / "backend_run_manifest_v1.json"),
            "--result-summary",
            str(output_dir / "backend_result_summary_v1.json"),
            "--writeback-map",
            str(output_dir / "backend_writeback_map_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "duplicate result_summary row" in result.stderr or "duplicate result_summary row" in result.stdout


def test_writeback_keeps_review_seed_when_result_summary_does_not_touch_review_object(tmp_path):
    output_dir = _prepare_environment(tmp_path)
    result_summary = [
        {
            "run_id": "RUN_importance_guided_R1_qkv_projection_dense_S1_register_pressure",
            "object_id": "R1_qkv_projection_dense",
            "family_id": "F1_dense_tiled_backbone",
            "regime_id": "R1_qkv_projection_dense",
            "priority_source": "importance-guided",
            "parameter_scenario_id": "S1_register_pressure",
            "observed_metric_values": {},
            "baseline_delta": {},
            "sensitivity_score": None,
            "coverage_gain": None,
            "tuning_gain": None,
            "result_status": "inconclusive",
            "notes": "leave review object untouched",
        }
    ]
    (output_dir / "backend_result_summary_v1.json").write_text(json.dumps(result_summary, indent=2))
    subprocess.run(
        [
            sys.executable,
            str(WRITEBACK_SCRIPT),
            "--run-manifest",
            str(output_dir / "backend_run_manifest_v1.json"),
            "--result-summary",
            str(output_dir / "backend_result_summary_v1.json"),
            "--writeback-map",
            str(output_dir / "backend_writeback_map_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    validation = json.loads((output_dir / "backend_validation_status_v1.json").read_text())
    layernorm = next(row for row in validation["regime_status"] if row["regime_id"] == "R7_layernorm_reduction")
    family = next(row for row in validation["family_status"] if row["family_id"] == "F2_reduction_normalize")
    assert layernorm["review_status"] == "keep-review"
    assert family["review_needed_regimes"] == ["R7_layernorm_reduction"]


def test_writeback_keeps_review_open_for_mixed_review_outcomes(tmp_path):
    output_dir = _prepare_environment(tmp_path)
    result_summary = [
        {
            "run_id": "RUN_importance_guided_R7_layernorm_reduction_S4_reduction_path",
            "object_id": "R7_layernorm_reduction",
            "family_id": "F2_reduction_normalize",
            "regime_id": "R7_layernorm_reduction",
            "priority_source": "importance-guided",
            "parameter_scenario_id": "S4_reduction_path",
            "observed_metric_values": {},
            "baseline_delta": {},
            "sensitivity_score": 1.0,
            "coverage_gain": 0.5,
            "tuning_gain": 0.5,
            "result_status": "success",
            "notes": "one strategy resolves review",
        },
        {
            "run_id": "RUN_time_only_R7_layernorm_reduction_S4_reduction_path",
            "object_id": "R7_layernorm_reduction",
            "family_id": "F2_reduction_normalize",
            "regime_id": "R7_layernorm_reduction",
            "priority_source": "time-only",
            "parameter_scenario_id": "S4_reduction_path",
            "observed_metric_values": {},
            "baseline_delta": {},
            "sensitivity_score": 0.0,
            "coverage_gain": 0.0,
            "tuning_gain": 0.0,
            "result_status": "failed",
            "notes": "another strategy keeps review open",
        },
    ]
    (output_dir / "backend_result_summary_v1.json").write_text(json.dumps(result_summary, indent=2))
    subprocess.run(
        [
            sys.executable,
            str(WRITEBACK_SCRIPT),
            "--run-manifest",
            str(output_dir / "backend_run_manifest_v1.json"),
            "--result-summary",
            str(output_dir / "backend_result_summary_v1.json"),
            "--writeback-map",
            str(output_dir / "backend_writeback_map_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    validation = json.loads((output_dir / "backend_validation_status_v1.json").read_text())
    layernorm = next(row for row in validation["regime_status"] if row["regime_id"] == "R7_layernorm_reduction")
    family = next(row for row in validation["family_status"] if row["family_id"] == "F2_reduction_normalize")
    assert layernorm["review_status"] == "keep-review"
    assert family["review_needed_regimes"] == ["R7_layernorm_reduction"]


def test_writeback_preserves_failed_status_even_if_another_run_validates(tmp_path):
    output_dir = _prepare_environment(tmp_path)
    result_summary = [
        {
            "run_id": "RUN_importance_guided_R1_qkv_projection_dense_S1_register_pressure",
            "object_id": "R1_qkv_projection_dense",
            "family_id": "F1_dense_tiled_backbone",
            "regime_id": "R1_qkv_projection_dense",
            "priority_source": "importance-guided",
            "parameter_scenario_id": "S1_register_pressure",
            "observed_metric_values": {},
            "baseline_delta": {},
            "sensitivity_score": 1.0,
            "coverage_gain": 0.5,
            "tuning_gain": 0.5,
            "result_status": "success",
            "notes": "success case",
        },
        {
            "run_id": "RUN_time_only_R1_qkv_projection_dense_S1_register_pressure",
            "object_id": "R1_qkv_projection_dense",
            "family_id": "F1_dense_tiled_backbone",
            "regime_id": "R1_qkv_projection_dense",
            "priority_source": "time-only",
            "parameter_scenario_id": "S1_register_pressure",
            "observed_metric_values": {},
            "baseline_delta": {},
            "sensitivity_score": 0.0,
            "coverage_gain": 0.0,
            "tuning_gain": 0.0,
            "result_status": "failed",
            "notes": "failure case",
        },
    ]
    (output_dir / "backend_result_summary_v1.json").write_text(json.dumps(result_summary, indent=2))
    subprocess.run(
        [
            sys.executable,
            str(WRITEBACK_SCRIPT),
            "--run-manifest",
            str(output_dir / "backend_run_manifest_v1.json"),
            "--result-summary",
            str(output_dir / "backend_result_summary_v1.json"),
            "--writeback-map",
            str(output_dir / "backend_writeback_map_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    validation = json.loads((output_dir / "backend_validation_status_v1.json").read_text())
    regime = next(row for row in validation["regime_status"] if row["regime_id"] == "R1_qkv_projection_dense")
    family = next(row for row in validation["family_status"] if row["family_id"] == "F1_dense_tiled_backbone")
    assert regime["current_status"] == "validated"
    assert "R1_qkv_projection_dense" not in family["failed_regimes"]


def test_writeback_keeps_parse_failed_main_object_pending(tmp_path):
    output_dir = _prepare_environment(tmp_path)
    result_summary = [
        {
            "run_id": "RUN_importance_guided_R1_qkv_projection_dense_S1_register_pressure",
            "object_id": "R1_qkv_projection_dense",
            "family_id": "F1_dense_tiled_backbone",
            "regime_id": "R1_qkv_projection_dense",
            "priority_source": "importance-guided",
            "parameter_scenario_id": "S1_register_pressure",
            "observed_metric_values": {},
            "baseline_delta": {},
            "sensitivity_score": None,
            "coverage_gain": None,
            "tuning_gain": None,
            "result_status": "parse-failed",
            "notes": "parser failed on smoke output",
        }
    ]
    (output_dir / "backend_result_summary_v1.json").write_text(json.dumps(result_summary, indent=2))
    subprocess.run(
        [
            sys.executable,
            str(WRITEBACK_SCRIPT),
            "--run-manifest",
            str(output_dir / "backend_run_manifest_v1.json"),
            "--result-summary",
            str(output_dir / "backend_result_summary_v1.json"),
            "--writeback-map",
            str(output_dir / "backend_writeback_map_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    updates = json.loads((output_dir / "backend_writeback_updates_v1.json").read_text())
    row = next(item for item in updates if item["regime_id"] == "R1_qkv_projection_dense")
    assert row["validation_status_update"] == "pending"
