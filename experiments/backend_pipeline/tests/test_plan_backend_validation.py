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
INPUT = REPO_ROOT / "experiments" / "mini_transformer" / "mini_transformer_v4_full.json"


def _prepare_outputs(tmp_path: Path) -> Path:
    output_dir = tmp_path / "backend"
    subprocess.run([sys.executable, str(BUILD_SCRIPT), "--input", str(INPUT), "--output-dir", str(output_dir)], check=True)
    subprocess.run([sys.executable, str(PLAN_SCRIPT), "--priority-lane-table", str(output_dir / "backend_priority_lane_table_v1.json"), "--validation-worksheet", str(output_dir / "backend_validation_worksheet_v1.json"), "--output-dir", str(output_dir)], check=True)
    return output_dir


def test_planner_writes_expected_execution_files(tmp_path):
    output_dir = _prepare_outputs(tmp_path)
    expected = {"backend_run_manifest_v1.json", "backend_scenario_matrix_v1.json", "backend_baseline_plan_v1.json", "backend_result_summary_v1.json"}
    assert expected.issubset({path.name for path in output_dir.iterdir()})


def test_planner_emits_empty_result_summary_template(tmp_path):
    output_dir = _prepare_outputs(tmp_path)
    result_summary = json.loads((output_dir / "backend_result_summary_v1.json").read_text())
    assert result_summary == []


def test_planner_marks_review_and_constraint_roles(tmp_path):
    output_dir = _prepare_outputs(tmp_path)
    manifest = json.loads((output_dir / "backend_run_manifest_v1.json").read_text())
    roles = {row["regime_id"]: row["validation_role"] for row in manifest if row["priority_source"] == "importance-guided"}
    assert roles["R4_layernorm_reduction"] == "review-object"
    assert roles["R6_residual_elementwise"] == "constraint-object"


def test_planner_uses_fixed_no_priority_order_and_four_baselines(tmp_path):
    output_dir = _prepare_outputs(tmp_path)
    manifest = json.loads((output_dir / "backend_run_manifest_v1.json").read_text())
    by_source = {}
    for row in manifest:
        by_source.setdefault(row["priority_source"], []).append(row)
    assert {"importance-guided", "time-only", "name-based", "no-priority"} == set(by_source)
    no_priority = by_source["no-priority"]
    ordered_regimes = []
    for row in no_priority:
        if row["regime_id"] not in ordered_regimes:
            ordered_regimes.append(row["regime_id"])
    assert ordered_regimes == ["R1_projection_dense", "R2_attention_score_dense", "R3_softmax_reduction", "R4_layernorm_reduction", "R5_context_streaming", "R6_residual_elementwise"]


def test_planner_fails_when_only_one_baseline_strategy_is_present(tmp_path):
    output_dir = tmp_path / "backend"
    subprocess.run([sys.executable, str(BUILD_SCRIPT), "--input", str(INPUT), "--output-dir", str(output_dir)], check=True)
    priority = json.loads((output_dir / "backend_priority_lane_table_v1.json").read_text())
    priority = [row for row in priority if row["priority_source"] == "importance-guided"]
    (output_dir / "priority_one_source.json").write_text(json.dumps(priority, indent=2))
    result = subprocess.run(
        [
            sys.executable,
            str(PLAN_SCRIPT),
            "--priority-lane-table",
            str(output_dir / "priority_one_source.json"),
            "--validation-worksheet",
            str(output_dir / "backend_validation_worksheet_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "comparison strategies" in result.stderr or "comparison strategies" in result.stdout


def test_planner_fails_when_worksheet_and_priority_sources_diverge(tmp_path):
    output_dir = tmp_path / "backend"
    subprocess.run([sys.executable, str(BUILD_SCRIPT), "--input", str(INPUT), "--output-dir", str(output_dir)], check=True)
    worksheet = json.loads((output_dir / "backend_validation_worksheet_v1.json").read_text())
    worksheet["budget_definition"]["comparison_strategies"] = ["importance-guided"]
    (output_dir / "worksheet_one_source.json").write_text(json.dumps(worksheet, indent=2))
    result = subprocess.run(
        [
            sys.executable,
            str(PLAN_SCRIPT),
            "--priority-lane-table",
            str(output_dir / "backend_priority_lane_table_v1.json"),
            "--validation-worksheet",
            str(output_dir / "worksheet_one_source.json"),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "comparison strategies" in result.stderr or "comparison strategies" in result.stdout


def test_planner_fails_on_missing_priority_source_field(tmp_path):
    output_dir = tmp_path / "backend"
    subprocess.run([sys.executable, str(BUILD_SCRIPT), "--input", str(INPUT), "--output-dir", str(output_dir)], check=True)
    priority = json.loads((output_dir / "backend_priority_lane_table_v1.json").read_text())
    priority[0].pop("priority_source")
    (output_dir / "priority_missing_field.json").write_text(json.dumps(priority, indent=2))
    result = subprocess.run(
        [
            sys.executable,
            str(PLAN_SCRIPT),
            "--priority-lane-table",
            str(output_dir / "priority_missing_field.json"),
            "--validation-worksheet",
            str(output_dir / "backend_validation_worksheet_v1.json"),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing required fields" in result.stderr or "missing required fields" in result.stdout


def test_planner_derives_budget_policy_from_worksheet_and_dedupes_selected_regimes(tmp_path):
    output_dir = tmp_path / "backend"
    subprocess.run([sys.executable, str(BUILD_SCRIPT), "--input", str(INPUT), "--output-dir", str(output_dir)], check=True)
    worksheet = json.loads((output_dir / "backend_validation_worksheet_v1.json").read_text())
    worksheet["budget_definition"]["family_preselection_count"] = 1
    worksheet["budget_definition"]["main_object_max_scenarios"] = 1
    worksheet["budget_definition"]["review_object_max_scenarios"] = 2
    (output_dir / "worksheet_custom.json").write_text(json.dumps(worksheet, indent=2))
    subprocess.run(
        [
            sys.executable,
            str(PLAN_SCRIPT),
            "--priority-lane-table",
            str(output_dir / "backend_priority_lane_table_v1.json"),
            "--validation-worksheet",
            str(output_dir / "worksheet_custom.json"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    plan = json.loads((output_dir / "backend_baseline_plan_v1.json").read_text())
    assert plan["budget_policy"]["family_preselection_count"] == 1
    assert plan["budget_policy"]["main_object_max_scenarios"] == 1
    assert plan["budget_policy"]["review_object_max_scenarios"] == 2
    assert plan["strategies"]["importance-guided"]["selected_families"] == [
        "F1_dense_tiled",
        "F4_elementwise_fusion",
    ]
    assert plan["strategies"]["importance-guided"]["selected_regimes"] == [
        "R1_projection_dense",
        "R2_attention_score_dense",
        "R6_residual_elementwise",
    ]
