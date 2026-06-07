import pytest

from experiments.gcl_phase_b.simulator_eval import evaluate_gate9_sampled_vs_full


def test_gate9_compares_sampled_against_full_baseline():
    report = evaluate_gate9_sampled_vs_full(
        sampled_metrics={"cycles": 90.0, "runtime_ms": 9.0},
        full_baseline_metrics={"cycles": 100.0, "runtime_ms": 10.0},
        measured_baseline_metrics={"cycles": 100.0, "runtime_ms": 10.0},
    )

    assert report["artifact_type"] == "gcl_resnet50_gate9_sampled_vs_full_evaluation"
    assert report["extension_label"] == "our_extension_not_original_gcl_sampler"
    assert report["full_vs_sampled_simulation_report"]["cycles"]["relative_error"] == 0.1
    assert report["sampled_speedup_report"]["runtime_ms_speedup"] > 1.0
    assert report["sampled_error_report"]["cycles_relative_error"] == 0.1
    assert report["gate9_simulator_evaluation_manifest"]["artifact_type"] == (
        "gcl_resnet50_gate9_simulator_evaluation_manifest"
    )
    assert report["gate9_simulator_evaluation_manifest"]["full_vs_sampled_simulation_report_hash"]
    assert report["gate9_simulator_evaluation_manifest"]["sampled_speedup_report_hash"]
    assert report["gate9_simulator_evaluation_manifest"]["sampled_error_report_hash"]
    assert report["gate9_simulator_evaluation_manifest"]["tuning_effect_report_hash"]


def test_gate9_rejects_speedup_claim_without_full_baseline():
    with pytest.raises(ValueError, match="baseline"):
        evaluate_gate9_sampled_vs_full(
            sampled_metrics={"cycles": 90.0},
            full_baseline_metrics=None,
            measured_baseline_metrics=None,
        )
