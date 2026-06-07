import pytest

from experiments.gcl_phase_b.simulator_eval import evaluate_gate9_sampled_vs_full


def test_gate9_compares_sampled_against_full_baseline():
    report = evaluate_gate9_sampled_vs_full(
        sampled_metrics={"cycles": 90.0, "runtime_ms": 9.0},
        full_baseline_metrics={"cycles": 100.0, "runtime_ms": 10.0},
        measured_baseline_metrics={"cycles": 100.0, "runtime_ms": 10.0},
        gate8_tuning_manifest={
            "artifact_type": "gcl_resnet50_gate8_tuning_manifest",
            "gate8_tuning_manifest_hash": "gate8-hash",
        },
        representative_anchor_table={
            "artifact_type": "gcl_resnet50_representative_anchor_table",
            "representative_anchor_table_hash": "anchor-hash",
        },
    )

    assert report["artifact_type"] == "gcl_resnet50_gate9_sampled_vs_full_evaluation"
    assert report["extension_label"] == "our_extension_not_original_gcl_sampler"
    assert report["full_vs_sampled_simulation_report"]["cycles"]["relative_error"] == 0.1
    assert report["sampled_speedup_report"]["runtime_ms_speedup"] > 1.0
    assert report["sampled_error_report"]["cycles_relative_error"] == 0.1
    assert report["sampled_error_report"]["p95_relative_error"] == 0.1
    assert report["sampled_error_report"]["high_weight_bad_case_count"] == 0
    assert report["tuning_effect_report"]["status"] == "evaluated_from_gate8_proposal"
    assert report["tuning_effect_report"]["source_gate8_tuning_manifest_hash"] == "gate8-hash"
    assert report["tuning_effect_report"]["representative_anchor_table_hash"] == "anchor-hash"
    assert report["gate9_simulator_evaluation_manifest"]["artifact_type"] == (
        "gcl_resnet50_gate9_simulator_evaluation_manifest"
    )
    assert report["gate9_simulator_evaluation_manifest"]["source_gate8_tuning_manifest_hash"] == (
        "gate8-hash"
    )
    assert report["gate9_simulator_evaluation_manifest"]["representative_anchor_table_hash"] == (
        "anchor-hash"
    )
    assert report["gate9_simulator_evaluation_manifest"]["full_vs_sampled_simulation_report_hash"]
    assert report["gate9_simulator_evaluation_manifest"]["sampled_speedup_report_hash"]
    assert report["gate9_simulator_evaluation_manifest"]["sampled_error_report_hash"]
    assert report["gate9_simulator_evaluation_manifest"]["tuning_effect_report_hash"]


def test_gate9_accuracy_uses_measured_baseline_when_full_baseline_also_exists():
    report = evaluate_gate9_sampled_vs_full(
        sampled_metrics={"cycles": 90.0, "runtime_ms": 9.0},
        full_baseline_metrics={"cycles": 100.0, "runtime_ms": 10.0},
        measured_baseline_metrics={"cycles": 120.0, "runtime_ms": 12.0},
        gate8_tuning_manifest={
            "artifact_type": "gcl_resnet50_gate8_tuning_manifest",
            "gate8_tuning_manifest_hash": "gate8-hash",
        },
        representative_anchor_table={
            "artifact_type": "gcl_resnet50_representative_anchor_table",
            "representative_anchor_table_hash": "anchor-hash",
        },
    )

    assert report["full_vs_sampled_simulation_report"]["cycles"]["baseline"] == 120.0
    assert report["full_vs_sampled_simulation_report"]["cycles"]["relative_error"] == 0.25
    assert report["sampled_error_report"]["cycles_relative_error"] == 0.25
    assert report["sampled_speedup_report"]["cycles_speedup"] == round(100.0 / 90.0, 8)


def test_gate9_rejects_speedup_claim_without_full_baseline():
    with pytest.raises(ValueError, match="baseline"):
        evaluate_gate9_sampled_vs_full(
            sampled_metrics={"cycles": 90.0},
            full_baseline_metrics=None,
            measured_baseline_metrics=None,
        )


def test_gate9_rejects_extension_evaluation_without_gate8_or_anchor_provenance():
    with pytest.raises(ValueError, match="Gate8"):
        evaluate_gate9_sampled_vs_full(
            sampled_metrics={"cycles": 90.0},
            full_baseline_metrics={"cycles": 100.0},
            measured_baseline_metrics=None,
        )
