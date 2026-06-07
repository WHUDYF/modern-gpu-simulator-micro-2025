"""Gate 9 sampled-vs-full simulator evaluation extension."""

from __future__ import annotations

from typing import Any

from .tuning import EXTENSION_LABEL
from .utils import stable_hash


def evaluate_gate9_sampled_vs_full(
    *,
    sampled_metrics: dict[str, float],
    full_baseline_metrics: dict[str, float] | None,
    measured_baseline_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not full_baseline_metrics and not measured_baseline_metrics:
        raise ValueError("full or measured baseline is required for speedup/accuracy claims")
    baseline = full_baseline_metrics or measured_baseline_metrics or {}
    comparable_keys = sorted(set(sampled_metrics).intersection(baseline))
    if not comparable_keys:
        raise ValueError("baseline has no comparable metric keys")
    comparison = {}
    error_report = {}
    for key in comparable_keys:
        sampled = float(sampled_metrics[key])
        expected = float(baseline[key])
        relative_error = abs(sampled - expected) / expected if expected else 0.0
        comparison[key] = {
            "sampled": sampled,
            "baseline": expected,
            "relative_error": round(relative_error, 8),
        }
        error_report[f"{key}_relative_error"] = round(relative_error, 8)
    speedup_report = {}
    if full_baseline_metrics:
        for key in comparable_keys:
            sampled = float(sampled_metrics[key])
            full = float(full_baseline_metrics[key])
            if sampled:
                speedup_report[f"{key}_speedup"] = round(full / sampled, 8)
    artifact = {
        "artifact_type": "gcl_resnet50_gate9_sampled_vs_full_evaluation",
        "artifact_version": "gate9_sampled_vs_full_evaluation_v1",
        "extension_label": EXTENSION_LABEL,
        "full_vs_sampled_simulation_report": comparison,
        "sampled_speedup_report": speedup_report,
        "sampled_error_report": error_report,
        "tuning_effect_report": {"status": "not_evaluated_without_tuning_run"},
    }
    artifact["gate9_simulator_evaluation_manifest"] = _gate9_manifest(artifact)
    artifact["gate9_sampled_vs_full_evaluation_hash"] = stable_hash(artifact)
    return artifact


def gate9_baseline_missing_report() -> dict[str, Any]:
    artifact = {
        "artifact_type": "gcl_resnet50_gate9_sampled_vs_full_evaluation",
        "artifact_version": "gate9_sampled_vs_full_evaluation_v1",
        "extension_label": EXTENSION_LABEL,
        "claim_status": "baseline_missing_no_speedup_or_accuracy_claim",
        "full_vs_sampled_simulation_report": {},
        "sampled_speedup_report": {},
        "sampled_error_report": {},
        "tuning_effect_report": {"status": "not_evaluated_without_baseline"},
    }
    artifact["gate9_simulator_evaluation_manifest"] = _gate9_manifest(artifact)
    artifact["gate9_sampled_vs_full_evaluation_hash"] = stable_hash(artifact)
    return artifact


def _gate9_manifest(artifact: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "artifact_type": "gcl_resnet50_gate9_simulator_evaluation_manifest",
        "artifact_version": "gate9_simulator_evaluation_manifest_v1",
        "extension_label": EXTENSION_LABEL,
        "full_vs_sampled_simulation_report_hash": stable_hash(
            artifact["full_vs_sampled_simulation_report"]
        ),
        "sampled_speedup_report_hash": stable_hash(artifact["sampled_speedup_report"]),
        "sampled_error_report_hash": stable_hash(artifact["sampled_error_report"]),
        "tuning_effect_report_hash": stable_hash(artifact["tuning_effect_report"]),
    }
    manifest["gate9_simulator_evaluation_manifest_hash"] = stable_hash(manifest)
    return manifest
