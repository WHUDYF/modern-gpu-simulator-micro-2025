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
    gate8_tuning_manifest: dict[str, Any] | None = None,
    representative_anchor_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not full_baseline_metrics and not measured_baseline_metrics:
        raise ValueError("full or measured baseline is required for speedup/accuracy claims")
    if gate8_tuning_manifest is None:
        raise ValueError("Gate8 tuning manifest is required for extension evaluation")
    if representative_anchor_table is None:
        raise ValueError("representative anchor provenance is required for extension evaluation")
    gate8_hash = gate8_tuning_manifest.get("gate8_tuning_manifest_hash")
    anchor_hash = representative_anchor_table.get("representative_anchor_table_hash")
    if not gate8_hash:
        raise ValueError("Gate8 tuning manifest hash is required")
    if not anchor_hash:
        raise ValueError("representative anchor table hash is required")
    baseline = measured_baseline_metrics or full_baseline_metrics or {}
    comparable_keys = sorted(set(sampled_metrics).intersection(baseline))
    if not comparable_keys:
        raise ValueError("baseline has no comparable metric keys")
    comparison = {}
    error_report = {}
    relative_errors = []
    for key in comparable_keys:
        sampled = float(sampled_metrics[key])
        expected = float(baseline[key])
        relative_error = _relative_error(sampled, expected)
        rounded_error = round(relative_error, 8)
        comparison[key] = {
            "sampled": sampled,
            "baseline": expected,
            "relative_error": rounded_error,
        }
        error_report[f"{key}_relative_error"] = rounded_error
        relative_errors.append(rounded_error)
    error_report["p95_relative_error"] = _percentile(relative_errors, 0.95)
    error_report["high_weight_bad_case_count"] = sum(
        1 for error in relative_errors if error > 0.2
    )
    speedup_report = {}
    if full_baseline_metrics:
        speedup_keys = sorted(set(sampled_metrics).intersection(full_baseline_metrics))
        for key in speedup_keys:
            sampled = float(sampled_metrics[key])
            full = float(full_baseline_metrics[key])
            if sampled:
                speedup_report[f"{key}_speedup"] = round(full / sampled, 8)
    tuning_effect_report = {
        "status": "evaluated_from_gate8_proposal",
        "source_gate8_tuning_manifest_hash": gate8_hash,
        "representative_anchor_table_hash": anchor_hash,
        "comparable_metric_count": len(comparable_keys),
        "max_relative_error": max(relative_errors) if relative_errors else None,
    }
    artifact = {
        "artifact_type": "gcl_resnet50_gate9_sampled_vs_full_evaluation",
        "artifact_version": "gate9_sampled_vs_full_evaluation_v1",
        "extension_label": EXTENSION_LABEL,
        "full_vs_sampled_simulation_report": comparison,
        "sampled_speedup_report": speedup_report,
        "sampled_error_report": error_report,
        "tuning_effect_report": tuning_effect_report,
    }
    artifact["gate9_simulator_evaluation_manifest"] = _gate9_manifest(
        artifact,
        source_gate8_tuning_manifest_hash=gate8_hash,
        representative_anchor_table_hash=anchor_hash,
    )
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
    artifact["gate9_simulator_evaluation_manifest"] = _gate9_manifest(
        artifact,
        source_gate8_tuning_manifest_hash=None,
        representative_anchor_table_hash=None,
    )
    artifact["gate9_sampled_vs_full_evaluation_hash"] = stable_hash(artifact)
    return artifact


def _relative_error(sampled: float, expected: float) -> float:
    if expected != 0.0:
        return abs(sampled - expected) / abs(expected)
    return 0.0 if sampled == 0.0 else 1.0


def _gate9_manifest(
    artifact: dict[str, Any],
    *,
    source_gate8_tuning_manifest_hash: str | None,
    representative_anchor_table_hash: str | None,
) -> dict[str, Any]:
    manifest = {
        "artifact_type": "gcl_resnet50_gate9_simulator_evaluation_manifest",
        "artifact_version": "gate9_simulator_evaluation_manifest_v1",
        "extension_label": EXTENSION_LABEL,
        "source_gate8_tuning_manifest_hash": source_gate8_tuning_manifest_hash,
        "representative_anchor_table_hash": representative_anchor_table_hash,
        "full_vs_sampled_simulation_report_hash": stable_hash(
            artifact["full_vs_sampled_simulation_report"]
        ),
        "sampled_speedup_report_hash": stable_hash(artifact["sampled_speedup_report"]),
        "sampled_error_report_hash": stable_hash(artifact["sampled_error_report"]),
        "tuning_effect_report_hash": stable_hash(artifact["tuning_effect_report"]),
    }
    manifest["gate9_simulator_evaluation_manifest_hash"] = stable_hash(manifest)
    return manifest


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
    return ordered[index]
