"""GNN trustworthiness acceptance for GCL ResNet-50 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import stable_hash

ACCEPTANCE_MANIFEST = "gnn_acceptance_manifest.json"
ACCEPTANCE_SUMMARY = "gnn_acceptance_summary.json"
ACCEPTANCE_REPORT = "gnn_acceptance_report.md"

CLAIM_NO_CORRECTNESS = "quantified_no_correctness_claim"
STATUS_WEAK = "weak_acceptance_structure_valid_but_correctness_unproven"
STATUS_MISSING = "not_evaluable_missing_artifacts"
STATUS_NO_GRAPH_SIGNAL = "rejected_no_graph_signal"
STATUS_TRAINING_INSUFFICIENT = "rejected_training_insufficient"
STATUS_UNSTABLE = "rejected_unstable_embedding"
STATUS_DOWNSTREAM_UNPROVEN = "rejected_downstream_unproven"

MIN_ASSIGNMENT_STABILITY_ARI = 0.8
MIN_ASSIGNMENT_STABILITY_NMI = 0.8
MIN_K_STABILITY = 0.8
MIN_REPRESENTATIVE_STABILITY_RATE = 0.8
MAX_CENTROID_DRIFT = 0.25
MAX_GLOBAL_WEIGHTED_MAPE = 1.0
MAX_GLOBAL_P95_RELATIVE_ERROR = 2.0


def evaluate_gnn_acceptance(
    *,
    full_trace_manifest: dict[str, Any],
    training_manifest: dict[str, Any],
    selector_artifacts: dict[str, Any],
    gate7_manifest: dict[str, Any],
    baseline_ablation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate GNN trustworthiness without mutating upstream gate artifacts."""

    acceptance_items = {
        "input_provenance": _evaluate_input_provenance(full_trace_manifest),
        "rgcn_structure": _evaluate_rgcn_structure(training_manifest),
        "training_adequacy": _evaluate_training_adequacy(training_manifest),
        "embedding_geometry_signal": _evaluate_embedding_geometry(gate7_manifest),
        "selector_cluster_result": _evaluate_selector_result(selector_artifacts),
        "baseline_ablation": _evaluate_baseline_ablation(baseline_ablation_report),
        "multi_seed_stability": _evaluate_multi_seed_stability(gate7_manifest),
        "semantic_cluster_correctness": _evaluate_semantic_alignment(gate7_manifest),
        "downstream_representative_usefulness": _evaluate_downstream_usefulness(
            gate7_manifest
        ),
    }
    blocking_gaps = _blocking_gaps(acceptance_items)
    status = _overall_status(acceptance_items)
    report = {
        "artifact_type": "gcl_gnn_acceptance_manifest",
        "artifact_version": "gnn_acceptance_manifest_v1",
        "workload_id": "resnet50",
        "input_artifact_hashes": {
            "full_trace_manifest_hash": full_trace_manifest.get(
                "full_trace_reproduction_manifest_hash"
            ),
            "training_run_manifest_hash": training_manifest.get(
                "training_run_manifest_hash"
            ),
            "selector_manifest_hash": selector_artifacts.get("selector_manifest_hash"),
            "gate7_cluster_correctness_manifest_hash": gate7_manifest.get(
                "gate7_cluster_correctness_manifest_hash"
            ),
        },
        "acceptance_items": acceptance_items,
        "blocking_gaps": blocking_gaps,
        "recommended_next_gates": _recommended_next_gates(acceptance_items),
        "gnn_acceptance_status": status,
        "claim_status": _claim_status_for(status, acceptance_items),
    }
    report["gnn_acceptance_manifest_hash"] = stable_hash(report)
    return report


def evaluate_gnn_acceptance_from_dir(root: Path) -> dict[str, Any]:
    root = Path(root)
    required = {
        "full_trace_manifest": "resnet50_full_trace_reproduction_manifest.json",
        "training_manifest": "rgcn_training_run_manifest.json",
        "selector_artifacts": "selector_artifacts.json",
        "gate7_manifest": "gate7_cluster_correctness_manifest.json",
    }
    loaded: dict[str, dict[str, Any]] = {}
    missing = []
    for key, filename in required.items():
        path = root / filename
        if not path.exists():
            missing.append(filename)
            continue
        loaded[key] = _read_json(path)
    if missing:
        blocker = {
            "artifact_type": "gcl_gnn_acceptance_blocker_report",
            "artifact_version": "gnn_acceptance_blocker_report_v1",
            "workload_id": "resnet50",
            "gnn_acceptance_status": STATUS_MISSING,
            "claim_status": CLAIM_NO_CORRECTNESS,
            "missing_artifacts": missing,
            "blocking_gaps": [
                "missing required acceptance input artifacts: " + ", ".join(missing)
            ],
            "acceptance_items": {},
            "input_artifact_hashes": {},
            "recommended_next_gates": ["rerun upstream formal gates before acceptance"],
        }
        blocker["gnn_acceptance_manifest_hash"] = stable_hash(blocker)
        return blocker
    baseline_path = root / "gnn_baseline_ablation_report.json"
    baseline = _read_json(baseline_path) if baseline_path.exists() else None
    return evaluate_gnn_acceptance(
        full_trace_manifest=loaded["full_trace_manifest"],
        training_manifest=loaded["training_manifest"],
        selector_artifacts=loaded["selector_artifacts"],
        gate7_manifest=loaded["gate7_manifest"],
        baseline_ablation_report=baseline,
    )


def write_gnn_acceptance_artifacts(
    out_dir: Path, report: dict[str, Any]
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_gnn_acceptance_markdown(report)
    report_hash = stable_hash({"markdown": markdown})
    manifest = dict(report)
    manifest["report_hash"] = report_hash
    manifest["gnn_acceptance_manifest_hash"] = stable_hash(
        {
            key: value
            for key, value in manifest.items()
            if key != "gnn_acceptance_manifest_hash"
        }
    )
    summary = {
        "artifact_type": "gcl_gnn_acceptance_summary",
        "artifact_version": "gnn_acceptance_summary_v1",
        "workload_id": manifest.get("workload_id"),
        "gnn_acceptance_status": manifest.get("gnn_acceptance_status"),
        "claim_status": manifest.get("claim_status"),
        "blocking_gap_count": len(manifest.get("blocking_gaps", [])),
        "manifest_hash": manifest["gnn_acceptance_manifest_hash"],
        "report_hash": report_hash,
    }
    summary["summary_hash"] = stable_hash(summary)
    _write_json(out_dir / ACCEPTANCE_MANIFEST, manifest)
    _write_json(out_dir / ACCEPTANCE_SUMMARY, summary)
    (out_dir / ACCEPTANCE_REPORT).write_text(markdown, encoding="utf-8")
    return {
        "manifest_path": out_dir / ACCEPTANCE_MANIFEST,
        "summary_path": out_dir / ACCEPTANCE_SUMMARY,
        "report_path": out_dir / ACCEPTANCE_REPORT,
    }


def validate_gnn_acceptance_manifest(
    manifest: dict[str, Any],
    *,
    summary: dict[str, Any] | None = None,
    markdown: str | None = None,
) -> None:
    if manifest.get("artifact_type") != "gcl_gnn_acceptance_manifest":
        raise ValueError("GNN acceptance manifest has invalid artifact_type")
    hashes = manifest.get("input_artifact_hashes")
    if not hashes:
        raise ValueError("GNN acceptance manifest requires input_artifact_hashes")
    required_hashes = [
        "full_trace_manifest_hash",
        "training_run_manifest_hash",
        "selector_manifest_hash",
        "gate7_cluster_correctness_manifest_hash",
    ]
    for key in required_hashes:
        if not hashes.get(key):
            raise ValueError(f"GNN acceptance manifest requires {key}")
    if not manifest.get("report_hash"):
        raise ValueError("GNN acceptance manifest requires report_hash")
    supplied_manifest_hash = manifest.get("gnn_acceptance_manifest_hash")
    if not supplied_manifest_hash:
        raise ValueError("GNN acceptance manifest requires gnn_acceptance_manifest_hash")
    computed_manifest_hash = stable_hash(
        {
            key: value
            for key, value in manifest.items()
            if key != "gnn_acceptance_manifest_hash"
        }
    )
    if supplied_manifest_hash != computed_manifest_hash:
        raise ValueError("GNN acceptance manifest_hash does not match manifest content")
    if markdown is not None and stable_hash({"markdown": markdown}) != manifest["report_hash"]:
        raise ValueError("GNN acceptance report_hash does not match markdown")
    if summary is not None:
        if summary.get("gnn_acceptance_status") != manifest.get("gnn_acceptance_status"):
            raise ValueError("GNN acceptance summary status does not match manifest")
        if summary.get("claim_status") != manifest.get("claim_status"):
            raise ValueError("GNN acceptance summary claim_status does not match manifest")
        if summary.get("manifest_hash") != supplied_manifest_hash:
            raise ValueError("GNN acceptance summary manifest_hash does not match manifest")
        if summary.get("report_hash") != manifest.get("report_hash"):
            raise ValueError("GNN acceptance summary report_hash does not match manifest")
        supplied_summary_hash = summary.get("summary_hash")
        if not supplied_summary_hash:
            raise ValueError("GNN acceptance summary requires summary_hash")
        computed_summary_hash = stable_hash(
            {key: value for key, value in summary.items() if key != "summary_hash"}
        )
        if supplied_summary_hash != computed_summary_hash:
            raise ValueError("GNN acceptance summary_hash does not match summary content")


def render_gnn_acceptance_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GCL ResNet-50 GNN Acceptance Report",
        "",
        "## Final Status",
        "",
        "```text",
        f"gnn_acceptance_status = {report.get('gnn_acceptance_status')}",
        f"claim_status = {report.get('claim_status')}",
        "```",
        "",
        "## Acceptance Items",
        "",
        "| Item | Status | Reason |",
        "| ---- | ------ | ------ |",
    ]
    for name, item in report.get("acceptance_items", {}).items():
        lines.append(f"| {name} | {item.get('status')} | {item.get('reason', '')} |")
    lines.extend(["", "## Blocking Gaps", ""])
    gaps = report.get("blocking_gaps", [])
    if gaps:
        for gap in gaps:
            lines.append(f"- {gap}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _evaluate_input_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    ok = (
        manifest.get("artifact_type") == "gcl_resnet50_full_trace_reproduction_manifest"
        and manifest.get("run_scope") == "real_resnet50_full_trace"
        and manifest.get("formal_full_trace_run") is True
        and _positive_int(manifest.get("input_kernel_invocation_count"))
        and _positive_int(manifest.get("input_cta_record_count"))
    )
    if ok:
        return _item("PASS", "formal real ResNet-50 full-trace provenance")
    return _item("FAIL", "input provenance is not a formal real full-trace run")


def _evaluate_rgcn_structure(manifest: dict[str, Any]) -> dict[str, Any]:
    arch = manifest.get("model_architecture", {})
    schema = manifest.get("edge_relation_schema", {})
    required_schema = {"control_flow", "data_source", "data_destination"}
    ok = (
        arch.get("layers") == 3
        and arch.get("input_dim") == 64
        and arch.get("hidden_dim") == 128
        and arch.get("kernel_embedding_dim") == 256
        and arch.get("relation_count") == 3
        and manifest.get("readout_hierarchy")
        == "node_to_warp_to_cta_to_selected_sm_to_kernel"
        and required_schema.issubset(schema.keys())
        and manifest.get("selector_embedding_source", "encoder_readout")
        != "projection_head_output"
    )
    if ok:
        return _item("PASS", "RGCN architecture and relation schema match contract")
    return _item("FAIL", "RGCN structure is missing relation-aware encoder contract")


def _evaluate_training_adequacy(manifest: dict[str, Any]) -> dict[str, Any]:
    steps = manifest.get("optimizer_config", {}).get("optimizer_step_count")
    has_curve = bool(manifest.get("loss_curve"))
    has_pairs = _positive_int(manifest.get("positive_pair_count")) and _positive_int(
        manifest.get("negative_pair_count")
    )
    ok = (
        _positive_int(manifest.get("train_graph_count"))
        and manifest["train_graph_count"] > 4
        and _positive_int(steps)
        and steps > 1
        and _positive_int(manifest.get("epoch_count"))
        and has_curve
        and has_pairs
    )
    if ok:
        return _item("PASS", "training run has multi-step evidence and loss curve")
    return _item("FAIL", "training is smoke-level or lacks training-curve evidence")


def _evaluate_embedding_geometry(gate7: dict[str, Any]) -> dict[str, Any]:
    metrics = gate7.get("embedding_geometry_metrics", {})
    if all(
        metrics.get(name) is not None
        for name in ["silhouette", "davies_bouldin", "inter_intra_ratio"]
    ):
        return _item("WEAK_PASS", "embedding geometry has separability signal")
    return _item("NOT_AVAILABLE", "embedding geometry metrics are missing")


def _evaluate_selector_result(selector: dict[str, Any]) -> dict[str, Any]:
    selected_k = selector.get("k_selection_report", {}).get("selected_k")
    assignments = selector.get("kmeans_cluster_assignment_table", {}).get("assignments", [])
    if selected_k and assignments:
        counts: dict[int, int] = {}
        for row in assignments:
            cluster = int(row["cluster_id"])
            counts[cluster] = counts.get(cluster, 0) + 1
        if len(counts) > 1:
            return _item("WEAK_PASS", f"selector produced {len(counts)} clusters")
    return _item("NOT_AVAILABLE", "selector clustering result is missing or degenerate")


def _evaluate_baseline_ablation(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return _item("NOT_AVAILABLE", "baseline ablation report is missing")
    normalized = dict(report)
    legacy_no_edge_key = False
    if (
        "node_feature_pooling_no_edge_baseline" not in normalized
        and "no_edge_baseline" in normalized
    ):
        normalized["node_feature_pooling_no_edge_baseline"] = normalized["no_edge_baseline"]
        legacy_no_edge_key = True
    required = [
        "full_rgcn",
        "random_embedding_baseline",
        "opcode_histogram_baseline",
        "node_feature_pooling_no_edge_baseline",
        "control_flow_only_rgcn",
        "data_flow_only_rgcn",
    ]
    metrics = [
        "silhouette",
        "davies_bouldin",
        "inter_intra_ratio",
        "assignment_stability_ari",
        "representative_metric_error",
    ]
    for key in required:
        row = normalized.get(key)
        if not isinstance(row, dict):
            return _item("FAIL", f"baseline ablation missing {key}")
        for metric in metrics:
            if _number(row.get(metric)) is None:
                return _item("FAIL", f"baseline ablation missing {key}.{metric}")
    full = normalized["full_rgcn"]
    higher_is_better = [
        "silhouette",
        "inter_intra_ratio",
        "assignment_stability_ari",
    ]
    lower_is_better = ["davies_bouldin", "representative_metric_error"]
    for key in required:
        if key == "full_rgcn":
            continue
        competitor = normalized[key]
        for metric in higher_is_better:
            if _number(full[metric]) <= _number(competitor[metric]):
                return _item(
                    "FAIL",
                    f"full RGCN does not beat {key} on {metric}",
                )
        for metric in lower_is_better:
            if _number(full[metric]) >= _number(competitor[metric]):
                return _item(
                    "FAIL",
                    f"full RGCN does not beat {key} on {metric}",
                )
    if legacy_no_edge_key:
        return _item(
            "PASS",
            "full RGCN beats required simple baselines; "
            "legacy no_edge_baseline normalized to canonical key",
        )
    return _item("PASS", "full RGCN beats required simple baselines")


def _evaluate_multi_seed_stability(gate7: dict[str, Any]) -> dict[str, Any]:
    stability = gate7.get("stability_report", {})
    if stability.get("stability_status") == "single_run_not_evaluated":
        return _item("NOT_AVAILABLE", "multi-seed stability was not evaluated")
    required = [
        "assignment_stability_ari",
        "assignment_stability_nmi",
        "centroid_drift",
        "k_stability",
        "representative_stability_rate",
    ]
    ok = (
        stability.get("training_seed_count", 0) >= 3
        and stability.get("kmeans_seed_count", 0) >= 5
        and all(stability.get(name) is not None for name in required)
        and float(stability["assignment_stability_ari"]) >= MIN_ASSIGNMENT_STABILITY_ARI
        and float(stability["assignment_stability_nmi"]) >= MIN_ASSIGNMENT_STABILITY_NMI
        and float(stability["centroid_drift"]) <= MAX_CENTROID_DRIFT
        and float(stability["k_stability"]) >= MIN_K_STABILITY
        and float(stability["representative_stability_rate"])
        >= MIN_REPRESENTATIVE_STABILITY_RATE
    )
    if ok:
        return _item("PASS", "multi-seed stability meets minimum seed counts")
    return _item("FAIL", "multi-seed stability report is under-seeded or incomplete")


def _evaluate_semantic_alignment(gate7: dict[str, Any]) -> dict[str, Any]:
    metrics = gate7.get("family_alignment_metrics", {})
    coverage = metrics.get("family_to_cluster_coverage", {})
    coarse_only = set(coverage.keys()) <= {"resnet50_real_trace"} if coverage else True
    if coarse_only:
        return _item("UNPROVEN", "family label granularity is too coarse")
    if (
        metrics.get("ari") is not None
        and metrics.get("nmi") is not None
        and metrics.get("v_measure") is not None
    ):
        return _item("PASS", "semantic labels are available for alignment")
    return _item("NOT_AVAILABLE", "semantic family alignment metrics are missing")


def _evaluate_downstream_usefulness(gate7: dict[str, Any]) -> dict[str, Any]:
    metric_report = gate7.get("metric_error_report", {})
    required = [
        "cluster_weighted_mape",
        "global_weighted_mape",
        "global_p95_relative_error",
    ]
    if metric_report.get("metric_claim_status") == "unavailable" or metric_report.get(
        "status"
    ) == "not_provided":
        return _item("NOT_AVAILABLE", "downstream metric rows were not provided")
    if not gate7.get("metric_source_manifest_hash"):
        return _item("FAIL", "downstream metric source provenance is missing")
    if not _representative_quality_complete(gate7.get("representative_quality_metrics", {})):
        return _item("FAIL", "representative quality evidence is missing")
    if all(metric_report.get(name) is not None for name in required):
        global_mape = _number(metric_report.get("global_weighted_mape"))
        p95_error = _number(metric_report.get("global_p95_relative_error"))
        if (
            global_mape is None
            or p95_error is None
            or global_mape > MAX_GLOBAL_WEIGHTED_MAPE
            or p95_error > MAX_GLOBAL_P95_RELATIVE_ERROR
        ):
            return _item("FAIL", "downstream representative error is too high")
        return _item("PASS", "representative downstream error metrics are available")
    return _item("NOT_AVAILABLE", "downstream representative error metrics are incomplete")


def _blocking_gaps(items: dict[str, dict[str, Any]]) -> list[str]:
    gaps = []
    gap_names = {
        "input_provenance": "input provenance is not a formal real full-trace run",
        "training_adequacy": "training is smoke-level or lacks training-curve evidence",
        "baseline_ablation": "baseline ablation evidence is missing or weak",
        "multi_seed_stability": "multi-seed stability evidence is missing or weak",
        "semantic_cluster_correctness": "semantic cluster correctness is unproven",
        "downstream_representative_usefulness": "downstream representative usefulness is unavailable",
    }
    for name, message in gap_names.items():
        status = items.get(name, {}).get("status")
        if status in {"FAIL", "NOT_AVAILABLE", "UNPROVEN"}:
            gaps.append(message)
    return gaps


def _overall_status(items: dict[str, dict[str, Any]]) -> str:
    if items.get("baseline_ablation", {}).get("status") == "FAIL":
        return STATUS_NO_GRAPH_SIGNAL
    if items.get("input_provenance", {}).get("status") == "FAIL":
        return STATUS_MISSING
    if items.get("multi_seed_stability", {}).get("status") == "FAIL":
        return STATUS_UNSTABLE
    if items.get("downstream_representative_usefulness", {}).get("status") == "FAIL":
        return STATUS_DOWNSTREAM_UNPROVEN
    if (
        items.get("training_adequacy", {}).get("status") == "FAIL"
        and items.get("baseline_ablation", {}).get("status") == "PASS"
        and items.get("multi_seed_stability", {}).get("status") == "PASS"
    ):
        return STATUS_TRAINING_INSUFFICIENT
    if all(item.get("status") == "PASS" for item in items.values()):
        return "accepted"
    return STATUS_WEAK


def _claim_status_for(status: str, items: dict[str, dict[str, Any]]) -> str:
    structure_tier = (
        items.get("input_provenance", {}).get("status") == "PASS"
        and items.get("rgcn_structure", {}).get("status") == "PASS"
        and items.get("embedding_geometry_signal", {}).get("status")
        in {"WEAK_PASS", "PASS"}
        and items.get("selector_cluster_result", {}).get("status")
        in {"WEAK_PASS", "PASS"}
    )
    stability_tier = (
        structure_tier and items.get("multi_seed_stability", {}).get("status") == "PASS"
    )
    semantic_tier = (
        stability_tier
        and items.get("semantic_cluster_correctness", {}).get("status") == "PASS"
    )
    downstream_tier = (
        semantic_tier
        and items.get("downstream_representative_usefulness", {}).get("status") == "PASS"
    )
    if (
        downstream_tier
        and status == "accepted"
        and all(item.get("status") == "PASS" for item in items.values())
    ):
        return "gnn_trustworthiness_accepted"
    if downstream_tier:
        return "representative_downstream_supported"
    if semantic_tier:
        return "semantic_cluster_supported"
    if stability_tier:
        return "cluster_stability_supported"
    if structure_tier:
        return "structure_valid_embedding_signal_only"
    return CLAIM_NO_CORRECTNESS


def _recommended_next_gates(items: dict[str, dict[str, Any]]) -> list[str]:
    recs = []
    if items.get("training_adequacy", {}).get("status") != "PASS":
        recs.append("run formal multi-step RGCN training")
    if items.get("baseline_ablation", {}).get("status") != "PASS":
        recs.append("run random histogram no-edge and edge-ablation baselines")
    if items.get("multi_seed_stability", {}).get("status") != "PASS":
        recs.append("run multi-seed training and clustering stability")
    if items.get("semantic_cluster_correctness", {}).get("status") != "PASS":
        recs.append("add fine-grained kernel semantic labels")
    if items.get("downstream_representative_usefulness", {}).get("status") != "PASS":
        recs.append("add measured or simulator metric representative validation")
    return recs


def _item(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _number(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _representative_quality_complete(report: dict[str, Any]) -> bool:
    clusters = report.get("cluster_reports")
    if not isinstance(clusters, list) or not clusters:
        return False
    required = [
        "mean_distance_to_representative",
        "p95_distance_to_representative",
        "max_distance_to_representative",
        "representative_rank_to_centroid",
        "outlier_member_ratio",
    ]
    return all(all(cluster.get(name) is not None for name in required) for cluster in clusters)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
