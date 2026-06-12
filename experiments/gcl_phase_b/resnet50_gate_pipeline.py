"""ResNet-50 Gate 1-7 formal GCL reproduction pipeline."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from .graph_builder import build_phase_b_graphs, validate_phase_b_graph_artifact
from .correctness import (
    GATE7_CLUSTER_CORRECTNESS_FILENAME,
    GATE7_REPORT_FILENAMES,
    evaluate_gate7_correctness_from_artifacts,
)
from .embedding_export import (
    GATE5_EXPORT_PROGRESS_FILENAME,
    READOUT_HIERARCHY,
    build_gate5_lineage_bundle,
    export_phase_b_embedding_table,
)
from .pipeline import create_augmentation_manifest_bundle
from .resnet50_adapter import build_resnet50_trace_adapter_bundle
from .resnet50_gate0 import GATE0_BLOCKER_FILENAME
from .resnet50_gate0 import GATE0_MANIFEST_FILENAME
from .resnet50_manifest import build_representative_sm_manifest_from_bundle
from .selector import select_phase_b_representatives
from .simulator_eval import evaluate_gate9_sampled_vs_full, gate9_baseline_missing_report
from .tensorizer import tensor_from_jsonable, tensor_to_jsonable, tensorize_phase_b_graphs
from .tuning import generate_gate8_tuning_vectors
from .trace_scope import build_phase_b_trace_records
from .utils import hash_without, read_json, stable_hash, write_json
from experiments.gcl_phase_a.train import train_minimal_contrastive
from experiments.gcl_phase_a.rgcn import MinimalRGCNEncoder, ProjectionHead
from experiments.gcl_phase_a.pipeline import load_checkpoint_weights_only

GATE1_7_PIPELINE_MANIFEST_FILENAME = "gate1_7_pipeline_manifest.json"
GATE1_PLUS_OUTPUT_FILENAMES = {
    "resnet50_trace_adapter_bundle.json",
    "representative_sm_trace_manifest.json",
    "selected_sm_policy_report.json",
    "scope_preview_report.json",
    "canonical_graph_bundle.json",
    "graph_tensor_bundle.json",
    "augmentation_manifest.json",
    "rgcn_training_run_manifest.json",
    "rgcn_checkpoint_manifest.json",
    "readout_manifest.json",
    "gate5_lineage_bundle.json",
    "kernel_embedding_table.json",
    "embedding_export_report.json",
    "selector_artifacts.json",
    GATE7_CLUSTER_CORRECTNESS_FILENAME,
    *GATE7_REPORT_FILENAMES.values(),
    "cluster_tuning_vector_table.json",
    "tuning_vector_provenance_report.json",
    "tuning_safety_report.json",
    "gate8_tuning_manifest.json",
    "gate8_tuning_vector_proposal.json",
    "full_vs_sampled_simulation_report.json",
    "sampled_speedup_report.json",
    "sampled_error_report.json",
    "tuning_effect_report.json",
    "gate9_simulator_evaluation_manifest.json",
    "gate9_sampled_vs_full_evaluation.json",
    "rgcn_checkpoint.pt",
}
GATE6_PLUS_OUTPUT_FILENAMES = {
    "selector_artifacts.json",
    GATE7_CLUSTER_CORRECTNESS_FILENAME,
    *GATE7_REPORT_FILENAMES.values(),
    "cluster_tuning_vector_table.json",
    "tuning_vector_provenance_report.json",
    "tuning_safety_report.json",
    "gate8_tuning_manifest.json",
    "gate8_tuning_vector_proposal.json",
    "full_vs_sampled_simulation_report.json",
    "sampled_speedup_report.json",
    "sampled_error_report.json",
    "tuning_effect_report.json",
    "gate9_simulator_evaluation_manifest.json",
    "gate9_sampled_vs_full_evaluation.json",
}
GATE5_TRAINING_GRAPH_LIMIT = 4


def run_resnet50_gate1_to_gate5(
    root: Path,
    out_dir: Path,
    seed: int = 20260606,
    invocation_limit: int | None = None,
    invocation_ids: list[str] | None = None,
) -> dict[str, Any]:
    return run_resnet50_gate1_to_gate7(
        root,
        out_dir,
        seed=seed,
        invocation_limit=invocation_limit,
        invocation_ids=invocation_ids,
        stop_after_gate5=True,
    )


def run_resnet50_gate1_to_gate7(
    root: Path,
    out_dir: Path,
    seed: int = 20260606,
    baseline_artifacts_path: Path | None = None,
    invocation_limit: int | None = None,
    invocation_ids: list[str] | None = None,
    stop_after_gate5: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    blocker_path = root / GATE0_BLOCKER_FILENAME
    if blocker_path.exists() and not (root / GATE0_MANIFEST_FILENAME).exists():
        _remove_resnet50_artifacts(out_dir, GATE1_PLUS_OUTPUT_FILENAMES)
        return _write_gate0_blocked_pipeline_manifest(root, out_dir, seed)

    adapter_bundle = build_resnet50_trace_adapter_bundle(
        root,
        invocation_limit=invocation_limit,
        invocation_ids=invocation_ids,
    )
    write_json(out_dir / "resnet50_trace_adapter_bundle.json", adapter_bundle)

    trace_manifest, report_bundle, preview = build_representative_sm_manifest_from_bundle(
        adapter_bundle
    )
    write_json(out_dir / "representative_sm_trace_manifest.json", trace_manifest)
    write_json(out_dir / "selected_sm_policy_report.json", report_bundle)
    write_json(out_dir / "scope_preview_report.json", preview)

    records = build_phase_b_trace_records(trace_manifest)
    graphs = build_phase_b_graphs(records)
    for graph in graphs:
        validate_phase_b_graph_artifact(graph)
    canonical_graph_bundle = {
        "artifact_type": "gcl_resnet50_canonical_graph_bundle",
        "artifact_version": "gate3_canonical_graph_bundle_v1",
        "source_trace_manifest_hash": trace_manifest["trace_manifest_hash"],
        "graphs": graphs,
    }
    canonical_graph_bundle["canonical_graph_bundle_hash"] = hash_without(
        canonical_graph_bundle, "canonical_graph_bundle_hash"
    )
    write_json(out_dir / "canonical_graph_bundle.json", canonical_graph_bundle)

    tensors = tensorize_phase_b_graphs(graphs)
    graph_tensor_bundle = {
        "artifact_type": "gcl_resnet50_graph_tensor_bundle",
        "artifact_version": "gate4_graph_tensor_bundle_v1",
        "source_canonical_graph_bundle_hash": canonical_graph_bundle[
            "canonical_graph_bundle_hash"
        ],
        "tensors": [tensor_to_jsonable(tensor) for tensor in tensors],
    }
    graph_tensor_bundle["graph_tensor_bundle_hash"] = hash_without(
        graph_tensor_bundle, "graph_tensor_bundle_hash"
    )
    write_json(out_dir / "graph_tensor_bundle.json", graph_tensor_bundle)

    augmentation_bundle = create_augmentation_manifest_bundle(tensors, seed=seed)
    write_json(out_dir / "augmentation_manifest.json", augmentation_bundle)

    embedding_table = _run_gate5_training_and_export(
        tensors=tensors,
        graph_tensor_bundle=graph_tensor_bundle,
        augmentation_bundle=augmentation_bundle,
        out_dir=out_dir,
        seed=seed,
    )
    if stop_after_gate5:
        _remove_resnet50_artifacts(out_dir, GATE6_PLUS_OUTPUT_FILENAMES)
        manifest = _gate5_pipeline_manifest(
            seed=seed,
            invocation_limit=invocation_limit,
            invocation_ids=invocation_ids,
            adapter_bundle=adapter_bundle,
            trace_manifest=trace_manifest,
            canonical_graph_bundle=canonical_graph_bundle,
            graph_tensor_bundle=graph_tensor_bundle,
            embedding_table=embedding_table,
        )
        write_json(out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME, manifest)
        return manifest

    selector_artifacts = select_phase_b_representatives(
        embedding_table,
        seed=seed,
        gate5_artifact_root=out_dir,
    )
    write_json(out_dir / "selector_artifacts.json", selector_artifacts)

    baseline_artifacts = _load_baseline_artifacts(baseline_artifacts_path)
    correctness_manifest = evaluate_gate7_correctness_from_artifacts(
        selector_artifacts=selector_artifacts,
        embedding_table=embedding_table,
        metric_rows=baseline_artifacts.get("metric_rows") if baseline_artifacts else None,
    )
    for report_key, filename in GATE7_REPORT_FILENAMES.items():
        write_json(out_dir / filename, correctness_manifest["gate7_report_artifacts"][report_key])
    write_json(out_dir / GATE7_CLUSTER_CORRECTNESS_FILENAME, correctness_manifest)
    gate8_proposal, gate9_report, final_gate = _emit_gate8_gate9_extension_artifacts(
        correctness_manifest=correctness_manifest,
        selector_artifacts=selector_artifacts,
        baseline_artifacts=baseline_artifacts,
        out_dir=out_dir,
    )

    manifest = {
        "artifact_type": "gcl_resnet50_gate1_7_pipeline_manifest",
        "final_gate": final_gate,
        "seed": seed,
        **_pipeline_run_scope_metadata(
            adapter_bundle=adapter_bundle,
            invocation_limit=invocation_limit,
            invocation_ids=invocation_ids,
        ),
        "hashes": {
            "adapter_bundle_hash": adapter_bundle["adapter_bundle_hash"],
            "trace_manifest_hash": trace_manifest["trace_manifest_hash"],
            "canonical_graph_bundle_hash": canonical_graph_bundle["canonical_graph_bundle_hash"],
            "graph_tensor_bundle_hash": graph_tensor_bundle["graph_tensor_bundle_hash"],
            "embedding_table_hash": _embedding_table_hash(embedding_table),
            "selector_manifest_hash": selector_artifacts["selector_manifest_hash"],
            "gate7_correctness_manifest_hash": correctness_manifest[
                "gate7_cluster_correctness_manifest_hash"
            ],
            "gate8_tuning_vector_proposal_hash": gate8_proposal[
                "gate8_tuning_vector_proposal_hash"
            ],
            "gate9_sampled_vs_full_evaluation_hash": gate9_report[
                "gate9_sampled_vs_full_evaluation_hash"
            ],
        },
    }
    manifest["pipeline_manifest_hash"] = stable_hash(manifest)
    write_json(out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME, manifest)
    return manifest


def resume_resnet50_gate5_to_gate9_from_disk(
    out_dir: Path,
    seed: int | None = None,
    baseline_artifacts_path: Path | None = None,
) -> dict[str, Any]:
    graph_tensor_bundle = read_json(out_dir / "graph_tensor_bundle.json")
    tensors = [tensor_from_jsonable(tensor) for tensor in graph_tensor_bundle.get("tensors", [])]
    if not tensors:
        raise ValueError("persisted graph_tensor_bundle.json does not contain tensors")
    adapter_bundle = read_json(out_dir / "resnet50_trace_adapter_bundle.json")
    trace_manifest = read_json(out_dir / "representative_sm_trace_manifest.json")
    canonical_graph_bundle = read_json(out_dir / "canonical_graph_bundle.json")
    previous_manifest_path = out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME
    previous_manifest = read_json(previous_manifest_path) if previous_manifest_path.exists() else {}
    resolved_seed = int(seed if seed is not None else _resume_seed_from_artifacts(out_dir, previous_manifest))
    invocation_limit, invocation_ids = _resume_invocation_scope(
        previous_manifest,
        adapter_bundle,
    )
    if graph_tensor_bundle.get("source_canonical_graph_bundle_hash") != canonical_graph_bundle.get(
        "canonical_graph_bundle_hash"
    ):
        raise ValueError("graph tensor bundle is not bound to canonical graph bundle")
    if trace_manifest.get("source_adapter_bundle_hash") != adapter_bundle.get("adapter_bundle_hash"):
        raise ValueError("representative SM trace manifest is not bound to adapter bundle")
    if canonical_graph_bundle.get("source_trace_manifest_hash") != trace_manifest.get(
        "trace_manifest_hash"
    ):
        raise ValueError(
            "canonical graph bundle is not bound to representative SM trace manifest"
        )
    if [tensor["input_graph_hash"] for tensor in tensors] != [
        graph["graph_hash"] for graph in canonical_graph_bundle.get("graphs", [])
    ]:
        raise ValueError("persisted tensors do not match persisted canonical graphs")

    augmentation_bundle = create_augmentation_manifest_bundle(tensors, seed=resolved_seed)
    write_json(out_dir / "augmentation_manifest.json", augmentation_bundle)
    embedding_table = _run_gate5_training_and_export(
        tensors=tensors,
        graph_tensor_bundle=graph_tensor_bundle,
        augmentation_bundle=augmentation_bundle,
        out_dir=out_dir,
        seed=resolved_seed,
    )

    selector_artifacts = select_phase_b_representatives(
        embedding_table,
        seed=resolved_seed,
        gate5_artifact_root=out_dir,
    )
    write_json(out_dir / "selector_artifacts.json", selector_artifacts)

    baseline_artifacts = _load_baseline_artifacts(baseline_artifacts_path)
    correctness_manifest = evaluate_gate7_correctness_from_artifacts(
        selector_artifacts=selector_artifacts,
        embedding_table=embedding_table,
        metric_rows=baseline_artifacts.get("metric_rows") if baseline_artifacts else None,
    )
    for report_key, filename in GATE7_REPORT_FILENAMES.items():
        write_json(out_dir / filename, correctness_manifest["gate7_report_artifacts"][report_key])
    write_json(out_dir / GATE7_CLUSTER_CORRECTNESS_FILENAME, correctness_manifest)
    gate8_proposal, gate9_report, final_gate = _emit_gate8_gate9_extension_artifacts(
        correctness_manifest=correctness_manifest,
        selector_artifacts=selector_artifacts,
        baseline_artifacts=baseline_artifacts,
        out_dir=out_dir,
    )

    manifest = {
        "artifact_type": "gcl_resnet50_gate1_7_pipeline_manifest",
        "final_gate": final_gate,
        "seed": resolved_seed,
        **_pipeline_run_scope_metadata(
            adapter_bundle=adapter_bundle,
            invocation_limit=invocation_limit,
            invocation_ids=invocation_ids,
        ),
        "resumed_from_persisted_gate4": True,
        "hashes": {
            "adapter_bundle_hash": adapter_bundle["adapter_bundle_hash"],
            "trace_manifest_hash": trace_manifest["trace_manifest_hash"],
            "canonical_graph_bundle_hash": canonical_graph_bundle["canonical_graph_bundle_hash"],
            "graph_tensor_bundle_hash": graph_tensor_bundle["graph_tensor_bundle_hash"],
            "embedding_table_hash": _embedding_table_hash(embedding_table),
            "selector_manifest_hash": selector_artifacts["selector_manifest_hash"],
            "gate7_correctness_manifest_hash": correctness_manifest[
                "gate7_cluster_correctness_manifest_hash"
            ],
            "gate8_tuning_vector_proposal_hash": gate8_proposal[
                "gate8_tuning_vector_proposal_hash"
            ],
            "gate9_sampled_vs_full_evaluation_hash": gate9_report[
                "gate9_sampled_vs_full_evaluation_hash"
            ],
        },
    }
    manifest["pipeline_manifest_hash"] = stable_hash(manifest)
    write_json(out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME, manifest)
    return manifest


def _resume_seed_from_artifacts(out_dir: Path, previous_manifest: dict[str, Any]) -> int:
    if previous_manifest.get("seed") is not None:
        return int(previous_manifest["seed"])
    training_manifest_path = out_dir / "rgcn_training_run_manifest.json"
    if training_manifest_path.exists():
        training_manifest = read_json(training_manifest_path)
        if training_manifest.get("random_seed") is not None:
            return int(training_manifest["random_seed"])
    return 20260606


def _remove_resnet50_artifacts(out_dir: Path, filenames: set[str]) -> None:
    for filename in filenames:
        (out_dir / filename).unlink(missing_ok=True)


def _resume_invocation_scope(
    previous_manifest: dict[str, Any],
    adapter_bundle: dict[str, Any],
) -> tuple[int | None, list[str] | None]:
    if previous_manifest:
        invocation_ids = previous_manifest.get("invocation_ids")
        return (
            previous_manifest.get("invocation_limit"),
            list(invocation_ids) if invocation_ids is not None else None,
        )
    report = adapter_bundle.get("adapter_validation_report", {})
    invocation_limit = report.get("formal_replay_invocation_limit")
    invocation_ids = report.get("formal_replay_invocation_ids")
    return (
        invocation_limit,
        list(invocation_ids) if invocation_ids is not None else None,
    )


def _gate5_pipeline_manifest(
    *,
    seed: int,
    invocation_limit: int | None,
    invocation_ids: list[str] | None,
    adapter_bundle: dict[str, Any],
    trace_manifest: dict[str, Any],
    canonical_graph_bundle: dict[str, Any],
    graph_tensor_bundle: dict[str, Any],
    embedding_table: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "artifact_type": "gcl_resnet50_gate1_7_pipeline_manifest",
        "final_gate": "gate5_embedding_exported",
        "seed": seed,
        **_pipeline_run_scope_metadata(
            adapter_bundle=adapter_bundle,
            invocation_limit=invocation_limit,
            invocation_ids=invocation_ids,
        ),
        "hashes": {
            "adapter_bundle_hash": adapter_bundle["adapter_bundle_hash"],
            "trace_manifest_hash": trace_manifest["trace_manifest_hash"],
            "canonical_graph_bundle_hash": canonical_graph_bundle["canonical_graph_bundle_hash"],
            "graph_tensor_bundle_hash": graph_tensor_bundle["graph_tensor_bundle_hash"],
            "embedding_table_hash": _embedding_table_hash(embedding_table),
            "selector_manifest_hash": None,
            "gate7_correctness_manifest_hash": None,
            "gate8_tuning_vector_proposal_hash": None,
            "gate9_sampled_vs_full_evaluation_hash": None,
        },
    }
    manifest["pipeline_manifest_hash"] = stable_hash(manifest)
    return manifest


def _pipeline_run_scope_metadata(
    *,
    adapter_bundle: dict[str, Any],
    invocation_limit: int | None,
    invocation_ids: list[str] | None,
) -> dict[str, Any]:
    return {
        "run_scope": "real_resnet50_full_trace"
        if invocation_limit is None and invocation_ids is None
        else "bounded_resnet50_trace_replay",
        "invocation_limit": invocation_limit,
        "invocation_ids": list(invocation_ids) if invocation_ids is not None else None,
        "input_kernel_invocation_count": len(adapter_bundle["kernel_invocation_table"]),
    }


def _emit_gate8_gate9_extension_artifacts(
    *,
    correctness_manifest: dict[str, Any],
    selector_artifacts: dict[str, Any],
    baseline_artifacts: dict[str, Any] | None,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    expected_anchor_hash = correctness_manifest["source_representative_anchor_table_hash"]
    selector_anchor_hash = selector_artifacts["representative_anchor_table"].get(
        "representative_anchor_table_hash",
        expected_anchor_hash,
    )
    if selector_anchor_hash != expected_anchor_hash:
        raise ValueError(
            "representative anchor table hash mismatch between Gate6 selector artifacts "
            "and Gate7 correctness manifest"
        )
    representative_anchor_table = {
        **selector_artifacts["representative_anchor_table"],
        "representative_anchor_table_hash": selector_anchor_hash,
    }
    tunable_component_schema = {
        "schema_version": "report_only_default_v1",
        "components": ["memory_latency_scale", "compute_latency_scale"],
    }
    try:
        gate8_proposal = generate_gate8_tuning_vectors(
            correctness_manifest,
            representative_anchor_table=representative_anchor_table,
            family_alignment_report=correctness_manifest["gate7_report_artifacts"][
                "family_alignment_report"
            ],
            metric_error_report=correctness_manifest["gate7_report_artifacts"]["metric_error_report"],
            tunable_component_schema=tunable_component_schema,
        )
    except ValueError as exc:
        gate8_proposal = _blocked_gate8_tuning_vector_proposal(
            correctness_manifest=correctness_manifest,
            representative_anchor_table=representative_anchor_table,
            tunable_component_schema=tunable_component_schema,
            blocker_message=str(exc),
        )
    write_json(out_dir / "cluster_tuning_vector_table.json", gate8_proposal["cluster_tuning_vector_table"])
    write_json(
        out_dir / "tuning_vector_provenance_report.json",
        gate8_proposal["tuning_vector_provenance_report"],
    )
    write_json(out_dir / "tuning_safety_report.json", gate8_proposal["tuning_safety_report"])
    write_json(out_dir / "gate8_tuning_manifest.json", gate8_proposal["gate8_tuning_manifest"])
    write_json(out_dir / "gate8_tuning_vector_proposal.json", gate8_proposal)
    gate8_blocked = gate8_proposal["gate8_tuning_manifest"].get(
        "tuning_safety_status"
    ) == "blocked_report_only"
    if baseline_artifacts and _has_gate9_baseline_metrics(baseline_artifacts) and not gate8_blocked:
        gate9_report = evaluate_gate9_sampled_vs_full(
            sampled_metrics=baseline_artifacts["sampled_metrics"],
            full_baseline_metrics=baseline_artifacts.get("full_baseline_metrics"),
            measured_baseline_metrics=baseline_artifacts.get("measured_baseline_metrics"),
            gate8_tuning_manifest=gate8_proposal["gate8_tuning_manifest"],
            representative_anchor_table={
                **selector_artifacts["representative_anchor_table"],
                "representative_anchor_table_hash": correctness_manifest[
                    "source_representative_anchor_table_hash"
                ],
            },
        )
        final_gate = "gate9_evaluated"
    else:
        gate9_report = gate9_baseline_missing_report()
        final_gate = "gate9_report_only"
    write_json(
        out_dir / "full_vs_sampled_simulation_report.json",
        gate9_report["full_vs_sampled_simulation_report"],
    )
    write_json(out_dir / "sampled_speedup_report.json", gate9_report["sampled_speedup_report"])
    write_json(out_dir / "sampled_error_report.json", gate9_report["sampled_error_report"])
    write_json(out_dir / "tuning_effect_report.json", gate9_report["tuning_effect_report"])
    write_json(
        out_dir / "gate9_simulator_evaluation_manifest.json",
        gate9_report["gate9_simulator_evaluation_manifest"],
    )
    write_json(out_dir / "gate9_sampled_vs_full_evaluation.json", gate9_report)
    return gate8_proposal, gate9_report, final_gate


def _has_gate9_baseline_metrics(baseline_artifacts: dict[str, Any]) -> bool:
    return bool(
        baseline_artifacts.get("sampled_metrics")
        and (
            baseline_artifacts.get("full_baseline_metrics")
            or baseline_artifacts.get("measured_baseline_metrics")
        )
    )


def _blocked_gate8_tuning_vector_proposal(
    *,
    correctness_manifest: dict[str, Any],
    representative_anchor_table: dict[str, Any],
    tunable_component_schema: dict[str, Any],
    blocker_message: str,
) -> dict[str, Any]:
    cluster_tuning_vector_table = {
        "artifact_type": "gcl_resnet50_cluster_tuning_vector_table",
        "artifact_version": "gate8_cluster_tuning_vector_table_v1",
        "source_gate7_correctness_manifest_hash": correctness_manifest[
            "gate7_cluster_correctness_manifest_hash"
        ],
        "tuning_vectors": [],
        "blocked_reason": blocker_message,
    }
    cluster_tuning_vector_table["cluster_tuning_vector_table_hash"] = stable_hash(
        cluster_tuning_vector_table
    )
    provenance_report = {
        "artifact_type": "gcl_resnet50_tuning_vector_provenance_report",
        "artifact_version": "gate8_tuning_vector_provenance_report_v1",
        "source_gate7_correctness_manifest_hash": correctness_manifest[
            "gate7_cluster_correctness_manifest_hash"
        ],
        "source_claim_status": correctness_manifest["claim_status"],
        "representative_anchor_table_hash": representative_anchor_table[
            "representative_anchor_table_hash"
        ],
        "representative_anchor_count": len(representative_anchor_table.get("anchors", [])),
        "tunable_component_schema": tunable_component_schema,
        "blocked_reason": blocker_message,
    }
    provenance_report["tuning_vector_provenance_report_hash"] = stable_hash(provenance_report)
    safety_report = {
        "artifact_type": "gcl_resnet50_tuning_safety_report",
        "artifact_version": "gate8_tuning_safety_report_v1",
        "safety_status": "blocked_report_only",
        "blocked_reason": blocker_message,
        "accuracy_claim": "not_claimed",
    }
    safety_report["tuning_safety_report_hash"] = stable_hash(safety_report)
    gate8_manifest = {
        "artifact_type": "gcl_resnet50_gate8_tuning_manifest",
        "artifact_version": "gate8_tuning_manifest_v1",
        "extension_label": "our_extension_not_original_gcl_sampler",
        "source_gate7_correctness_manifest_hash": correctness_manifest[
            "gate7_cluster_correctness_manifest_hash"
        ],
        "representative_anchor_table_hash": representative_anchor_table[
            "representative_anchor_table_hash"
        ],
        "cluster_tuning_vector_table_hash": cluster_tuning_vector_table[
            "cluster_tuning_vector_table_hash"
        ],
        "tuning_vector_provenance_report_hash": provenance_report[
            "tuning_vector_provenance_report_hash"
        ],
        "tuning_safety_report_hash": safety_report["tuning_safety_report_hash"],
        "tuning_safety_status": "blocked_report_only",
    }
    gate8_manifest["gate8_tuning_manifest_hash"] = stable_hash(gate8_manifest)
    artifact = {
        "artifact_type": "gcl_resnet50_gate8_tuning_vector_proposal",
        "artifact_version": "gate8_tuning_vector_proposal_v1",
        "extension_label": "our_extension_not_original_gcl_sampler",
        "source_gate7_correctness_manifest_hash": correctness_manifest[
            "gate7_cluster_correctness_manifest_hash"
        ],
        "tunable_component_schema": tunable_component_schema,
        "proposals": [],
        "cluster_tuning_vector_table": cluster_tuning_vector_table,
        "tuning_vector_provenance_report": provenance_report,
        "tuning_safety_report": safety_report,
        "gate8_tuning_manifest": gate8_manifest,
        "blocked_reason": blocker_message,
    }
    artifact["gate8_tuning_vector_proposal_hash"] = stable_hash(artifact)
    return artifact


def _write_gate0_blocked_pipeline_manifest(root: Path, out_dir: Path, seed: int) -> dict[str, Any]:
    from .utils import read_json

    blocker_report = read_json(root / GATE0_BLOCKER_FILENAME)
    write_json(out_dir / GATE0_BLOCKER_FILENAME, blocker_report)
    manifest = {
        "artifact_type": "gcl_resnet50_gate1_7_pipeline_manifest",
        "final_gate": "gate0_blocked",
        "artifact_status": "formal_blocked",
        "formal_input_eligible": False,
        "seed": seed,
        "blocked_gate": "gate0",
        "blocker_report_hash": blocker_report["gate0_blocker_report_hash"],
        "hashes": {
            "gate0_blocker_report_hash": blocker_report["gate0_blocker_report_hash"],
            "adapter_bundle_hash": None,
            "trace_manifest_hash": None,
            "canonical_graph_bundle_hash": None,
            "graph_tensor_bundle_hash": None,
            "embedding_table_hash": None,
            "selector_manifest_hash": None,
            "gate7_correctness_manifest_hash": None,
            "gate8_tuning_vector_proposal_hash": None,
            "gate9_sampled_vs_full_evaluation_hash": None,
        },
    }
    manifest["pipeline_manifest_hash"] = stable_hash(manifest)
    write_json(out_dir / GATE1_7_PIPELINE_MANIFEST_FILENAME, manifest)
    return manifest


def _load_baseline_artifacts(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    baseline = read_json(path)
    if not baseline.get("metric_rows") and not baseline.get("sampled_metrics"):
        raise ValueError("baseline artifacts require metric_rows or sampled_metrics")
    if baseline.get("sampled_metrics") and not (
        baseline.get("full_baseline_metrics") or baseline.get("measured_baseline_metrics")
    ):
        raise ValueError("baseline artifacts require full or measured baseline metrics")
    return baseline


def _run_gate5_training_and_export(
    *,
    tensors: list[dict[str, Any]],
    graph_tensor_bundle: dict[str, Any],
    augmentation_bundle: dict[str, Any],
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    existing_embedding = _load_existing_gate5_embedding_table(
        out_dir,
        graph_tensor_bundle=graph_tensor_bundle,
        seed=seed,
    )
    if existing_embedding is not None:
        return existing_embedding
    training_source_tensors = _select_gate5_training_tensors(tensors)
    existing_training = _load_existing_gate5_training(
        out_dir,
        graph_tensor_bundle=graph_tensor_bundle,
        seed=seed,
    )
    if existing_training is None:
        training_tensors = [_phase_a_compatible_tensor(tensor) for tensor in training_source_tensors]
        training_report = train_minimal_contrastive(training_tensors, out_dir, seed=seed)
    else:
        training_report = existing_training
    training_run_manifest = _training_run_manifest(
        graph_tensor_bundle=graph_tensor_bundle,
        tensors=training_source_tensors,
        training_report=training_report,
        augmentation_bundle=augmentation_bundle,
        seed=seed,
        export_graph_count=len(tensors),
        checkpoint_reuse=existing_training is not None,
    )
    checkpoint_manifest = _checkpoint_manifest(
        training_report["checkpoint_manifest"],
        training_run_manifest["training_run_manifest_hash"],
    )
    write_json(out_dir / "rgcn_training_run_manifest.json", training_run_manifest)
    write_json(out_dir / "rgcn_checkpoint_manifest.json", checkpoint_manifest)
    embedding_table, readout_bundle = export_phase_b_embedding_table(
        tensors,
        training_report["encoder"],
        training_report["checkpoint_manifest"],
        source_graph_tensor_bundle_hash=graph_tensor_bundle["graph_tensor_bundle_hash"],
        progress_dir=out_dir,
    )
    write_json(out_dir / "readout_manifest.json", readout_bundle)
    export_report = {
        "artifact_type": "gcl_resnet50_embedding_export_report",
        "artifact_version": "gate5_embedding_export_report_v1",
        "source_graph_tensor_bundle_hash": graph_tensor_bundle["graph_tensor_bundle_hash"],
        "encoder_manifest_hash": embedding_table["encoder_manifest_hash"],
        "checkpoint_hash": embedding_table["checkpoint_hash"],
        "readout_manifest_bundle_hash": readout_bundle["readout_manifest_bundle_hash"],
        "failed_graphs": [],
    }
    export_report["embedding_export_report_hash"] = hash_without(
        export_report, "embedding_export_report_hash"
    )
    _bind_embedding_table_to_persisted_gate5_manifests(
        embedding_table,
        training_run_manifest=training_run_manifest,
        checkpoint_manifest=checkpoint_manifest,
        readout_bundle=readout_bundle,
        export_report=export_report,
    )
    lineage_bundle = build_gate5_lineage_bundle(
        embedding_table["gate5_lineage"],
        readout_bundle,
    )
    write_json(out_dir / "gate5_lineage_bundle.json", lineage_bundle)
    write_json(out_dir / "kernel_embedding_table.json", embedding_table)
    write_json(out_dir / "embedding_export_report.json", export_report)
    return embedding_table


def _load_existing_gate5_embedding_table(
    out_dir: Path,
    *,
    graph_tensor_bundle: dict[str, Any],
    seed: int,
) -> dict[str, Any] | None:
    embedding_path = out_dir / "kernel_embedding_table.json"
    if not embedding_path.exists():
        return None
    training_manifest_path = out_dir / "rgcn_training_run_manifest.json"
    if not training_manifest_path.exists():
        return None
    try:
        training_manifest = read_json(training_manifest_path)
        embedding_table = read_json(embedding_path)
    except (OSError, json.JSONDecodeError):
        return None
    if training_manifest.get("random_seed") != seed:
        return None
    if embedding_table.get("source_graph_tensor_bundle_hash") != graph_tensor_bundle[
        "graph_tensor_bundle_hash"
    ]:
        return None
    from .embedding_export import validate_phase_b_embedding_table

    try:
        validate_phase_b_embedding_table(embedding_table)
    except ValueError:
        return None
    if not _gate5_side_artifacts_match_embedding_table(out_dir, embedding_table):
        return None
    (out_dir / GATE5_EXPORT_PROGRESS_FILENAME).unlink(missing_ok=True)
    return embedding_table


def _gate5_side_artifacts_match_embedding_table(
    out_dir: Path,
    embedding_table: dict[str, Any],
) -> bool:
    paths = {
        "training_run_manifest": out_dir / "rgcn_training_run_manifest.json",
        "checkpoint_manifest": out_dir / "rgcn_checkpoint_manifest.json",
        "readout_manifest": out_dir / "readout_manifest.json",
        "lineage_bundle": out_dir / "gate5_lineage_bundle.json",
        "export_report": out_dir / "embedding_export_report.json",
    }
    if any(not path.exists() for path in paths.values()):
        return False
    try:
        training_run_manifest = read_json(paths["training_run_manifest"])
        checkpoint_manifest = read_json(paths["checkpoint_manifest"])
        readout_bundle = read_json(paths["readout_manifest"])
        lineage_bundle = read_json(paths["lineage_bundle"])
        export_report = read_json(paths["export_report"])
    except (OSError, json.JSONDecodeError):
        return False

    expected_hashes = {
        "training_run_manifest_hash": training_run_manifest.get("training_run_manifest_hash"),
        "checkpoint_manifest_hash": checkpoint_manifest.get("rgcn_checkpoint_manifest_hash"),
        "readout_manifest_bundle_hash": readout_bundle.get("readout_manifest_bundle_hash"),
        "embedding_export_report_hash": export_report.get("embedding_export_report_hash"),
    }
    if not all(expected_hashes.values()):
        return False
    if training_run_manifest.get("training_run_manifest_hash") != hash_without(
        training_run_manifest,
        "training_run_manifest_hash",
    ):
        return False
    if checkpoint_manifest.get("rgcn_checkpoint_manifest_hash") != hash_without(
        checkpoint_manifest,
        "rgcn_checkpoint_manifest_hash",
    ):
        return False
    if readout_bundle.get("readout_manifest_bundle_hash") != hash_without(
        readout_bundle,
        "readout_manifest_bundle_hash",
    ):
        return False
    if export_report.get("embedding_export_report_hash") != hash_without(
        export_report,
        "embedding_export_report_hash",
    ):
        return False
    if lineage_bundle.get("gate5_lineage_bundle_hash") != hash_without(
        lineage_bundle,
        "gate5_lineage_bundle_hash",
    ):
        return False
    if lineage_bundle.get("persisted_manifest_hashes") != expected_hashes:
        return False
    if lineage_bundle.get("lineage") != embedding_table.get("gate5_lineage"):
        return False
    if lineage_bundle.get("gate5_lineage_bundle_hash") != embedding_table.get(
        "gate5_lineage_bundle_hash"
    ):
        return False
    if expected_hashes != {
        "training_run_manifest_hash": embedding_table["gate5_lineage"][
            "training_run_manifest_hash"
        ],
        "checkpoint_manifest_hash": embedding_table["gate5_lineage"][
            "checkpoint_manifest_hash"
        ],
        "readout_manifest_bundle_hash": embedding_table["gate5_lineage"][
            "readout_manifest_bundle_hash"
        ],
        "embedding_export_report_hash": embedding_table["gate5_lineage"][
            "embedding_export_report_hash"
        ],
    }:
        return False
    return True


def _select_gate5_training_tensors(tensors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(tensors) <= GATE5_TRAINING_GRAPH_LIMIT:
        return tensors
    return tensors[:GATE5_TRAINING_GRAPH_LIMIT]


def _load_existing_gate5_training(
    out_dir: Path,
    *,
    graph_tensor_bundle: dict[str, Any],
    seed: int,
) -> dict[str, Any] | None:
    checkpoint_path = out_dir / "rgcn_checkpoint.pt"
    training_manifest_path = out_dir / "rgcn_training_run_manifest.json"
    checkpoint_manifest_path = out_dir / "rgcn_checkpoint_manifest.json"
    if not checkpoint_path.exists():
        return None
    if not training_manifest_path.exists():
        return None
    try:
        checkpoint = load_checkpoint_weights_only(checkpoint_path)
        checkpoint_hash = hash_without({"checkpoint_bytes": checkpoint_path.read_bytes().hex()})
    except (OSError, RuntimeError, ValueError, KeyError, pickle.UnpicklingError):
        return None
    progress_path = out_dir / GATE5_EXPORT_PROGRESS_FILENAME
    try:
        progress = read_json(progress_path) if progress_path.exists() else {}
        training_manifest = read_json(training_manifest_path)
    except (OSError, json.JSONDecodeError):
        return None
    if training_manifest.get("source_graph_tensor_bundle_hash") != graph_tensor_bundle[
        "graph_tensor_bundle_hash"
    ]:
        return None
    if training_manifest.get("random_seed") != seed:
        return None
    try:
        checkpoint_manifest = (
            read_json(checkpoint_manifest_path)
            if checkpoint_manifest_path.exists()
            else {
                "encoder_architecture": checkpoint["model_config"],
                "encoder_manifest_hash": progress.get("encoder_manifest_hash"),
                "checkpoint_hash": checkpoint_hash,
            }
        )
    except (OSError, json.JSONDecodeError, KeyError):
        return None
    if checkpoint_manifest.get("checkpoint_hash") != checkpoint_hash:
        return None
    if checkpoint_manifest.get("encoder_manifest_hash") != hash_without(
        {
            "encoder_name": "minimal_phase_a_rgcn",
            "encoder_version": 1,
            "model_config": checkpoint["model_config"],
            "source_tensor_hashes": checkpoint.get("source_tensor_hashes", []),
            "seed": checkpoint.get("seed"),
            "checkpoint_hash": checkpoint_hash,
            "encoder_batch_size": checkpoint.get("encoder_batch_size", 16),
            "partitioned_encoding": checkpoint.get("partitioned_encoding", True),
        },
        "encoder_manifest_hash",
        "checkpoint_path",
    ):
        return None
    expected_checkpoint_tensor_hashes = training_manifest.get("checkpoint_source_tensor_hashes", [])
    if checkpoint.get("source_tensor_hashes", []) != expected_checkpoint_tensor_hashes:
        return None
    if checkpoint.get("seed") != seed:
        return None
    encoder = MinimalRGCNEncoder()
    try:
        encoder.load_state_dict(checkpoint["encoder"])
        projection_head = ProjectionHead()
        projection_head.load_state_dict(checkpoint["projection_head"])
    except (RuntimeError, KeyError, TypeError):
        return None
    phase_a_manifest = {
        "encoder_name": "minimal_phase_a_rgcn",
        "encoder_version": 1,
        "model_config": checkpoint_manifest["encoder_architecture"],
        "source_tensor_hashes": checkpoint.get("source_tensor_hashes", []),
        "seed": checkpoint.get("seed"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": checkpoint_manifest.get("checkpoint_hash", checkpoint_hash),
        "encoder_batch_size": checkpoint.get("encoder_batch_size", 16),
        "partitioned_encoding": checkpoint.get("partitioned_encoding", True),
        "encoder_manifest_hash": checkpoint_manifest.get("encoder_manifest_hash")
        or progress.get("encoder_manifest_hash"),
    }
    if not phase_a_manifest["encoder_manifest_hash"]:
        return None
    train_graph_count = training_manifest.get(
        "train_graph_count",
        len(phase_a_manifest["source_tensor_hashes"]),
    )
    return {
        "training_mode": "minimal_rgcn_contrastive_smoke",
        "loss": float(training_manifest.get("final_loss", 0.0)),
        "optimizer_step_count": training_manifest.get("optimizer_config", {}).get(
            "optimizer_step_count",
            1,
        ),
        "encoder_batch_size": phase_a_manifest["encoder_batch_size"],
        "partitioned_encoding": phase_a_manifest["partitioned_encoding"],
        "augmentation_retry_count": 0,
        "kernel_embedding_shape": [
            train_graph_count,
            checkpoint_manifest["encoder_architecture"]["kernel_embedding_dim"],
        ],
        "projection_output_shape": [
            train_graph_count,
            checkpoint_manifest["encoder_architecture"]["projection_output_dim"],
        ],
        "checkpoint_manifest": phase_a_manifest,
        "encoder": encoder,
        "projection_head": projection_head,
    }


def _jsonable_training_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key not in {"encoder", "projection_head"}}


def _embedding_table_hash(table: dict[str, Any]) -> str:
    return table["kernel_embedding_table_hash"]


def _training_run_manifest(
    graph_tensor_bundle: dict[str, Any],
    tensors: list[dict[str, Any]],
    training_report: dict[str, Any],
    augmentation_bundle: dict[str, Any],
    seed: int,
    export_graph_count: int | None = None,
    checkpoint_reuse: bool = False,
) -> dict[str, Any]:
    checkpoint_manifest = training_report["checkpoint_manifest"]
    representation_modes = sorted({tensor["representation_mode"] for tensor in tensors})
    pseudo_node_modes = sorted({tensor["pseudo_node_mode"] for tensor in tensors})
    manifest = {
        "artifact_type": "gcl_resnet50_rgcn_training_run_manifest",
        "artifact_version": "gate5_rgcn_training_run_manifest_v1",
        "source_graph_tensor_bundle_hash": graph_tensor_bundle["graph_tensor_bundle_hash"],
        "representation_mode": representation_modes[0]
        if len(representation_modes) == 1
        else "mixed:" + ",".join(representation_modes),
        "pseudo_node_mode": pseudo_node_modes[0]
        if len(pseudo_node_modes) == 1
        else "mixed:" + ",".join(pseudo_node_modes),
        "model_architecture": checkpoint_manifest["model_config"],
        "source_tensor_hashes": [tensor["tensor_hash"] for tensor in tensors],
        "checkpoint_source_tensor_hashes": checkpoint_manifest["source_tensor_hashes"],
        "edge_relation_schema": tensors[0]["edge_relation_schema"],
        "readout_hierarchy": READOUT_HIERARCHY,
        "augmentation_config": {
            "augmentation_manifest_bundle_hash": augmentation_bundle[
                "augmentation_manifest_bundle_hash"
            ],
            "node_drop_rate": 0.15,
            "edge_drop_rate": 0.15,
            "noise_sigma": 0.01,
        },
        "contrastive_loss_config": {
            "loss": "InfoNCE",
            "temperature": 0.2,
            "projection_output_dim": 64,
        },
        "optimizer_config": {
            "optimizer": "Adam",
            "learning_rate": 0.005,
            "optimizer_step_count": training_report["optimizer_step_count"],
        },
        "random_seed": seed,
        "train_graph_count": len(tensors),
        "export_graph_count": export_graph_count if export_graph_count is not None else len(tensors),
        "training_subset_policy": "deterministic_prefix_for_full_trace_scalability"
        if export_graph_count is not None and export_graph_count > len(tensors)
        else "all_graphs",
        "checkpoint_reuse": checkpoint_reuse,
        "training_status": "formal_gate5_complete"
        if len(tensors) >= 2
        else "debug_single_graph_not_formal",
        "final_loss": round(float(training_report["loss"]), 8),
        "best_checkpoint_hash": checkpoint_manifest["checkpoint_hash"],
    }
    manifest["training_run_manifest_hash"] = hash_without(
        manifest, "training_run_manifest_hash"
    )
    return manifest


def _checkpoint_manifest(
    phase_a_checkpoint_manifest: dict[str, Any],
    training_run_manifest_hash: str,
) -> dict[str, Any]:
    model_config = phase_a_checkpoint_manifest["model_config"]
    manifest = {
        "artifact_type": "gcl_resnet50_rgcn_checkpoint_manifest",
        "artifact_version": "gate5_rgcn_checkpoint_manifest_v1",
        "encoder_architecture": model_config,
        "encoder_state_hash": phase_a_checkpoint_manifest["encoder_manifest_hash"],
        "projection_head_state_hash": stable_hash(
            {
                "projection_hidden_dim": model_config["projection_hidden_dim"],
                "projection_output_dim": model_config["projection_output_dim"],
                "checkpoint_hash": phase_a_checkpoint_manifest["checkpoint_hash"],
            }
        ),
        "optimizer_state_hash": stable_hash(
            {
                "optimizer": "Adam",
                "seed": phase_a_checkpoint_manifest["seed"],
                "checkpoint_hash": phase_a_checkpoint_manifest["checkpoint_hash"],
            }
        ),
        "encoder_manifest_hash": phase_a_checkpoint_manifest["encoder_manifest_hash"],
        "checkpoint_hash": phase_a_checkpoint_manifest["checkpoint_hash"],
        "checkpoint_created_from_training_run_manifest_hash": training_run_manifest_hash,
    }
    manifest["rgcn_checkpoint_manifest_hash"] = hash_without(
        manifest, "rgcn_checkpoint_manifest_hash"
    )
    return manifest


def _bind_embedding_table_to_persisted_gate5_manifests(
    embedding_table: dict[str, Any],
    *,
    training_run_manifest: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    readout_bundle: dict[str, Any],
    export_report: dict[str, Any],
) -> None:
    lineage = dict(embedding_table["gate5_lineage"])
    lineage["training_run_manifest_hash"] = training_run_manifest["training_run_manifest_hash"]
    lineage["checkpoint_manifest_hash"] = checkpoint_manifest["rgcn_checkpoint_manifest_hash"]
    lineage["readout_manifest_bundle_hash"] = readout_bundle["readout_manifest_bundle_hash"]
    lineage["embedding_export_report_hash"] = export_report["embedding_export_report_hash"]
    embedding_table["gate5_lineage"] = lineage
    embedding_table["gate5_lineage_hash"] = hash_without(lineage)
    lineage_bundle = build_gate5_lineage_bundle(lineage, readout_bundle)
    embedding_table["gate5_lineage_bundle_hash"] = lineage_bundle["gate5_lineage_bundle_hash"]
    for row in embedding_table["embeddings"]:
        row["gate5_lineage_hash"] = embedding_table["gate5_lineage_hash"]
        row["embedding_hash"] = hash_without(row, "embedding_hash")
    embedding_table["kernel_embedding_table_hash"] = hash_without(
        embedding_table,
        "kernel_embedding_table_hash",
    )


def _phase_a_compatible_tensor(tensor: dict[str, Any]) -> dict[str, Any]:
    from experiments.gcl_phase_a.tensorizer import TENSORIZER_VERSION as PHASE_A_TENSORIZER_VERSION
    from experiments.gcl_phase_a.tensorizer import _tensor_hash as phase_a_tensor_hash

    compatible = dict(tensor)
    compatible["artifact_type"] = "graph_tensor"
    compatible["tensorizer_version"] = PHASE_A_TENSORIZER_VERSION
    compatible.pop("phase_b_tensorizer_version", None)
    compatible["tensor_hash"] = phase_a_tensor_hash(compatible)
    return compatible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--baseline-artifacts", type=Path)
    args = parser.parse_args()
    manifest = run_resnet50_gate1_to_gate7(
        args.input_root,
        args.out,
        seed=args.seed,
        baseline_artifacts_path=args.baseline_artifacts,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
