"""ResNet-50 Gate 1-7 formal GCL reproduction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .graph_builder import build_phase_b_graphs, validate_phase_b_graph_artifact
from .correctness import (
    GATE7_CLUSTER_CORRECTNESS_FILENAME,
    GATE7_REPORT_FILENAMES,
    evaluate_gate7_correctness_from_artifacts,
)
from .embedding_export import (
    READOUT_HIERARCHY,
    build_gate5_lineage_bundle,
    export_phase_b_embedding_table,
)
from .pipeline import create_augmentation_manifest_bundle
from .resnet50_adapter import build_resnet50_trace_adapter_bundle
from .resnet50_gate0 import GATE0_BLOCKER_FILENAME
from .resnet50_manifest import build_representative_sm_manifest_from_bundle
from .selector import select_phase_b_representatives
from .simulator_eval import evaluate_gate9_sampled_vs_full, gate9_baseline_missing_report
from .tensorizer import tensor_to_jsonable, tensorize_phase_b_graphs
from .tuning import generate_gate8_tuning_vectors
from .trace_scope import build_phase_b_trace_records
from .utils import hash_without, read_json, stable_hash, write_json
from experiments.gcl_phase_a.train import train_minimal_contrastive

GATE1_7_PIPELINE_MANIFEST_FILENAME = "gate1_7_pipeline_manifest.json"


def run_resnet50_gate1_to_gate5(
    root: Path,
    out_dir: Path,
    seed: int = 20260606,
) -> dict[str, Any]:
    return run_resnet50_gate1_to_gate7(root, out_dir, seed=seed)


def run_resnet50_gate1_to_gate7(
    root: Path,
    out_dir: Path,
    seed: int = 20260606,
    baseline_artifacts_path: Path | None = None,
    invocation_limit: int | None = None,
    invocation_ids: list[str] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    blocker_path = root / GATE0_BLOCKER_FILENAME
    if blocker_path.exists():
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

    training_tensors = [_phase_a_compatible_tensor(tensor) for tensor in tensors]
    training_report = train_minimal_contrastive(training_tensors, out_dir, seed=seed)
    embedding_table, readout_bundle = export_phase_b_embedding_table(
        tensors,
        training_report["encoder"],
        training_report["checkpoint_manifest"],
        source_graph_tensor_bundle_hash=graph_tensor_bundle["graph_tensor_bundle_hash"],
    )
    training_run_manifest = _training_run_manifest(
        graph_tensor_bundle=graph_tensor_bundle,
        tensors=tensors,
        training_report=training_report,
        augmentation_bundle=augmentation_bundle,
        seed=seed,
    )
    checkpoint_manifest = _checkpoint_manifest(
        training_report["checkpoint_manifest"],
        training_run_manifest["training_run_manifest_hash"],
    )
    write_json(out_dir / "rgcn_training_run_manifest.json", training_run_manifest)
    write_json(out_dir / "rgcn_checkpoint_manifest.json", checkpoint_manifest)
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


def _emit_gate8_gate9_extension_artifacts(
    *,
    correctness_manifest: dict[str, Any],
    selector_artifacts: dict[str, Any],
    baseline_artifacts: dict[str, Any] | None,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    representative_anchor_table = {
        **selector_artifacts["representative_anchor_table"],
        "representative_anchor_table_hash": selector_artifacts["representative_anchor_table"].get(
            "representative_anchor_table_hash",
            correctness_manifest["source_representative_anchor_table_hash"],
        ),
    }
    gate8_proposal = generate_gate8_tuning_vectors(
        correctness_manifest,
        representative_anchor_table=representative_anchor_table,
        family_alignment_report=correctness_manifest["gate7_report_artifacts"][
            "family_alignment_report"
        ],
        metric_error_report=correctness_manifest["gate7_report_artifacts"]["metric_error_report"],
        tunable_component_schema={
            "schema_version": "report_only_default_v1",
            "components": ["memory_latency_scale", "compute_latency_scale"],
        },
    )
    write_json(out_dir / "cluster_tuning_vector_table.json", gate8_proposal["cluster_tuning_vector_table"])
    write_json(
        out_dir / "tuning_vector_provenance_report.json",
        gate8_proposal["tuning_vector_provenance_report"],
    )
    write_json(out_dir / "tuning_safety_report.json", gate8_proposal["tuning_safety_report"])
    write_json(out_dir / "gate8_tuning_manifest.json", gate8_proposal["gate8_tuning_manifest"])
    write_json(out_dir / "gate8_tuning_vector_proposal.json", gate8_proposal)
    if baseline_artifacts:
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
    required = {"sampled_metrics"}
    missing = required.difference(baseline)
    if missing:
        raise ValueError(f"baseline artifacts missing required fields: {sorted(missing)}")
    if not baseline.get("full_baseline_metrics") and not baseline.get("measured_baseline_metrics"):
        raise ValueError("baseline artifacts require full or measured baseline metrics")
    return baseline


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
