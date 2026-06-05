"""ResNet-50 Gate 1-5 pipeline. Stops before Gate 6 clustering/classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .graph_builder import build_phase_b_graphs, validate_phase_b_graph_artifact
from .embedding_export import READOUT_HIERARCHY, export_phase_b_embedding_table
from .pipeline import create_augmentation_manifest_bundle
from .resnet50_adapter import build_resnet50_trace_adapter_bundle
from .resnet50_manifest import build_representative_sm_manifest_from_bundle
from .tensorizer import tensor_to_jsonable, tensorize_phase_b_graphs
from .trace_scope import build_phase_b_trace_records
from .utils import hash_without, stable_hash, write_json
from experiments.gcl_phase_a.train import train_minimal_contrastive


def run_resnet50_gate1_to_gate5(
    root: Path,
    out_dir: Path,
    seed: int = 20260606,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_bundle = build_resnet50_trace_adapter_bundle(root)
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
    write_json(out_dir / "kernel_embedding_table.json", embedding_table)
    export_report = {
        "artifact_type": "gcl_resnet50_embedding_export_report",
        "artifact_version": "gate5_embedding_export_report_v1",
        "source_graph_tensor_bundle_hash": graph_tensor_bundle["graph_tensor_bundle_hash"],
        "embedding_table_hash": _embedding_table_hash(embedding_table),
        "failed_graphs": [],
    }
    export_report["embedding_export_report_hash"] = hash_without(
        export_report, "embedding_export_report_hash"
    )
    write_json(out_dir / "embedding_export_report.json", export_report)

    manifest = {
        "artifact_type": "gcl_resnet50_gate1_5_pipeline_manifest",
        "final_gate": "gate5",
        "seed": seed,
        "hashes": {
            "adapter_bundle_hash": adapter_bundle["adapter_bundle_hash"],
            "trace_manifest_hash": trace_manifest["trace_manifest_hash"],
            "canonical_graph_bundle_hash": canonical_graph_bundle["canonical_graph_bundle_hash"],
            "graph_tensor_bundle_hash": graph_tensor_bundle["graph_tensor_bundle_hash"],
            "embedding_table_hash": _embedding_table_hash(embedding_table),
        },
    }
    manifest["pipeline_manifest_hash"] = stable_hash(manifest)
    write_json(out_dir / "gate1_5_pipeline_manifest.json", manifest)
    return manifest


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
    args = parser.parse_args()
    manifest = run_resnet50_gate1_to_gate5(args.input_root, args.out, seed=args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
