"""ResNet-50 Gate 1-5 pipeline. Stops before Gate 6 clustering/classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .graph_builder import build_phase_b_graphs, validate_phase_b_graph_artifact
from .pipeline import create_augmentation_manifest_bundle, run_embedding_export
from .resnet50_adapter import build_resnet50_trace_adapter_bundle
from .resnet50_manifest import build_representative_sm_manifest_from_bundle
from .tensorizer import tensor_to_jsonable, tensorize_phase_b_graphs
from .trace_scope import build_phase_b_trace_records
from .utils import hash_without, stable_hash, write_json


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

    embedding_table, training_report = run_embedding_export(tensors, out_dir, seed=seed)
    write_json(out_dir / "rgcn_training_run_manifest.json", _jsonable_training_report(training_report))
    write_json(out_dir / "rgcn_checkpoint_manifest.json", training_report["checkpoint_manifest"])
    write_json(out_dir / "kernel_embedding_table.json", embedding_table)
    export_report = {
        "artifact_type": "gcl_resnet50_embedding_export_report",
        "artifact_version": "gate5_embedding_export_report_v1",
        "source_graph_tensor_bundle_hash": graph_tensor_bundle["graph_tensor_bundle_hash"],
        "embedding_table_hash": embedding_table["embedding_table_hash"],
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
            "embedding_table_hash": embedding_table["embedding_table_hash"],
        },
    }
    manifest["pipeline_manifest_hash"] = stable_hash(manifest)
    write_json(out_dir / "gate1_5_pipeline_manifest.json", manifest)
    return manifest


def _jsonable_training_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key not in {"encoder", "projection_head"}}


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
