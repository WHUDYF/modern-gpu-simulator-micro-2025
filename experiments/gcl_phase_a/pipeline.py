"""End-to-end CLI for the GCL Phase A semantic path."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .embedding_export import export_embedding_table, validate_embedding_table
from .graph_builder import build_canonical_graphs, validate_graph_artifact
from .rgcn import MinimalRGCNEncoder, require_torch
from .selector import select_representatives
from .tensorizer import tensor_from_jsonable, tensor_to_jsonable, tensorize_graphs
from .trace_fixture import build_controlled_trace_fixture, fixture_summary, validate_trace_fixture
from .train import train_minimal_contrastive
from .utils import hash_without, read_json, stable_hash, write_json

ARTIFACT_FILENAMES = {
    "trace_fixture": "trace_fixture.json",
    "graph_bundle": "graph_bundle.json",
    "tensor_bundle": "tensor_bundle.json",
    "training_report": "training_report.json",
    "checkpoint_manifest": "checkpoint_manifest.json",
    "embedding_table": "embedding_table.json",
    "selector_artifacts": "selector_artifacts.json",
    "pipeline_manifest": "pipeline_manifest.json",
}


def require_pipeline_artifact(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def _jsonable_training_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"encoder", "projection_head"}
    }


def run_pipeline(out_dir: Path, seed: int = 20260602) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture = build_controlled_trace_fixture(seed=seed)
    validate_trace_fixture(fixture)
    graphs = build_canonical_graphs(fixture)
    tensors = tensorize_graphs(graphs)
    training_report = train_minimal_contrastive(tensors, out_dir, seed=seed)
    embedding_table = export_embedding_table(
        tensors,
        training_report["encoder"],
        training_report["checkpoint_manifest"],
    )
    selector_artifacts = select_representatives(embedding_table, seed=seed)

    trace_fixture_path = out_dir / ARTIFACT_FILENAMES["trace_fixture"]
    graph_bundle_path = out_dir / ARTIFACT_FILENAMES["graph_bundle"]
    tensor_bundle_path = out_dir / ARTIFACT_FILENAMES["tensor_bundle"]
    training_report_path = out_dir / ARTIFACT_FILENAMES["training_report"]
    checkpoint_manifest_path = out_dir / ARTIFACT_FILENAMES["checkpoint_manifest"]
    embedding_table_path = out_dir / ARTIFACT_FILENAMES["embedding_table"]
    selector_artifacts_path = out_dir / ARTIFACT_FILENAMES["selector_artifacts"]
    pipeline_manifest_path = out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]

    write_json(trace_fixture_path, fixture)
    write_json(graph_bundle_path, {"artifact_type": "graph_bundle", "graphs": graphs})
    write_json(
        tensor_bundle_path,
        {
            "artifact_type": "tensor_bundle",
            "tensors": [tensor_to_jsonable(tensor) for tensor in tensors],
        },
    )
    write_json(training_report_path, _jsonable_training_report(training_report))
    write_json(checkpoint_manifest_path, training_report["checkpoint_manifest"])
    write_json(embedding_table_path, embedding_table)
    write_json(selector_artifacts_path, selector_artifacts)

    manifest = {
        "artifact_type": "gcl_phase_a_pipeline_manifest",
        "seed": seed,
        "fixture_summary": fixture_summary(fixture),
        "paths": {
            "trace_fixture": str(trace_fixture_path),
            "graph_bundle": str(graph_bundle_path),
            "tensor_bundle": str(tensor_bundle_path),
            "training_report": str(training_report_path),
            "checkpoint_manifest": str(checkpoint_manifest_path),
            "embedding_table": str(embedding_table_path),
            "selector_artifacts": str(selector_artifacts_path),
        },
        "hashes": {
            "trace_fixture_hash": fixture["fixture_hash"],
            "graph_hashes": [graph["graph_hash"] for graph in graphs],
            "tensor_hashes": [tensor["tensor_hash"] for tensor in tensors],
            "encoder_manifest_hash": training_report["checkpoint_manifest"]["encoder_manifest_hash"],
            "embedding_table_hash": embedding_table["embedding_table_hash"],
            "selector_manifest_hash": selector_artifacts["selector_manifest_hash"],
        },
    }
    manifest["pipeline_manifest_hash"] = stable_hash(manifest)
    write_json(pipeline_manifest_path, manifest)
    return manifest


def _refresh_pipeline_manifest_hashes(out_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    pipeline_manifest_path = require_pipeline_artifact(
        out_dir / ARTIFACT_FILENAMES["pipeline_manifest"], "pipeline manifest"
    )
    manifest = read_json(pipeline_manifest_path)
    manifest["paths"] = {
        key: str(out_dir / filename)
        for key, filename in ARTIFACT_FILENAMES.items()
        if key != "pipeline_manifest"
    }
    manifest.setdefault("hashes", {}).update(updates)
    manifest["pipeline_manifest_hash"] = stable_hash(
        {key: value for key, value in manifest.items() if key != "pipeline_manifest_hash"}
    )
    write_json(pipeline_manifest_path, manifest)
    return manifest


def run_embedding_export_stage_from_disk(out_dir: Path) -> dict[str, Any]:
    graph_bundle_path = require_pipeline_artifact(
        out_dir / ARTIFACT_FILENAMES["graph_bundle"], "graph bundle"
    )
    tensor_bundle_path = require_pipeline_artifact(
        out_dir / ARTIFACT_FILENAMES["tensor_bundle"], "tensor bundle"
    )
    checkpoint_manifest_path = require_pipeline_artifact(
        out_dir / ARTIFACT_FILENAMES["checkpoint_manifest"], "checkpoint manifest"
    )
    checkpoint_path = require_pipeline_artifact(out_dir / "rgcn_checkpoint.pt", "RGCN checkpoint")

    graph_bundle = read_json(graph_bundle_path)
    graphs = graph_bundle.get("graphs", [])
    if not graphs:
        raise ValueError("graph bundle must contain graphs")
    for graph in graphs:
        validate_graph_artifact(graph)
    graph_hashes = {graph["graph_hash"] for graph in graphs}

    tensor_bundle = read_json(tensor_bundle_path)
    tensors = [tensor_from_jsonable(tensor) for tensor in tensor_bundle.get("tensors", [])]
    if not tensors:
        raise ValueError("tensor bundle must contain tensors")
    if {tensor["input_graph_hash"] for tensor in tensors} != graph_hashes:
        raise ValueError("tensor bundle input_graph_hash values must match graph bundle")

    checkpoint_manifest = read_json(checkpoint_manifest_path)
    checkpoint_hash = hash_without({"checkpoint_bytes": checkpoint_path.read_bytes().hex()})
    if checkpoint_manifest.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("checkpoint_hash does not match rgcn_checkpoint.pt")
    expected_encoder_manifest_hash = hash_without(
        checkpoint_manifest, "encoder_manifest_hash", "checkpoint_path"
    )
    if checkpoint_manifest.get("encoder_manifest_hash") != expected_encoder_manifest_hash:
        raise ValueError("encoder_manifest_hash is not reproducible")
    source_tensor_hashes = [tensor["tensor_hash"] for tensor in tensors]
    if checkpoint_manifest.get("source_tensor_hashes") != source_tensor_hashes:
        raise ValueError("checkpoint manifest source_tensor_hashes do not match tensor bundle")

    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = MinimalRGCNEncoder()
    encoder.load_state_dict(checkpoint["encoder"])
    embedding_table = export_embedding_table(tensors, encoder, checkpoint_manifest)
    validate_embedding_table(embedding_table)
    write_json(out_dir / ARTIFACT_FILENAMES["embedding_table"], embedding_table)
    selector_artifacts_path = out_dir / ARTIFACT_FILENAMES["selector_artifacts"]
    if selector_artifacts_path.exists():
        selector_artifacts_path.unlink()
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {
            "embedding_table_hash": embedding_table["embedding_table_hash"],
            "selector_manifest_hash": None,
        },
    )
    return embedding_table


def run_selector_stage_from_disk(out_dir: Path) -> dict[str, Any]:
    embedding_table_path = require_pipeline_artifact(
        out_dir / ARTIFACT_FILENAMES["embedding_table"], "embedding table"
    )
    pipeline_manifest_path = require_pipeline_artifact(
        out_dir / ARTIFACT_FILENAMES["pipeline_manifest"], "pipeline manifest"
    )
    table = read_json(embedding_table_path)
    pipeline_manifest = read_json(pipeline_manifest_path)
    validate_embedding_table(table)
    selector_artifacts = select_representatives(table, seed=pipeline_manifest["seed"])
    write_json(out_dir / ARTIFACT_FILENAMES["selector_artifacts"], selector_artifacts)
    _refresh_pipeline_manifest_hashes(
        out_dir,
        {
            "embedding_table_hash": table["embedding_table_hash"],
            "selector_manifest_hash": selector_artifacts["selector_manifest_hash"],
        },
    )
    return selector_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GCL Phase A semantic end-to-end pipeline")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260602)
    args = parser.parse_args(argv)
    manifest = run_pipeline(args.out, seed=args.seed)
    print(manifest["pipeline_manifest_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
