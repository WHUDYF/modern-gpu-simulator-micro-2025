"""End-to-end pipeline for GCL Phase B representative-SM scope."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.gcl_phase_a.embedding_export import export_embedding_table, validate_embedding_table
from experiments.gcl_phase_a.train import train_minimal_contrastive
from experiments.gcl_phase_a.tensorizer import TENSORIZER_VERSION as PHASE_A_TENSORIZER_VERSION
from experiments.gcl_phase_a.tensorizer import _tensor_hash as phase_a_tensor_hash

from .graph_audit import build_graph_size_audit, validate_graph_size_audit
from .graph_builder import build_phase_b_graphs, validate_phase_b_graph_artifact
from .readout import build_readout_manifest, validate_readout_manifest
from .selector import select_phase_b_representatives
from .sm_selection import validate_selected_sm_policy_report
from .tensorizer import (
    tensor_from_jsonable,
    tensor_to_jsonable,
    tensorize_phase_b_graphs,
    validate_phase_b_tensor_artifact,
)
from .training import create_augmented_training_views
from .trace_scope import (
    build_phase_b_trace_records,
    build_scope_audit,
    validate_phase_b_trace_manifest,
    validate_scope_audit,
)
from .utils import hash_without, read_json, stable_hash, write_json

ARTIFACT_FILENAMES = {
    "trace_manifest": "trace_manifest.json",
    "selected_sm_policy_report": "selected_sm_policy_report.json",
    "scope_audits": "scope_audits.json",
    "graph_bundle": "graph_bundle.json",
    "graph_size_audits": "graph_size_audits.json",
    "tensor_bundle": "tensor_bundle.json",
    "augmentation_manifests": "augmentation_manifests.json",
    "training_report": "training_report.json",
    "checkpoint_manifest": "checkpoint_manifest.json",
    "readout_manifest": "readout_manifest.json",
    "embedding_table": "embedding_table.json",
    "selector_artifacts": "selector_artifacts.json",
    "resource_blocked_artifact": "resource_blocked_artifact.json",
    "pipeline_manifest": "pipeline_manifest.json",
}


class PhaseBResourceError(RuntimeError):
    """Raised when Phase B cannot continue because of a concrete resource limit."""


def _jsonable_training_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key not in {"encoder", "projection_head"}}


def _write_bundle_artifacts(
    out_dir: Path,
    manifest: dict[str, Any],
    selected_sm_policy_reports: list[dict[str, Any]],
    scope_audits: list[dict[str, Any]],
    graphs: list[dict[str, Any]],
    graph_size_audits: list[dict[str, Any]],
    tensors: list[dict[str, Any]],
) -> None:
    write_json(out_dir / ARTIFACT_FILENAMES["trace_manifest"], manifest)
    write_json(
        out_dir / ARTIFACT_FILENAMES["selected_sm_policy_report"],
        {
            "artifact_type": "gcl_phase_b_selected_sm_policy_report_bundle",
            "reports": selected_sm_policy_reports,
        },
    )
    write_json(
        out_dir / ARTIFACT_FILENAMES["scope_audits"],
        {"artifact_type": "gcl_phase_b_scope_audit_bundle", "audits": scope_audits},
    )
    write_json(
        out_dir / ARTIFACT_FILENAMES["graph_bundle"],
        {"artifact_type": "gcl_phase_b_graph_bundle", "graphs": graphs},
    )
    write_json(
        out_dir / ARTIFACT_FILENAMES["graph_size_audits"],
        {"artifact_type": "gcl_phase_b_graph_size_audit_bundle", "audits": graph_size_audits},
    )
    write_json(
        out_dir / ARTIFACT_FILENAMES["tensor_bundle"],
        {
            "artifact_type": "gcl_phase_b_tensor_bundle",
            "tensors": [tensor_to_jsonable(tensor) for tensor in tensors],
        },
    )


def require_pipeline_artifact(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def run_embedding_export(tensors: list[dict[str, Any]], out_dir: Path, seed: int = 20260602):
    out_dir.mkdir(parents=True, exist_ok=True)
    for tensor in tensors:
        validate_phase_b_tensor_artifact(tensor)
    training_tensors = [_phase_a_compatible_tensor(tensor) for tensor in tensors]
    training_report = train_minimal_contrastive(training_tensors, out_dir, seed=seed)
    embedding_table = export_embedding_table(
        training_tensors,
        training_report["encoder"],
        training_report["checkpoint_manifest"],
    )
    validate_embedding_table(embedding_table)
    return embedding_table, training_report


def _phase_a_compatible_tensor(tensor: dict[str, Any]) -> dict[str, Any]:
    compatible = dict(tensor)
    compatible["artifact_type"] = "graph_tensor"
    compatible["tensorizer_version"] = PHASE_A_TENSORIZER_VERSION
    compatible.pop("phase_b_tensorizer_version", None)
    compatible["tensor_hash"] = phase_a_tensor_hash(compatible)
    return compatible


def create_augmentation_manifest_bundle(tensors: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    views = []
    for index, tensor in enumerate(tensors):
        view_a, view_b = create_augmented_training_views(tensor, seed=seed + index * 2)
        for view in (view_a, view_b):
            views.append(view["phase_b_augmentation_manifest"])
    bundle = {
        "artifact_type": "gcl_phase_b_augmentation_manifest_bundle",
        "manifests": views,
    }
    bundle["augmentation_manifest_bundle_hash"] = hash_without(
        bundle, "augmentation_manifest_bundle_hash"
    )
    return bundle


def build_readout_manifest_bundle(tensors: list[dict[str, Any]], encoder) -> dict[str, Any]:
    import torch

    manifests = []
    encoder.eval()
    with torch.no_grad():
        for tensor in tensors:
            node_features = torch.as_tensor(tensor["node_features"], dtype=torch.float32)
            edge_index = torch.as_tensor(tensor["edge_index"], dtype=torch.long)
            edge_type = torch.as_tensor(tensor["edge_type"], dtype=torch.long)
            node_embeddings = encoder(node_features, edge_index, edge_type)
            manifest, _kernel_embedding = build_readout_manifest(tensor, node_embeddings)
            manifests.append(manifest)
    bundle = {
        "artifact_type": "gcl_phase_b_readout_manifest_bundle",
        "manifests": manifests,
    }
    bundle["readout_manifest_bundle_hash"] = hash_without(bundle, "readout_manifest_bundle_hash")
    return bundle


def _resource_blocked_artifact(
    graphs: list[dict[str, Any]],
    graph_size_audits: list[dict[str, Any]],
    failed_stage: str,
    exc: Exception,
) -> dict[str, Any]:
    artifact = {
        "artifact_type": "gcl_phase_b_resource_blocked_artifact",
        "graph_hashes": [graph["graph_hash"] for graph in graphs],
        "size_audits": graph_size_audits,
        "failed_stage": failed_stage,
        "resource_failure_reason": str(exc),
        "attempted_batch_config": {
            "graph_count": len(graphs),
            "training_mode": "minimal_rgcn_contrastive_smoke",
        },
        "suggested_next_spec_boundary": "phase_c_compression_abstraction",
    }
    artifact["resource_blocked_hash"] = hash_without(artifact, "resource_blocked_hash")
    return artifact


def _resource_not_blocked_artifact(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    artifact = {
        "artifact_type": "gcl_phase_b_resource_blocked_artifact",
        "resource_blocked": False,
        "graph_hashes": [graph["graph_hash"] for graph in graphs],
        "failed_stage": None,
        "resource_failure_reason": None,
    }
    artifact["resource_blocked_hash"] = hash_without(artifact, "resource_blocked_hash")
    return artifact


def run_pipeline(input_manifest_path: Path, out_dir: Path, seed: int = 20260602) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_manifest = read_json(input_manifest_path)
    validate_phase_b_trace_manifest(trace_manifest)
    selected_sm_policy_reports = []
    for invocation in trace_manifest["kernel_invocations"]:
        validate_selected_sm_policy_report(invocation["selected_sm_policy_report"])
        if invocation["selected_sm_policy_report_hash"] != invocation["selected_sm_policy_report"]["selection_hash"]:
            raise ValueError("selected_sm_policy_report_hash mismatch")
        selected_sm_policy_reports.append(invocation["selected_sm_policy_report"])

    records = build_phase_b_trace_records(trace_manifest)
    scope_audits = [build_scope_audit(invocation) for invocation in trace_manifest["kernel_invocations"]]
    for audit, invocation in zip(scope_audits, trace_manifest["kernel_invocations"]):
        validate_scope_audit(audit, invocation)
    graphs = build_phase_b_graphs(records)
    for graph in graphs:
        validate_phase_b_graph_artifact(graph)
    graph_size_audits = [build_graph_size_audit(graph) for graph in graphs]
    for audit, graph in zip(graph_size_audits, graphs):
        validate_graph_size_audit(audit, graph)
    tensors = tensorize_phase_b_graphs(graphs)
    augmentation_bundle = create_augmentation_manifest_bundle(tensors, seed=seed)

    _write_bundle_artifacts(
        out_dir,
        trace_manifest,
        selected_sm_policy_reports,
        scope_audits,
        graphs,
        graph_size_audits,
        tensors,
    )
    write_json(out_dir / ARTIFACT_FILENAMES["augmentation_manifests"], augmentation_bundle)
    base_hashes = {
        "selection_hashes": [
            invocation["selected_sm_policy_report_hash"]
            for invocation in trace_manifest["kernel_invocations"]
        ],
        "trace_scope_hashes": [audit["trace_scope_hash"] for audit in scope_audits],
        "graph_hashes": [graph["graph_hash"] for graph in graphs],
        "graph_size_audit_hashes": [audit["graph_size_audit_hash"] for audit in graph_size_audits],
        "tensor_hashes": [tensor["tensor_hash"] for tensor in tensors],
        "augmentation_manifest_hashes": [
            manifest["augmentation_manifest_hash"]
            for manifest in augmentation_bundle["manifests"]
        ],
        "augmentation_manifest_bundle_hash": augmentation_bundle["augmentation_manifest_bundle_hash"],
    }

    try:
        embedding_table, training_report = run_embedding_export(tensors, out_dir, seed=seed)
    except (RuntimeError, PhaseBResourceError) as exc:
        blocked = _resource_blocked_artifact(graphs, graph_size_audits, "training", exc)
        write_json(out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"], blocked)
        manifest = {
            "artifact_type": "gcl_phase_b_pipeline_manifest",
            "seed": seed,
            "resource_blocked": True,
            "paths": _paths(out_dir),
            "hashes": {
                **base_hashes,
                "resource_blocked_hash": blocked["resource_blocked_hash"],
                "embedding_table_hash": None,
                "selector_manifest_hash": None,
            },
        }
        manifest["pipeline_manifest_hash"] = stable_hash(manifest)
        write_json(out_dir / ARTIFACT_FILENAMES["pipeline_manifest"], manifest)
        return manifest

    selector_artifacts = select_phase_b_representatives(embedding_table, seed=seed)
    readout_bundle = build_readout_manifest_bundle(tensors, training_report["encoder"])
    resource_status = _resource_not_blocked_artifact(graphs)
    write_json(out_dir / ARTIFACT_FILENAMES["training_report"], _jsonable_training_report(training_report))
    write_json(out_dir / ARTIFACT_FILENAMES["checkpoint_manifest"], training_report["checkpoint_manifest"])
    write_json(out_dir / ARTIFACT_FILENAMES["readout_manifest"], readout_bundle)
    write_json(out_dir / ARTIFACT_FILENAMES["embedding_table"], embedding_table)
    write_json(out_dir / ARTIFACT_FILENAMES["selector_artifacts"], selector_artifacts)
    write_json(out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"], resource_status)

    manifest = {
        "artifact_type": "gcl_phase_b_pipeline_manifest",
        "seed": seed,
        "resource_blocked": False,
        "paths": _paths(out_dir),
        "hashes": {
            **base_hashes,
            "encoder_manifest_hash": training_report["checkpoint_manifest"]["encoder_manifest_hash"],
            "readout_manifest_hashes": [
                manifest["readout_manifest_hash"]
                for manifest in readout_bundle["manifests"]
            ],
            "readout_manifest_bundle_hash": readout_bundle["readout_manifest_bundle_hash"],
            "embedding_table_hash": embedding_table["embedding_table_hash"],
            "selector_manifest_hash": selector_artifacts["selector_manifest_hash"],
            "resource_blocked_hash": resource_status["resource_blocked_hash"],
        },
    }
    manifest["pipeline_manifest_hash"] = stable_hash(manifest)
    write_json(out_dir / ARTIFACT_FILENAMES["pipeline_manifest"], manifest)
    return manifest


def _paths(out_dir: Path) -> dict[str, str]:
    return {
        key: str(out_dir / filename)
        for key, filename in ARTIFACT_FILENAMES.items()
        if key != "pipeline_manifest"
    }


def run_embedding_export_stage_from_disk(out_dir: Path, seed: int = 20260602) -> dict[str, Any]:
    require_pipeline_artifact(
        out_dir / ARTIFACT_FILENAMES["graph_size_audits"], "graph size audit bundle"
    )
    tensor_bundle = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["tensor_bundle"], "tensor bundle")
    )
    tensors = [tensor_from_jsonable(tensor) for tensor in tensor_bundle.get("tensors", [])]
    table, training_report = run_embedding_export(tensors, out_dir, seed=seed)
    write_json(out_dir / ARTIFACT_FILENAMES["training_report"], _jsonable_training_report(training_report))
    write_json(out_dir / ARTIFACT_FILENAMES["checkpoint_manifest"], training_report["checkpoint_manifest"])
    write_json(out_dir / ARTIFACT_FILENAMES["embedding_table"], table)
    return table


def run_selector_stage_from_disk(out_dir: Path, seed: int = 20260602) -> dict[str, Any]:
    table = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["embedding_table"], "embedding table")
    )
    artifacts = select_phase_b_representatives(table, seed=seed)
    write_json(out_dir / ARTIFACT_FILENAMES["selector_artifacts"], artifacts)
    return artifacts


def run_graph_construction_stage_from_disk(out_dir: Path) -> list[dict[str, Any]]:
    manifest = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["trace_manifest"], "trace manifest")
    )
    report_bundle = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["selected_sm_policy_report"],
            "selected SM policy report",
        )
    )
    reports = report_bundle.get("reports", [])
    if len(reports) != len(manifest.get("kernel_invocations", [])):
        raise ValueError("selected SM policy report count mismatch")
    for invocation, report in zip(manifest["kernel_invocations"], reports):
        validate_selected_sm_policy_report(report)
        if invocation["selected_sm_policy_report_hash"] != report["selection_hash"]:
            raise ValueError("selected_sm_policy_report_hash mismatch")
    records = build_phase_b_trace_records(manifest)
    graphs = build_phase_b_graphs(records)
    write_json(
        out_dir / ARTIFACT_FILENAMES["graph_bundle"],
        {"artifact_type": "gcl_phase_b_graph_bundle", "graphs": graphs},
    )
    return graphs


def run_tensorization_stage_from_disk(out_dir: Path) -> list[dict[str, Any]]:
    graph_bundle = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["graph_bundle"], "graph bundle")
    )
    audit_bundle = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["graph_size_audits"], "graph size audit bundle"
        )
    )
    graphs = graph_bundle.get("graphs", [])
    audits = audit_bundle.get("audits", [])
    if len(graphs) != len(audits):
        raise ValueError("graph size audit count mismatch")
    for graph, audit in zip(graphs, audits):
        validate_graph_size_audit(audit, graph)
    tensors = tensorize_phase_b_graphs(graphs)
    write_json(
        out_dir / ARTIFACT_FILENAMES["tensor_bundle"],
        {
            "artifact_type": "gcl_phase_b_tensor_bundle",
            "tensors": [tensor_to_jsonable(tensor) for tensor in tensors],
        },
    )
    return tensors


def validate_phase_b_replay_from_disk(out_dir: Path) -> dict[str, Any]:
    pipeline_manifest = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["pipeline_manifest"], "pipeline manifest"
        )
    )
    checkpoint_manifest = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["checkpoint_manifest"], "checkpoint manifest"
        )
    )
    checkpoint_path = require_pipeline_artifact(out_dir / "rgcn_checkpoint.pt", "RGCN checkpoint")
    checkpoint_hash = hash_without({"checkpoint_bytes": checkpoint_path.read_bytes().hex()})
    if checkpoint_manifest.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("checkpoint_hash does not match rgcn_checkpoint.pt")
    expected_encoder_hash = hash_without(
        checkpoint_manifest, "encoder_manifest_hash", "checkpoint_path"
    )
    if checkpoint_manifest.get("encoder_manifest_hash") != expected_encoder_hash:
        raise ValueError("encoder_manifest_hash is not reproducible")

    tensor_bundle = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["tensor_bundle"], "tensor bundle")
    )
    tensors = [tensor_from_jsonable(tensor) for tensor in tensor_bundle.get("tensors", [])]
    source_tensor_hashes = [tensor["tensor_hash"] for tensor in tensors]
    if checkpoint_manifest.get("source_tensor_hashes") != source_tensor_hashes:
        raise ValueError("checkpoint manifest source_tensor_hashes do not match tensor bundle")

    embedding_table = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["embedding_table"], "embedding table")
    )
    validate_embedding_table(embedding_table)
    if embedding_table.get("encoder_manifest_hash") != checkpoint_manifest["encoder_manifest_hash"]:
        raise ValueError("embedding table encoder_manifest_hash mismatch")
    if pipeline_manifest["hashes"].get("encoder_manifest_hash") != checkpoint_manifest["encoder_manifest_hash"]:
        raise ValueError("pipeline manifest encoder_manifest_hash mismatch")
    if pipeline_manifest["hashes"].get("embedding_table_hash") != embedding_table["embedding_table_hash"]:
        raise ValueError("pipeline manifest embedding_table_hash mismatch")

    selector_artifacts = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["selector_artifacts"], "selector artifacts"
        )
    )
    if selector_artifacts.get("source_embedding_table_hash") != embedding_table["embedding_table_hash"]:
        raise ValueError("selector source_embedding_table_hash mismatch")
    if pipeline_manifest["hashes"].get("selector_manifest_hash") != selector_artifacts["selector_manifest_hash"]:
        raise ValueError("pipeline manifest selector_manifest_hash mismatch")

    readout_bundle = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["readout_manifest"], "readout manifest")
    )
    for manifest, tensor in zip(readout_bundle.get("manifests", []), tensors):
        validate_readout_manifest(manifest, tensor)
    readout_hashes = [manifest["readout_manifest_hash"] for manifest in readout_bundle.get("manifests", [])]
    if pipeline_manifest["hashes"].get("readout_manifest_hashes") != readout_hashes:
        raise ValueError("pipeline manifest readout_manifest_hashes mismatch")
    if readout_bundle.get("readout_manifest_bundle_hash") != hash_without(
        readout_bundle, "readout_manifest_bundle_hash"
    ):
        raise ValueError("readout_manifest_bundle_hash is not reproducible")
    if pipeline_manifest["hashes"].get("readout_manifest_bundle_hash") != readout_bundle["readout_manifest_bundle_hash"]:
        raise ValueError("pipeline manifest readout_manifest_bundle_hash mismatch")

    expected_pipeline_hash = stable_hash(
        {key: value for key, value in pipeline_manifest.items() if key != "pipeline_manifest_hash"}
    )
    if pipeline_manifest.get("pipeline_manifest_hash") != expected_pipeline_hash:
        raise ValueError("pipeline_manifest_hash is not reproducible")
    return {
        "artifact_type": "gcl_phase_b_replay_validation",
        "checkpoint_hash": checkpoint_hash,
        "encoder_manifest_hash": checkpoint_manifest["encoder_manifest_hash"],
        "embedding_table_hash": embedding_table["embedding_table_hash"],
        "selector_manifest_hash": selector_artifacts["selector_manifest_hash"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GCL Phase B representative-SM pipeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260602)
    args = parser.parse_args(argv)
    manifest = run_pipeline(args.input, args.out, seed=args.seed)
    print(manifest["pipeline_manifest_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
