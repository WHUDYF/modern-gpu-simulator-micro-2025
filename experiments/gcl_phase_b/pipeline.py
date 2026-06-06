"""End-to-end pipeline for GCL Phase B representative-SM scope."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.gcl_phase_a.train import train_minimal_contrastive
from experiments.gcl_phase_a.tensorizer import TENSORIZER_VERSION as PHASE_A_TENSORIZER_VERSION
from experiments.gcl_phase_a.tensorizer import _tensor_hash as phase_a_tensor_hash

from .graph_audit import build_graph_size_audit, validate_graph_size_audit
from .graph_builder import build_phase_b_graphs, validate_phase_b_graph_artifact
from .embedding_export import export_phase_b_embedding_table, validate_phase_b_embedding_table
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


SUCCESS_ARTIFACT_KEYS = {
    "training_report",
    "checkpoint_manifest",
    "readout_manifest",
    "embedding_table",
    "selector_artifacts",
}
TENSOR_DOWNSTREAM_ARTIFACT_KEYS = {
    "augmentation_manifests",
    "resource_blocked_artifact",
    *SUCCESS_ARTIFACT_KEYS,
}
CHECKPOINT_FILENAME = "rgcn_checkpoint.pt"
EMBEDDING_DOWNSTREAM_HASH_NULLS = {
    "encoder_manifest_hash": None,
    "readout_manifest_hashes": None,
    "readout_manifest_bundle_hash": None,
    "embedding_table_hash": None,
    "selector_manifest_hash": None,
}
TENSOR_DOWNSTREAM_HASH_NULLS = {
    "augmentation_manifest_hashes": None,
    "augmentation_manifest_bundle_hash": None,
    "resource_blocked_hash": None,
    **EMBEDDING_DOWNSTREAM_HASH_NULLS,
}
GRAPH_DOWNSTREAM_HASH_NULLS = {
    "tensor_hashes": None,
    **TENSOR_DOWNSTREAM_HASH_NULLS,
}


class PhaseBResourceError(RuntimeError):
    """Raised when Phase B cannot continue because of a concrete resource limit."""


def _is_resource_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, PhaseBResourceError | MemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    resource_markers = (
        "out of memory",
        "cuda oom",
        "cuda memory",
        "cublas",
        "cudnn",
        "allocation failed",
        "failed to allocate",
        "resource exhausted",
    )
    return any(marker in message for marker in resource_markers)


def _jsonable_training_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key not in {"encoder", "projection_head"}}


def _embedding_table_hash(table: dict[str, Any]) -> str:
    return table["kernel_embedding_table_hash"]


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


def _validate_selected_sm_report_matches_invocation(
    invocation: dict[str, Any],
    report: dict[str, Any],
) -> None:
    expected_fields = {
        "kernel_invocation_id",
        "selected_sm_policy",
        "selected_sm",
        "selected_sm_reason",
        "candidate_sm_count",
    }
    for field in expected_fields:
        if report.get(field) != invocation.get(field):
            raise ValueError(f"selected_sm_policy_report {field} mismatch")


def _remove_success_artifacts(out_dir: Path) -> None:
    for key in SUCCESS_ARTIFACT_KEYS:
        (out_dir / ARTIFACT_FILENAMES[key]).unlink(missing_ok=True)
    (out_dir / CHECKPOINT_FILENAME).unlink(missing_ok=True)


def _remove_tensor_and_downstream_artifacts(out_dir: Path) -> None:
    (out_dir / ARTIFACT_FILENAMES["tensor_bundle"]).unlink(missing_ok=True)
    _remove_tensor_downstream_artifacts(out_dir)


def _remove_tensor_downstream_artifacts(out_dir: Path) -> None:
    for key in TENSOR_DOWNSTREAM_ARTIFACT_KEYS:
        (out_dir / ARTIFACT_FILENAMES[key]).unlink(missing_ok=True)
    (out_dir / CHECKPOINT_FILENAME).unlink(missing_ok=True)


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
    embedding_table, _readout_bundle = export_phase_b_embedding_table(
        tensors,
        training_report["encoder"],
        training_report["checkpoint_manifest"],
    )
    validate_phase_b_embedding_table(embedding_table)
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
        "resource_blocked": True,
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


def _validate_resource_status_artifact(
    artifact: dict[str, Any],
    graphs: list[dict[str, Any]],
    graph_size_audits: list[dict[str, Any]],
    expected_resource_blocked: bool,
) -> None:
    if artifact.get("artifact_type") != "gcl_phase_b_resource_blocked_artifact":
        raise ValueError("resource blocked artifact_type mismatch")
    if artifact.get("resource_blocked_hash") != hash_without(artifact, "resource_blocked_hash"):
        raise ValueError("resource_blocked_hash is not reproducible")
    if artifact.get("resource_blocked") is not expected_resource_blocked:
        raise ValueError("resource_blocked status mismatch")
    if artifact.get("graph_hashes") != [graph["graph_hash"] for graph in graphs]:
        raise ValueError("resource blocked graph_hashes mismatch")
    if expected_resource_blocked:
        if artifact.get("size_audits") != graph_size_audits:
            raise ValueError("resource blocked size_audits mismatch")
        if not artifact.get("failed_stage"):
            raise ValueError("resource blocked failed_stage is required")
        if not artifact.get("resource_failure_reason"):
            raise ValueError("resource blocked resource_failure_reason is required")
    else:
        if artifact.get("failed_stage") is not None:
            raise ValueError("non-blocked failed_stage must be null")
        if artifact.get("resource_failure_reason") is not None:
            raise ValueError("non-blocked resource_failure_reason must be null")


def _validate_selector_artifacts_cover_embedding_table(
    selector_artifacts: dict[str, Any],
    embedding_table: dict[str, Any],
) -> None:
    rows = embedding_table["embeddings"]
    expected_by_record = {
        row["record_id"]: row["kernel_invocation_id"]
        for row in rows
    }
    assignments = _selector_assignments(selector_artifacts)
    if len(assignments) != len(expected_by_record):
        raise ValueError("selector cluster_assignments do not cover embedding table")
    assignment_by_record = {}
    for assignment in assignments:
        record_id = assignment.get("record_id")
        if record_id in assignment_by_record:
            raise ValueError("selector cluster_assignments contain duplicate record_id")
        assignment_by_record[record_id] = assignment
    if set(assignment_by_record) != set(expected_by_record):
        raise ValueError("selector cluster_assignments do not match embedding table records")
    for record_id, expected_invocation_id in expected_by_record.items():
        if assignment_by_record[record_id].get("kernel_invocation_id") != expected_invocation_id:
            raise ValueError("selector cluster_assignments kernel_invocation_id mismatch")

    assignment_cluster_ids = {
        int(assignment["cluster_id"])
        for assignment in assignments
    }
    anchors = _selector_anchors(selector_artifacts)
    if not anchors:
        raise ValueError("selector representative_anchor_table must not be empty")
    anchor_cluster_ids = set()
    for anchor in anchors:
        record_id = anchor.get("representative_record_id")
        if record_id not in expected_by_record:
            raise ValueError("selector representative_anchor_table references unknown record")
        cluster_id = int(anchor["cluster_id"])
        if cluster_id not in assignment_cluster_ids:
            raise ValueError("selector representative_anchor_table references unknown cluster")
        if assignment_by_record[record_id]["cluster_id"] != cluster_id:
            raise ValueError("selector anchor cluster_id does not match assignment")
        if anchor.get("kernel_invocation_id") != expected_by_record[record_id]:
            raise ValueError("selector anchor kernel_invocation_id mismatch")
        anchor_cluster_ids.add(cluster_id)
    if anchor_cluster_ids != assignment_cluster_ids:
        raise ValueError("selector representative_anchor_table does not cover clusters")


def _selector_assignments(selector_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    if "kmeans_cluster_assignment_table" in selector_artifacts:
        return selector_artifacts["kmeans_cluster_assignment_table"].get("assignments", [])
    return selector_artifacts.get("cluster_assignments", [])


def _selector_anchors(selector_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    table = selector_artifacts.get("representative_anchor_table", [])
    if isinstance(table, dict):
        return table.get("anchors", [])
    return table


def run_pipeline(input_manifest_path: Path, out_dir: Path, seed: int = 20260602) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_manifest = read_json(input_manifest_path)
    validate_phase_b_trace_manifest(trace_manifest)
    selected_sm_policy_reports = []
    for invocation in trace_manifest["kernel_invocations"]:
        if "selected_sm_policy_report" not in invocation:
            raise ValueError("selected_sm_policy_report is required in Phase B trace manifest")
        validate_selected_sm_policy_report(invocation["selected_sm_policy_report"])
        if invocation["selected_sm_policy_report_hash"] != invocation["selected_sm_policy_report"]["selection_hash"]:
            raise ValueError("selected_sm_policy_report_hash mismatch")
        _validate_selected_sm_report_matches_invocation(
            invocation,
            invocation["selected_sm_policy_report"],
        )
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
    except (PhaseBResourceError, MemoryError, RuntimeError) as exc:
        if not _is_resource_limit_error(exc):
            raise
        return _write_resource_blocked_pipeline_manifest(
            out_dir,
            seed,
            base_hashes,
            graphs,
            graph_size_audits,
            "training",
            exc,
        )

    try:
        selector_artifacts = select_phase_b_representatives(
            embedding_table, seed=seed, allow_debug=True
        )
    except (PhaseBResourceError, MemoryError, RuntimeError) as exc:
        if not _is_resource_limit_error(exc):
            raise
        return _write_resource_blocked_pipeline_manifest(
            out_dir,
            seed,
            base_hashes,
            graphs,
            graph_size_audits,
            "selector",
            exc,
        )
    try:
        readout_bundle = build_readout_manifest_bundle(tensors, training_report["encoder"])
    except (PhaseBResourceError, MemoryError, RuntimeError) as exc:
        if not _is_resource_limit_error(exc):
            raise
        return _write_resource_blocked_pipeline_manifest(
            out_dir,
            seed,
            base_hashes,
            graphs,
            graph_size_audits,
            "readout",
            exc,
        )
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
            "embedding_table_hash": _embedding_table_hash(embedding_table),
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


def _recorded_seed(out_dir: Path, fallback: int) -> int:
    manifest_path = out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]
    if not manifest_path.exists():
        return fallback
    return int(read_json(manifest_path).get("seed", fallback))


def _refresh_pipeline_manifest_hashes(
    out_dir: Path,
    hash_updates: dict[str, Any],
    top_level_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["pipeline_manifest"], "pipeline manifest"
        )
    )
    manifest = dict(manifest)
    if top_level_updates:
        manifest.update(top_level_updates)
    manifest["paths"] = _paths(out_dir)
    manifest["hashes"] = {**manifest.get("hashes", {}), **hash_updates}
    manifest["pipeline_manifest_hash"] = stable_hash(
        {key: value for key, value in manifest.items() if key != "pipeline_manifest_hash"}
    )
    write_json(out_dir / ARTIFACT_FILENAMES["pipeline_manifest"], manifest)
    return manifest


def _refresh_pipeline_manifest_hashes_if_present(
    out_dir: Path,
    hash_updates: dict[str, Any],
    top_level_updates: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not (out_dir / ARTIFACT_FILENAMES["pipeline_manifest"]).exists():
        return None
    return _refresh_pipeline_manifest_hashes(out_dir, hash_updates, top_level_updates)


def _write_resource_blocked_pipeline_manifest(
    out_dir: Path,
    seed: int,
    base_hashes: dict[str, Any],
    graphs: list[dict[str, Any]],
    graph_size_audits: list[dict[str, Any]],
    failed_stage: str,
    exc: Exception,
) -> dict[str, Any]:
    _remove_success_artifacts(out_dir)
    blocked = _resource_blocked_artifact(graphs, graph_size_audits, failed_stage, exc)
    write_json(out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"], blocked)
    manifest = {
        "artifact_type": "gcl_phase_b_pipeline_manifest",
        "seed": seed,
        "resource_blocked": True,
        "paths": _paths(out_dir),
        "hashes": {
            **base_hashes,
            "resource_blocked_hash": blocked["resource_blocked_hash"],
            **EMBEDDING_DOWNSTREAM_HASH_NULLS,
        },
    }
    manifest["pipeline_manifest_hash"] = stable_hash(manifest)
    write_json(out_dir / ARTIFACT_FILENAMES["pipeline_manifest"], manifest)
    return manifest


def _mark_embedding_stage_resource_blocked(
    out_dir: Path,
    augmentation_bundle: dict[str, Any],
    graphs: list[dict[str, Any]],
    graph_size_audits: list[dict[str, Any]],
    failed_stage: str,
    exc: Exception,
) -> None:
    _remove_success_artifacts(out_dir)
    blocked = _resource_blocked_artifact(graphs, graph_size_audits, failed_stage, exc)
    write_json(out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"], blocked)
    write_json(out_dir / ARTIFACT_FILENAMES["augmentation_manifests"], augmentation_bundle)
    _refresh_pipeline_manifest_hashes_if_present(
        out_dir,
        {
            "augmentation_manifest_hashes": [
                manifest["augmentation_manifest_hash"]
                for manifest in augmentation_bundle["manifests"]
            ],
            "augmentation_manifest_bundle_hash": augmentation_bundle[
                "augmentation_manifest_bundle_hash"
            ],
            "resource_blocked_hash": blocked["resource_blocked_hash"],
            **EMBEDDING_DOWNSTREAM_HASH_NULLS,
        },
        top_level_updates={"resource_blocked": True},
    )


def run_embedding_export_stage_from_disk(out_dir: Path, seed: int | None = None) -> dict[str, Any]:
    resolved_seed = _recorded_seed(out_dir, 20260602 if seed is None else seed)
    require_pipeline_artifact(
        out_dir / ARTIFACT_FILENAMES["graph_size_audits"], "graph size audit bundle"
    )
    tensor_bundle = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["tensor_bundle"], "tensor bundle")
    )
    tensors = [tensor_from_jsonable(tensor) for tensor in tensor_bundle.get("tensors", [])]
    augmentation_bundle = create_augmentation_manifest_bundle(tensors, seed=resolved_seed)
    graph_bundle = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["graph_bundle"], "graph bundle")
    )
    graphs = graph_bundle.get("graphs", [])
    if [tensor["input_graph_hash"] for tensor in tensors] != [
        graph["graph_hash"] for graph in graphs
    ]:
        raise ValueError("tensor bundle input_graph_hash values do not match graph bundle")
    audit_bundle = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["graph_size_audits"], "graph size audit bundle"
        )
    )
    graph_size_audits = audit_bundle.get("audits", [])
    if len(graph_size_audits) != len(graphs):
        raise ValueError("graph size audit count mismatch")
    for audit, graph in zip(graph_size_audits, graphs):
        validate_graph_size_audit(audit, graph)
    try:
        table, training_report = run_embedding_export(tensors, out_dir, seed=resolved_seed)
    except (PhaseBResourceError, MemoryError, RuntimeError) as exc:
        if not _is_resource_limit_error(exc):
            raise
        _mark_embedding_stage_resource_blocked(
            out_dir,
            augmentation_bundle,
            graphs,
            graph_size_audits,
            "training",
            exc,
        )
        raise
    try:
        selector_artifacts = select_phase_b_representatives(
            table, seed=resolved_seed, allow_debug=True
        )
    except (PhaseBResourceError, MemoryError, RuntimeError) as exc:
        if not _is_resource_limit_error(exc):
            raise
        _mark_embedding_stage_resource_blocked(
            out_dir,
            augmentation_bundle,
            graphs,
            graph_size_audits,
            "selector",
            exc,
        )
        raise
    try:
        readout_bundle = build_readout_manifest_bundle(tensors, training_report["encoder"])
    except (PhaseBResourceError, MemoryError, RuntimeError) as exc:
        if not _is_resource_limit_error(exc):
            raise
        _mark_embedding_stage_resource_blocked(
            out_dir,
            augmentation_bundle,
            graphs,
            graph_size_audits,
            "readout",
            exc,
        )
        raise
    resource_status = _resource_not_blocked_artifact(graphs)
    write_json(out_dir / ARTIFACT_FILENAMES["training_report"], _jsonable_training_report(training_report))
    write_json(out_dir / ARTIFACT_FILENAMES["checkpoint_manifest"], training_report["checkpoint_manifest"])
    write_json(out_dir / ARTIFACT_FILENAMES["readout_manifest"], readout_bundle)
    write_json(out_dir / ARTIFACT_FILENAMES["embedding_table"], table)
    write_json(out_dir / ARTIFACT_FILENAMES["selector_artifacts"], selector_artifacts)
    write_json(out_dir / ARTIFACT_FILENAMES["augmentation_manifests"], augmentation_bundle)
    write_json(out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"], resource_status)
    _refresh_pipeline_manifest_hashes_if_present(
        out_dir,
        {
            "augmentation_manifest_hashes": [
                manifest["augmentation_manifest_hash"]
                for manifest in augmentation_bundle["manifests"]
            ],
            "augmentation_manifest_bundle_hash": augmentation_bundle[
                "augmentation_manifest_bundle_hash"
            ],
            "encoder_manifest_hash": training_report["checkpoint_manifest"]["encoder_manifest_hash"],
            "readout_manifest_hashes": [
                manifest["readout_manifest_hash"]
                for manifest in readout_bundle["manifests"]
            ],
            "readout_manifest_bundle_hash": readout_bundle["readout_manifest_bundle_hash"],
            "embedding_table_hash": _embedding_table_hash(table),
            "selector_manifest_hash": selector_artifacts["selector_manifest_hash"],
            "resource_blocked_hash": resource_status["resource_blocked_hash"],
        },
        top_level_updates={"resource_blocked": False},
    )
    return table


def run_selector_stage_from_disk(out_dir: Path, seed: int | None = None) -> dict[str, Any]:
    resolved_seed = _recorded_seed(out_dir, 20260602 if seed is None else seed)
    table = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["embedding_table"], "embedding table")
    )
    artifacts = select_phase_b_representatives(table, seed=resolved_seed, allow_debug=True)
    write_json(out_dir / ARTIFACT_FILENAMES["selector_artifacts"], artifacts)
    _refresh_pipeline_manifest_hashes_if_present(
        out_dir,
        {
            "embedding_table_hash": _embedding_table_hash(table),
            "selector_manifest_hash": artifacts["selector_manifest_hash"],
        },
    )
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
        _validate_selected_sm_report_matches_invocation(invocation, report)
    scope_audits = [build_scope_audit(invocation) for invocation in manifest["kernel_invocations"]]
    for audit, invocation in zip(scope_audits, manifest["kernel_invocations"]):
        validate_scope_audit(audit, invocation)
    records = build_phase_b_trace_records(manifest)
    graphs = build_phase_b_graphs(records)
    graph_size_audits = [build_graph_size_audit(graph) for graph in graphs]
    _remove_tensor_and_downstream_artifacts(out_dir)
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
    _refresh_pipeline_manifest_hashes_if_present(
        out_dir,
        {
            "selection_hashes": [
                invocation["selected_sm_policy_report_hash"]
                for invocation in manifest["kernel_invocations"]
            ],
            "trace_scope_hashes": [
                audit["trace_scope_hash"]
                for audit in scope_audits
            ],
            "graph_hashes": [graph["graph_hash"] for graph in graphs],
            "graph_size_audit_hashes": [
                audit["graph_size_audit_hash"]
                for audit in graph_size_audits
            ],
            **GRAPH_DOWNSTREAM_HASH_NULLS,
        },
        top_level_updates={"resource_blocked": False},
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
    _remove_tensor_downstream_artifacts(out_dir)
    write_json(
        out_dir / ARTIFACT_FILENAMES["tensor_bundle"],
        {
            "artifact_type": "gcl_phase_b_tensor_bundle",
            "tensors": [tensor_to_jsonable(tensor) for tensor in tensors],
        },
    )
    _refresh_pipeline_manifest_hashes_if_present(
        out_dir,
        {
            "tensor_hashes": [tensor["tensor_hash"] for tensor in tensors],
            **TENSOR_DOWNSTREAM_HASH_NULLS,
        },
        top_level_updates={"resource_blocked": False},
    )
    return tensors


def validate_phase_b_replay_from_disk(out_dir: Path) -> dict[str, Any]:
    pipeline_manifest = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["pipeline_manifest"], "pipeline manifest"
        )
    )
    if pipeline_manifest.get("paths") != _paths(out_dir):
        raise ValueError("pipeline_manifest paths mismatch")
    trace_manifest = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["trace_manifest"], "trace manifest")
    )
    validate_phase_b_trace_manifest(trace_manifest)
    scope_bundle = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["scope_audits"], "scope audit bundle")
    )
    scope_audits = scope_bundle.get("audits", [])
    if len(scope_audits) != len(trace_manifest.get("kernel_invocations", [])):
        raise ValueError("scope audit count mismatch")
    for audit, invocation in zip(scope_audits, trace_manifest["kernel_invocations"]):
        validate_scope_audit(audit, invocation)
    if pipeline_manifest["hashes"].get("trace_scope_hashes") != [
        audit["trace_scope_hash"] for audit in scope_audits
    ]:
        raise ValueError("pipeline manifest trace_scope_hashes mismatch")

    report_bundle = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["selected_sm_policy_report"],
            "selected SM policy report",
        )
    )
    selected_reports = report_bundle.get("reports", [])
    if len(selected_reports) != len(trace_manifest.get("kernel_invocations", [])):
        raise ValueError("selected_sm_policy_report count mismatch")
    selection_hashes = []
    for invocation, report in zip(trace_manifest["kernel_invocations"], selected_reports):
        try:
            validate_selected_sm_policy_report(report)
        except ValueError as exc:
            raise ValueError(f"selected_sm_policy_report validation failed: {exc}") from exc
        if invocation["selected_sm_policy_report_hash"] != report["selection_hash"]:
            raise ValueError("selected_sm_policy_report hash mismatch")
        _validate_selected_sm_report_matches_invocation(invocation, report)
        selection_hashes.append(report["selection_hash"])
    if pipeline_manifest["hashes"].get("selection_hashes") != selection_hashes:
        raise ValueError("pipeline manifest selection_hashes mismatch")

    graph_bundle = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["graph_bundle"], "graph bundle")
    )
    graphs = graph_bundle.get("graphs", [])
    if not graphs:
        raise ValueError("graph bundle must contain graphs")
    for graph in graphs:
        validate_phase_b_graph_artifact(graph)
    graph_hashes = [graph["graph_hash"] for graph in graphs]
    if pipeline_manifest["hashes"].get("graph_hashes") != graph_hashes:
        raise ValueError("pipeline manifest graph_hashes mismatch")

    audit_bundle = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["graph_size_audits"], "graph size audit bundle"
        )
    )
    graph_size_audits = audit_bundle.get("audits", [])
    if len(graph_size_audits) != len(graphs):
        raise ValueError("graph size audit count mismatch")
    for audit, graph in zip(graph_size_audits, graphs):
        validate_graph_size_audit(audit, graph)
    if pipeline_manifest["hashes"].get("graph_size_audit_hashes") != [
        audit["graph_size_audit_hash"] for audit in graph_size_audits
    ]:
        raise ValueError("pipeline manifest graph_size_audit_hashes mismatch")

    tensor_bundle = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["tensor_bundle"], "tensor bundle")
    )
    tensors = [tensor_from_jsonable(tensor) for tensor in tensor_bundle.get("tensors", [])]
    if [tensor["input_graph_hash"] for tensor in tensors] != graph_hashes:
        raise ValueError("tensor bundle input_graph_hash values do not match graph bundle")
    if pipeline_manifest["hashes"].get("tensor_hashes") != [tensor["tensor_hash"] for tensor in tensors]:
        raise ValueError("pipeline manifest tensor_hashes mismatch")

    augmentation_bundle = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["augmentation_manifests"],
            "augmentation manifest bundle",
        )
    )
    augmentation_manifests = augmentation_bundle.get("manifests", [])
    expected_augmentation_count = len(tensors) * 2
    if len(augmentation_manifests) != expected_augmentation_count:
        raise ValueError("augmentation manifest count mismatch")
    augmentation_hashes = []
    for index, manifest in enumerate(augmentation_manifests):
        if manifest.get("augmentation_manifest_hash") != hash_without(
            manifest, "augmentation_manifest_hash"
        ):
            raise ValueError("augmentation_manifest_hash is not reproducible")
        tensor_index = index // 2
        tensor = tensors[tensor_index]
        if manifest.get("input_graph_hash") != tensor["input_graph_hash"]:
            raise ValueError("augmentation manifest source tensor mismatch")
        augmentation_hashes.append(manifest["augmentation_manifest_hash"])
    if pipeline_manifest["hashes"].get("augmentation_manifest_hashes") != augmentation_hashes:
        raise ValueError("pipeline manifest augmentation_manifest_hashes mismatch")
    if augmentation_bundle.get("augmentation_manifest_bundle_hash") != hash_without(
        augmentation_bundle, "augmentation_manifest_bundle_hash"
    ):
        raise ValueError("augmentation_manifest_bundle_hash is not reproducible")
    if pipeline_manifest["hashes"].get("augmentation_manifest_bundle_hash") != augmentation_bundle[
        "augmentation_manifest_bundle_hash"
    ]:
        raise ValueError("pipeline manifest augmentation_manifest_bundle_hash mismatch")

    expected_pipeline_hash = stable_hash(
        {key: value for key, value in pipeline_manifest.items() if key != "pipeline_manifest_hash"}
    )
    if pipeline_manifest.get("pipeline_manifest_hash") != expected_pipeline_hash:
        raise ValueError("pipeline_manifest_hash is not reproducible")

    resource_status = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["resource_blocked_artifact"],
            "resource blocked artifact",
        )
    )
    _validate_resource_status_artifact(
        resource_status,
        graphs,
        graph_size_audits,
        bool(pipeline_manifest.get("resource_blocked")),
    )
    if resource_status.get("resource_blocked_hash") != pipeline_manifest["hashes"].get(
        "resource_blocked_hash"
    ):
        raise ValueError("resource_blocked_hash mismatch")

    if pipeline_manifest.get("resource_blocked"):
        for key in EMBEDDING_DOWNSTREAM_HASH_NULLS:
            if pipeline_manifest["hashes"].get(key) is not None:
                raise ValueError("resource-blocked replay contains stale success hash")
        for key in SUCCESS_ARTIFACT_KEYS:
            if (out_dir / ARTIFACT_FILENAMES[key]).exists():
                raise ValueError("resource-blocked replay contains stale success artifact")
        if (out_dir / CHECKPOINT_FILENAME).exists():
            raise ValueError("resource-blocked replay contains stale success artifact")
        return {
            "artifact_type": "gcl_phase_b_replay_validation",
            "resource_blocked": True,
            "resource_blocked_hash": resource_status["resource_blocked_hash"],
        }

    checkpoint_manifest = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["checkpoint_manifest"], "checkpoint manifest"
        )
    )
    checkpoint_path = require_pipeline_artifact(out_dir / CHECKPOINT_FILENAME, "RGCN checkpoint")
    checkpoint_hash = hash_without({"checkpoint_bytes": checkpoint_path.read_bytes().hex()})
    if checkpoint_manifest.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("checkpoint_hash does not match rgcn_checkpoint.pt")
    expected_encoder_hash = hash_without(
        checkpoint_manifest, "encoder_manifest_hash", "checkpoint_path"
    )
    if checkpoint_manifest.get("encoder_manifest_hash") != expected_encoder_hash:
        raise ValueError("encoder_manifest_hash is not reproducible")
    source_tensor_hashes = [
        _phase_a_compatible_tensor(tensor)["tensor_hash"]
        for tensor in tensors
    ]
    if checkpoint_manifest.get("source_tensor_hashes") != source_tensor_hashes:
        raise ValueError("checkpoint manifest source_tensor_hashes do not match tensor bundle")

    embedding_table = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["embedding_table"], "embedding table")
    )
    validate_phase_b_embedding_table(embedding_table)
    expected_graph_hashes = [tensor["input_graph_hash"] for tensor in tensors]
    expected_tensor_hashes = [tensor["tensor_hash"] for tensor in tensors]
    expected_invocation_ids = [tensor["kernel_invocation_id"] for tensor in tensors]
    embedding_rows = embedding_table.get("embeddings", [])
    if len(embedding_rows) != len(tensors):
        raise ValueError("embedding table row count does not match tensor batch")
    if [row["source_graph_hash"] for row in embedding_rows] != expected_graph_hashes:
        raise ValueError("embedding table source_graph_hash coverage mismatch")
    if [row["source_tensor_hash"] for row in embedding_rows] != expected_tensor_hashes:
        raise ValueError("embedding table source_tensor_hash coverage mismatch")
    if [row["kernel_invocation_id"] for row in embedding_rows] != expected_invocation_ids:
        raise ValueError("embedding table kernel_invocation_id coverage mismatch")
    for row in embedding_rows:
        if row["embedding_hash"] != hash_without(row, "embedding_hash"):
            raise ValueError("embedding row embedding_hash is not reproducible")
    if embedding_table.get("encoder_manifest_hash") != checkpoint_manifest["encoder_manifest_hash"]:
        raise ValueError("embedding table encoder_manifest_hash mismatch")
    if pipeline_manifest["hashes"].get("encoder_manifest_hash") != checkpoint_manifest["encoder_manifest_hash"]:
        raise ValueError("pipeline manifest encoder_manifest_hash mismatch")
    if pipeline_manifest["hashes"].get("embedding_table_hash") != _embedding_table_hash(embedding_table):
        raise ValueError("pipeline manifest embedding_table_hash mismatch")

    selector_artifacts = read_json(
        require_pipeline_artifact(
            out_dir / ARTIFACT_FILENAMES["selector_artifacts"], "selector artifacts"
        )
    )
    if selector_artifacts.get("source_embedding_table_hash") != _embedding_table_hash(embedding_table):
        raise ValueError("selector source_embedding_table_hash mismatch")
    if selector_artifacts.get("selector_manifest_hash") != hash_without(
        selector_artifacts, "selector_manifest_hash"
    ):
        raise ValueError("selector_manifest_hash is not reproducible")
    if pipeline_manifest["hashes"].get("selector_manifest_hash") != selector_artifacts["selector_manifest_hash"]:
        raise ValueError("pipeline manifest selector_manifest_hash mismatch")
    _validate_selector_artifacts_cover_embedding_table(selector_artifacts, embedding_table)

    readout_bundle = read_json(
        require_pipeline_artifact(out_dir / ARTIFACT_FILENAMES["readout_manifest"], "readout manifest")
    )
    readout_manifests = readout_bundle.get("manifests", [])
    if len(readout_manifests) != len(tensors):
        raise ValueError("readout manifest count mismatch")
    for manifest, tensor in zip(readout_manifests, tensors):
        validate_readout_manifest(manifest, tensor)
    readout_hashes = [manifest["readout_manifest_hash"] for manifest in readout_manifests]
    if [row["readout_manifest_hash"] for row in embedding_rows] != readout_hashes:
        raise ValueError("embedding table readout_manifest_hash coverage mismatch")
    if [row["kernel_embedding_hash"] for row in embedding_rows] != [
        manifest["kernel"]["kernel_embedding_hash"]
        for manifest in readout_manifests
    ]:
        raise ValueError("embedding table kernel_embedding_hash coverage mismatch")
    if pipeline_manifest["hashes"].get("readout_manifest_hashes") != readout_hashes:
        raise ValueError("pipeline manifest readout_manifest_hashes mismatch")
    if readout_bundle.get("readout_manifest_bundle_hash") != hash_without(
        readout_bundle, "readout_manifest_bundle_hash"
    ):
        raise ValueError("readout_manifest_bundle_hash is not reproducible")
    if pipeline_manifest["hashes"].get("readout_manifest_bundle_hash") != readout_bundle["readout_manifest_bundle_hash"]:
        raise ValueError("pipeline manifest readout_manifest_bundle_hash mismatch")

    return {
        "artifact_type": "gcl_phase_b_replay_validation",
        "checkpoint_hash": checkpoint_hash,
        "encoder_manifest_hash": checkpoint_manifest["encoder_manifest_hash"],
        "embedding_table_hash": _embedding_table_hash(embedding_table),
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
