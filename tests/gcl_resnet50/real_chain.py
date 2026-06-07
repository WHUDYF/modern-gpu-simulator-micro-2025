from pathlib import Path

from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate7
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records
from experiments.gcl_phase_b.utils import read_json


FORMAL_ROOT = Path("artifacts/gcl_resnet50_gate0_formal_trace/traces")
NONDEGENERATE_INVOCATION_IDS = [
    "d_0_s_0_k_267",
    "d_0_s_0_k_276",
    "d_0_s_0_k_291",
]


def build_real_adapter_bundle():
    return build_resnet50_trace_adapter_bundle(FORMAL_ROOT)


def build_real_adapter_bundle_subset(limit: int = 1):
    """Build a small formal replay slice from the real Gate0 root for regression tests."""
    return build_resnet50_trace_adapter_bundle(FORMAL_ROOT, invocation_limit=limit)


def build_real_adapter_bundle_invocations(invocation_ids=None):
    return build_resnet50_trace_adapter_bundle(
        FORMAL_ROOT,
        invocation_ids=list(invocation_ids or NONDEGENERATE_INVOCATION_IDS),
    )


def build_real_trace_manifest(limit: int = 1):
    bundle = build_real_adapter_bundle_subset(limit=limit)
    manifest, reports, preview = build_representative_sm_manifest_from_bundle(bundle)
    return bundle, manifest, reports, preview


def build_real_graphs(limit: int = 1):
    _bundle, manifest, reports, preview = build_real_trace_manifest(limit=limit)
    records = build_phase_b_trace_records(manifest)
    return manifest, reports, preview, build_phase_b_graphs(records)


def build_real_tensors(limit: int = 1):
    manifest, reports, preview, graphs = build_real_graphs(limit=limit)
    return manifest, reports, preview, graphs, tensorize_phase_b_graphs(graphs)


def run_real_gate1_to_gate7_artifacts(out_dir, limit: int = 1, seed: int = 20260607):
    manifest = run_resnet50_gate1_to_gate7(
        FORMAL_ROOT,
        out_dir,
        seed=seed,
        invocation_limit=limit,
    )
    return {
        "artifact_root": out_dir,
        "pipeline_manifest": manifest,
        "embedding_table": read_json(out_dir / "kernel_embedding_table.json"),
        "selector_artifacts": read_json(out_dir / "selector_artifacts.json"),
        "correctness_manifest": read_json(out_dir / "gate7_cluster_correctness_manifest.json"),
    }


def run_real_nondegenerate_gate1_to_gate7_artifacts(out_dir, seed: int = 20260607):
    manifest = run_resnet50_gate1_to_gate7(
        FORMAL_ROOT,
        out_dir,
        seed=seed,
        invocation_ids=NONDEGENERATE_INVOCATION_IDS,
    )
    return {
        "artifact_root": out_dir,
        "pipeline_manifest": manifest,
        "embedding_table": read_json(out_dir / "kernel_embedding_table.json"),
        "selector_artifacts": read_json(out_dir / "selector_artifacts.json"),
        "correctness_manifest": read_json(out_dir / "gate7_cluster_correctness_manifest.json"),
    }
