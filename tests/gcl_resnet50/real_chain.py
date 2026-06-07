from pathlib import Path

from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records
from experiments.gcl_phase_b.utils import hash_without


FORMAL_ROOT = Path("artifacts/gcl_resnet50_gate0_formal_trace/traces")


def build_real_adapter_bundle():
    return build_resnet50_trace_adapter_bundle(FORMAL_ROOT)


def build_real_adapter_bundle_subset(limit: int = 1):
    """Build a small formal replay slice from the real Gate0 root for regression tests."""
    bundle = build_real_adapter_bundle()
    kept_invocations = bundle["kernel_invocation_table"][:limit]
    kept_ids = {row["kernel_invocation_id"] for row in kept_invocations}
    subset = dict(bundle)
    subset["kernel_invocation_table"] = kept_invocations
    subset["cta_scheduler_records"] = [
        record
        for record in bundle["cta_scheduler_records"]
        if record["kernel_invocation_id"] in kept_ids
    ]
    subset["per_warp_trace_records"] = [
        record
        for record in bundle["per_warp_trace_records"]
        if record["kernel_invocation_id"] in kept_ids
    ]
    subset["adapter_bundle_hash"] = hash_without(subset, "adapter_bundle_hash")
    return subset


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
