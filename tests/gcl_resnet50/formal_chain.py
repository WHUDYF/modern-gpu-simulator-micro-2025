from pathlib import Path

from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.resnet50_adapter import build_resnet50_artifact_shape_trace_adapter_bundle
from experiments.gcl_phase_b.resnet50_manifest import build_representative_sm_manifest_from_bundle
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records

from gcl_resnet50.formal_fixture import write_minimal_artifact_shape_resnet50_root


def build_artifact_shape_adapter_bundle(tmp_path: Path):
    root = write_minimal_artifact_shape_resnet50_root(
        tmp_path / "artifact_shape_trace",
    )
    return build_resnet50_artifact_shape_trace_adapter_bundle(root)


def build_artifact_shape_trace_manifest(tmp_path: Path):
    bundle = build_artifact_shape_adapter_bundle(tmp_path)
    manifest, reports, preview = build_representative_sm_manifest_from_bundle(bundle)
    return bundle, manifest, reports, preview


def build_artifact_shape_graphs(tmp_path: Path):
    _bundle, manifest, _reports, _preview = build_artifact_shape_trace_manifest(tmp_path)
    records = build_phase_b_trace_records(manifest)
    return manifest, build_phase_b_graphs(records)


def build_artifact_shape_tensors(tmp_path: Path):
    _manifest, graphs = build_artifact_shape_graphs(tmp_path)
    return tensorize_phase_b_graphs(graphs)
