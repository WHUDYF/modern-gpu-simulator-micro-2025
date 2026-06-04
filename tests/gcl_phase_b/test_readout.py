import pytest

from experiments.gcl_phase_a.rgcn import MinimalRGCNEncoder, require_torch
from experiments.gcl_phase_b.graph_builder import build_phase_b_graphs
from experiments.gcl_phase_b.readout import build_readout_manifest, validate_readout_manifest
from experiments.gcl_phase_b.tensorizer import tensorize_phase_b_graphs
from experiments.gcl_phase_b.trace_fixture import build_representative_sm_trace_manifest
from experiments.gcl_phase_b.trace_scope import build_phase_b_trace_records


def _tensor():
    records = build_phase_b_trace_records(build_representative_sm_trace_manifest())
    graph = build_phase_b_graphs(records)[0]
    return tensorize_phase_b_graphs([graph])[0]


def test_hierarchical_readout_pools_nodes_to_warps_to_kernel():
    torch = require_torch()
    tensor = _tensor()
    encoder = MinimalRGCNEncoder()
    node_features = torch.as_tensor(tensor["node_features"], dtype=torch.float32)
    edge_index = torch.as_tensor(tensor["edge_index"], dtype=torch.long)
    edge_type = torch.as_tensor(tensor["edge_type"], dtype=torch.long)
    node_embeddings = encoder(node_features, edge_index, edge_type)

    manifest, kernel_embedding = build_readout_manifest(tensor, node_embeddings)

    validate_readout_manifest(manifest, tensor)
    assert manifest["kernel"]["pooling_method"] == "average"
    assert manifest["kernel"]["kernel_embedding_dim"] == 256
    assert kernel_embedding.shape[0] == 256
    assert all(row["pooling_method"] == "mean" for row in manifest["warps"])


def test_readout_rejects_empty_warp_partition():
    torch = require_torch()
    tensor = _tensor()
    first_key = next(iter(tensor["warp_partitions"]))
    tensor["warp_partitions"][first_key] = []
    node_embeddings = torch.zeros((tensor["node_features"].shape[0], 256))

    with pytest.raises(ValueError, match="warp partition"):
        build_readout_manifest(tensor, node_embeddings)
