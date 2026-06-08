"""Minimal relation-aware GCN encoder used by GCL Phase A."""

from __future__ import annotations

from typing import Any
from collections import defaultdict

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - exercised only on missing dependency.
    torch = None
    nn = None
    F = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def require_torch():
    if torch is None:
        raise RuntimeError(f"GCL Phase A RGCN requires torch: {TORCH_IMPORT_ERROR}")
    return torch


if torch is not None:

    class RGCNLayer(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, relation_count: int, dropout: float):
            super().__init__()
            self.self_linear = nn.Linear(input_dim, output_dim)
            self.relation_linears = nn.ModuleList(
                nn.Linear(input_dim, output_dim, bias=False) for _ in range(relation_count)
            )
            self.norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)
            self.use_dropout = dropout > 0.0

        def forward(self, node_features, edge_index, edge_type):
            output = self.self_linear(node_features)
            if edge_index.numel() > 0:
                source = edge_index[0]
                target = edge_index[1]
                for relation_id, relation_linear in enumerate(self.relation_linears):
                    relation_mask = edge_type == relation_id
                    if relation_mask.any():
                        messages = relation_linear(node_features[source[relation_mask]])
                        output.index_add_(0, target[relation_mask], messages)
            output = self.norm(output)
            output = F.relu(output)
            if self.use_dropout:
                output = self.dropout(output)
            return output


    class MinimalRGCNEncoder(nn.Module):
        def __init__(
            self,
            input_dim: int = 64,
            hidden_dim: int = 128,
            output_dim: int = 256,
            relation_count: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.relation_count = relation_count
            self.layers = nn.ModuleList(
                [
                    RGCNLayer(input_dim, hidden_dim, relation_count, dropout),
                    RGCNLayer(hidden_dim, hidden_dim, relation_count, dropout),
                    RGCNLayer(hidden_dim, output_dim, relation_count, 0.0),
                ]
            )

        def forward(self, node_features, edge_index, edge_type):
            x = node_features
            for layer in self.layers:
                x = layer(x, edge_index, edge_type)
            return x

        def readout(self, node_embeddings, warp_partitions: dict[str, list[int]]):
            warp_embeddings = []
            for _, indices in sorted(warp_partitions.items()):
                if not indices:
                    raise ValueError("warp partition must not be empty")
                index_tensor = torch.tensor(indices, dtype=torch.long, device=node_embeddings.device)
                warp_embeddings.append(node_embeddings.index_select(0, index_tensor).mean(dim=0))
            if not warp_embeddings:
                raise ValueError("at least one warp partition is required")
            return torch.stack(warp_embeddings, dim=0).mean(dim=0)

        def encode_kernel(self, tensor_artifact: dict[str, Any]):
            node_features = torch.as_tensor(
                tensor_artifact["node_features"], dtype=torch.float32
            )
            edge_index = torch.as_tensor(tensor_artifact["edge_index"], dtype=torch.long)
            edge_type = torch.as_tensor(tensor_artifact["edge_type"], dtype=torch.long)
            node_embeddings = self.forward(node_features, edge_index, edge_type)
            return self.readout(node_embeddings, tensor_artifact["warp_partitions"])

        def encode_kernel_partitioned(self, tensor_artifact: dict[str, Any]):
            node_features = torch.as_tensor(
                tensor_artifact["node_features"], dtype=torch.float32
            )
            edge_index = torch.as_tensor(tensor_artifact["edge_index"], dtype=torch.long)
            edge_type = torch.as_tensor(tensor_artifact["edge_type"], dtype=torch.long)
            partition_embeddings_by_cta = defaultdict(list)
            for partition_id, indices in sorted(tensor_artifact["warp_partitions"].items()):
                if not indices:
                    raise ValueError("warp partition must not be empty")
                partition_tensor = tensor_artifact.get("warp_partition_tensors", {}).get(partition_id)
                index_tensor = torch.tensor(indices, dtype=torch.long, device=node_features.device)
                local_features = node_features.index_select(0, index_tensor)
                local_edge_index, local_edge_type = self._partition_edges(
                    edge_index,
                    edge_type,
                    indices,
                    partition_tensor,
                )
                local_node_embeddings = self.forward(local_features, local_edge_index, local_edge_type)
                cta_id = partition_tensor.get("cta_id") if partition_tensor else partition_id
                partition_embeddings_by_cta[cta_id].append(local_node_embeddings.mean(dim=0))
            if not partition_embeddings_by_cta:
                raise ValueError("at least one warp partition is required")
            cta_embeddings = []
            for cta_id in sorted(partition_embeddings_by_cta):
                cta_embeddings.append(torch.stack(partition_embeddings_by_cta[cta_id], dim=0).mean(dim=0))
            return torch.stack(cta_embeddings, dim=0).mean(dim=0)

        def _partition_edges(self, edge_index, edge_type, indices, partition_tensor):
            if edge_index.numel() == 0:
                return edge_index, edge_type
            old_to_new = {int(old): new for new, old in enumerate(indices)}
            if (
                partition_tensor is not None
                and "edge_indices" in partition_tensor
                and partition_tensor["edge_indices"]
                and max(partition_tensor["edge_indices"]) < edge_index.shape[1]
            ):
                selected_edge_indices = torch.tensor(
                    partition_tensor["edge_indices"],
                    dtype=torch.long,
                    device=edge_index.device,
                )
                selected_edges = edge_index.index_select(1, selected_edge_indices)
                selected_types = edge_type.index_select(0, selected_edge_indices)
            else:
                selected_edges = edge_index
                selected_types = edge_type
            local_edges = []
            local_types = []
            for column, relation in zip(selected_edges.T, selected_types):
                source = int(column[0])
                target = int(column[1])
                if source in old_to_new and target in old_to_new:
                    local_edges.append([old_to_new[source], old_to_new[target]])
                    local_types.append(int(relation))
            if not local_edges:
                return (
                    torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
                    torch.empty((0,), dtype=torch.long, device=edge_type.device),
                )
            return (
                torch.tensor(local_edges, dtype=torch.long, device=edge_index.device).T,
                torch.tensor(local_types, dtype=torch.long, device=edge_type.device),
            )


    class ProjectionHead(nn.Module):
        def __init__(self, input_dim: int = 256, hidden_dim: int = 128, output_dim: int = 64):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, kernel_embeddings):
            return self.layers(kernel_embeddings)
else:

    class MinimalRGCNEncoder:  # pragma: no cover - exercised in torch-less subprocess tests.
        def __init__(self, *args, **kwargs):
            require_torch()


    class ProjectionHead:  # pragma: no cover - exercised in torch-less subprocess tests.
        def __init__(self, *args, **kwargs):
            require_torch()


def model_config() -> dict[str, int]:
    return {
        "input_dim": 64,
        "hidden_dim": 128,
        "kernel_embedding_dim": 256,
        "projection_hidden_dim": 128,
        "projection_output_dim": 64,
        "relation_count": 3,
        "layers": 3,
    }
