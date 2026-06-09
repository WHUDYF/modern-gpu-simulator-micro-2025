import pytest
import subprocess
import sys

import numpy as np

from experiments.gcl_phase_a.graph_builder import build_canonical_graphs
from experiments.gcl_phase_a.tensorizer import tensorize_graphs
from experiments.gcl_phase_a.trace_fixture import build_controlled_trace_fixture
from experiments.gcl_phase_a.train import augment_tensor, train_minimal_contrastive, validate_training_inputs


def _tensors():
    fixture = build_controlled_trace_fixture()
    return tensorize_graphs(build_canonical_graphs(fixture))


def test_minimal_training_smoke(tmp_path):
    report = train_minimal_contrastive(_tensors(), tmp_path)

    assert report["loss"] > 0.0
    assert report["optimizer_step_count"] == 1
    assert report["kernel_embedding_shape"] == [12, 256]
    assert report["projection_output_shape"] == [12, 64]
    assert report["encoder_batch_size"] > 0
    assert report["checkpoint_manifest"]["model_config"]["input_dim"] == 64
    assert report["checkpoint_manifest"]["model_config"]["hidden_dim"] == 128
    assert report["checkpoint_manifest"]["model_config"]["kernel_embedding_dim"] == 256
    assert report["checkpoint_manifest"]["model_config"]["projection_output_dim"] == 64
    assert (tmp_path / "rgcn_checkpoint.pt").exists()


def test_minimal_training_accepts_explicit_encoder_batch_size(tmp_path):
    report = train_minimal_contrastive(_tensors(), tmp_path, encoder_batch_size=3)

    assert report["encoder_batch_size"] == 3
    assert report["kernel_embedding_shape"] == [12, 256]
    assert report["projection_output_shape"] == [12, 64]
    assert report["checkpoint_manifest"]["encoder_batch_size"] == 3


def test_partitioned_encoder_returns_kernel_embedding_without_whole_graph_forward():
    from experiments.gcl_phase_a.rgcn import MinimalRGCNEncoder, require_torch

    torch = require_torch()
    tensor = _tensors()[0]
    encoder = MinimalRGCNEncoder()
    encoder.eval()

    with torch.no_grad():
        partitioned = encoder.encode_kernel_partitioned(tensor)

    assert list(partitioned.shape) == [256]
    assert torch.isfinite(partitioned).all()


def test_encoder_manifest_hash_changes_when_checkpoint_payload_changes(monkeypatch, tmp_path):
    import experiments.gcl_phase_a.train as train_module

    first = train_minimal_contrastive(_tensors(), tmp_path / "first", seed=1)
    torch = train_module.require_torch()
    original_save = torch.save

    def save_with_marker(payload, path):
        original_save(payload, path)
        with open(path, "ab") as handle:
            handle.write(b"\nmutated-checkpoint")

    monkeypatch.setattr(torch, "save", save_with_marker)

    second = train_minimal_contrastive(_tensors(), tmp_path / "second", seed=1)

    assert first["checkpoint_manifest"]["source_tensor_hashes"] == second["checkpoint_manifest"]["source_tensor_hashes"]
    assert first["checkpoint_manifest"]["seed"] == second["checkpoint_manifest"]["seed"]
    assert first["checkpoint_manifest"]["encoder_manifest_hash"] != second["checkpoint_manifest"]["encoder_manifest_hash"]


def test_phase_a_train_imports_without_torch_until_training_invocation():
    script = """
import builtins
original_import = builtins.__import__
def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError('No module named torch')
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = blocked_import
import experiments.gcl_phase_a.train as train
try:
    train.train_minimal_contrastive([], __import__('pathlib').Path('/tmp/noop'))
except RuntimeError as exc:
    assert 'requires torch' in str(exc)
else:
    raise AssertionError('expected RuntimeError')
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_training_rejects_wrong_feature_width():
    tensors = _tensors()
    tensors[0] = dict(tensors[0])
    tensors[0]["node_features"] = tensors[0]["node_features"][:, 0:63]

    with pytest.raises(ValueError, match="shape \\[node_count, 64\\]|feature width 64"):
        validate_training_inputs(tensors)


def test_augmentation_preserves_strict_zero_padding_columns():
    tensor = _tensors()[0]
    augmented, _ = augment_tensor(tensor, seed=7, node_drop_rate=0.0, edge_drop_rate=0.0)

    for index, node_type in enumerate(augmented["node_types"]):
        if node_type in {"register_version", "input_variable", "unknown_variable"}:
            assert (augmented["node_features"][index, 40:64] == 0.0).all()
        if node_type == "pseudo":
            assert (augmented["node_features"][index, 16:64] == 0.0).all()


def test_augmentation_rebuilds_warp_partition_tensors_after_node_and_edge_drops():
    tensor = {
        "node_features": np.zeros((4, 64), dtype=np.float32),
        "node_types": ["instruction", "register_version", "instruction", "pseudo"],
        "edge_index": np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        "edge_type": np.asarray([0, 1, 2], dtype=np.int64),
        "warp_partitions": {"cta_0:warp_0": [0, 1, 2, 3]},
        "warp_partition_tensors": {
            "cta_0:warp_0": {
                "partition_id": "cta_0:warp_0",
                "cta_id": "cta_0",
                "warp_id": 0,
                "node_indices": [0, 1, 2, 3],
                "edge_indices": [0, 1, 2],
                "instruction_count": 2,
            }
        },
    }

    augmented, _ = augment_tensor(
        tensor,
        seed=11,
        node_drop_rate=1.0,
        edge_drop_rate=0.0,
        noise_sigma=0.0,
    )

    partition = augmented["warp_partition_tensors"]["cta_0:warp_0"]
    assert partition["node_indices"] == augmented["warp_partitions"]["cta_0:warp_0"]
    assert partition["node_indices"] == [0, 1]
    assert partition["edge_indices"] == []


def test_training_rejects_empty_warp_partition():
    tensors = _tensors()
    tensors[0] = dict(tensors[0])
    tensors[0]["warp_partitions"] = dict(tensors[0]["warp_partitions"])
    first_warp = next(iter(tensors[0]["warp_partitions"]))
    tensors[0]["warp_partitions"][first_warp] = []

    with pytest.raises(ValueError, match="warp partition"):
        validate_training_inputs(tensors)


def test_training_records_retry_count_when_augmentation_regenerates(monkeypatch, tmp_path):
    import experiments.gcl_phase_a.train as train_module

    original_augment = train_module.augment_tensor
    calls = {"count": 0}

    def flaky_augment(tensor, seed, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("augmentation produced empty warp partition")
        return original_augment(tensor, seed, *args, **kwargs)

    monkeypatch.setattr(train_module, "augment_tensor", flaky_augment)

    report = train_module.train_minimal_contrastive(_tensors(), tmp_path)

    assert report["augmentation_retry_count"] >= 1
