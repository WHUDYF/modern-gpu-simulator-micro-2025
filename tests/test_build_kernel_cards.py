from pathlib import Path
import sys

REPO_ROOT = Path("/home/dyf/modern-gpu-simulator-micro-2025")
sys.path.insert(0, str(REPO_ROOT))

from experiments.baseline_diagnosis.build_kernel_cards import (
    default_kernel_names,
    load_sources,
)


def test_load_sources_contains_expected_reports():
    sources = load_sources(REPO_ROOT)

    assert "E0_baseline" in sources
    assert "E1_squash" in sources
    assert "E2_batch" in sources
    assert "E4_full" in sources
    assert "E5_stageC_validation" in sources
    assert "baseline_ape" in sources


def test_load_sources_paths_exist_on_disk():
    sources = load_sources(REPO_ROOT)

    for path in sources.values():
        assert path.exists(), f"Missing canonical evidence path: {path}"


def test_default_kernel_names_are_the_expected_representatives():
    assert default_kernel_names() == [
        "gemm_tiled",
        "attention_score",
        "residual_add",
        "softmax_kernel",
        "context_mul",
        "layernorm_kernel",
    ]
