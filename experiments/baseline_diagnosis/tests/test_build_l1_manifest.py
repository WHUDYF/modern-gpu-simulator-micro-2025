import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = REPO_ROOT / "experiments" / "baseline_diagnosis"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))


from build_l1_manifest import _build_entries  # noqa: E402


def test_build_entries_normalizes_old_checkout_absolute_links():
    rows = [
        {
            "id": "L1_JSON",
            "来源": "ai workload",
            "对象": "gemm_tiled",
            "本地路径 / 来源路径": (
                "[mini_transformer_v4_full.json]"
                "(/home/dyf/modern-gpu-simulator-micro-2025/experiments/mini_transformer/mini_transformer_v4_full.json)"
            ),
            "优先级": "P0",
            "面向线路": "A+B",
            "预期行为轴": "dense",
            "当前状态": "ready_local",
        }
    ]

    entries = _build_entries(rows)

    assert entries[0]["local_input_path"] == "experiments/mini_transformer/mini_transformer_v4_full.json"
