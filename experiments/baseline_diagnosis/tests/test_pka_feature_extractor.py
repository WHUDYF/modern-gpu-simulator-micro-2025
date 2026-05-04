import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = REPO_ROOT / "experiments" / "baseline_diagnosis"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))


from pka_feature_extractor import PKA_FEATURES, _adapt_mini_transformer  # noqa: E402


def _measured_metrics(value: float = 1.0) -> dict[str, float]:
    return {
        spec["canonical_metric"]: value
        for spec in PKA_FEATURES.values()
    }


def test_mini_transformer_adapter_preserves_json_launch_order(tmp_path):
    source = tmp_path / "mini_transformer_full.json"
    source.write_text(
        json.dumps(
            {
                "per_kernel": {
                    "kernel_1": {
                        "kernel_name": "gemm_launch_1",
                        "hardware_metrics": {"duration_ns": 10.0},
                        "compression_features": _measured_metrics(1.0),
                    },
                    "kernel_2": {
                        "kernel_name": "gemm_launch_2",
                        "hardware_metrics": {"duration_ns": 20.0},
                        "compression_features": _measured_metrics(2.0),
                    },
                    "kernel_11": {
                        "kernel_name": "gemm_launch_11",
                        "hardware_metrics": {"duration_ns": 30.0},
                        "compression_features": _measured_metrics(3.0),
                    },
                }
            }
        )
    )
    entry = {"id": "L1_JSON", "priority": "P0", "source_type": "local_ai_workload", "kernel_or_case": "gemm"}

    records = _adapt_mini_transformer(entry, source)

    assert [record["kernel_name"] for record in records] == ["gemm_launch_1", "gemm_launch_2", "gemm_launch_11"]
    assert [record["trace_order"] for record in records] == [0, 1, 2]


def test_mini_transformer_adapter_preserves_timing_provenance(tmp_path):
    source = tmp_path / "mini_transformer_full.json"
    source.write_text(
        json.dumps(
            {
                "per_kernel": {
                    "kernel_1": {
                        "kernel_name": "gemm_launch_1",
                        "hardware_metrics": {"elapsed_cycles": 12345},
                        "compression_features": _measured_metrics(1.0),
                    }
                }
            }
        )
    )
    entry = {"id": "L1_JSON", "priority": "P0", "source_type": "local_ai_workload", "kernel_or_case": "gemm"}

    record = _adapt_mini_transformer(entry, source)[0]

    assert record["outcome"] == "measured"
    assert record["timing_basis"] == "elapsed_cycles"
    assert record["timing_value"] == 12345.0
