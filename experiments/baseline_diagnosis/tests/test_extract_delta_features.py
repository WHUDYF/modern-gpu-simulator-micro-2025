"""Test Delta mechanism."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SCRIPT = ROOT / "mechanisms" / "extract_delta_features.py"
INPUT = ROOT / "results" / "rodinia" / "backprop_mechanisms" / "backprop_4096_per_tb.json"
CONFIG = ROOT / "schemas" / "mechanism_config.json"


def run_delta(output_path):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(INPUT),
         "--config", str(CONFIG), "--output", str(output_path)],
        check=True,
    )
    return json.loads(output_path.read_text())


def test_mechanism_field(tmp_path):
    out = tmp_path / "delta.json"
    result = run_delta(out)
    assert result["mechanism"] == "delta"


def test_two_levels(tmp_path):
    out = tmp_path / "delta.json"
    result = run_delta(out)
    assert "kernel_level" in result
    assert "tb_level" in result


def test_kernel_level_has_fields(tmp_path):
    out = tmp_path / "delta.json"
    result = run_delta(out)
    kl = result["kernel_level"]
    assert "field_temperature" in kl
    assert "hot_fields" in kl
    assert "cold_fields" in kl


def test_backprop_kernel_level_fp64_is_hot(tmp_path):
    """The uses_fp64 field should be HOT at kernel-level in backprop."""
    out = tmp_path / "delta.json"
    result = run_delta(out)
    kl = result["kernel_level"]
    hot = set(kl["hot_fields"])
    # uses_fp64 changes between forward and adjust_weights → should be hot
    assert "uses_fp64" in hot, (
        f"Expected uses_fp64 in hot_fields at kernel-level, got {hot}"
    )


def test_tb_level_per_kernel(tmp_path):
    out = tmp_path / "delta.json"
    result = run_delta(out)
    tl = result["tb_level"]
    assert len(tl) == 2
    for kid, data in tl.items():
        assert "field_temperature" in data
        assert "hot_fields" in data
        assert "cold_fields" in data


def test_no_spurious_correlations_on_constant_fields(tmp_path):
    """Regression test: ensure pairwise_correlation filters out fields
    whose std is just floating-point noise.

    Uses a synthetic per-TB file where all TBs have identical feature
    values. Delta should report 0 correlations and 0 outlier_diffs
    at TB-level in this case.
    """
    # Build a synthetic workload: 1 kernel with 100 identical TBs
    synthetic = {
        "workload": "synthetic_constant",
        "kernels": [
            {
                "kernel_id": 1,
                "kernel_name": "constant_kernel",
                "kernel_summary": {
                    "top_opcodes": [],
                    "total_static_instructions": 10,
                    "total_dynamic_instructions": 100,
                    "uses_fp64": False,
                    "uses_shared_memory": False,
                    "num_barriers": 0,
                    "grid_dim": "1x1x1",
                    "block_dim": "32x1x1",
                },
                "per_tb": [
                    {
                        "tb_index": i,
                        "features": {
                            "num_warps": 1.0,
                            "instructions_per_warp_mean": 133.6631130063966,
                            "opcode_ffma_ratio": 0.5,
                            "opcode_ldg_ratio": 0.3,
                            "address_override_count": 0,
                            "is_full_encoding": False,
                        },
                    }
                    for i in range(100)
                ],
            }
        ],
    }
    synthetic_path = tmp_path / "synthetic_per_tb.json"
    synthetic_path.write_text(json.dumps(synthetic))

    out = tmp_path / "delta_synthetic.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(synthetic_path),
            "--config",
            str(CONFIG),
            "--output",
            str(out),
        ],
        check=True,
    )
    result = json.loads(out.read_text())

    # All TBs are identical → no hot fields, no correlations, no outliers
    tb_level = result["tb_level"]["1"]
    assert tb_level["hot_fields"] == [], (
        f"Expected 0 hot fields on constant data, got {tb_level['hot_fields']}"
    )
    assert len(tb_level.get("field_correlations", [])) == 0, (
        f"Regression: spurious correlations on constant fields. "
        f"Found {len(tb_level['field_correlations'])}."
    )
    assert len(tb_level.get("outlier_diffs", [])) == 0, (
        f"Regression: spurious outlier diffs on constant fields. "
        f"Found {len(tb_level['outlier_diffs'])}."
    )
