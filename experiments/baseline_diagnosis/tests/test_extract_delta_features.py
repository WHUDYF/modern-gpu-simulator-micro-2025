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
