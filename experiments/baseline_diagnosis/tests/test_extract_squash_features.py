"""Test Squash mechanism."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SCRIPT = ROOT / "mechanisms" / "extract_squash_features.py"
INPUT = ROOT / "results" / "rodinia" / "backprop_mechanisms" / "backprop_4096_per_tb.json"
CONFIG = ROOT / "schemas" / "mechanism_config.json"


def run_squash(output_path):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(INPUT),
         "--config", str(CONFIG), "--output", str(output_path)],
        check=True,
    )
    return json.loads(output_path.read_text())


def test_output_has_two_levels(tmp_path):
    out = tmp_path / "squash.json"
    result = run_squash(out)
    assert result["mechanism"] == "squash"
    assert "kernel_level" in result
    assert "tb_level" in result


def test_kernel_level_has_segments(tmp_path):
    out = tmp_path / "squash.json"
    result = run_squash(out)
    kl = result["kernel_level"]
    assert "squash_segments" in kl
    assert "boundary_count" in kl
    assert "total_kernels" in kl
    assert kl["total_kernels"] == 2


def test_backprop_kernel_level_finds_fp32_fp64_boundary(tmp_path):
    out = tmp_path / "squash.json"
    result = run_squash(out)
    kl = result["kernel_level"]
    assert kl["boundary_count"] >= 1, (
        f"Expected >= 1 boundary between FP32 forward and FP64 adjust_weights, "
        f"got {kl['boundary_count']}"
    )


def test_tb_level_has_entry_per_kernel(tmp_path):
    out = tmp_path / "squash.json"
    result = run_squash(out)
    tl = result["tb_level"]
    assert "1" in tl or 1 in tl
    assert "2" in tl or 2 in tl


def test_reuse_hint_present(tmp_path):
    out = tmp_path / "squash.json"
    result = run_squash(out)
    assert "_simulation_reuse_hint" in result
