"""Test Batch mechanism."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SCRIPT = ROOT / "mechanisms" / "extract_batch_features.py"
INPUT = ROOT / "results" / "rodinia" / "backprop_mechanisms" / "backprop_4096_per_tb.json"
CONFIG = ROOT / "schemas" / "mechanism_config.json"


def run_batch(output_path):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(INPUT),
         "--config", str(CONFIG), "--output", str(output_path)],
        check=True,
    )
    return json.loads(output_path.read_text())


def test_mechanism_field(tmp_path):
    out = tmp_path / "batch.json"
    result = run_batch(out)
    assert result["mechanism"] == "batch"


def test_two_levels_present(tmp_path):
    out = tmp_path / "batch.json"
    result = run_batch(out)
    assert "kernel_level" in result
    assert "tb_level" in result


def test_kernel_level_has_clusters(tmp_path):
    out = tmp_path / "batch.json"
    result = run_batch(out)
    kl = result["kernel_level"]
    assert "batch_clusters" in kl
    assert "outlier_kernels" in kl
    assert "homogeneity_score" in kl


def test_tb_level_per_kernel(tmp_path):
    out = tmp_path / "batch.json"
    result = run_batch(out)
    tl = result["tb_level"]
    assert len(tl) == 2


def test_backprop_tb_level_high_homogeneity(tmp_path):
    """backprop TBs in each kernel are highly similar - expect one big cluster."""
    out = tmp_path / "batch.json"
    result = run_batch(out)
    tl = result["tb_level"]
    for kid, data in tl.items():
        assert data["homogeneity_score"] >= 0.9, (
            f"Expected high homogeneity for backprop kernel {kid}, "
            f"got {data['homogeneity_score']}"
        )
