"""Test per-TB feature extraction."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent

SCRIPT = ROOT / "mechanisms" / "extract_per_tb_features.py"
INPUT = ROOT / "results" / "rodinia" / "backprop_4096_full.json"
SCHEMA = ROOT / "schemas" / "per_tb_features_schema.json"


def run_extractor(output_path):
    """Call the extractor as a CLI and return parsed JSON."""
    subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(INPUT), "--output", str(output_path)],
        check=True,
    )
    return json.loads(output_path.read_text())


def test_output_has_workload_and_kernels(tmp_path):
    out = tmp_path / "out.json"
    result = run_extractor(out)
    assert "workload" in result
    assert "kernels" in result
    assert isinstance(result["kernels"], list)
    assert len(result["kernels"]) == 2


def test_each_kernel_has_required_fields(tmp_path):
    out = tmp_path / "out.json"
    result = run_extractor(out)
    for kernel in result["kernels"]:
        assert "kernel_id" in kernel
        assert "kernel_name" in kernel
        assert "per_tb" in kernel
        assert "kernel_summary" in kernel


def test_per_tb_entries_have_features(tmp_path):
    out = tmp_path / "out.json"
    result = run_extractor(out)
    for kernel in result["kernels"]:
        assert len(kernel["per_tb"]) > 0
        for tb in kernel["per_tb"]:
            assert "tb_index" in tb
            assert "features" in tb
            assert isinstance(tb["features"], dict)
            for field in ["num_warps", "instructions_per_warp_mean"]:
                assert field in tb["features"], f"Missing {field} in TB {tb['tb_index']}"


def test_kernel_summary_has_opcodes(tmp_path):
    out = tmp_path / "out.json"
    result = run_extractor(out)
    for kernel in result["kernels"]:
        summary = kernel["kernel_summary"]
        assert "top_opcodes" in summary
        assert "uses_fp64" in summary
        if "layerforward" in kernel["kernel_name"]:
            assert summary["uses_fp64"] is False
        elif "adjust_weights" in kernel["kernel_name"]:
            assert summary["uses_fp64"] is True
