import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SCRIPT = ROOT / "build_invocation_table.py"
PIPELINE = ROOT / "build_frontend_anchor_outputs.py"
FULL_JSON = ROOT.parent / "mini_transformer" / "mini_transformer_v4_full.json"
SQUASH_JSON = ROOT.parent / "mini_transformer" / "mechanisms" / "squash.json"


def run_builder(tmp_path, *extra):
    out = tmp_path / "invocation_table.json"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--full-json",
        str(FULL_JSON),
        "--output",
        str(out),
        *extra,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc, out


def test_builder_writes_records(tmp_path):
    proc, out = run_builder(tmp_path)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    assert payload["source_mode"] == "full_json_shortcut"
    assert len(payload["records"]) == 14
    first = payload["records"][0]
    assert first["kernel_invocation_id"] == "_Z10gemm_tiledPKfS0_Pfiii#1"
    assert first["trace_order"] == 1
    assert "feature_vector" in first


def test_builder_can_attach_squash_context(tmp_path):
    proc, out = run_builder(tmp_path, "--squash-json", str(SQUASH_JSON))
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    first = payload["records"][0]
    assert first["kernel_squash_segment_id"] == 0
    assert "tb_squash_segment_count" in first


def test_builder_rejects_missing_input(tmp_path):
    out = tmp_path / "invocation_table.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--full-json",
            str(tmp_path / "missing.json"),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "full_json not found" in proc.stderr


def test_frontend_pipeline_writes_anchor_outputs(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE),
            "--full-json",
            str(FULL_JSON),
            "--squash-json",
            str(SQUASH_JSON),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    anchor_table = json.loads((tmp_path / "representative_anchor_table_v1.json").read_text())
    methods = json.loads((tmp_path / "comparison_table_v1.json").read_text())
    case_note = (tmp_path / "case_note_v1.md").read_text()

    assert anchor_table
    assert {"rep_kernel_id", "kernel_name", "cluster_id", "member_invocations", "coverage_weight", "time_weight"} <= set(anchor_table[0].keys())
    assert any(row["method"] == "hybrid" for row in methods)
    assert "Representative split cases" in case_note
