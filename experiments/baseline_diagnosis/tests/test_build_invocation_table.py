import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT = ROOT / "build_invocation_table.py"
PIPELINE = ROOT / "build_frontend_anchor_outputs.py"
FULL_JSON = ROOT.parent / "mini_transformer" / "mini_transformer_v4_full.json"
SQUASH_JSON = ROOT.parent / "mini_transformer" / "mechanisms" / "squash.json"

from experiments.baseline_diagnosis.frontend_anchor.selector import run_selector
from experiments.baseline_diagnosis.frontend_anchor.invocation_table import build_records_from_full_json


def run_builder(tmp_path, *extra):
    out = tmp_path / "invocation_table.json"
    cmd = [
        sys.executable,
        str(SCRIPT),
            "--identity-json",
            str(FULL_JSON),
            "--features-json",
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
    assert payload["source_mode"] == "explicit_dual_source"
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
            "--identity-json",
            str(tmp_path / "missing.json"),
            "--features-json",
            str(FULL_JSON),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "identity_json not found" in proc.stderr


def test_builder_rejects_unalignable_sources(tmp_path):
    bad_features = tmp_path / "bad_features.json"
    data = json.loads(FULL_JSON.read_text())
    first_key = next(iter(data["per_kernel"]))
    data["per_kernel"].pop(first_key)
    bad_features.write_text(json.dumps(data))

    out = tmp_path / "invocation_table.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--identity-json",
            str(FULL_JSON),
            "--features-json",
            str(bad_features),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "dual-source alignment failed" in proc.stderr


def test_frontend_pipeline_writes_anchor_outputs(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE),
            "--identity-json",
            str(FULL_JSON),
            "--features-json",
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
    assert anchor_table[0]["output_role"] == "mainline_anchor"
    assert any(row["method"] == "hybrid" for row in methods)
    assert all(row["output_role"] == "evidence_only" for row in methods)
    assert "Representative split cases" in case_note
    assert "evidence_only" in case_note


def test_selector_rejects_unknown_mode():
    records = build_records_from_full_json(FULL_JSON)["records"]

    try:
        run_selector(records, "unknown-mode")
    except ValueError as exc:
        assert "unknown selector mode" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown selector mode")
