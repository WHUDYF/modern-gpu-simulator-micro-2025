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
IDENTITY_JSON = ROOT.parent / "mini_transformer" / "frontend_anchor_sources" / "mini_transformer_v4_identity.json"
FEATURES_JSON = ROOT.parent / "mini_transformer" / "frontend_anchor_sources" / "mini_transformer_v4_features.json"
SQUASH_JSON = ROOT.parent / "mini_transformer" / "mechanisms" / "squash.json"

from experiments.baseline_diagnosis.frontend_anchor.selector import run_selector
from experiments.baseline_diagnosis.frontend_anchor.invocation_table import (
    build_records_from_dual_sources,
)
from experiments.baseline_diagnosis.frontend_anchor.exporter import export_anchor_table


def run_builder(tmp_path, *extra):
    out = tmp_path / "invocation_table.json"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--identity-json",
        str(IDENTITY_JSON),
        "--features-json",
        str(FEATURES_JSON),
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
            str(FEATURES_JSON),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "identity_json not found" in proc.stderr


def test_builder_rejects_missing_feature_source(tmp_path):
    out = tmp_path / "invocation_table.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--identity-json",
            str(IDENTITY_JSON),
            "--features-json",
            str(tmp_path / "missing_features.json"),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "features_json not found" in proc.stderr


def test_builder_rejects_unalignable_sources(tmp_path):
    bad_features = tmp_path / "bad_features.json"
    data = json.loads(FEATURES_JSON.read_text())
    data["feature_records"] = data["feature_records"][1:]
    bad_features.write_text(json.dumps(data))

    out = tmp_path / "invocation_table.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--identity-json",
            str(IDENTITY_JSON),
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
            str(IDENTITY_JSON),
            "--features-json",
            str(FEATURES_JSON),
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
    records = build_records_from_dual_sources(IDENTITY_JSON, FEATURES_JSON)["records"]

    try:
        run_selector(records, "unknown-mode")
    except ValueError as exc:
        assert "unknown selector mode" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown selector mode")


def test_hybrid_groups_stay_within_coarse_buckets():
    records = build_records_from_dual_sources(IDENTITY_JSON, FEATURES_JSON)["records"]
    coarse = run_selector(records, "pka-like-coarse")
    hybrid = run_selector(records, "hybrid")
    coarse_sets = {
        tuple(sorted(m["kernel_invocation_id"] for m in group["members"]))
        for group in coarse
    }
    for group in hybrid:
        member_ids = {m["kernel_invocation_id"] for m in group["members"]}
        assert any(member_ids <= set(coarse_members) for coarse_members in coarse_sets)


def test_exporter_rejects_forbidden_downstream_keys():
    groups = [
        {
            "method": "hybrid",
            "cluster_id": "hybrid-1",
            "anchor_record": {
                "kernel_name": "k",
                "trace_order": 1,
                "grid_dim": "1x1x1",
                "block_dim": "1x1x1",
            },
            "members": [{"kernel_invocation_id": "k#1", "trace_order": 1, "exec_time": 1.0}],
            "member_count": 1,
            "heterogeneity_flag": False,
            "squash_boundary_crossing_flag": False,
            "guardrail_note": None,
        }
    ]
    table = export_anchor_table(groups)
    table[0]["family_id"] = "f-1"
    from experiments.baseline_diagnosis.frontend_anchor import exporter as exporter_mod

    try:
        exporter_mod._validate_anchor_table(table)
    except ValueError as exc:
        assert "forbidden downstream keys" in str(exc)
    else:
        raise AssertionError("expected forbidden downstream key rejection")


def test_exporter_rejects_missing_required_fields():
    from experiments.baseline_diagnosis.frontend_anchor import exporter as exporter_mod

    table = [{"kernel_name": "k"}]
    try:
        exporter_mod._validate_anchor_table(table)
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("expected missing required field rejection")
