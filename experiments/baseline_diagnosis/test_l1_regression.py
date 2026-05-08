"""L1 regression tests - isolated, using temp artifact roots for synthetic tests."""

from __future__ import annotations

import json, math, os, re, sys, tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

failures = 0

def check(name, cond):
    global failures
    if not cond:
        failures += 1
        print(f"  FAIL: {name}")
        raise AssertionError(name)
    else:
        print(f"  PASS: {name}")

def _tmpdir():
    return Path(tempfile.mkdtemp())


# ══════════════════════════════════════════════════════════════════
# Manifest tests
# ══════════════════════════════════════════════════════════════════

def test_manifest():
    from build_l1_manifest import _validate_manifest, _check_paths_and_structure, REPO_ROOT as MROOT, MANIFEST_DOC, _parse_markdown_table, _build_entries
    schema = json.loads((REPO_ROOT / "experiments/baseline_diagnosis/schemas/kernel_validation_manifest_schema.json").read_text())

    # Live manifest should parse
    rows = _parse_markdown_table(MANIFEST_DOC.read_text())
    check("markdown parses 17 rows", len(rows) == 17)

    # Duplicate ID
    dup_manifest = {"manifest_name": "test", "dataset_level": "L1",
                     "goal": "test", "entries": [
                         {"id": "L1_X", "source_type": "local_microbench", "benchmark_name": "b",
                          "kernel_or_case": "k", "priority": "P0", "target_line": "A+B",
                          "expected_behavior_axis": "a", "status": "ready_local"},
                         {"id": "L1_X", "source_type": "local_microbench", "benchmark_name": "b",
                          "kernel_or_case": "k", "priority": "P0", "target_line": "A+B",
                          "expected_behavior_axis": "a", "status": "ready_local"},
                     ]}
    errs = _validate_manifest(dup_manifest, schema)
    check("duplicate ID rejected", any("duplicate" in e for e in errs))

    # Broken path
    bad = {"manifest_name": "t", "dataset_level": "L1", "goal": "t", "entries": [
        {"id": "L1_X", "source_type": "local_microbench", "benchmark_name": "b",
         "kernel_or_case": "k", "local_input_path": "nonexistent/path.json",
         "priority": "P0", "target_line": "A+B", "expected_behavior_axis": "a", "status": "ready_local"},
    ]}
    errs2 = _check_paths_and_structure(bad["entries"], MROOT)
    check("broken path rejected", any("does not exist" in e for e in errs2))


# ══════════════════════════════════════════════════════════════════
# Feature extractor tests
# ══════════════════════════════════════════════════════════════════

def test_extractor():
    from pka_feature_extractor import (
        _extract_pka_features, _is_fully_measured, _collect_missing,
        _make_record, _validate_outcomes, ADAPTERS, PKA_FEATURES, _match_kernel_name,
    )

    # Complete measured fixture
    full = {}
    for name, spec in PKA_FEATURES.items():
        full[spec["canonical_metric"]] = 1.0 if name != "divergence_efficiency" else 0.95
    feats = _extract_pka_features(full, "test.json")
    check("fully measured with 12 canonical", _is_fully_measured(feats))
    check("num_thread_blocks from launch_grid_size",
          feats["num_thread_blocks"]["value"] == 1.0 and feats["num_thread_blocks"]["status"] == "measured")

    # Partial
    feats2 = _extract_pka_features({}, "test.json")
    check("empty -> 12 missing", not _is_fully_measured(feats2) and len(_collect_missing(feats2)) == 12)
    check("missing_reason populated", all("missing_reason" in feats2[n] for n in feats2))

    feats3 = _extract_pka_features(
        {PKA_FEATURES["coalesced_global_loads"]["canonical_metric"]: 100.0}, "test.json")
    check("partial: 11 missing", len(_collect_missing(feats3)) == 11)

    # Deterministic
    f1 = _extract_pka_features(full, "a.json")
    f2 = _extract_pka_features(full, "a.json")
    check("deterministic", all(f1[n]["value"] == f2[n]["value"] for n in PKA_FEATURES))

    # make_record
    entry = {"id": "L1_MB_X", "priority": "P0", "source_type": "local_microbench", "kernel_or_case": "test"}
    rec = _make_record(entry, "test#1", "kernel_test", feats)
    check("measured record has manifest_id", rec["manifest_id"] == "L1_MB_X")
    check("measured record outcome", rec["outcome"] == "measured")
    rec2 = _make_record(entry, "test#2", "kernel_test", feats2)
    check("gap record outcome", rec2["outcome"] == "acquisition_gap")

    # Outcome validation
    errs = _validate_outcomes([rec, rec2], {"entries": [entry]})
    check("P0 outcomes ok for 1 entry", len(errs) == 0)

    # Missing P0
    errs2 = _validate_outcomes([], {"entries": [entry]})
    check("missing P0 outcome detected", any("zero outcomes" in e for e in errs2))

    # Duplicate within entry
    rec3 = _make_record(entry, "test#1", "kernel_test", feats)
    errs3 = _validate_outcomes([rec, rec3], {"entries": [entry]})
    check("dup within entry detected", any("duplicate" in e for e in errs3))

    # Global duplicate across P0 entries
    entry2 = {"id": "L1_MB_Y", "priority": "P0", "source_type": "local_microbench", "kernel_or_case": "test2"}
    rec4 = _make_record(entry2, "test#1", "kernel_test2", feats)
    errs4 = _validate_outcomes([rec, rec4], {"entries": [entry, entry2]})
    check("global dup detected", any("across P0" in e for e in errs4))

    # Adapter map
    check("adapters complete", all(k in ADAPTERS for k in ["local_microbench", "local_benchmark_result", "local_ai_workload"]))

    # Kernel name matching
    check("mangled name match", _match_kernel_name("_Z10gemm_tiledPKfS0_Pfiii", "gemm_tiled"))
    check("mangled name mismatch", not _match_kernel_name("_Z10gemm_tiledPKfS0_Pfiii", "softmax_kernel"))


# ══════════════════════════════════════════════════════════════════
# Selector tests with temp artifacts
# ══════════════════════════════════════════════════════════════════

def test_selector():
    from pka_baseline_selector import (
        _validate_allowlist, _build_matrix, ALLOWED_FEATURES, FORBIDDEN,
        _standardize, _pca_reduce, _kmeans, COUNT_FEATURES,
    )

    # Exact allowlist
    check("clean allowlist passes", len(_validate_allowlist(ALLOWED_FEATURES)) == 0)
    dup = ALLOWED_FEATURES + [ALLOWED_FEATURES[0]]
    check("13-field allowlist rejected", any("length" in e for e in _validate_allowlist(dup)))
    check("duplicate in allowlist rejected", any("Duplicate" in e for e in _validate_allowlist(dup)))
    check("kernel_name rejected",
          any("Forbidden" in e or "position" in e for e in _validate_allowlist(ALLOWED_FEATURES[:11] + ["kernel_name"])))
    check("grid_dim/block_dim rejected",
          any(e for e in _validate_allowlist(ALLOWED_FEATURES[:10] + ["grid_dim", ALLOWED_FEATURES[0]]) if "position" in e or "Forbidden" in e))

    # Build synthetic records
    names = ALLOWED_FEATURES
    records = []
    for i in range(5):
        feats = {}
        for n in names:
            val = 10.0 + i * 5.0 if n != "divergence_efficiency" else 0.9 + i * 0.02
            feats[n] = {"value": val, "status": "measured",
                         "canonical_metric": f"canon_{n}", "actual_source_metric": f"canon_{n}",
                         "source_artifact_path": f"synthetic/test_{i}.json"}
        records.append({"manifest_id": f"L1_T_{i}", "priority": "P0",
                        "kernel_invocation_id": f"test_{i}#1", "kernel_name": f"kernel_{i}",
                        "features": feats, "feature_mode": "pka_l1_measured_only"})

    matrix, meta, _ = _build_matrix(records)
    check("matrix 5x12", len(matrix) == 5 and len(matrix[0]) == 12)

    import math
    for i in range(len(matrix)):
        for j, fn in enumerate(names):
            if fn in COUNT_FEATURES:
                matrix[i][j] = math.log1p(matrix[i][j])

    std, means, stds, zv = _standardize(matrix)
    check("standardize output", len(std) == 5 and len(std[0]) == 12)

    reduced, comps, explained, total = _pca_reduce(std, min(3, 4, 12))
    check("PCA produced components", len(comps) > 0)
    check("total explained variance > 0", total > 0)

    k = min(3, 5)
    assignments = _kmeans(reduced, k, seed=42)
    check("k-means: all assigned", len(assignments) == 5)
    check("k-means: at least 1 cluster", len(set(assignments)) >= 1)
    # Deterministic re-run
    assignments2 = _kmeans(reduced, k, seed=42)
    check("k-means deterministic", assignments == assignments2)

    # Test with temp artifacts
    import subprocess
    tmp = _tmpdir()
    os.makedirs(tmp, exist_ok=True)
    with open(tmp / "pka_feature_table_l1.json", "w") as f:
        json.dump(records, f)
    with open(tmp / "l1_stage_gate_report_l1.json", "w") as f:
        json.dump({"stages": {"stage_3_selector": "ready", "stage_4_b_line_consumption": "pending"},
                   "run_status": "full_closure_success"}, f)

    # Run selector against temp dir
    import pka_baseline_selector as sel_mod
    old_ft = sel_mod.FT_PATH
    old_sg = sel_mod.SG_PATH
    sel_mod.FT_PATH = tmp / "pka_feature_table_l1.json"
    sel_mod.SG_PATH = tmp / "l1_stage_gate_report_l1.json"
    sel_mod.ARTIFACT_DIR = tmp
    exitcode = sel_mod.main()
    sel_mod.FT_PATH = old_ft
    sel_mod.SG_PATH = old_sg
    sel_mod.ARTIFACT_DIR = REPO_ROOT / "artifacts" / "a_line" / "l1"

    check("selector exits 0 with synthetic", exitcode == 0)
    check("selector config", (tmp / "pka_selector_config_l1.json").exists())
    check("dim report", (tmp / "pka_dimensionality_reduction_report_l1.json").exists())
    check("reduced table", (tmp / "pka_reduced_feature_table_l1.json").exists())
    check("cluster assignment", (tmp / "pka_cluster_assignment_l1.json").exists())
    check("anchor table", (tmp / "representative_anchor_table_l1.json").exists())

    # Rejection test: invalid feature row should fail matrix construction
    invalid_records = list(records)
    bad_rec = dict(invalid_records[0])
    bad_feats = dict(bad_rec["features"])
    del bad_feats["coalesced_global_loads"]
    bad_rec["features"] = bad_feats
    invalid_records[0] = bad_rec
    try:
        _build_matrix(invalid_records)
        check("selector rejects invalid feature row", False)
    except ValueError as e:
        check("selector rejects invalid feature row", "invalid" in str(e).lower())

    # feature_mode presence
    check("measured record has feature_mode", records[0].get("feature_mode") == "pka_l1_measured_only")


# ══════════════════════════════════════════════════════════════════
# B-line tests
# ══════════════════════════════════════════════════════════════════

def test_bline():
    from b_line_consumer_l1 import _validate, _report, REQUIRED, FORBIDDEN, TYPE_CHECKS

    valid_row = {"rep_kernel_id": "r1", "kernel_name": "k", "cluster_id": "c1",
                 "member_invocations": ["k#1"], "coverage_count": 1,
                 "coverage_weight": 0.5, "time_weight": 0.3}
    results = _validate([valid_row])
    check("valid row passes", results[0]["required_fields_present"] and results[0]["forbidden_fields_absent"] and not results[0].get("type_errors"))

    bad_row = {"rep_kernel_id": "r1"}
    results2 = _validate([bad_row])
    check("bad row fails required", not results2[0]["required_fields_present"])

    leaked_row = {**valid_row, "family_id": "F1", "execution_template": "compute"}
    results3 = _validate([leaked_row])
    check("leaked forbidden", not results3[0]["forbidden_fields_absent"])

    comp_row = {**valid_row, "cross_tb_offset_coverage": 0.5, "dominant_format": "compressed"}
    results4 = _validate([comp_row])
    check("compression fields rejected", not results4[0]["forbidden_fields_absent"])
    check("compression fields named", len(results4[0]["leaked_forbidden_fields"]) >= 2)

    type_row = {**valid_row, "coverage_count": "1", "member_invocations": "not-a-list"}
    results5 = _validate([type_row])
    check("type errors detected", len(results5[0].get("type_errors", [])) >= 2)

    # Null required fields
    null_row = {**valid_row, "rep_kernel_id": None, "coverage_weight": None}
    results6 = _validate([null_row])
    check("null required fields rejected", not results6[0]["required_fields_present"])
    check("null fields in missing list", "rep_kernel_id" in results6[0]["missing_required_fields"])

    report = _report(results)
    check("report has per-row results", "rep_kernel_id" in report)
    check("report has ALL_PASS or ISSUES_FOUND", "ALL_PASS" in report or "ISSUES_FOUND" in report)


# ══════════════════════════════════════════════════════════════════
# Stage gate tests
# ══════════════════════════════════════════════════════════════════

def test_stagegate():
    from build_l1_audit import _stage_gate

    manifest = {
        "entries": [
            {"id": "L1_MEASURED", "priority": "P0"},
            {"id": "L1_GAP", "priority": "P0"},
        ]
    }
    sg = _stage_gate(
        [
            {
                "manifest_id": "L1_MEASURED",
                "kernel_invocation_id": "measured#1",
                "kernel_name": "measured",
                "outcome": "measured",
                "timing_basis": "duration_ns",
            },
            {
                "manifest_id": "L1_GAP",
                "kernel_invocation_id": "gap#1",
                "kernel_name": "gap",
                "outcome": "acquisition_gap",
                "source_path": "synthetic.json",
                "missing_metrics": ["coalesced_global_loads"],
            },
        ],
        manifest,
    )
    check("valid run status", sg["run_status"] in {
        "acquisition_gate_success", "full_closure_success",
        "selector_insufficient_records", "weight_unit_conflict",
        "validation_failed"})

    # When acquisition_gate_success, no downstream artifacts should exist
    if sg["run_status"] == "acquisition_gate_success":
        check("stage 3 blocked", sg["stages"]["stage_3_selector"] == "blocked")
        check("stage 4 blocked", sg["stages"]["stage_4_b_line_consumption"] == "blocked")

    # provenance rejection: empty actual_source_metric detected
    from pka_feature_extractor import _extract_pka_features, PKA_FEATURES, _make_record
    prov_rec = _make_record(
        {"id": "L1_PRV", "priority": "P0", "source_type": "local_microbench", "kernel_or_case": "test"},
        "test#1", "k", _extract_pka_features(
            {s["canonical_metric"]: 1.0 for s in PKA_FEATURES.values()}, "test.json"))
    prov_rec["features"]["coalesced_global_loads"]["actual_source_metric"] = ""
    from build_l1_audit import _audit_json
    audit_p = _audit_json([prov_rec])
    pve_p = audit_p.get("provenance_validation_errors", [])
    check("audit detects empty provenance", any("actual_source_metric" in e for e in pve_p))


# ══════════════════════════════════════════════════════════════════
# M1 measured-loop tests
# ══════════════════════════════════════════════════════════════════

def test_m1_measured_loop_blocked_terminal_state():
    import m1_measured_feature_extractor as gate3
    import m1_ncu_capture_dispatcher as gate2
    import m1_selector_eligibility as gate4
    import m1_workload_resolver as gate1
    import run_m1_measured_loop as loop

    tmp = _tmpdir()
    manifest_path = tmp / "kernel_validation_manifest_l1.json"
    registry_path = tmp / "workload_registry_l1.json"
    manifest_path.write_text(json.dumps({"entries": [
        {"id": "L1_M1_A", "priority": "P0", "benchmark_name": "ok", "kernel_or_case": "kernel_a", "source_type": "local_microbench"},
        {"id": "L1_M1_B", "priority": "P0", "benchmark_name": "ok", "kernel_or_case": "kernel_b", "source_type": "local_microbench"},
    ]}))
    registry_path.write_text(json.dumps([
        {
            "workload_id": "ok",
            "binary_path": sys.executable,
            "build_command": None,
            "run_args": [],
            "run_command_template": [sys.executable, "-c", "print('ok')"],
            "working_directory": str(tmp),
            "smoke_timeout_seconds": 5,
            "capture_timeout_seconds": 5,
        }
    ]))

    old_values = {
        (gate1, "MANIFEST_PATH"): gate1.MANIFEST_PATH,
        (gate1, "REGISTRY_PATH"): gate1.REGISTRY_PATH,
        (gate1, "RESOLUTION_PATH"): gate1.RESOLUTION_PATH,
        (gate1, "GAP_PATH"): gate1.GAP_PATH,
        (gate1, "SMOKE_DIR"): gate1.SMOKE_DIR,
        (gate2, "RESOLUTION_PATH"): gate2.RESOLUTION_PATH,
        (gate2, "ATTEMPTS_PATH"): gate2.ATTEMPTS_PATH,
        (gate2, "GAP_PATH"): gate2.GAP_PATH,
        (gate2, "QUERY_PATH"): gate2.QUERY_PATH,
        (gate2, "RESOLUTION_TABLE_PATH"): gate2.RESOLUTION_TABLE_PATH,
        (gate2, "RESULTS_DIR"): gate2.RESULTS_DIR,
        (gate3, "ATTEMPTS_PATH"): gate3.ATTEMPTS_PATH,
        (gate3, "FEATURE_TABLE_PATH"): gate3.FEATURE_TABLE_PATH,
        (gate3, "GAP_PATH"): gate3.GAP_PATH,
        (gate3, "FEATURE_AUDIT_PATH"): gate3.FEATURE_AUDIT_PATH,
        (gate3, "JOIN_AUDIT_PATH"): gate3.JOIN_AUDIT_PATH,
        (gate4, "MANIFEST_PATH"): gate4.MANIFEST_PATH,
        (gate4, "RESOLUTION_GAP_PATH"): gate4.RESOLUTION_GAP_PATH,
        (gate4, "CAPTURE_GAP_PATH"): gate4.CAPTURE_GAP_PATH,
        (gate4, "FEATURE_TABLE_PATH"): gate4.FEATURE_TABLE_PATH,
        (gate4, "ACQ_GAP_PATH"): gate4.ACQ_GAP_PATH,
        (gate4, "ELIGIBILITY_PATH"): gate4.ELIGIBILITY_PATH,
        (gate4, "SELECTOR_INPUT_PATH"): gate4.SELECTOR_INPUT_PATH,
        (gate4, "REPAIR_JSON_PATH"): gate4.REPAIR_JSON_PATH,
        (gate4, "REPAIR_MD_PATH"): gate4.REPAIR_MD_PATH,
        (loop, "STATUS_PATH"): loop.STATUS_PATH,
    }
    try:
        gate1.MANIFEST_PATH = manifest_path
        gate1.REGISTRY_PATH = registry_path
        gate1.RESOLUTION_PATH = tmp / "m1_workload_resolution_l1.json"
        gate1.GAP_PATH = tmp / "m1_workload_resolution_gap_l1.json"
        gate1.SMOKE_DIR = tmp / "smoke"

        gate2.RESOLUTION_PATH = gate1.RESOLUTION_PATH
        gate2.ATTEMPTS_PATH = tmp / "m1_ncu_capture_attempts_l1.json"
        gate2.GAP_PATH = tmp / "m1_ncu_capture_gap_l1.json"
        gate2.QUERY_PATH = tmp / "ncu_metric_query_l1.json"
        gate2.RESOLUTION_TABLE_PATH = tmp / "ncu_metric_resolution_table_l1.json"
        gate2.RESULTS_DIR = tmp / "ncu"

        gate3.ATTEMPTS_PATH = gate2.ATTEMPTS_PATH
        gate3.FEATURE_TABLE_PATH = tmp / "pka_feature_table_l1.json"
        gate3.GAP_PATH = tmp / "pka_acquisition_gap_l1.json"
        gate3.FEATURE_AUDIT_PATH = tmp / "pka_feature_audit_l1.json"
        gate3.JOIN_AUDIT_PATH = tmp / "pka_join_audit_l1.json"

        gate4.MANIFEST_PATH = manifest_path
        gate4.RESOLUTION_GAP_PATH = gate1.GAP_PATH
        gate4.CAPTURE_GAP_PATH = gate2.GAP_PATH
        gate4.FEATURE_TABLE_PATH = gate3.FEATURE_TABLE_PATH
        gate4.ACQ_GAP_PATH = gate3.GAP_PATH
        gate4.ELIGIBILITY_PATH = tmp / "m1_selector_eligibility_l1.json"
        gate4.SELECTOR_INPUT_PATH = tmp / "m1_selector_input_l1.json"
        gate4.REPAIR_JSON_PATH = tmp / "m1_backward_repair_report_l1.json"
        gate4.REPAIR_MD_PATH = tmp / "m1_backward_repair_report_l1.md"
        loop.STATUS_PATH = tmp / "m1_measured_loop_status_l1.json"

        report = loop.run(dry_run_capture=True, dry_run_smoke=True)
    finally:
        for (module, name), value in old_values.items():
            setattr(module, name, value)

    check("M1 dry-run terminal state is blocked with repair",
          report["status"] == "blocked_on_acquisition_with_repair_report")
    check("M1 Gate 4 blocks insufficient measured rows",
          report["gate4"]["selector_eligibility_state"] == "selector_blocked_insufficient_measured_records")
    check("M1 repair report exists", (tmp / "m1_backward_repair_report_l1.json").exists())
    repair = json.loads((tmp / "m1_backward_repair_report_l1.json").read_text())
    check("M1 repair report covers all P0 entries", len(repair["entries"]) == 2)
    check("M1 dry-run did not write Gate 5 anchors", not (tmp / "representative_anchor_table_l1.json").exists())


def test_m1_completion_classifier():
    import run_m1_measured_loop as loop

    check("M1 classifier allows blocked with complete repair",
          loop.classify_completion({
              "gate5_allowed": False,
              "measured_rows": 2,
              "backward_repair_report_exists": True,
              "per_entry_earliest_gate_complete": True,
          }) == "blocked_on_acquisition_with_repair_report")
    check("M1 classifier rejects missing repair report",
          loop.classify_completion({
              "gate5_allowed": False,
              "measured_rows": 2,
              "backward_repair_report_exists": False,
              "per_entry_earliest_gate_complete": False,
          }) == "stop_fail_missing_backward_repair_report")
    check("M1 classifier rejects invalid feature table",
          loop.classify_completion({
              "selector_eligibility_state": "selector_blocked_invalid_feature_table",
              "gate5_allowed": False,
          }) == "stop_fail_invalid_feature_table")


# ══════════════════════════════════════════════════════════════════

def run_all():
    global failures
    failures = 0
    tests = [
        ("manifest", test_manifest),
        ("extractor", test_extractor),
        ("selector", test_selector),
        ("b-line", test_bline),
        ("stage-gate", test_stagegate),
        ("m1-measured-loop", test_m1_measured_loop_blocked_terminal_state),
        ("m1-completion", test_m1_completion_classifier),
    ]
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as exc:
            failures += 1
            import traceback
            traceback.print_exc()
    print(f"\n{'='*60}")
    print(f"RESULTS: {failures} failures")
    return failures


if __name__ == "__main__":
    sys.exit(run_all())
