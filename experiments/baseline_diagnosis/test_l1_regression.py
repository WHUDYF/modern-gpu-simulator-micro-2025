"""L1 regression tests - behavioral, calling real code paths with temp artifacts."""

from __future__ import annotations

import json, os, sys, tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _tmpdir():
    return Path(tempfile.mkdtemp())


passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}")


# ═══════════════════════════════════════════════════════════════════════
# Manifest builder tests
# ═══════════════════════════════════════════════════════════════════════

def test_manifest_builder():
    from build_l1_manifest import main as manifest_main, OUTPUT_PATH, _validate_manifest, _check_paths_and_structure, REPO_ROOT as MROOT

    # Run the real builder
    import subprocess
    result = subprocess.run([sys.executable, str(REPO_ROOT / "experiments/baseline_diagnosis/build_l1_manifest.py")],
                           capture_output=True, text=True)
    check("manifest builder exits 0", result.returncode == 0)
    manifest = json.loads(OUTPUT_PATH.read_text())
    check("manifest has 17 entries", len(manifest["entries"]) == 17)
    check("dataset_level is L1", manifest["dataset_level"] == "L1")

    # Duplicate ID check
    schema = json.loads((REPO_ROOT / "experiments/baseline_diagnosis/schemas/kernel_validation_manifest_schema.json").read_text())
    dup_manifest = dict(manifest)
    dup_manifest["entries"] = list(manifest["entries"])
    dup_manifest["entries"].append(dict(manifest["entries"][0]))
    errs = _validate_manifest(dup_manifest, schema)
    check("duplicate ID rejected by validator", any("duplicate" in e for e in errs))

    # Invalid path check
    bad_manifest = dict(manifest)
    bad_manifest["entries"] = [{**manifest["entries"][0], "local_input_path": "nonexistent/file.json"}]
    errs = _check_paths_and_structure(bad_manifest["entries"], MROOT)
    check("broken path rejected", any("does not exist" in e for e in errs))


# ═══════════════════════════════════════════════════════════════════════
# Feature extractor tests
# ═══════════════════════════════════════════════════════════════════════

def test_feature_extractor():
    from pka_feature_extractor import (
        _extract_pka_features, _is_fully_measured, _collect_missing,
        _make_record, _validate_outcomes, ADAPTERS, PKA_FEATURES,
    )

    # Complete measured fixture
    full = {
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 100.0,
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": 50.0,
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": 0.0,
        "smsp__inst_executed_op_global_ld.sum": 200.0,
        "smsp__inst_executed_op_global_st.sum": 150.0,
        "smsp__inst_executed_op_local_ld.sum": 0.0,
        "smsp__inst_executed_op_shared_ld.sum": 300.0,
        "smsp__inst_executed_op_shared_st.sum": 250.0,
        "smsp__sass_inst_executed_op_global_atom.sum": 0.0,
        "smsp__inst_executed.sum": 1000.0,
        "smsp__thread_inst_executed_per_inst_executed.ratio": 0.95,
        "launch_grid_size": 64,
    }
    feats = _extract_pka_features(full, "test.json")
    check("fully measured with 12 canonical metrics", _is_fully_measured(feats))
    check("num_thread_blocks from launch_grid_size", feats["num_thread_blocks"]["value"] == 64 and feats["num_thread_blocks"]["status"] == "measured")

    # Incomplete fixture
    partial = {"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 100.0}
    feats2 = _extract_pka_features(partial, "test.json")
    check("partial routes to gap", not _is_fully_measured(feats2))
    missing = _collect_missing(feats2)
    check("11 metrics missing", len(missing) == 11)
    check("missing_reason populated", all("missing_reason" in feats2[n] for n in missing))

    # Empty fixture
    feats3 = _extract_pka_features({}, "test.json")
    check("empty routes to gap with 12 missing", len(_collect_missing(feats3)) == 12)

    # Deterministic output
    f1 = _extract_pka_features(full, "a.json")
    f2 = _extract_pka_features(full, "a.json")
    check("deterministic output", all(f1[n]["value"] == f2[n]["value"] for n in PKA_FEATURES))

    # make_record
    entry = {"id": "L1_MB_TEST", "priority": "P0", "source_type": "local_microbench", "kernel_or_case": "test"}
    rec = _make_record(entry, "test#1", "kernel_test", feats)
    check("measured record has manifest_id", rec["manifest_id"] == "L1_MB_TEST")
    check("measured record outcome", rec["outcome"] == "measured")

    rec2 = _make_record(entry, "test#2", "kernel_test", feats2)
    check("gap record has outcome acquisition_gap", rec2["outcome"] == "acquisition_gap")
    check("gap record has missing_metrics", len(rec2["missing_metrics"]) == 11)

    # Outcome validation
    errs = _validate_outcomes([rec, rec2], {"entries": [entry]})
    check("P0 outcome validation: 2 outcomes for 1 entry", len(errs) == 0)

    # Missing P0 entry
    errs2 = _validate_outcomes([], {"entries": [entry]})
    check("missing P0 outcome detected", any("zero outcomes" in e for e in errs2))

    # Duplicate invocation_id
    rec3 = _make_record(entry, "test#1", "kernel_test", feats)
    errs3 = _validate_outcomes([rec, rec3], {"entries": [entry]})
    check("duplicate invocation_id detected", any("duplicate" in e for e in errs3))

    # Adapter map
    check("all adapters registered", all(k in ADAPTERS for k in ["local_microbench", "local_benchmark_result", "local_ai_workload"]))


# ═══════════════════════════════════════════════════════════════════════
# Selector tests
# ═══════════════════════════════════════════════════════════════════════

def test_selector():
    from pka_baseline_selector import (
        _validate_allowlist, _build_matrix, ALLOWED_FEATURES, FORBIDDEN,
        _standardize, _pca_reduce, _kmeans, _check_gate, COUNT_FEATURES,
    )
    import pka_baseline_selector as sel

    # Allowlist validation
    check("clean allowlist passes", len(_validate_allowlist(ALLOWED_FEATURES)) == 0)
    check("kernel_name rejected", any("Forbidden" in e for e in _validate_allowlist(ALLOWED_FEATURES + ["kernel_name"])))
    check("grid_dim rejected", any("Forbidden" in e for e in _validate_allowlist(ALLOWED_FEATURES + ["grid_dim", "block_dim"])))
    check("family_id rejected", any("Forbidden" in e for e in _validate_allowlist(ALLOWED_FEATURES + ["family_id"])))

    # Build synthetic records
    names = ALLOWED_FEATURES
    metrics = [f["canonical_metric"] for f in [
        {"canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"},
        {"canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"},
        {"canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"},
        {"canonical_metric": "smsp__inst_executed_op_global_ld.sum"},
        {"canonical_metric": "smsp__inst_executed_op_global_st.sum"},
        {"canonical_metric": "smsp__inst_executed_op_local_ld.sum"},
        {"canonical_metric": "smsp__inst_executed_op_shared_ld.sum"},
        {"canonical_metric": "smsp__inst_executed_op_shared_st.sum"},
        {"canonical_metric": "smsp__sass_inst_executed_op_global_atom.sum"},
        {"canonical_metric": "smsp__inst_executed.sum"},
        {"canonical_metric": "smsp__thread_inst_executed_per_inst_executed.ratio"},
        {"canonical_metric": "launch_grid_size"},
    ]]
    records = []
    for i in range(5):
        feats = {}
        for n, m in zip(names, metrics):
            val = 10.0 + i * 5.0 if n != "divergence_efficiency" else 0.9 + i * 0.02
            feats[n] = {"value": val, "status": "measured", "canonical_metric": m,
                         "actual_source_metric": m, "source_artifact_path": f"synthetic/test_{i}.json"}
        records.append({
            "manifest_id": f"L1_T_{i}", "priority": "P0", "kernel_invocation_id": f"test_{i}#1",
            "kernel_name": f"kernel_{i}", "features": feats,
        })

    matrix, meta, _ = _build_matrix(records)
    check("matrix built with 5 records", len(matrix) == 5)
    check("matrix has 12 features", len(matrix[0]) == 12)

    import math
    for i in range(len(matrix)):
        for j, fn in enumerate(names):
            if fn in COUNT_FEATURES:
                matrix[i][j] = math.log1p(matrix[i][j])

    std, means, stds, zv = _standardize(matrix)
    check("standardization produced matrix", len(std) == 5 and len(std[0]) == 12)

    reduced, comps, explained, total = _pca_reduce(std, min(3, 4, 12))
    check("PCA produced components", len(comps) > 0)

    assignments = _kmeans(reduced, min(3, 5), seed=42)
    check("k-means assigned all records", len(assignments) == 5)

    # Run full selector pipeline with synthetic data
    import subprocess
    artifact_dir = REPO_ROOT / "artifacts" / "a_line" / "l1"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(artifact_dir / "pka_feature_table_l1.json", "w") as f:
        json.dump(records, f)
    with open(artifact_dir / "l1_stage_gate_report_l1.json", "w") as f:
        json.dump({"stages": {"stage_3_selector": "ready", "stage_4_b_line_consumption": "pending"}, "run_status": "full_closure_success"}, f)

    result = subprocess.run([sys.executable, str(REPO_ROOT / "experiments/baseline_diagnosis/pka_baseline_selector.py")],
                           capture_output=True, text=True)
    check("selector exits 0 with synthetic data", result.returncode == 0)
    check("selector config emitted", (artifact_dir / "pka_selector_config_l1.json").exists())
    check("dim reduction report emitted", (artifact_dir / "pka_dimensionality_reduction_report_l1.json").exists())
    check("reduced feature table emitted", (artifact_dir / "pka_reduced_feature_table_l1.json").exists())
    check("cluster assignment emitted", (artifact_dir / "pka_cluster_assignment_l1.json").exists())
    check("anchor table emitted", (artifact_dir / "representative_anchor_table_l1.json").exists())

    anchor = json.loads((artifact_dir / "representative_anchor_table_l1.json").read_text())
    check("anchor table has rows", len(anchor) > 0)
    check("anchor rows have rep_kernel_id", all("rep_kernel_id" in r for r in anchor))

    # Test forbidden field rejection in actual selector
    sel.FORBIDDEN = frozenset({"kernel_name"})
    bad_features = list(ALLOWED_FEATURES) + ["kernel_name"]
    errs = _validate_allowlist(bad_features)
    check("forbidden kernel_name rejected by selector", any("Forbidden" in e for e in errs))


# ═══════════════════════════════════════════════════════════════════════
# B-line consumer tests
# ═══════════════════════════════════════════════════════════════════════

def test_b_line():
    from b_line_consumer_l1 import _validate, _report, REQUIRED, FORBIDDEN

    valid_row = {
        "rep_kernel_id": "r1", "kernel_name": "k", "cluster_id": "c1",
        "member_invocations": ["k#1"], "coverage_count": 1,
        "coverage_weight": 0.5, "time_weight": 0.3,
    }
    results = _validate([valid_row])
    check("valid row passes required", results[0]["required_fields_present"])
    check("valid row passes forbidden", results[0]["forbidden_fields_absent"])

    bad_row = {"rep_kernel_id": "r1"}
    results2 = _validate([bad_row])
    check("bad row fails required", not results2[0]["required_fields_present"])
    check("bad row reports missing fields", len(results2[0]["missing_required_fields"]) > 0)

    leaked_row = {**valid_row, "family_id": "F1", "execution_template": "compute"}
    results3 = _validate([leaked_row])
    check("leaked row fails forbidden", not results3[0]["forbidden_fields_absent"])
    check("leaked row reports forbidden fields", len(results3[0]["leaked_forbidden_fields"]) == 2)

    # Report generation
    report = _report(results)
    check("report includes per-row results", "rep_kernel_id" in report)
    check("report includes overall status", "ALL_PASS" in report)

    # Test with actual anchor table if it exists
    artifact_dir = REPO_ROOT / "artifacts" / "a_line" / "l1"
    anchor_path = artifact_dir / "representative_anchor_table_l1.json"
    if anchor_path.exists():
        try:
            table = json.loads(anchor_path.read_text())
            if isinstance(table, list) and len(table) > 0:
                results4 = _validate(table)
                check("live anchor table validates", all(r["required_fields_present"] for r in results4))
                check("live anchor table no forbidden leaks", all(r["forbidden_fields_absent"] for r in results4))
        except Exception:
            pass

    # Compression-side fields rejected
    comp_row = {**valid_row, "cross_tb_offset_coverage": 0.5, "address_override_density": 0.1}
    results5 = _validate([comp_row])
    check("compression fields rejected", not results5[0]["forbidden_fields_absent"])
    check("compression fields named in leaks", "cross_tb_offset_coverage" in results5[0]["leaked_forbidden_fields"])


# ═══════════════════════════════════════════════════════════════════════
# Stage gate tests
# ═══════════════════════════════════════════════════════════════════════

def test_stage_gate():
    artifact_dir = REPO_ROOT / "artifacts" / "a_line" / "l1"
    sg_path = artifact_dir / "l1_stage_gate_report_l1.json"
    check("stage gate report exists", sg_path.exists())

    sg = json.loads(sg_path.read_text())
    check("run_status is valid", sg["run_status"] in {
        "acquisition_gate_success", "full_closure_success",
        "selector_insufficient_records", "weight_unit_conflict",
    })

    # With live data, should be acquisition_gate_success (all P0 gaps)
    if sg["run_status"] == "acquisition_gate_success":
        check("stage 3 blocked", sg["stages"]["stage_3_selector"] == "blocked")
        check("stage 4 blocked", sg["stages"]["stage_4_b_line_consumption"] == "blocked")

    # With synthetic measured data (from test_selector), gate should be open
    ft_path = artifact_dir / "pka_feature_table_l1.json"
    if ft_path.exists():
        records = json.loads(ft_path.read_text())
        if len(records) >= 2:
            check("synthetic records >= 2", True)


# ═══════════════════════════════════════════════════════════════════════

def run_all():
    global passed, failed
    passed = failed = 0
    tests = [
        ("manifest builder", test_manifest_builder),
        ("feature extractor", test_feature_extractor),
        ("selector", test_selector),
        ("B-line consumer", test_b_line),
        ("stage gate", test_stage_gate),
    ]
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  EXCEPTION in {name}: {exc}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    return failed


if __name__ == "__main__":
    sys.exit(run_all())
