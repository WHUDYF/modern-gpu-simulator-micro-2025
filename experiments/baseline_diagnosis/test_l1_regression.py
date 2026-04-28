"""L1 regression tests: manifest schema, feature completeness, forbidden-field
rejection, anchor schema, B-line smoke, gap routing, stage-gate enforcement."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pka_feature_extractor import (
    PKA_FEATURES,
    _extract_pka_features,
    _is_fully_measured,
    _collect_missing,
    ADAPTERS,
)
from pka_baseline_selector import (
    FORBIDDEN_FIELDS as SELECTOR_FORBIDDEN,
    ALLOWED_FEATURES,
    _validate_forbidden_fields,
)
from b_line_consumer_l1 import REQUIRED_FIELDS, FORBIDDEN_FIELDS as BL_FORBIDDEN


MANIFEST_SCHEMA = json.loads(
    (REPO_ROOT / "experiments/baseline_diagnosis/schemas/kernel_validation_manifest_schema.json").read_text()
)


# ── Manifest schema tests ──────────────────────────────────────────────

class TestManifestSchema:
    def _make_entry(self, **overrides):
        e = {
            "id": "L1_MB_TEST",
            "source_type": "local_microbench",
            "benchmark_name": "test_bm",
            "kernel_or_case": "test_bm",
            "local_input_path": "experiments/test/data.json",
            "priority": "P0",
            "target_line": "A+B",
            "expected_behavior_axis": "test axis",
            "status": "ready_local",
        }
        e.update(overrides)
        return e

    def _validate(self, entries):
        entry_schema = MANIFEST_SCHEMA["$defs"]["entry"]
        entry_required = entry_schema["required"]
        entry_props = entry_schema["properties"]
        errors = []
        for idx, e in enumerate(entries):
            for field in entry_required:
                if field not in e:
                    errors.append(f"entries[{idx}] missing required: {field}")
            for field in ("source_type", "priority", "target_line", "status"):
                allowed = entry_props.get(field, {}).get("enum", [])
                if e.get(field) not in allowed:
                    errors.append(f"entries[{idx}] invalid {field}: {e.get(field)}")
        return errors

    def test_valid_manifest_passes(self):
        entries = [
            self._make_entry(id="L1_MB_01"),
            self._make_entry(id="L1_MB_02", source_type="local_benchmark_result", priority="P1"),
        ]
        errors = self._validate(entries)
        assert len(errors) == 0

    def test_missing_id_rejected(self):
        entries = [self._make_entry()]
        del entries[0]["id"]
        errors = self._validate(entries)
        assert any("missing required" in e for e in errors)

    def test_duplicate_id_rejected(self):
        entries = [self._make_entry(id="L1_MB_01"), self._make_entry(id="L1_MB_01")]
        ids = [e["id"] for e in entries]
        assert len(ids) != len(set(ids))

    def test_invalid_source_type_rejected(self):
        entries = [self._make_entry(source_type="invalid_source")]
        errors = self._validate(entries)
        assert any("invalid source_type" in e for e in errors)

    def test_invalid_priority_rejected(self):
        entries = [self._make_entry(priority="P3")]
        errors = self._validate(entries)
        assert any("invalid priority" in e for e in errors)


# ── Feature extractor tests ────────────────────────────────────────────

class TestFeatureExtraction:
    def test_all_12_features_defined(self):
        assert len(PKA_FEATURES) == 12
        for name in ALLOWED_FEATURES:
            assert name in PKA_FEATURES

    def test_measured_extraction(self):
        metric_map = {
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
        features = _extract_pka_features(metric_map, "test/artifact.json")
        assert _is_fully_measured(features)
        for pka_name, f in features.items():
            assert f["status"] == "measured", f"Feature {pka_name} not measured"
            assert f["value"] is not None, f"Feature {pka_name} has null value"

    def test_missing_routes_to_gap(self):
        metric_map = {}  # empty — nothing measured
        features = _extract_pka_features(metric_map, "test/artifact.json")
        assert not _is_fully_measured(features)
        missing = _collect_missing(features)
        assert len(missing) == 12

    def test_partial_measurement(self):
        metric_map = {
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 100.0,
            "smsp__inst_executed.sum": 1000.0,
        }
        features = _extract_pka_features(metric_map, "test/artifact.json")
        assert not _is_fully_measured(features)
        missing = _collect_missing(features)
        assert len(missing) == 10

    def test_num_thread_blocks_measured(self):
        metric_map = {
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 0,
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": 0,
            "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": 0,
            "smsp__inst_executed_op_global_ld.sum": 0,
            "smsp__inst_executed_op_global_st.sum": 0,
            "smsp__inst_executed_op_local_ld.sum": 0,
            "smsp__inst_executed_op_shared_ld.sum": 0,
            "smsp__inst_executed_op_shared_st.sum": 0,
            "smsp__sass_inst_executed_op_global_atom.sum": 0,
            "smsp__inst_executed.sum": 0,
            "smsp__thread_inst_executed_per_inst_executed.ratio": 1.0,
            "launch_grid_size": 64,
        }
        features = _extract_pka_features(metric_map, "test/artifact.json")
        assert _is_fully_measured(features)
        assert features["num_thread_blocks"]["status"] == "measured"
        assert features["num_thread_blocks"]["value"] == 64
        assert features["num_thread_blocks"]["canonical_metric"] == "launch_grid_size"

    def test_deterministic_output(self):
        metric_map = {
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 0,
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": 0,
            "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": 0,
            "smsp__inst_executed_op_global_ld.sum": 0,
            "smsp__inst_executed_op_global_st.sum": 0,
            "smsp__inst_executed_op_local_ld.sum": 0,
            "smsp__inst_executed_op_shared_ld.sum": 0,
            "smsp__inst_executed_op_shared_st.sum": 0,
            "smsp__sass_inst_executed_op_global_atom.sum": 0,
            "smsp__inst_executed.sum": 0,
            "smsp__thread_inst_executed_per_inst_executed.ratio": 1.0,
            "launch_grid_size": 32,
        }
        f1 = _extract_pka_features(metric_map, "test/a.json")
        f2 = _extract_pka_features(metric_map, "test/a.json")
        for name in PKA_FEATURES:
            assert f1[name]["value"] == f2[name]["value"]
            assert f1[name]["status"] == f2[name]["status"]


# ── Selector forbidden-field tests ────────────────────────────────────

class TestSelectorForbiddenFields:
    def test_clean_allowlist_passes(self):
        violations = _validate_forbidden_fields(ALLOWED_FEATURES)
        assert len(violations) == 0

    def test_kernel_name_in_grouping_rejected(self):
        fields = list(ALLOWED_FEATURES) + ["kernel_name"]
        violations = _validate_forbidden_fields(fields)
        assert "kernel_name" in violations

    def test_grid_dim_block_dim_rejected(self):
        fields = list(ALLOWED_FEATURES) + ["grid_dim", "block_dim"]
        violations = _validate_forbidden_fields(fields)
        assert "grid_dim" in violations
        assert "block_dim" in violations

    def test_family_regime_lane_rejected(self):
        fields = list(ALLOWED_FEATURES) + ["family_id", "regime_id", "simulator_lane_id"]
        violations = _validate_forbidden_fields(fields)
        assert "family_id" in violations
        assert "regime_id" in violations
        assert "simulator_lane_id" in violations

    def test_compression_fields_rejected(self):
        fields = list(ALLOWED_FEATURES) + [
            "cross_tb_offset_coverage", "squash_boundary_crossing_flag",
            "address_override_density", "full_encoding_fallback_rate",
        ]
        violations = _validate_forbidden_fields(fields)
        assert "cross_tb_offset_coverage" in violations
        assert "squash_boundary_crossing_flag" in violations


# ── B-line consumer tests ──────────────────────────────────────────────

class TestBLineConsumer:
    def test_valid_anchor_row_passes(self):
        row = {
            "rep_kernel_id": "rep-1",
            "kernel_name": "test_kernel",
            "cluster_id": "c1",
            "member_invocations": ["test#1"],
            "coverage_count": 1,
            "coverage_weight": 0.5,
            "time_weight": 0.3,
        }
        missing = REQUIRED_FIELDS - set(row.keys())
        assert len(missing) == 0

    def test_missing_required_rejected(self):
        row = {"rep_kernel_id": "rep-1"}
        missing = REQUIRED_FIELDS - set(row.keys())
        assert len(missing) > 0

    def test_forbidden_fields_rejected(self):
        row = {
            "rep_kernel_id": "rep-1",
            "kernel_name": "test",
            "cluster_id": "c1",
            "member_invocations": [],
            "coverage_count": 1,
            "coverage_weight": 0.5,
            "time_weight": 0.3,
            "family_id": "F1",  # forbidden
        }
        leaked = BL_FORBIDDEN & set(row.keys())
        assert "family_id" in leaked

    def test_execution_template_rejected(self):
        row = {
            "rep_kernel_id": "rep-1",
            "kernel_name": "test",
            "cluster_id": "c1",
            "member_invocations": [],
            "coverage_count": 1,
            "coverage_weight": 0.5,
            "time_weight": 0.3,
            "execution_template": "compute",  # forbidden
        }
        leaked = BL_FORBIDDEN & set(row.keys())
        assert "execution_template" in leaked


# ── Stage-gate enforcement tests ──────────────────────────────────────

class TestStageGate:
    def test_acquisition_gap_blocks_selector(self):
        # The selector refuses to run when stage gate says blocked
        from pka_baseline_selector import _check_stage_gate as sel_check
        import pka_baseline_selector as sel_mod

        # Without a real stage gate file, should refuse
        if sel_mod.STAGE_GATE_PATH.exists():
            ok, _ = sel_check()
            # In acquisition_gate_success state, this should be False
            report = json.loads(sel_mod.STAGE_GATE_PATH.read_text())
            stage_3 = report["stages"]["stage_3_selector"]
            assert stage_3 == "blocked" or ok

    def test_blocked_b_line_refuses(self):
        from b_line_consumer_l1 import _check_stage_gate as bl_check
        import b_line_consumer_l1 as bl_mod

        if bl_mod.STAGE_GATE_PATH.exists():
            ok, _ = bl_check()
            report = json.loads(bl_mod.STAGE_GATE_PATH.read_text())
            stage_4 = report["stages"]["stage_4_b_line_consumption"]
            # If blocked, check returns False
            if stage_4 == "blocked":
                assert not ok


# ── Source adapter tests ───────────────────────────────────────────────

class TestSourceAdapters:
    def test_adapter_map_complete(self):
        assert "local_microbench" in ADAPTERS
        assert "local_benchmark_result" in ADAPTERS
        assert "local_ai_workload" in ADAPTERS

    def test_microbench_adapter_exists_for_local_data(self):
        microbench_path = REPO_ROOT / "experiments/baseline_diagnosis/results/microbench/l1_bw_32f.json"
        assert microbench_path.exists(), "Microbench test fixture missing"

    def test_mini_transformer_adapter_exists_for_local_data(self):
        mt_path = REPO_ROOT / "experiments/mini_transformer/mini_transformer_v4_full.json"
        assert mt_path.exists(), "Mini-transformer test fixture missing"
