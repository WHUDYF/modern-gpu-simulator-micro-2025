import copy
from pathlib import Path

import pytest

from experiments.gcl_phase_b.resnet50_adapter import (
    build_resnet50_trace_adapter_bundle,
    load_resnet50_trace_sources,
    validate_resnet50_trace_adapter_bundle,
)

FIXTURE_ROOT = Path("tests/fixtures/gcl_resnet50_gate1")


def test_resnet50_gate1_fixture_sources_are_loadable():
    sources = load_resnet50_trace_sources(FIXTURE_ROOT)

    assert sources.scheduler_metadata["scheduler_metadata_source"] == "real_nvbit_smid"
    assert sources.dynamic_trace["kernel_invocations"]
    assert sources.threadblocks["threadblocks"]
    assert sources.enhanced_execution_info["instructions"]
    assert sources.stats_rows


def test_gate1_builds_resnet50_trace_adapter_bundle():
    bundle = build_resnet50_trace_adapter_bundle(FIXTURE_ROOT)

    validate_resnet50_trace_adapter_bundle(bundle)
    assert bundle["artifact_type"] == "gcl_resnet50_trace_adapter_bundle"
    assert bundle["workload_id"] == "resnet50"
    assert bundle["scheduler_metadata_source"] == "real_nvbit_smid"
    assert bundle["adapter_validation_report"]["status"] == "passed"
    assert len(bundle["kernel_invocation_table"]) == 2


def test_gate1_rejects_non_real_scheduler_metadata():
    bundle = build_resnet50_trace_adapter_bundle(FIXTURE_ROOT)
    corrupted = copy.deepcopy(bundle)
    corrupted["scheduler_metadata_source"] = "file_order_fallback"

    with pytest.raises(ValueError, match="real_nvbit_smid"):
        validate_resnet50_trace_adapter_bundle(corrupted)


def test_gate1_rejects_non_reproducible_adapter_hash():
    bundle = build_resnet50_trace_adapter_bundle(FIXTURE_ROOT)
    bundle["kernel_invocation_table"][0]["kernel_name"] = "changed"

    with pytest.raises(ValueError, match="adapter_bundle_hash"):
        validate_resnet50_trace_adapter_bundle(bundle)
