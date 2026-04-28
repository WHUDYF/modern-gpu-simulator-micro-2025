from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.trace_bottleneck_map.catalog import ESTIMATED_SUITE_OBSERVATIONS


def test_catalog_keeps_multigpu_suites_out_of_main_table():
    by_suite = {entry.suite: entry for entry in ESTIMATED_SUITE_OBSERVATIONS}
    assert by_suite["NCCL-tests"].status == "excluded"
    assert by_suite["OSU micro-benchmarks"].status == "excluded"


def test_catalog_marks_mlperf_as_appendix_anchor():
    by_suite = {entry.suite: entry for entry in ESTIMATED_SUITE_OBSERVATIONS}
    assert by_suite["MLPerf Inference / Training"].status == "appendix_only"
    assert by_suite["MLPerf Inference / Training"].category == "full workload anchor"


def test_catalog_includes_main_single_gpu_suites():
    suites = {entry.suite for entry in ESTIMATED_SUITE_OBSERVATIONS if entry.status == "estimated"}
    assert {
        "BabelStream",
        "nvbandwidth",
        "nvbench",
        "CUTLASS profiler",
        "Rodinia",
        "Parboil",
        "PolyBench/GPU",
    } <= suites
