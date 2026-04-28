from experiments.trace_bottleneck_map.model import (
    BenchmarkObservation,
    classify_seconds,
    classify_trace_size_mib,
    infer_bottleneck,
    record_from_observation,
)


def test_classify_seconds_boundaries():
    assert classify_seconds(None) == "unknown"
    assert classify_seconds(0.4) == "sub-second"
    assert classify_seconds(2.0) == "seconds"
    assert classify_seconds(34.0) == "tens of seconds"
    assert classify_seconds(120.0) == "minutes"
    assert classify_seconds(7200.0) == "hours"
    assert classify_seconds(200000.0) == "infeasible"


def test_classify_trace_size_mib_boundaries():
    assert classify_trace_size_mib(None) == "unknown"
    assert classify_trace_size_mib(0.3) == "tiny"
    assert classify_trace_size_mib(4.0) == "small"
    assert classify_trace_size_mib(47.0) == "medium"
    assert classify_trace_size_mib(568.0) == "large"
    assert classify_trace_size_mib(2048.0) == "huge"


def test_infer_bottleneck_export_dominates_large_trace():
    assert infer_bottleneck(trace_size_mib=568.192, export_time_s=69.41, sim_time_s=17.0) == "trace export / I/O"


def test_infer_bottleneck_simulator_dominates():
    assert infer_bottleneck(trace_size_mib=5.423, export_time_s=2.27, sim_time_s=10.13) == "simulator throughput"


def test_infer_bottleneck_fixed_overhead_for_tiny_trace():
    assert infer_bottleneck(trace_size_mib=0.288, export_time_s=2.11, sim_time_s=1.26) == "capture / fixed overhead"


def test_record_from_observation_classifies_fields():
    obs = BenchmarkObservation(
        suite="GPU_Microbenchmark",
        representative_case="l2_bw_32f",
        category="bandwidth",
        trace_size_mib=568.192,
        export_time_s=69.41,
        sim_time_s=17.0,
        status="measured",
        evidence="docs/trace-benchmark-2026-04-03.md",
    )
    record = record_from_observation(obs)
    assert record["trace_size_class"] == "large"
    assert record["export_cost_class"] == "minutes"
    assert record["simulator_cost_class"] == "tens of seconds"
    assert record["dominant_bottleneck"] == "trace export / I/O"


def test_record_from_observation_uses_catalog_hint():
    obs = BenchmarkObservation(
        suite="nvbench",
        representative_case="runtime and compile-time parameter sweeps",
        category="generic kernel benchmark",
        status="estimated",
        evidence="https://github.com/NVIDIA/nvbench",
        dominant_bottleneck_hint="benchmark sweep explosion",
    )
    record = record_from_observation(obs)
    assert record["dominant_bottleneck"] == "benchmark sweep explosion"
