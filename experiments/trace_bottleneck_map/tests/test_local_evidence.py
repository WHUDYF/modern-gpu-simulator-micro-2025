from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.trace_bottleneck_map.local_evidence import load_trace_benchmark_table


def test_load_trace_benchmark_table_parses_measured_rows(tmp_path):
    source = tmp_path / "trace-benchmark.md"
    source.write_text(
        "\n".join(
            [
                "| benchmark | trace status | trace size (MiB) | export time (s) | export speed | dynamic insts | sim status | sim time (s) |",
                "|---|---|---:|---:|---:|---:|---|---:|",
                "| l2_bw_32f | ok | 568.192 | 69.41 | 8.186 | 56820736 | ok | about 17 |",
                "| shared_lat | ok | 0.340 | 2.18 | 0.156 | 17344 | ok | 1.29 |",
                "| l2_bw_64f | partial_or_timeout | 0.000 | 90.18 | 0.000 | 0 | missing |  |",
            ]
        )
    )

    observations = load_trace_benchmark_table(source)

    cases = {obs.representative_case: obs for obs in observations}
    assert cases["l2_bw_32f"].trace_size_mib == 568.192
    assert cases["l2_bw_32f"].export_time_s == 69.41
    assert cases["l2_bw_32f"].sim_time_s == 17.0
    assert cases["shared_lat"].category == "latency"
    assert "l2_bw_64f" not in cases


def test_load_trace_benchmark_table_rejects_missing_file(tmp_path):
    missing = tmp_path / "missing.md"
    try:
        load_trace_benchmark_table(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing trace benchmark file was accepted")
