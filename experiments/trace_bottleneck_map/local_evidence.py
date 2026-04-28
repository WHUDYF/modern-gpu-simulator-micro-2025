from __future__ import annotations

from pathlib import Path

from experiments.trace_bottleneck_map.model import BenchmarkObservation


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_float(text: str) -> float | None:
    cleaned = text.strip().replace("about", "").strip()
    if cleaned == "":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _category_for_case(name: str) -> str:
    lowered = name.lower()
    if "lat" in lowered:
        return "latency"
    if "atomic" in lowered:
        return "atomic / bandwidth"
    if "flop" in lowered:
        return "compute"
    if "bw" in lowered:
        return "bandwidth"
    return "mixed"


def load_trace_benchmark_table(path: str | Path) -> list[BenchmarkObservation]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"trace benchmark file not found: {source}")

    observations: list[BenchmarkObservation] = []
    for line in source.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = _cells(stripped)
        if not cells or cells[0] in {"benchmark", "---"} or set(cells[0]) == {"-"}:
            continue
        if len(cells) < 8:
            continue

        benchmark = cells[0].strip("` ")
        trace_status = cells[1]
        sim_status = cells[6]
        if trace_status != "ok" or sim_status != "ok":
            continue

        observations.append(
            BenchmarkObservation(
                suite="GPU_Microbenchmark",
                representative_case=benchmark,
                category=_category_for_case(benchmark),
                trace_size_mib=_parse_float(cells[2]),
                export_time_s=_parse_float(cells[3]),
                sim_time_s=_parse_float(cells[7]),
                status="measured",
                evidence=str(source),
            )
        )
    return observations
