from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkObservation:
    suite: str
    representative_case: str
    category: str
    trace_size_mib: float | None = None
    export_time_s: float | None = None
    sim_time_s: float | None = None
    native_runtime_s: float | None = None
    status: str = "estimated"
    evidence: str = ""
    dominant_bottleneck_hint: str | None = None


def classify_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 1:
        return "sub-second"
    if seconds < 10:
        return "seconds"
    if seconds < 60:
        return "tens of seconds"
    if seconds < 3600:
        return "minutes"
    if seconds < 86400:
        return "hours"
    return "infeasible"


def classify_trace_size_mib(size_mib: float | None) -> str:
    if size_mib is None:
        return "unknown"
    if size_mib < 1:
        return "tiny"
    if size_mib < 10:
        return "small"
    if size_mib < 128:
        return "medium"
    if size_mib < 1024:
        return "large"
    return "huge"


def infer_bottleneck(
    *,
    trace_size_mib: float | None,
    export_time_s: float | None,
    sim_time_s: float | None,
) -> str:
    if export_time_s is None and sim_time_s is None:
        return "insufficient evidence"
    if export_time_s is None:
        return "simulator throughput"
    if sim_time_s is None:
        return "trace export / I/O"
    if trace_size_mib is not None and trace_size_mib < 1 and max(export_time_s, sim_time_s) <= 3:
        return "capture / fixed overhead"
    if trace_size_mib is not None and trace_size_mib < 10 and max(export_time_s, sim_time_s) < 10 and export_time_s >= sim_time_s:
        return "balanced / mixed"
    if export_time_s >= sim_time_s:
        return "trace export / I/O"
    if sim_time_s >= 1.5 * export_time_s:
        return "simulator throughput"
    return "balanced / mixed"


def record_from_observation(observation: BenchmarkObservation) -> dict[str, Any]:
    record = asdict(observation)
    record.update(
        {
            "native_runtime_class": classify_seconds(observation.native_runtime_s),
            "trace_size_class": classify_trace_size_mib(observation.trace_size_mib),
            "export_cost_class": classify_seconds(observation.export_time_s),
            "simulator_cost_class": classify_seconds(observation.sim_time_s),
            "dominant_bottleneck": observation.dominant_bottleneck_hint or infer_bottleneck(
                trace_size_mib=observation.trace_size_mib,
                export_time_s=observation.export_time_s,
                sim_time_s=observation.sim_time_s,
            ),
        }
    )
    return record
