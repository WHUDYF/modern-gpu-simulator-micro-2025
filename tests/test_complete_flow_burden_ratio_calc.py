import json
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "artifacts"
    / "gpu_trace_frontend_difftest_necessity"
    / "complete_flow_burden_ratio_calc.py"
)
spec = importlib.util.spec_from_file_location("complete_flow_burden_ratio_calc", MODULE_PATH)
complete_flow_burden_ratio_calc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(complete_flow_burden_ratio_calc)


def test_build_workloads_uses_measured_frontend_timing_without_complete_flow_records(
    tmp_path,
    monkeypatch,
):
    wid = "bert-base-encoder-layer-slice"
    timing = {
        "trace_read_s": 1.0,
        "parse_pb_s": 2.0,
        "static_bind_s": 3.0,
        "warp_trace_build_s": 4.0,
        "tb_load_s": 5.0,
        "get_next_inst_s": 6.0,
    }
    (tmp_path / f"frontend_timing_breakdown_{wid}.json").write_text(
        json.dumps(timing),
        encoding="utf-8",
    )
    monkeypatch.setattr(complete_flow_burden_ratio_calc, "ARTIFACT_DIR", str(tmp_path))

    workloads = complete_flow_burden_ratio_calc.build_workloads()

    workload = next(item for item in workloads if item["workload_id"] == wid)
    assert workload["T_trace_to_sim_s"] == {"value": 21.0, "label": "measured"}
    assert workload["T_kernel_or_trace_export_s"]["label"] == "placeholder"
    assert workload["T_sim_backend_execution_s"]["label"] == "placeholder"
    assert workload["T_result_analysis_s"]["label"] == "placeholder"
