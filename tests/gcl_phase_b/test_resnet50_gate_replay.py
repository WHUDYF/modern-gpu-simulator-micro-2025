from experiments.gcl_phase_b.resnet50_gate0 import write_resnet50_gate0_blocker_report
from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate5


def _blocked_gate0_root(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    write_resnet50_gate0_blocker_report(
        root,
        reason="real ResNet-50 NVBit trace is not available",
        missing_requirements=["dynamic_trace.pb", "threadblocks/"],
    )
    return root


def test_resnet50_gate0_blocker_pipeline_hashes_are_replayable(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    manifest_a = run_resnet50_gate1_to_gate5(
        _blocked_gate0_root(tmp_path, "blocked_a"), out_a, seed=20260606
    )
    manifest_b = run_resnet50_gate1_to_gate5(
        _blocked_gate0_root(tmp_path, "blocked_b"), out_b, seed=20260606
    )

    assert manifest_a["final_gate"] == "gate0_blocked"
    assert manifest_a["pipeline_manifest_hash"] == manifest_b["pipeline_manifest_hash"]
    assert manifest_a["hashes"]["gate0_blocker_report_hash"] == manifest_b["hashes"][
        "gate0_blocker_report_hash"
    ]
