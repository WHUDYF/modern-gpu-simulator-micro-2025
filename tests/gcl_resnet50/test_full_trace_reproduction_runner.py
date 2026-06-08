import json
import subprocess
import sys

import pytest

from scripts import run_resnet50_full_trace_gcl


def _write_gate0_manifest(root, *, artifact_status="formal", eligible=True, scope="full_resnet50_inference_trace"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "gate0_trace_acquisition_manifest.json").write_text(
        json.dumps(
            {
                "artifact_status": artifact_status,
                "formal_input_eligible": eligible,
                "input_scope": scope,
                "gate0_manifest_hash": "gate0-hash",
            }
        ),
        encoding="utf-8",
    )


def _fake_success_pipeline(calls):
    def fake_pipeline(
        root,
        out_dir,
        seed,
        baseline_artifacts_path=None,
        invocation_limit=None,
        invocation_ids=None,
    ):
        calls["root"] = root
        calls["out_dir"] = out_dir
        calls["seed"] = seed
        calls["baseline_artifacts_path"] = baseline_artifacts_path
        calls["invocation_limit"] = invocation_limit
        calls["invocation_ids"] = invocation_ids
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "resnet50_trace_adapter_bundle.json").write_text(
            json.dumps({"adapter_validation_report": {"status": "passed"}}),
            encoding="utf-8",
        )
        (out_dir / "kernel_embedding_table.json").write_text(
            json.dumps({"embeddings": [1, 2]}),
            encoding="utf-8",
        )
        (out_dir / "selector_artifacts.json").write_text(
            json.dumps({"k_selection_report": {"selected_k": 2}}),
            encoding="utf-8",
        )
        (out_dir / "gate7_cluster_correctness_manifest.json").write_text(
            json.dumps({"gate7_cluster_correctness_manifest_hash": "gate7-hash"}),
            encoding="utf-8",
        )
        return {
            "artifact_type": "gcl_resnet50_gate1_7_pipeline_manifest",
            "final_gate": "gate9_report_only",
            "hashes": {
                "embedding_table_hash": "embedding-hash",
                "selector_manifest_hash": "selector-hash",
                "gate7_correctness_manifest_hash": "gate7-hash",
                "gate8_tuning_vector_proposal_hash": "gate8-hash",
                "gate9_sampled_vs_full_evaluation_hash": "gate9-hash",
            },
            "pipeline_manifest_hash": "pipeline-hash",
        }

    return fake_pipeline


def test_full_trace_runner_calls_pipeline_without_invocation_slicing(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "run_resnet50_gate1_to_gate7",
        _fake_success_pipeline(calls),
    )
    root = tmp_path / "formal_root"
    _write_gate0_manifest(root)

    result = run_resnet50_full_trace_gcl.run_full_trace_reproduction(
        input_root=root,
        out_dir=tmp_path / "out",
        seed=20260608,
        baseline_artifacts=None,
    )

    assert calls["invocation_limit"] is None
    assert calls["invocation_ids"] is None
    assert result["run_scope"] == "real_resnet50_full_trace"
    assert result["formal_full_trace_run"] is True
    assert result["source_gate0_manifest_hash"] == "gate0-hash"
    assert result["invocation_limit"] is None
    assert result["invocation_ids"] is None


def test_full_trace_runner_rejects_non_full_gate0_scope(tmp_path):
    root = tmp_path / "root"
    _write_gate0_manifest(root, scope="bounded_invocation_slice")

    with pytest.raises(ValueError, match="full ResNet50"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=tmp_path / "out",
            seed=20260608,
            baseline_artifacts=None,
        )


def test_full_trace_runner_rejects_debug_gate0_manifest(tmp_path):
    root = tmp_path / "root"
    _write_gate0_manifest(root, artifact_status="debug_not_formal", eligible=False)

    with pytest.raises(ValueError, match="not formal"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=tmp_path / "out",
            seed=20260608,
            baseline_artifacts=None,
        )


def test_full_trace_runner_cli_writes_manifest(tmp_path, monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(
        run_resnet50_full_trace_gcl,
        "run_resnet50_gate1_to_gate7",
        _fake_success_pipeline(calls),
    )
    root = tmp_path / "root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"

    run_resnet50_full_trace_gcl.main_args(
        [
            "--input-root",
            str(root),
            "--out",
            str(out_dir),
            "--seed",
            "20260608",
        ]
    )

    manifest = json.loads(
        (out_dir / "resnet50_full_trace_reproduction_manifest.json").read_text()
    )
    assert manifest["formal_full_trace_run"] is True
    assert manifest["artifact_presence"]["kernel_embedding_table.json"] is True
    assert "real_resnet50_full_trace" in capsys.readouterr().out


def test_full_trace_runner_writes_blocker_report_on_resource_failure(tmp_path, monkeypatch):
    def fake_pipeline(*args, **kwargs):
        raise RuntimeError("out of memory while tensorizing full trace")

    monkeypatch.setattr(run_resnet50_full_trace_gcl, "run_resnet50_gate1_to_gate7", fake_pipeline)
    root = tmp_path / "root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match="out of memory"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=out_dir,
            seed=20260608,
            baseline_artifacts=None,
        )

    blocker = json.loads(
        (out_dir / "resnet50_full_trace_reproduction_blocker_report.json").read_text()
    )
    assert blocker["run_scope"] == "real_resnet50_full_trace"
    assert blocker["formal_full_trace_run"] is False
    assert blocker["blocker_reason"] == "RuntimeError"
    assert not (out_dir / "resnet50_full_trace_reproduction_manifest.json").exists()


def test_full_trace_runner_writes_blocker_report_on_deadline(tmp_path, monkeypatch):
    def fake_pipeline(*args, **kwargs):
        raise TimeoutError("full trace reproduction exceeded 1 seconds")

    monkeypatch.setattr(run_resnet50_full_trace_gcl, "run_resnet50_gate1_to_gate7", fake_pipeline)
    root = tmp_path / "root"
    _write_gate0_manifest(root)
    out_dir = tmp_path / "out"

    with pytest.raises(TimeoutError, match="exceeded"):
        run_resnet50_full_trace_gcl.run_full_trace_reproduction(
            input_root=root,
            out_dir=out_dir,
            seed=20260608,
            baseline_artifacts=None,
            deadline_seconds=1,
        )

    blocker = json.loads(
        (out_dir / "resnet50_full_trace_reproduction_blocker_report.json").read_text()
    )
    assert blocker["blocker_reason"] == "TimeoutError"
    assert blocker["resource_status"] == "blocked"
    assert blocker["deadline_seconds"] == 1
    assert not (out_dir / "resnet50_full_trace_reproduction_manifest.json").exists()


def test_full_trace_runner_script_entrypoint_is_importable_from_repo_root():
    result = subprocess.run(
        [sys.executable, "scripts/run_resnet50_full_trace_gcl.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--input-root" in result.stdout
