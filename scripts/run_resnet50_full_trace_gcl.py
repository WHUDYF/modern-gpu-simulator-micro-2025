#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.gcl_phase_b.resnet50_gate0 import load_gate0_trace_acquisition_manifest
from experiments.gcl_phase_b.resnet50_gate0 import (
    GATE0_BLOCKER_FILENAME,
    GATE0_MANIFEST_FILENAME,
)
from experiments.gcl_phase_b.resnet50_gate_pipeline import (
    resume_resnet50_gate5_to_gate9_from_disk,
    run_resnet50_gate1_to_gate7,
)
from experiments.gcl_phase_b.trustworthiness import (
    evaluate_gnn_acceptance_from_dir,
    write_gnn_acceptance_artifacts,
)
from experiments.gcl_phase_b.utils import stable_hash


FULL_TRACE_MANIFEST = "resnet50_full_trace_reproduction_manifest.json"
FULL_TRACE_BLOCKER = "resnet50_full_trace_reproduction_blocker_report.json"
ADAPTER_BUNDLE = "resnet50_trace_adapter_bundle.json"
DEFAULT_DEADLINE_SECONDS = 2400
GATE4_RESUME_REQUIRED_ARTIFACTS = [
    ADAPTER_BUNDLE,
    "representative_sm_trace_manifest.json",
    "canonical_graph_bundle.json",
    "graph_tensor_bundle.json",
]
GATE1_PLUS_ARTIFACTS = [
    ADAPTER_BUNDLE,
    "representative_sm_trace_manifest.json",
    "selected_sm_policy_report.json",
    "scope_preview_report.json",
    "canonical_graph_bundle.json",
    "graph_tensor_bundle.json",
    "augmentation_manifest.json",
    "rgcn_checkpoint.pt",
    "rgcn_training_run_manifest.json",
    "rgcn_checkpoint_manifest.json",
    "gate5_embedding_export_progress.json",
    "kernel_embedding_table.json",
    "embedding_export_report.json",
    "readout_manifest.json",
    "gate5_lineage_bundle.json",
    "selector_artifacts.json",
    "cluster_embedding_quality_report.json",
    "cluster_family_alignment_report.json",
    "representative_quality_report.json",
    "cluster_metric_error_report.json",
    "cluster_stability_report.json",
    "gate7_cluster_correctness_manifest.json",
    "cluster_tuning_vector_table.json",
    "tuning_vector_provenance_report.json",
    "tuning_safety_report.json",
    "gate8_tuning_manifest.json",
    "gate8_tuning_vector_proposal.json",
    "full_vs_sampled_simulation_report.json",
    "sampled_speedup_report.json",
    "sampled_error_report.json",
    "tuning_effect_report.json",
    "gate9_simulator_evaluation_manifest.json",
    "gate9_sampled_vs_full_evaluation.json",
    "gate1_7_pipeline_manifest.json",
    "gnn_acceptance_manifest.json",
    "gnn_acceptance_summary.json",
    "gnn_acceptance_report.md",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _input_cta_record_count(input_root: Path) -> int:
    scheduler_path = input_root / "scheduler_metadata.json"
    if not scheduler_path.exists():
        raise ValueError(f"missing scheduler metadata for CTA count: {scheduler_path}")
    scheduler_metadata = _read_json(scheduler_path)
    invocations = scheduler_metadata.get("kernel_invocations")
    if not isinstance(invocations, list):
        raise ValueError("scheduler metadata requires kernel_invocations for CTA count")
    count = 0
    for invocation in invocations:
        cta_records = invocation.get("cta_records")
        if not isinstance(cta_records, list):
            raise ValueError("scheduler metadata kernel invocation requires cta_records")
        count += len(cta_records)
    return count


def _artifact_presence(out_dir: Path) -> dict[str, bool]:
    filenames = [
        ADAPTER_BUNDLE,
        "kernel_embedding_table.json",
        "selector_artifacts.json",
        "gate7_cluster_correctness_manifest.json",
        "gate1_7_pipeline_manifest.json",
        "gnn_acceptance_manifest.json",
        "gnn_acceptance_summary.json",
        "gnn_acceptance_report.md",
    ]
    return {filename: (out_dir / filename).exists() for filename in filenames}


def _reject_bounded_adapter_bundle(out_dir: Path) -> None:
    adapter_path = out_dir / ADAPTER_BUNDLE
    if not adapter_path.exists():
        raise ValueError(f"missing adapter bundle after pipeline run: {adapter_path}")
    adapter = _read_json(adapter_path)
    report = adapter.get("adapter_validation_report", {})
    bounded_fields = [
        field
        for field in ["formal_replay_invocation_limit", "formal_replay_invocation_ids"]
        if field in report
    ]
    if bounded_fields:
        raise ValueError(
            "bounded replay adapter cannot be labeled full trace: "
            + ", ".join(sorted(bounded_fields))
        )


def _adapter_has_bounded_replay(adapter: dict[str, Any]) -> bool:
    report = adapter.get("adapter_validation_report", {})
    return any(
        field in report
        for field in ["formal_replay_invocation_limit", "formal_replay_invocation_ids"]
    )


def _clear_gate1_plus_artifacts(out_dir: Path) -> None:
    for filename in GATE1_PLUS_ARTIFACTS:
        (out_dir / filename).unlink(missing_ok=True)


def _has_complete_gate4_resume_artifact_set(out_dir: Path) -> bool:
    return all((out_dir / filename).exists() for filename in GATE4_RESUME_REQUIRED_ARTIFACTS)


def _has_gate0_blocker_without_manifest(input_root: Path) -> bool:
    return (input_root / GATE0_BLOCKER_FILENAME).exists() and not (
        input_root / GATE0_MANIFEST_FILENAME
    ).exists()


def _validate_resume_artifacts_match_gate0(
    out_dir: Path,
    *,
    gate0_manifest: dict[str, Any],
) -> bool:
    adapter_path = out_dir / ADAPTER_BUNDLE
    if not adapter_path.exists():
        raise ValueError(f"missing adapter bundle for Gate4 resume: {adapter_path}")
    adapter = _read_json(adapter_path)
    expected_hash = gate0_manifest.get("gate0_manifest_hash")
    actual_hash = adapter.get("source_gate0_manifest_hash")
    if actual_hash != expected_hash:
        raise ValueError(
            "Gate4 resume artifacts do not match requested Gate0 root: "
            f"adapter source_gate0_manifest_hash={actual_hash!r}, "
            f"gate0_manifest_hash={expected_hash!r}"
        )
    return _adapter_has_bounded_replay(adapter)


def _run_manifest_backed_pipeline(
    *,
    input_root: Path,
    out_dir: Path,
    seed: int,
    baseline_artifacts: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    gate0_manifest = load_gate0_trace_acquisition_manifest(input_root)
    input_cta_record_count = _input_cta_record_count(input_root)
    should_resume = _has_complete_gate4_resume_artifact_set(out_dir)
    if not should_resume and (out_dir / "graph_tensor_bundle.json").exists():
        _clear_gate1_plus_artifacts(out_dir)
    if should_resume:
        is_bounded_resume = _validate_resume_artifacts_match_gate0(
            out_dir,
            gate0_manifest=gate0_manifest,
        )
        if is_bounded_resume:
            _clear_gate1_plus_artifacts(out_dir)
            should_resume = False
    if should_resume:
        pipeline_manifest = resume_resnet50_gate5_to_gate9_from_disk(
            out_dir,
            seed=seed,
            baseline_artifacts_path=baseline_artifacts,
        )
    else:
        pipeline_manifest = run_resnet50_gate1_to_gate7(
            input_root,
            out_dir,
            seed=seed,
            baseline_artifacts_path=baseline_artifacts,
            invocation_limit=None,
            invocation_ids=None,
        )
    _reject_bounded_adapter_bundle(out_dir)
    return pipeline_manifest, gate0_manifest, input_cta_record_count


def _write_blocker(
    *,
    out_dir: Path,
    input_root: Path,
    seed: int,
    started: float,
    exc: Exception,
    deadline_seconds: int | None,
) -> dict[str, Any]:
    (out_dir / FULL_TRACE_MANIFEST).unlink(missing_ok=True)
    blocker = {
        "artifact_type": "gcl_resnet50_full_trace_reproduction_blocker_report",
        "artifact_version": "full_trace_reproduction_blocker_report_v1",
        "run_scope": "real_resnet50_full_trace",
        "formal_full_trace_run": False,
        "seed": seed,
        "input_root": str(input_root),
        "blocker_reason": type(exc).__name__,
        "blocker_message": str(exc),
        "deadline_seconds": deadline_seconds,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "resource_status": "blocked",
    }
    blocker["blocker_report_hash"] = stable_hash(blocker)
    _write_json(out_dir / FULL_TRACE_BLOCKER, blocker)
    return blocker


def _write_gnn_acceptance(out_dir: Path) -> dict[str, Any]:
    report = evaluate_gnn_acceptance_from_dir(out_dir)
    write_gnn_acceptance_artifacts(out_dir, report)
    return _read_json(out_dir / "gnn_acceptance_manifest.json")


def append_gnn_acceptance_report(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    manifest_path = out_dir / FULL_TRACE_MANIFEST
    if not manifest_path.exists():
        raise ValueError(f"missing full trace manifest for GNN acceptance: {manifest_path}")
    return _write_gnn_acceptance(out_dir)


def run_full_trace_reproduction(
    *,
    input_root: Path,
    out_dir: Path,
    seed: int,
    baseline_artifacts: Path | None,
    deadline_seconds: int | None = DEFAULT_DEADLINE_SECONDS,
) -> dict[str, Any]:
    started = time.monotonic()
    previous_handler = None
    if deadline_seconds is not None:
        if deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive or None")
        previous_handler = signal.signal(
            signal.SIGALRM,
            lambda _signum, _frame: (_ for _ in ()).throw(
                TimeoutError(
                    f"full trace reproduction exceeded {deadline_seconds} seconds"
                )
            ),
        )
        signal.alarm(deadline_seconds)
    try:
        gate0_manifest: dict[str, Any] | None = None
        input_cta_record_count: int | None = None
        if _has_gate0_blocker_without_manifest(input_root):
            pipeline_manifest = run_resnet50_gate1_to_gate7(
                input_root,
                out_dir,
                seed=seed,
                baseline_artifacts_path=baseline_artifacts,
                invocation_limit=None,
                invocation_ids=None,
            )
        else:
            pipeline_manifest, gate0_manifest, input_cta_record_count = (
                _run_manifest_backed_pipeline(
                    input_root=input_root,
                    out_dir=out_dir,
                    seed=seed,
                    baseline_artifacts=baseline_artifacts,
                )
            )
    except Exception as exc:
        _write_blocker(
            out_dir=out_dir,
            input_root=input_root,
            seed=seed,
            started=started,
            exc=exc,
            deadline_seconds=deadline_seconds,
        )
        raise
    finally:
        if deadline_seconds is not None:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)

    manifest = {
        "artifact_type": "gcl_resnet50_full_trace_reproduction_manifest",
        "artifact_version": "full_trace_reproduction_manifest_v1",
        "run_scope": "real_resnet50_full_trace",
        "formal_full_trace_run": pipeline_manifest["final_gate"] != "gate0_blocked",
        "seed": seed,
        "input_root": str(input_root),
        "source_gate0_manifest_hash": (
            gate0_manifest.get("gate0_manifest_hash") if gate0_manifest else None
        ),
        "input_kernel_invocation_count": pipeline_manifest.get("input_kernel_invocation_count"),
        "input_cta_record_count": input_cta_record_count,
        "invocation_limit": None,
        "invocation_ids": None,
        "baseline_artifacts_path": str(baseline_artifacts) if baseline_artifacts else None,
        "deadline_seconds": deadline_seconds,
        "final_gate": pipeline_manifest["final_gate"],
        "pipeline_manifest_hash": pipeline_manifest["pipeline_manifest_hash"],
        "pipeline_hashes": pipeline_manifest["hashes"],
        "artifact_presence": _artifact_presence(out_dir),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "resource_status": (
            "blocked" if pipeline_manifest["final_gate"] == "gate0_blocked" else "completed"
        ),
    }
    manifest["full_trace_reproduction_manifest_hash"] = stable_hash(manifest)
    (out_dir / FULL_TRACE_BLOCKER).unlink(missing_ok=True)
    _write_json(out_dir / FULL_TRACE_MANIFEST, manifest)
    acceptance_report = _write_gnn_acceptance(out_dir)
    manifest["pre_gnn_acceptance_manifest_hash"] = manifest[
        "full_trace_reproduction_manifest_hash"
    ]
    manifest["gnn_acceptance_status"] = acceptance_report["gnn_acceptance_status"]
    manifest["gnn_acceptance_claim_status"] = acceptance_report["claim_status"]
    manifest["gnn_acceptance_manifest_hash"] = acceptance_report[
        "gnn_acceptance_manifest_hash"
    ]
    manifest["artifact_presence"] = _artifact_presence(out_dir)
    manifest["full_trace_reproduction_manifest_hash"] = stable_hash(manifest)
    _write_json(out_dir / FULL_TRACE_MANIFEST, manifest)
    return manifest


def main_args(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--baseline-artifacts", type=Path)
    parser.add_argument("--deadline-seconds", type=int, default=DEFAULT_DEADLINE_SECONDS)
    args = parser.parse_args(argv)
    manifest = run_full_trace_reproduction(
        input_root=args.input_root,
        out_dir=args.out,
        seed=args.seed,
        baseline_artifacts=args.baseline_artifacts,
        deadline_seconds=args.deadline_seconds,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    main_args()


if __name__ == "__main__":
    main()
