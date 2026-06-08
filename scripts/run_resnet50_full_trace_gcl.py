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

from experiments.gcl_phase_b.resnet50_gate_pipeline import run_resnet50_gate1_to_gate7
from experiments.gcl_phase_b.utils import stable_hash


FULL_TRACE_MANIFEST = "resnet50_full_trace_reproduction_manifest.json"
FULL_TRACE_BLOCKER = "resnet50_full_trace_reproduction_blocker_report.json"
GATE0_MANIFEST = "gate0_trace_acquisition_manifest.json"
ADAPTER_BUNDLE = "resnet50_trace_adapter_bundle.json"
DEFAULT_DEADLINE_SECONDS = 2400


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_gate0_manifest(input_root: Path) -> dict[str, Any]:
    manifest_path = input_root / GATE0_MANIFEST
    if not manifest_path.exists():
        raise ValueError(f"missing Gate0 formal manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_status") != "formal":
        raise ValueError("Gate0 manifest is not formal")
    if manifest.get("formal_input_eligible") is not True:
        raise ValueError("Gate0 manifest is not formal-input eligible")
    if manifest.get("input_scope") != "full_resnet50_inference_trace":
        raise ValueError(f"Gate0 input_scope is not full ResNet50: {manifest.get('input_scope')}")
    return manifest


def _artifact_presence(out_dir: Path) -> dict[str, bool]:
    filenames = [
        ADAPTER_BUNDLE,
        "kernel_embedding_table.json",
        "selector_artifacts.json",
        "gate7_cluster_correctness_manifest.json",
        "gate1_7_pipeline_manifest.json",
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


def run_full_trace_reproduction(
    *,
    input_root: Path,
    out_dir: Path,
    seed: int,
    baseline_artifacts: Path | None,
    deadline_seconds: int | None = DEFAULT_DEADLINE_SECONDS,
) -> dict[str, Any]:
    gate0_manifest = _load_gate0_manifest(input_root)
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
        pipeline_manifest = run_resnet50_gate1_to_gate7(
            input_root,
            out_dir,
            seed=seed,
            baseline_artifacts_path=baseline_artifacts,
            invocation_limit=None,
            invocation_ids=None,
        )
        _reject_bounded_adapter_bundle(out_dir)
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
        "formal_full_trace_run": True,
        "seed": seed,
        "input_root": str(input_root),
        "source_gate0_manifest_hash": gate0_manifest.get("gate0_manifest_hash"),
        "input_kernel_invocation_count": pipeline_manifest.get("input_kernel_invocation_count"),
        "input_cta_record_count": gate0_manifest.get("cta_record_count"),
        "invocation_limit": None,
        "invocation_ids": None,
        "baseline_artifacts_path": str(baseline_artifacts) if baseline_artifacts else None,
        "deadline_seconds": deadline_seconds,
        "final_gate": pipeline_manifest["final_gate"],
        "pipeline_manifest_hash": pipeline_manifest["pipeline_manifest_hash"],
        "pipeline_hashes": pipeline_manifest["hashes"],
        "artifact_presence": _artifact_presence(out_dir),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "resource_status": "completed",
    }
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
