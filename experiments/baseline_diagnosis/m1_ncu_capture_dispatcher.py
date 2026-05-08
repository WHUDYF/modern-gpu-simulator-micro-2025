"""Gate 2 exact-metric NCU capture dispatcher for PKA-M1."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from shared_acquisition import (
    ARTIFACT_DIR,
    REPO_ROOT,
    artifact_ref,
    command_hash,
    environment_signature,
    file_hash,
    sanitize_token,
    selected_metric_records,
    selected_ncu_metrics,
    stable_hash,
    write_json,
)

RESOLUTION_PATH = ARTIFACT_DIR / "m1_workload_resolution_l1.json"
ATTEMPTS_PATH = ARTIFACT_DIR / "m1_ncu_capture_attempts_l1.json"
GAP_PATH = ARTIFACT_DIR / "m1_ncu_capture_gap_l1.json"
QUERY_PATH = ARTIFACT_DIR / "ncu_metric_query_l1.json"
RESOLUTION_TABLE_PATH = ARTIFACT_DIR / "ncu_metric_resolution_table_l1.json"
RESULTS_DIR = REPO_ROOT / "experiments" / "baseline_diagnosis" / "results" / "m1_ncu"


def _write_query_artifacts() -> None:
    query_command = ["ncu", "--query-metrics"]
    ncu_path = shutil.which("ncu")
    stdout = ""
    stderr = ""
    exit_code = None
    query_status = "environment_blocked"
    if ncu_path is not None:
        try:
            proc = subprocess.run(
                query_command,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            stdout = proc.stdout[-20000:]
            stderr = proc.stderr[-20000:]
            exit_code = proc.returncode
            query_status = "completed" if proc.returncode == 0 else "query_failed"
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout[-20000:] if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr[-20000:] if isinstance(exc.stderr, str) else ""
            query_status = "query_timeout"
    query = {
        "tool": "ncu",
        "query_command": query_command,
        "environment_signature": environment_signature(),
        "query_status": query_status,
        "query_exit_code": exit_code,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
        "metrics": selected_ncu_metrics(),
    }
    write_json(QUERY_PATH, query)
    write_json(RESOLUTION_TABLE_PATH, selected_metric_records())


def _classify(exit_code: int | None, stderr: str, csv_path: Path, timed_out: bool = False) -> tuple[str, bool, str | None]:
    stderr_lower = stderr.lower()
    if "err_nvgpuctrperm" in stderr_lower or "permission" in stderr_lower:
        return "permission_blocked", False, "ncu_permission_blocked"
    if timed_out:
        return "ncu_capture_timeout", False, "timeout"
    if exit_code == 0 and csv_path.exists() and csv_path.stat().st_size > 0:
        return "captured", True, None
    if exit_code not in (0, None) and csv_path.exists() and csv_path.stat().st_size > 0:
        return "capture_non_zero_exit_with_partial_csv", True, "non_zero_exit_with_partial_csv"
    if shutil.which("ncu") is None:
        return "environment_blocked", False, "ncu_not_found"
    return "malformed_ncu_csv", False, "missing_or_empty_csv"


def dispatch(dry_run: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _write_query_artifacts()
    records = json.loads(RESOLUTION_PATH.read_text()) if RESOLUTION_PATH.exists() else []
    resolved = [row for row in records if row.get("resolution_status") == "resolved"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in resolved:
        key = stable_hash(row.get("resolved_run_command", []))
        grouped.setdefault(key, []).append(row)

    attempts: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    metrics = selected_ncu_metrics()
    for index, rows in enumerate(grouped.values()):
        first = rows[0]
        token = sanitize_token(first.get("workload_id", "workload"))
        short_hash = command_hash(first["resolved_run_command"])[:8]
        job_id = f"m1cap_{index:03d}_{token}_{short_hash}"
        job_dir = RESULTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        csv_path = job_dir / "capture.csv"
        stdout_path = job_dir / "capture_stdout.log"
        stderr_path = job_dir / "capture_stderr.log"
        exit_path = job_dir / "capture_exit_code.txt"
        env_path = job_dir / "capture_env_manifest.json"
        selected_path = job_dir / "selected_metrics.json"
        cmd_path = job_dir / "capture_command.json"
        ncu_command = [
            "ncu",
            "--csv",
            "--target-processes",
            "all",
            "--metrics",
            ",".join(metrics),
            "--log-file",
            str(csv_path),
            *first["resolved_run_command"],
        ]
        timed_out = False
        if dry_run:
            stdout = ""
            stderr = "dry-run capture skipped"
            exit_code = 0
            csv_path.write_text("")
            status, eligible, reason = "dry_run_capture_skipped", False, "dry_run"
        elif shutil.which("ncu") is None:
            stdout = ""
            stderr = "ncu executable not found"
            exit_code = None
            status, eligible, reason = "environment_blocked", False, "ncu_not_found"
        else:
            try:
                proc = subprocess.run(
                    ncu_command,
                    cwd=first["working_directory"],
                    text=True,
                    capture_output=True,
                    timeout=int(first.get("capture_timeout_seconds", 120)),
                    check=False,
                )
                stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr = exc.stderr if isinstance(exc.stderr, str) else ""
                exit_code = None
                timed_out = True
            status, eligible, reason = _classify(exit_code, stderr, csv_path, timed_out=timed_out)
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)
        exit_path.write_text("" if exit_code is None else str(exit_code))
        write_json(env_path, environment_signature())
        write_json(selected_path, selected_metric_records())
        write_json(cmd_path, {"ncu_capture_command": ncu_command})
        attempt = {
            "capture_job_id": job_id,
            "capture_job_index": index,
            "resolved_run_command_hash": stable_hash(first["resolved_run_command"]),
            "target_run_command": first["resolved_run_command"],
            "ncu_capture_command": ncu_command,
            "working_directory": first["working_directory"],
            "consuming_manifest_entry_ids": [row["manifest_entry_id"] for row in rows],
            "consuming_workload_ids": [row["workload_id"] for row in rows],
            "consuming_kernel_or_cases": [row["kernel_or_case"] for row in rows],
            "selected_metrics": metrics,
            "query_artifact_path": artifact_ref(QUERY_PATH),
            "query_artifact_hash": file_hash(QUERY_PATH),
            "resolution_table_path": artifact_ref(RESOLUTION_TABLE_PATH),
            "resolution_table_hash": file_hash(RESOLUTION_TABLE_PATH),
            "environment_manifest_path": artifact_ref(env_path),
            "capture_stdout_path": artifact_ref(stdout_path),
            "capture_stderr_path": artifact_ref(stderr_path),
            "capture_csv_path": artifact_ref(csv_path),
            "capture_exit_code_path": artifact_ref(exit_path),
            "capture_timeout_seconds": first.get("capture_timeout_seconds", 120),
            "capture_exit_code": exit_code,
            "capture_status": status,
            "gate3_eligible": eligible,
            "gap_reason": reason,
        }
        attempts.append(attempt)
        if not eligible:
            gaps.append({
                "capture_job_id": job_id,
                "consuming_manifest_entry_ids": attempt["consuming_manifest_entry_ids"],
                "capture_status": status,
                "gap_reason": reason,
            })
    write_json(ATTEMPTS_PATH, attempts)
    write_json(GAP_PATH, gaps)
    return attempts, gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    attempts, gaps = dispatch(dry_run=args.dry_run)
    print(f"Gate2 NCU dispatcher: {len(attempts)} attempts, {len(gaps)} gaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
