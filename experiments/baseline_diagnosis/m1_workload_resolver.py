"""Gate 1 workload resolver for the PKA-M1 measured loop."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from shared_acquisition import ARTIFACT_DIR, REPO_ROOT, artifact_ref, repo_path, sanitize_token, write_json

MANIFEST_PATH = ARTIFACT_DIR / "kernel_validation_manifest_l1.json"
REGISTRY_PATH = REPO_ROOT / "experiments" / "baseline_diagnosis" / "workload_registry_l1.json"
RESOLUTION_PATH = ARTIFACT_DIR / "m1_workload_resolution_l1.json"
GAP_PATH = ARTIFACT_DIR / "m1_workload_resolution_gap_l1.json"
SMOKE_DIR = REPO_ROOT / "experiments" / "baseline_diagnosis" / "results" / "m1_smoke"


def _registry_key(entry: dict[str, Any]) -> str:
    if entry.get("source_type") == "local_ai_workload":
        return "mini_transformer_v4"
    if entry.get("benchmark_name") == "nn":
        return "rodinia_nn"
    return str(entry.get("benchmark_name") or entry.get("kernel_or_case"))


def _format_command(template: list[str], binary_path: Path, args: list[str]) -> list[str]:
    rendered = [part.format(binary_path=str(binary_path), args=" ".join(args)) for part in template]
    expanded: list[str] = []
    for part in rendered:
        if part == "{args}":
            expanded.extend(args)
        elif "{args}" in part:
            expanded.extend(part.split())
        else:
            expanded.append(part)
    return expanded


def _smoke(command: list[str], cwd: Path, timeout: int, token: str, expected_output_regex: str | None = None) -> dict[str, Any]:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = SMOKE_DIR / f"{token}.stdout.log"
    stderr_path = SMOKE_DIR / f"{token}.stderr.log"
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout_path.write_text(proc.stdout[-4000:])
        stderr_path.write_text(proc.stderr[-4000:])
        combined_output = f"{proc.stdout}\n{proc.stderr}"
        regex_ok = expected_output_regex is None or re.search(expected_output_regex, combined_output) is not None
        status = "passed" if proc.returncode == 0 and regex_ok else "failed"
        reason = None
        if proc.returncode != 0:
            reason = "non_zero_exit"
        elif not regex_ok:
            reason = "expected_output_regex_mismatch"
        code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text((exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "")
        stderr_path.write_text((exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "")
        status = "failed"
        reason = "timeout"
        code = None
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return {
        "enabled": True,
        "command": command,
        "timeout_seconds": timeout,
        "exit_code": code,
        "elapsed_ms": elapsed_ms,
        "stdout_tail_path": artifact_ref(stdout_path),
        "stderr_tail_path": artifact_ref(stderr_path),
        "expected_output_regex": expected_output_regex,
        "status": status,
        "failure_reason": reason,
    }


def resolve(dry_run_smoke: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    registry_rows = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else []
    registry = {row["workload_id"]: row for row in registry_rows}
    resolutions: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []

    for entry in [row for row in manifest.get("entries", []) if row.get("priority") == "P0"]:
        key = _registry_key(entry)
        reg = registry.get(key)
        base = {
            "manifest_entry_id": entry["id"],
            "workload_id": key,
            "benchmark_name": entry.get("benchmark_name"),
            "kernel_or_case": entry.get("kernel_or_case"),
            "source_type": entry.get("source_type"),
            "registry_entry_id": key if reg else None,
            "dispatcher_path": None,
            "dispatcher_arg": None,
        }
        if not reg:
            gaps.append({**base, "resolution_status": "gap", "gap_reason": "registry_missing"})
            continue
        binary_path = repo_path(reg["binary_path"])
        cwd = repo_path(reg.get("working_directory", "."))
        build_attempted = False
        build_status = "not_needed"
        if not binary_path.exists() and reg.get("build_command"):
            build_attempted = True
            build_cmd = [str(x) for x in reg["build_command"]]
            build = subprocess.run(build_cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
            build_status = "passed" if build.returncode == 0 else "failed"
        binary_exists = binary_path.exists()
        binary_executable = binary_path.is_file() and binary_path.stat().st_mode & 0o111 != 0
        if not binary_exists:
            gaps.append({**base, "resolution_status": "gap", "gap_reason": "binary_missing", "resolved_binary_path": str(binary_path)})
            continue
        if not binary_executable:
            gaps.append({**base, "resolution_status": "gap", "gap_reason": "binary_not_executable", "resolved_binary_path": str(binary_path)})
            continue
        if not cwd.exists():
            gaps.append({**base, "resolution_status": "gap", "gap_reason": "working_directory_missing", "working_directory": str(cwd)})
            continue
        run_args = [str(x) for x in reg.get("run_args", [])]
        smoke_args = [str(x) for x in reg.get("smoke_args", [])]
        command = _format_command(reg.get("run_command_template", ["{binary_path}"]), binary_path, run_args)
        smoke_command = _format_command(reg.get("run_command_template", ["{binary_path}"]), binary_path, run_args + smoke_args)
        token = f"{entry['id']}_{sanitize_token(key)}"
        if dry_run_smoke:
            smoke = {
                "enabled": True,
                "command": smoke_command,
                "timeout_seconds": reg.get("smoke_timeout_seconds", 10),
                "exit_code": 0,
                "elapsed_ms": 0,
                "stdout_tail_path": None,
                "stderr_tail_path": None,
                "expected_output_regex": reg.get("expected_output_regex"),
                "status": "passed",
                "failure_reason": None,
                "dry_run": True,
            }
        else:
            smoke = _smoke(
                smoke_command,
                cwd,
                int(reg.get("smoke_timeout_seconds", 10)),
                token,
                reg.get("expected_output_regex"),
            )
        if smoke["status"] != "passed":
            gaps.append({**base, "resolution_status": "gap", "gap_reason": f"smoke_{smoke['failure_reason']}", "smoke_run": smoke})
            continue
        resolutions.append({
            **base,
            "resolved_binary_path": str(binary_path),
            "resolved_run_args": run_args,
            "resolved_run_command": command,
            "working_directory": str(cwd),
            "build_command": reg.get("build_command"),
            "build_attempted": build_attempted,
            "build_status": build_status,
            "binary_exists": binary_exists,
            "binary_executable": bool(binary_executable),
            "capture_timeout_seconds": reg.get("capture_timeout_seconds", 120),
            "smoke_run": smoke,
            "resolution_status": "resolved",
            "gap_reason": None,
        })
    return resolutions, gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run-smoke", action="store_true")
    args = parser.parse_args(argv)
    resolutions, gaps = resolve(dry_run_smoke=args.dry_run_smoke)
    write_json(RESOLUTION_PATH, resolutions)
    write_json(GAP_PATH, gaps)
    print(f"Gate1 workload resolver: {len(resolutions)} resolved, {len(gaps)} gaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
