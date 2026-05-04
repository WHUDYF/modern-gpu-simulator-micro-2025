"""Build L1 kernel validation manifest JSON from the manifest document.

Parses the markdown table in docs/a-line-l1-validation-manifest-2026-04-26.md
and produces artifacts/a_line/l1/kernel_validation_manifest_l1.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DOC = REPO_ROOT / "docs" / "a-line-l1-validation-manifest-2026-04-26.md"
SCHEMA_PATH = REPO_ROOT / "experiments" / "baseline_diagnosis" / "schemas" / "kernel_validation_manifest_schema.json"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "a_line" / "l1" / "kernel_validation_manifest_l1.json"

SOURCE_TYPE_MAP = {
    "microbench": "local_microbench",
    "rodinia": "local_benchmark_result",
    "ai workload": "local_ai_workload",
    "ai_workload": "local_ai_workload",
}

# Per-source-type required top-level keys in the source file
SOURCE_REQUIRED_KEYS = {
    "local_microbench": ["enhanced_execution_info"],
    "local_benchmark_result": ["enhanced_execution_info"],
    "local_ai_workload": ["per_kernel"],
}


def _parse_markdown_table(md_text: str) -> list[dict[str, str]]:
    """Parse the L1 validation manifest markdown table.

    Standard markdown table format: | cell1 | cell2 | ... |
    """
    lines = md_text.split("\n")

    def _cells(line: str) -> list[str]:
        s = line.strip().strip("|")
        return [c.strip() for c in s.split("|")]

    # Find header line followed by a separator line (|---|...)
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = _cells(stripped)
            # Check if any cell contains only dashes (separator line)
            if all(c.replace("-", "").strip() == "" for c in cells if c.strip()):
                header_idx = i - 1
                break

    if header_idx is None or header_idx < 0:
        raise ValueError("No markdown table found in manifest document")

    headers = _cells(lines[header_idx].strip())
    if not headers:
        raise ValueError("Failed to parse table headers")

    rows = []
    for line in lines[header_idx + 2:]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        cells = _cells(stripped)
        if len(cells) < len(headers):
            continue
        # Build dict from header positions
        row = {}
        for idx, h in enumerate(headers):
            if idx < len(cells):
                row[h] = cells[idx]
        eid = row.get("ID", "").strip("`")
        if eid.startswith("L1_"):
            # Normalize keys
            row["id"] = eid
            row["来源"] = row.get("来源", row.get("来源 ", ""))
            row["对象"] = row.get("对象", "").strip("`")
            row["本地路径 / 来源路径"] = row.get("本地路径 / 来源路径", row.get("本地路径 / 来源路径 ", ""))
            row["优先级"] = row.get("优先级", "").strip("`")
            row["面向线路"] = row.get("面向线路", "").strip("`")
            row["预期行为轴"] = row.get("预期行为轴", "").strip("`")
            row["当前状态"] = row.get("当前状态", "").strip("`")
            rows.append(row)
    return rows


def _extract_path_from_md_link(cell: str) -> str:
    """Extract the path from a markdown link like [fname.json](/path/to/fname.json)."""
    m = re.search(r'\]\(([^)]+)\)', cell)
    if m:
        return m.group(1)
    # If not a link, use the cell text directly
    return cell.strip()


def _map_source_type(raw: str) -> str:
    raw_lower = raw.strip().lower()
    return SOURCE_TYPE_MAP.get(raw_lower, raw_lower)


def _map_target_line(raw: str) -> str:
    raw_upper = raw.strip().upper()
    if "A+B" in raw_upper or "A + B" in raw_upper:
        return "A+B"
    if raw_upper == "A":
        return "A"
    if raw_upper == "B":
        return "B"
    return "A+B"


def _map_status(raw: str) -> str:
    raw_lower = raw.strip().lower()
    if "ready_local" in raw_lower or "ready" in raw_lower:
        return "ready_local"
    if "need" in raw_lower and "profile" in raw_lower:
        return "needs_profile"
    if "need" in raw_lower and "acquisition" in raw_lower:
        return "needs_acquisition"
    if "block" in raw_lower:
        return "blocked"
    return "ready_local"


def _normalize_repo_relative_path(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        return path_text

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        pass

    parts = path.parts
    for marker in ("experiments", "artifacts", "docs"):
        if marker in parts:
            idx = parts.index(marker)
            return str(Path(*parts[idx:]))
    return path_text


def _build_entries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    entries = []
    for row in rows:
        entry_id = row["id"].strip()
        source_type = _map_source_type(row["来源"])
        local_path = _extract_path_from_md_link(row["本地路径 / 来源路径"])
        local_path = _normalize_repo_relative_path(local_path)

        entry = {
            "id": entry_id,
            "source_type": source_type,
            "benchmark_name": row["对象"].strip(),
            "kernel_or_case": row["对象"].strip(),
            "local_input_path": local_path,
            "priority": row["优先级"].strip(),
            "target_line": _map_target_line(row["面向线路"]),
            "expected_behavior_axis": row["预期行为轴"].strip(),
            "status": _map_status(row["当前状态"]),
        }
        entries.append(entry)
    return entries


def _validate_manifest(manifest: dict, schema: dict) -> list[str]:
    errors = []
    required_top = schema.get("required", [])
    for field in required_top:
        if field not in manifest:
            errors.append(f"manifest missing required top-level field: {field}")

    dataset_level = manifest.get("dataset_level")
    allowed_levels = schema.get("properties", {}).get("dataset_level", {}).get("enum", [])
    if dataset_level not in allowed_levels:
        errors.append(f"dataset_level '{dataset_level}' not in allowed values: {allowed_levels}")

    entry_schema = schema.get("$defs", {}).get("entry", {})
    entry_required = entry_schema.get("required", [])
    entry_props = entry_schema.get("properties", {})

    seen_ids = set()
    for idx, entry in enumerate(manifest.get("entries", [])):
        eid = entry.get("id")
        if eid in seen_ids:
            errors.append(f"entries[{idx}] duplicate id: {eid}")
        seen_ids.add(eid)

        for field in entry_required:
            if field not in entry:
                errors.append(f"entries[{idx}] ({eid}) missing required field: {field}")

        for field, allowed in [
            ("source_type", entry_props.get("source_type", {}).get("enum", [])),
            ("priority", entry_props.get("priority", {}).get("enum", [])),
            ("target_line", entry_props.get("target_line", {}).get("enum", [])),
            ("status", entry_props.get("status", {}).get("enum", [])),
        ]:
            val = entry.get(field)
            if val is not None and val not in allowed:
                errors.append(f"entries[{idx}] ({eid}) invalid {field}: {val}")

        validation_role = entry.get("validation_role")
        if validation_role is not None:
            allowed_vr = entry_props.get("validation_role", {}).get("enum", [])
            if allowed_vr and validation_role not in allowed_vr:
                errors.append(f"entries[{idx}] ({eid}) invalid validation_role: {validation_role}")

    return errors


def _check_paths_and_structure(entries: list[dict], repo_root: Path) -> list[str]:
    errors = []
    for entry in entries:
        eid = entry["id"]
        local_path = entry.get("local_input_path", "")
        source_type = entry.get("source_type", "")

        if not local_path:
            if source_type in SOURCE_REQUIRED_KEYS:
                errors.append(f"{eid}: local_input_path is empty but source_type={source_type} requires a local file")
            continue

        full_path = repo_root / local_path
        if not full_path.exists():
            errors.append(f"{eid}: local_input_path does not exist: {full_path}")
            continue

        # Source-type-specific structure validation
        if full_path.suffix == ".md":
            # Prescription file — accepted, no content validation
            continue

        if full_path.suffix == ".json":
            try:
                data = json.loads(full_path.read_text())
            except json.JSONDecodeError as exc:
                errors.append(f"{eid}: local_input_path is not valid JSON: {full_path} — {exc}")
                continue

            required_keys = SOURCE_REQUIRED_KEYS.get(source_type)
            if required_keys:
                missing_keys = [k for k in required_keys if k not in data]
                if missing_keys:
                    errors.append(
                        f"{eid}: source_type={source_type} requires top-level keys {missing_keys} "
                        f"in {full_path}"
                    )

    return errors


def main() -> int:
    md_text = MANIFEST_DOC.read_text()
    rows = _parse_markdown_table(md_text)
    if not rows:
        print("Error: no table entries found in manifest document")
        return 1

    entries = _build_entries(rows)
    manifest = {
        "manifest_name": "L1 Kernel Validation Manifest",
        "dataset_level": "L1",
        "goal": (
            "Functionality gate, feature sanity gate, and downstream interface gate "
            "for PKA baseline input, 12-D feature extraction, anchor output, and "
            "B-line consumption on a small set of interpretable kernels."
        ),
        "notes": (
            "Parsed from docs/a-line-l1-validation-manifest-2026-04-26.md. "
            "P0 entries block stage-gate; P1 entries are non-blocking."
        ),
        "entries": entries,
    }

    schema = json.loads(SCHEMA_PATH.read_text())
    schema_errors = _validate_manifest(manifest, schema)
    if schema_errors:
        print("Schema validation failed:")
        for err in schema_errors:
            print(f"  - {err}")
        return 1

    path_errors = _check_paths_and_structure(manifest["entries"], REPO_ROOT)
    blocking = [e for e in path_errors]
    if blocking:
        print("Path and structure validation failed:")
        for err in blocking:
            print(f"  - {err}")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    p0 = sum(1 for e in entries if e["priority"] == "P0")
    p1 = sum(1 for e in entries if e["priority"] == "P1")
    print(f"Manifest written: {OUTPUT_PATH}")
    print(f"  P0 entries: {p0}")
    print(f"  P1 entries: {p1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
