from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    label: str
    role: str
    source_path: Path
    record_pointer: str


@dataclass(frozen=True)
class Catalog:
    catalog_id: str
    description: str
    entries: list[CatalogEntry]


def load_catalog(path: Path) -> Catalog:
    path = Path(path)
    data = json.loads(path.read_text())
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"catalog must contain a non-empty entries list: {path}")
    _validate_unique_entry_ids(entries, path)

    return Catalog(
        catalog_id=str(data["catalog_id"]),
        description=str(data.get("description", "")),
        entries=[
            CatalogEntry(
                id=str(entry["id"]),
                label=str(entry["label"]),
                role=str(entry["role"]),
                source_path=_resolve_source_path(str(entry["source_path"]), path.parent),
                record_pointer=str(entry["record_pointer"]),
            )
            for entry in entries
        ],
    )


def _validate_unique_entry_ids(entries: list[dict[str, Any]], path: Path) -> None:
    seen: set[str] = set()
    for entry in entries:
        entry_id = str(entry["id"])
        if entry_id in seen:
            raise ValueError(f"duplicate catalog entry id in {path}: {entry_id}")
        seen.add(entry_id)


def load_catalog_records(catalog: Catalog) -> dict[str, Any]:
    return {
        entry.id: _read_json_pointer(json.loads(entry.source_path.read_text()), entry.record_pointer)
        for entry in catalog.entries
    }


def _resolve_source_path(source_path: str, root: Path) -> Path:
    path = Path(source_path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _read_json_pointer(data: Any, pointer: str) -> Any:
    if pointer == "":
        return data
    if not pointer.startswith("/"):
        raise ValueError(f"record_pointer must be an absolute JSON pointer: {pointer}")
    current = data
    for raw_part in pointer.lstrip("/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise ValueError(f"record_pointer descends into non-container value: {pointer}")
    return current
