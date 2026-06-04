"""Shared deterministic artifact helpers for GCL Phase A."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def hash_without(value: dict[str, Any], *keys: str) -> str:
    payload = {key: item for key, item in value.items() if key not in set(keys)}
    return stable_hash(payload)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
