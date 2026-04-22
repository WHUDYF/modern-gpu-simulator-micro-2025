#!/usr/bin/env python3
"""Build a v1 KernelInvocationRecord table for frontend anchor work."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontend_anchor.invocation_table import build_records_from_full_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v1 frontend invocation table from repository inputs."
    )
    parser.add_argument(
        "--full-json",
        help="Path to premerged per-invocation full JSON (v1 shortcut input).",
    )
    parser.add_argument(
        "--squash-json",
        help="Optional squash mechanism JSON for context/guardrail summaries.",
    )
    parser.add_argument("--output", required=True, help="Path to output JSON table.")
    args = parser.parse_args()
    if not args.full_json:
        parser.error("v1 currently requires --full-json")
    return args


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.full_json):
        print(f"ERROR: full_json not found: {args.full_json}", file=sys.stderr)
        return 1
    if args.squash_json and not os.path.isfile(args.squash_json):
        print(f"ERROR: squash_json not found: {args.squash_json}", file=sys.stderr)
        return 1

    try:
        table = build_records_from_full_json(args.full_json, args.squash_json)
    except Exception as exc:
        print(f"ERROR: failed to build invocation table: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(table, indent=2))
    print(
        f"Wrote {output_path} with {len(table['records'])} KernelInvocationRecord rows "
        f"(mode={table['source_mode']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

