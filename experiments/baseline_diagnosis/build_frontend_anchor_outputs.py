#!/usr/bin/env python3
"""Build v1 frontend anchor outputs for the A-line pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontend_anchor.exporter import (
    build_case_note,
    build_comparison_table,
    export_anchor_table,
)
from frontend_anchor.invocation_table import build_records_from_full_json
from frontend_anchor.selector import run_selector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v1 frontend anchor outputs from mini_transformer-style inputs."
    )
    parser.add_argument("--full-json", required=True, help="Path to v1 full.json input.")
    parser.add_argument("--squash-json", help="Optional squash summary JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for output artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = build_records_from_full_json(args.full_json, args.squash_json)
    records = table["records"]

    by_method = {
        method: run_selector(records, method)
        for method in ["name-only", "pka-like-coarse", "hybrid"]
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "kernel_invocation_table_v1.json").write_text(json.dumps(table, indent=2))
    anchor_table = export_anchor_table(by_method["hybrid"])
    (output_dir / "representative_anchor_table_v1.json").write_text(json.dumps(anchor_table, indent=2))
    comparison_table = build_comparison_table(by_method)
    (output_dir / "comparison_table_v1.json").write_text(json.dumps(comparison_table, indent=2))
    (output_dir / "case_note_v1.md").write_text(build_case_note(by_method))
    (output_dir / "frontend_compression_note_v1.md").write_text(
        "\n".join(
            [
                "# Frontend Compression Note",
                "",
                "- Anchors are generated from a v1 full.json shortcut input and synthetic kernel_invocation_id values.",
                "- member_invocations are emitted as full lists in this v1 pass.",
                "- coverage_weight is derived from member counts.",
                "- time_weight is derived from exec_time (preferring duration_ns, then elapsed_cycles).",
                "- Current likely bias sources include premerged input shortcuts, synthetic invocation IDs, and provisional squash summaries.",
                "",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

