#!/usr/bin/env python3
"""Build v1 frontend anchor outputs for the A-line pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .frontend_anchor.exporter import (
        build_case_note,
        build_comparison_table,
        export_anchor_table,
    )
    from .frontend_anchor.selector import run_selector
except ImportError:
    from frontend_anchor.exporter import (
        build_case_note,
        build_comparison_table,
        export_anchor_table,
    )
    from frontend_anchor.selector import run_selector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v1 frontend anchor outputs from mini_transformer-style inputs."
    )
    parser.add_argument("--identity-json", required=True, help="Path to identity/context JSON.")
    parser.add_argument("--features-json", required=True, help="Path to feature/weight JSON.")
    parser.add_argument("--squash-json", help="Optional squash summary JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for output artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from .frontend_anchor.invocation_table import build_records_from_dual_sources
    except ImportError:
        from frontend_anchor.invocation_table import build_records_from_dual_sources

    table = build_records_from_dual_sources(
        args.identity_json,
        args.features_json,
        args.squash_json,
    )
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
                "## Anchor generation",
                f"- Anchors are generated from an explicit dual-source CLI path using `{Path(args.identity_json).name}` as the identity/context source and `{Path(args.features_json).name}` as the feature/weight source.",
                "- `kernel_invocation_id` is synthetic in v1 and follows `<kernel_name>#<trace_order>`.",
                "- `member_invocations` are emitted as full lists in this v1 pass.",
                "",
                "## Field status",
                "- `coverage_weight`: derived from member counts.",
                "- `time_weight`: derived from exec_time, preferring `duration_ns` and falling back to `elapsed_cycles`.",
                "- `grid_dim` / `block_dim`: measured from the committed input data.",
                "- `kernel_squash_*` / `tb_squash_*`: derived from squash summaries and used as context/guardrail support.",
                "- `shape_hint_summary`: placeholder (`null`) in this v1 pass.",
                "",
                "## Output boundary",
                "- `Representative Anchor Table` is the mainline A-line output.",
                "- `Comparison Table` and `Case Note` are evidence-only outputs and must not be treated as downstream mainline input tables.",
                "",
                "## Bias sources",
                "- Current likely bias sources include synthetic invocation IDs, source-pair derivation choices, and still-lightweight squash guardrail integration.",
                "",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
