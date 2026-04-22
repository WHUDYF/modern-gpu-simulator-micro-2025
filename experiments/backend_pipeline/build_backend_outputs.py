#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.backend_pipeline.backend_builder import build_backend_outputs, load_full_features, write_backend_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build backend outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = build_backend_outputs(load_full_features(args.input))
    write_backend_outputs(outputs, args.output_dir)
    print(f"[backend-builder] wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
