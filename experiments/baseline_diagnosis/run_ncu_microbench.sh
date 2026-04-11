#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: bash $0 <binary> <output_csv>"
    exit 1
fi

BINARY="$1"
OUTPUT="$2"

if [[ ! -x "$BINARY" ]]; then
    echo "Error: $BINARY is not executable or does not exist"
    exit 1
fi

ncu --set full --csv --target-processes all "$BINARY" > "$OUTPUT"

echo "Profiling complete. Output written to $OUTPUT"
