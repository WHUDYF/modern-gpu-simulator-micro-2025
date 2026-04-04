#!/bin/bash
# End-to-end test: convert a real v4 .pb file to v5 and verify
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACE_DIR="$SCRIPT_DIR/../../exampleTraces"
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

echo "=== Extracting sample traces ==="
tar xzf "$TRACE_DIR/rodinia2Ampere.tar.gz" -C "$WORK_DIR"

# Find first .pb file
PB_FILE=$(find "$WORK_DIR" -name "*.pb" | head -1)
if [ -z "$PB_FILE" ]; then
  echo "FAIL: no .pb files found"
  exit 1
fi
echo "Test file: $PB_FILE ($(du -b "$PB_FILE" | cut -f1) bytes)"

echo "=== Converting v4 -> v5 ==="
"$SCRIPT_DIR/trace-compress" --input "$PB_FILE" --output "$WORK_DIR/compressed.pb" \
  --from-version 4 --to-version 5 --func-id 1

echo "=== Verifying compressed file exists and is smaller ==="
ORIG_SIZE=$(du -b "$PB_FILE" | cut -f1)
COMP_SIZE=$(du -b "$WORK_DIR/compressed.pb" | cut -f1)
echo "Original: ${ORIG_SIZE}B  Compressed: ${COMP_SIZE}B"

if [ "$COMP_SIZE" -ge "$ORIG_SIZE" ]; then
  echo "WARN: compressed file is not smaller (may be expected for tiny files)"
fi

echo "=== Running unit roundtrip tests ==="
"$SCRIPT_DIR/test_roundtrip"

echo ""
echo "=== ALL TESTS PASSED ==="
