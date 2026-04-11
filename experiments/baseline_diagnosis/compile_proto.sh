#!/usr/bin/env bash
set -euo pipefail

PROTOC="/home/dyf/opt/protobuf-3.21.12/bin/protoc"
PROTO_SRC="/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/util/traces_enhanced/dynamic_trace"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/proto_gen"

mkdir -p "${OUT_DIR}"

"${PROTOC}" \
  --proto_path="${PROTO_SRC}" \
  --python_out="${OUT_DIR}" \
  "${PROTO_SRC}"/*.proto

# protoc generates bare imports (e.g. "import foo_pb2 as ...") which fail
# when proto_gen is used as a Python package. Convert to relative imports.
for f in "${OUT_DIR}"/*_pb2.py; do
  sed -i 's/^import \([a-z0-9_]*_pb2\) as/from . import \1 as/' "$f"
done

echo "Proto compilation complete. Generated files:"
ls -1 "${OUT_DIR}"/*_pb2.py 2>/dev/null || echo "No _pb2.py files found"
