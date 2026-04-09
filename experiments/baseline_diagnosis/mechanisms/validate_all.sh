#!/usr/bin/env bash
set -euo pipefail

# Validate Phase 1: run all three mechanisms end-to-end on backprop.
# Assumes: backprop_4096_full.json exists in results/rodinia/.

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP="$ROOT/experiments/baseline_diagnosis"
RESULTS="$EXP/results/rodinia"
MECH="$RESULTS/backprop_mechanisms"
SCHEMAS="$EXP/schemas"

mkdir -p "$MECH"

echo "=== Step 1: Extract per-TB features ==="
python3 "$EXP/mechanisms/extract_per_tb_features.py" \
  --input "$RESULTS/backprop_4096_full.json" \
  --output "$MECH/backprop_4096_per_tb.json"

echo "=== Step 2: Run Squash ==="
python3 "$EXP/mechanisms/extract_squash_features.py" \
  --input "$MECH/backprop_4096_per_tb.json" \
  --config "$SCHEMAS/mechanism_config.json" \
  --output "$MECH/backprop_4096_squash.json"

echo "=== Step 3: Run Batch ==="
python3 "$EXP/mechanisms/extract_batch_features.py" \
  --input "$MECH/backprop_4096_per_tb.json" \
  --config "$SCHEMAS/mechanism_config.json" \
  --output "$MECH/backprop_4096_batch.json"

echo "=== Step 4: Run Delta ==="
python3 "$EXP/mechanisms/extract_delta_features.py" \
  --input "$MECH/backprop_4096_per_tb.json" \
  --config "$SCHEMAS/mechanism_config.json" \
  --output "$MECH/backprop_4096_delta.json"

echo "=== Step 5: Verify all outputs exist and are non-empty ==="
for f in per_tb squash batch delta; do
  path="$MECH/backprop_4096_${f}.json"
  if [ ! -s "$path" ]; then
    echo "FAIL: $path is missing or empty"
    exit 1
  fi
  echo "OK: $path ($(wc -c < "$path") bytes)"
done

echo "=== Step 6: Print key findings from each mechanism ==="
python3 <<PYEOF
import json
from pathlib import Path
mech = Path("$MECH")
for name in ["squash", "batch", "delta"]:
    path = mech / f"backprop_4096_{name}.json"
    d = json.loads(path.read_text())
    print(f"\n--- {name.upper()} ---")
    print(f"mechanism={d['mechanism']}")
    kl = d["kernel_level"]
    if name == "squash":
        print(f"  kernel_segments={len(kl['squash_segments'])}, boundaries={kl['boundary_count']}")
    elif name == "batch":
        print(f"  kernel_clusters={len(kl['batch_clusters'])}, outliers={len(kl['outlier_kernels'])}")
    elif name == "delta":
        print(f"  hot_fields={kl['hot_fields']}")
PYEOF

echo ""
echo "=== Phase 1 Validation PASSED ==="
