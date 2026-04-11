#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/dyf/modern-gpu-simulator-micro-2025
TRACER_TOOL=$PROJECT_ROOT/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.so
OUT_ROOT=$PROJECT_ROOT/experiments/gpt2_decode/results
MODEL=${MODEL:-gpt2}
CONTEXTS=${CONTEXTS:-"128 512 1024"}
RUNS=${RUNS:-"1 2 3"}

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate trace_gen

mkdir -p "$OUT_ROOT"

for CTX in $CONTEXTS; do
  for RUN in $RUNS; do
    OUT_DIR=$OUT_ROOT/model_${MODEL}_ctx${CTX}_gen1_run${RUN}
    mkdir -p "$OUT_DIR"

    export ACTIVE_FROM_START=0
    export USER_DEFINED_FOLDERS=1
    export TRACES_FOLDER=$OUT_DIR/traces
    export CUDA_INJECTION64_PATH=$TRACER_TOOL
    export LD_PRELOAD=$TRACER_TOOL

    python $PROJECT_ROOT/experiments/gpt2_decode/run_decode.py \
      --model "$MODEL" \
      --context-len "$CTX" \
      --gen-tokens 1 \
      2>&1 | tee "$OUT_DIR/run.log"

    unset LD_PRELOAD
    unset CUDA_INJECTION64_PATH
    unset TRACES_FOLDER
    unset USER_DEFINED_FOLDERS
    unset ACTIVE_FROM_START
  done
done

python $PROJECT_ROOT/experiments/gpt2_decode/summarize_runs.py \
  --results-root "$OUT_ROOT" \
  --output "$OUT_ROOT/summary.csv"
