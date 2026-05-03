# Llama 3.1 8B Full-Step Validation Attempt

**Attempt**: 0
**Date**: 2026-05-04
**Status**: Not run — prerequisites incomplete

## Prerequisites

| Prerequisite | Status |
|-------------|--------|
| task10 (central evidence table with measured data) | Incomplete |
| task13 (BERT batch-scaling records) | Incomplete |

Per AC-10, the optional Llama 3.1 8B full-step validation is attempted only after the required evidence line is complete.

## Infrastructure Requirements

- NVBit-instrumented Llama 3.1 8B pretraining harness
- GPU with >= 48 GiB VRAM (A6000 or A100)
- >= 500 GiB storage for trace + artifacts

## Required Artifacts Preserved

All required evidence artifacts remain intact under `artifacts/gpu_trace_frontend_difftest_necessity/`.

## Next Steps

Complete task10 and task13 with measured AI-training data, then rerun this tail attempt with a real Llama 3.1 8B full-step trace.
