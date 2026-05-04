# Llama 3.1 8B Full-Step Validation Attempt

**Attempt**: 0
**Date**: 2026-05-04
**Status**: Not run — prerequisites incomplete

## Prerequisites

| Prerequisite | Status |
|-------------|--------|
| task-E1 (BERT-base pretraining full step) | Blocked before trace export; see `extension_workload_attempts.json` |
| task-E2 (Llama 3.1 8B decoder-layer slice) | Blocked before trace export; see `extension_workload_attempts.json` |
| task-D2 (honest final status summary) | In progress |

Per the current tracked plan, the optional Llama 3.1 8B full-step validation is attempted only after task-E1 and task-E2 are executed or formally settled.

## Infrastructure Requirements

- NVBit-instrumented Llama 3.1 8B pretraining harness
- GPU with >= 48 GiB VRAM (A6000 or A100)
- >= 500 GiB storage for trace + artifacts

## Required Artifacts Preserved

All required evidence artifacts remain intact under `artifacts/gpu_trace_frontend_difftest_necessity/`.

## Next Steps

Complete or unblock task-E1 and task-E2 with measured AI-training trace/replay data, then reconsider this optional tail attempt with a real Llama 3.1 8B full-step trace.
