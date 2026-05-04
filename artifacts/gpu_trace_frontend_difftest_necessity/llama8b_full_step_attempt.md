# Llama 3.1 8B Full-Step Validation Attempt

**Attempt**: 0
**Date**: 2026-05-05
**Status**: Not run — prerequisites incomplete

## Prerequisites

| Prerequisite | Status |
|-------------|--------|
| task-E1 (BERT-base pretraining full step) | Trace export passed; replay blocked on unsupported opcode `UF2I.FTZ.U32.TRUNC.NTZ`; see `bert_full_step_attempt.json` |
| task-E2 (Llama 3.1 8B decoder-layer slice) | Trace export and bounded replay passed; see `llama_decoder_layer_attempt.json` |
| task-D2 (honest final status summary) | In progress |

Per the current tracked plan, the optional Llama 3.1 8B full-step validation is attempted only after task-E1 and task-E2 are executed or formally settled.

## Infrastructure Requirements

- NVBit-instrumented Llama 3.1 8B pretraining harness
- GPU with >= 48 GiB VRAM (A6000 or A100)
- >= 500 GiB storage for trace + artifacts

## Required Artifacts Preserved

All required evidence artifacts remain intact under `artifacts/gpu_trace_frontend_difftest_necessity/`.

## Next Steps

Add simulator opcode support for the BERT full-step replay blocker, then reconsider this optional tail attempt with a real Llama 3.1 8B full-step trace.
