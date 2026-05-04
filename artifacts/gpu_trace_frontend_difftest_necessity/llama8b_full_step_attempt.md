# Llama 3.1 8B Full-Step Validation Attempt

**Attempt**: 1
**Date**: 2026-05-05
**Status**: Blocked by local GPU memory

## Prerequisites

| Prerequisite | Status |
|-------------|--------|
| task-E1 (BERT-base pretraining full step) | Trace export and bounded replay passed; see `bert_full_step_attempt.json` |
| task-E2 (Llama 3.1 8B decoder-layer slice) | Trace export and bounded replay passed; see `llama_decoder_layer_attempt.json` |
| task-D2 (honest final status summary) | In progress |

Per the current tracked plan, the optional Llama 3.1 8B full-step validation is attempted only after task-E1 and task-E2 are settled. Round 4 ran the smallest local validation attempt:

```bash
CUDA_VISIBLE_DEVICES=0 LLAMA_FULL_TRACE_BATCH=1 LLAMA_FULL_TRACE_SEQ_LEN=8 python3 docs/llama-full-step-training.py
```

The attempt failed during fp16 model placement with `torch.OutOfMemoryError`. The local RTX 5090 reports 32607 MiB total memory; PyTorch reported 30.40 GiB already in use by the process and failed a further 1002 MiB allocation.

## Infrastructure Requirements

- NVBit-instrumented Llama 3.1 8B pretraining harness
- GPU with >= 48 GiB VRAM (A6000 or A100)
- >= 500 GiB storage for trace + artifacts

## Required Artifacts Preserved

All required evidence artifacts remain intact under `artifacts/gpu_trace_frontend_difftest_necessity/`.

## Next Steps

Run this optional tail attempt on a host with materially more GPU memory, or request explicit user approval for a reduced/non-8B full-step surrogate.
