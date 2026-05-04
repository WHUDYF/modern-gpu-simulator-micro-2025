# BERT/Llama Trace Acquisition Smoke Test

**Date**: 2026-05-04
**Gate**: Gate A — Trace Acquisition Ready
**Status**: BLOCKED

## Verified Finding

NVBit v1.7.6 loads correctly (banner confirmed) but its kernel-launch callback never writes trace files. This affects both compiled CUDA binaries (l1_bw_32f) and PyTorch (BERT-base).

## Control Experiment

The exact flow documented in `docs/5090-trace-to-sim.md` (which previously produced traces on 2026-04-12) was reproduced 3 times:

| Run | NVBit Loaded | Binary Ran | Trace Output |
|-----|-------------|-----------|-------------|
| 1 | Yes | Yes (63.81 byte/clk/SM) | Empty |
| 2 | Yes | Yes | Empty |
| 3 | Yes (default path) | Yes | Empty |

## Root Cause Location

`tracer_tool.cu:1006-1024` — the kernel-launch callback that opens `stats.csv` and sets up trace output never executes. The printf at line 1015 ("Traces location is %s") never appears in any output stream.

## Existing Traces Available

- `manual-l1_bw_32f-5090-fixed` (2026-04-12): Usable for control validation
- Rodinia `nn` (exampleTraces): Previously used for instrumentation validation
- BERT/Llama: Not yet generated — blocked by NVBit issue

## Resolution

Debug NVBit kernel-launch callback on driver 580.105.08, update NVBit, or roll back driver.
