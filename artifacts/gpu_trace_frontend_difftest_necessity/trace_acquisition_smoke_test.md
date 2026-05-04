# BERT/Llama Trace Acquisition Smoke Test

**Date**: 2026-05-04
**Gate**: Gate A — Trace Acquisition Ready
**Status**: BLOCKED

## Test Result

| Field | Value |
|-------|-------|
| Workload | bert-base-encoder-slice |
| Export attempt | Yes |
| Status | Failed |
| Failure stage | NVBit GPU architecture support |

## Failure Details

NVBit v1.7.6 does not support the available GPU hardware:

- GPU: NVIDIA GeForce RTX 5090 (Blackwell, compute capability 12.0)
- NVBit: v1.7.6 (likely supports up to Hopper sm_90)
- Error: `unsupported binary version: 0`

The NVBit `get_sm_family()` function does not recognize sm_120 (Blackwell).

## Environment

| Component | Version |
|-----------|---------|
| CUDA Driver | 13.0 |
| CUDA Toolkit | 12.8 |
| NVBit | 1.7.6 |
| GPU | RTX 5090 (32 GiB) |
| PyTorch | 2.11.0+cu130 |
| Transformers | Available |

## Prerequisites Check

| Prerequisite | Status |
|-------------|--------|
| NVBit harness exists | Yes |
| BERT inference script exists | Yes |
| PyTorch CUDA available | Yes |
| GPU accessible | Yes |
| NVBit GPU arch supported | No |

## Resolution

- Upgrade NVBit to a Blackwell-compatible version
- Or use a Hopper (sm_90) or earlier GPU
