# BERT/Llama Trace Acquisition Smoke Test

**Date**: 2026-05-04  
**Gate**: Gate A - Trace Acquisition Ready  
**Status**: PASS  
**Data label**: measured

## Verified Finding

The earlier diagnosis that the NVBit kernel-launch callback never triggers is obsolete. Fresh runs show the callback is reached and trace artifacts are produced. The real blockers were:

- `cuobjdump` was invoked after `cd traces/extra_info/cubin` while still using a relative executable path.
- `USER_DEFINED_FOLDERS=1` updated `stats.csv` paths but not `dynamic_trace.pb`, threadblock traces, or enhanced metadata paths.
- GCC 8 requires `-lstdc++fs` for binaries using `std::filesystem`.
- Python/PyTorch workloads can report `binary_version=0` and the Python executable itself contains no device code, so BERT needs device-version fallback plus no-binary enhanced trace fallback.

## Control Experiment

| Workload | Exit | Trace path | Size | Files | Kernel rows |
|---|---:|---|---:|---:|---:|
| `l1_bw_32f` | 0 | `/tmp/tracer_tool_repro/traces` | 3,105,111 bytes | 4 | 1 |

Required files were present:

- `dynamic_trace.pb`
- `stats.csv`
- `extra_info/enhanced_execution_info.json`
- `threadblocks/device_0/stream_0/kernel_1/...pb`

## Claim-Bearing BERT Slice

| Workload | Exit | Trace path | Size | Files | Threadblock PBs | Kernel rows |
|---|---:|---|---:|---:|---:|---:|
| `bert-base-encoder-layer-slice` | 0 | `/tmp/bert_trace_repro/traces` | 4,680,065,040 bytes | 19,942 | 19,939 | 77 |

The BERT harness ran a training-style encoder layer slice: `BertLayer` forward pass plus `loss.backward()` on random CUDA input. Output included:

```text
Training loss: 0.0001
Kernel binary version is 0; falling back to current device sm_120
Enhanced tracer has parsed 28/28 kernels
```

For Python/PyTorch, the traced command must include:

```bash
export ALLOW_CUOBJDUMP_NO_DEVICE_CODE=1
```

This allows enhanced tracing to continue when `cuobjdump` reports that `/usr/bin/python3.11` has no embedded CUDA device code. The enhanced trace then uses the no-binary fallback from NVBit-captured SASS.

## Fix Applied

Changed local tracer sources under:

- `/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu`
- `/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/Makefile`

Main changes:

- Resolve executable path via `/proc/self/exe`, with backtrace as fallback.
- Normalize executable path before `cuobjdump`.
- Propagate `TRACES_FOLDER` to all trace outputs, not only `stats.csv`.
- Link `tracer_tool.so` and `trace_printer` with `-lstdc++fs`.
- Fallback `binary_version=0` to the current device compute capability.
- Add `ALLOW_CUOBJDUMP_NO_DEVICE_CODE=1` for interpreter workloads that need no-binary enhanced trace fallback.

## Backup Check

Checked the local backup worktree:

```text
/home/dyf/modern-gpu-simulator-micro-2025/.worktrees/difftest-doc
```

Its `tracer_tool.cu` and `Makefile` matched the old broken implementation and did not contain a ready-made fix.

## Remaining Scope

Gate A is now satisfied for trace acquisition. This artifact does not claim simulator replay, frontend timing breakdown, redundancy profile, or burden-ratio completion. Those belong to the next gates.
