# GPU Trace Frontend Necessity: Workload Evidence Catalog

## Overview

This catalog defines the workload evidence line for the trace-driven GPU simulator frontend necessity study. It follows the confirmed study scope: T1 baselines (BERT-base slice and pretraining full step), T2 representative (Llama 3.1 8B decoder layer slice), and T2 nice-to-have (Llama 3.1 8B full step).

Microbenchmark and HPC cases from the existing bottleneck map are listed as **controls** or **appendix** material, not as primary AI-training evidence. GPT-2 small is explicitly excluded from the evidence line as too small for meaningful signal.

## Go/No-Go Rule

- `P_trace_to_sim = T_trace_to_sim / T_kernel_to_sim_done`
- `T_kernel_to_sim_done = T_kernel_or_trace_export + T_trace_to_sim + T_sim_backend_execution + T_result_analysis`
- **Gate**: `P_trace_to_sim_slice > 15%` OR `P_trace_to_sim_step > 15%` → proceed to prototype investigation
- This is an early-stage engineering gate, not a final paper claim threshold.

## Claim-Bearing Workloads

### T1 Baseline: BERT-base Encoder Layer Slice

| Field | Value |
|-------|-------|
| Workload ID | `bert-base-encoder-layer-slice` |
| Model | BERT-base (~110M parameters) |
| Slice Type | Single encoder layer (~7M parameters) |
| Trace Granularity | Instruction-level (NVBit) |
| Expected Trace Size Tier | Medium-to-Large |
| Measurement Unit | Slice |
| Role in Argument | T1 baseline — primary claim-bearing at slice granularity |
| Status | Measured |

A single transformer encoder layer captures representative compute (matmul, softmax, layer norm) and memory (attention, FFN) patterns at a granularity suitable for iterative measurement.

### T1 Baseline: BERT-base Pretraining Full Step

| Field | Value |
|-------|-------|
| Workload ID | `bert-base-pretraining-full-step` |
| Model | BERT-base (~110M parameters) |
| Slice Type | Full forward+backward pass |
| Trace Granularity | Instruction-level (NVBit) |
| Expected Trace Size Tier | Large-to-Very-Large |
| Measurement Unit | Step |
| Role in Argument | T1 baseline — primary claim-bearing at step granularity |
| Status | Measured |
| Batch Scaling | Start small, scale upward until resource ceiling |

Resource ceiling:
- Per-GPU memory: <= 28 GiB
- Trace + artifact size per workload unit: <= 500 GiB
- Single complete iteration time: <= 2 hours

### T2 Representative: Llama 3.1 8B Decoder Layer Slice

| Field | Value |
|-------|-------|
| Workload ID | `llama3.1-8b-decoder-layer-slice` |
| Model | Llama 3.1 8B |
| Slice Type | Single decoder layer (~200M params/layer, grouped-query attention, SwiGLU FFN) |
| Trace Granularity | Instruction-level (NVBit) |
| Expected Trace Size Tier | Large |
| Measurement Unit | Slice |
| Role in Argument | T2 representative — industrial-scale decoder architecture |
| Status | Measured or Modeled |

Represents modern decoder-only architecture. Larger per-layer trace than BERT due to grouped-query attention and wider SwiGLU feed-forward network.

### T2 Nice-to-Have: Llama 3.1 8B Full Step

| Field | Value |
|-------|-------|
| Workload ID | `llama3.1-8b-full-step` |
| Model | Llama 3.1 8B |
| Slice Type | Full forward+backward pass |
| Trace Granularity | Instruction-level (NVBit) |
| Expected Trace Size Tier | Very-Large-to-Extreme |
| Measurement Unit | Step |
| Role in Argument | T2 nice-to-have — RLCR tail attempt only |
| Status | Modeled or Attempted |

Non-blocking. Only attempted after required evidence line is complete. Repeated failures do not invalidate completed required artifacts.

## Control Workloads

### GPU Microbenchmark Suite (Control)

Source: `artifacts/trace_bottleneck_map/benchmark_cost_map.json`

Representative cases: MaxFlops, atomic_add_bw, atomic_add_bw_conflict, l1_bw_128, l1_lat, l2_bw_32f, l1_shared_bw, mem_bw, shared_bw, shared_lat

These measured microbenchmarks serve as **controls** for trace-size, export-time, and sim-time baselines. They are not used as primary AI-training evidence because they lack the static metadata reuse and threadblock diversity that makes AI-training frontend cost interesting.

### HPC Benchmarks (Appendix)

Source: Existing bottleneck map estimates for BabelStream, Rodinia (nn, backprop, bfs, lud, nw), Parboil (sgemm, stencil, cutcp, mri-q, histo, bfs), PolyBench/GPU (gemm, 3mm, 3DConvolution, atax, bicg, syrk)

HPC benchmarks provide diversity in trace access patterns but are not AI-training workloads. Listed as appendix material.

### MLPerf Inference/Training (Scale Anchor)

Source: MLPerf workload estimates from bottleneck map.

Full MLPerf workloads (BERT, ResNet, DLRM, Llama2, Mixtral) are scale anchors at infeasible simulation cost. Used only to contextualize the gap between measured workloads and production scale. Not part of claim-bearing evidence.

## Explicitly Excluded

| Candidate | Reason |
|-----------|--------|
| GPT-2 small | Too small for meaningful `P_trace_to_sim` signal at AI-training scale |
| NCCL-tests | Multi-GPU communication class — different problem domain |
| OSU micro-benchmarks | MPI/network class — irrelevant to GPU trace-driven frontend study |

## Measurement Feasibility

| Workload | Expected Trace Size | Single-Run Feasibility | Sweep Feasibility |
|----------|-------------------|----------------------|-------------------|
| BERT-base encoder layer slice | 10s-100s MiB | Feasible (seconds) | Feasible (minutes to 10s of minutes) |
| BERT-base pretraining full step | 100s MiB - 10s GiB | Feasible (seconds to minutes) | Feasible with batching (10s of minutes to ~1h) |
| Llama 3.1 8B decoder layer slice | 100s MiB - GiB | Feasible (seconds to minutes) | Feasible with care (10s of minutes) |
| Llama 3.1 8B full step | 10s - 100s GiB | Challenging | Modeled only in first round |
