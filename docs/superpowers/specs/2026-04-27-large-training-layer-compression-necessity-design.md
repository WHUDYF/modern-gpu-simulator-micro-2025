# Large Training Layer Compression Necessity Design

Date: 2026-04-27

## 1. Purpose

This spec defines the first large-training-workload experiment for proving why representative compression is necessary before exact-cycle GPU simulation.

The experiment is not meant to train a full model, reproduce an MLPerf submission, or immediately run a complete simulator validation. Its job is to create a controlled but realistic scale proof:

```text
one real large-model training layer -> kernel timeline -> invocation scale -> compression opportunity -> projected simulation cost
```

The central claim is:

> For modern training workloads, even a single large model layer can produce enough kernel invocations, runtime, and trace volume that direct full-trace exact-cycle simulation is not a practical default path. Representative compression is therefore a prerequisite, not only an optimization.

## 2. Context

The current repository already has a working small-scale method chain:

```text
frontend anchor -> middle structure -> backend planning -> execution bridge -> result summary -> writeback
```

That chain has been exercised mostly on `mini_transformer_v4`, microbenchmarks, and classic benchmark kernels. Those inputs are useful for correctness gates, schema stability, and interface bring-up, but they are too small to prove the necessity of compression.

This experiment shifts the evidence target from "can the method run" to "why must the method exist for modern training workloads".

## 3. Experiment Unit

The first workload unit is one Llama-style decoder block training step:

```text
single decoder layer, random synthetic tokens/activations, forward + backward
```

The layer should include the main structures of a modern decoder-only Transformer block:

- RMSNorm or LayerNorm
- QKV projection
- attention score computation
- softmax
- attention value/context computation
- output projection
- MLP up/gate/down projections
- activation
- residual paths
- loss proxy
- backward pass

The first target shape is:

| Parameter | Value |
|---|---:|
| batch size | 1 |
| sequence length | 2048 |
| hidden size | 4096 |
| intermediate size | 14336 |
| attention heads | 32 |
| dtype | bf16 preferred, fp16 fallback |
| device | CUDA GPU |
| measured region | one forward + backward after warmup |

This shape is large enough to resemble an 8B-class decoder layer, but small enough to fit on a 32GB RTX 5090 without loading a full pretrained model.

## 4. Non-Goals

This experiment deliberately does not attempt to:

- load full Llama-8B or larger pretrained weights;
- run a full training step across all model layers;
- run a full dataset, epoch, or MLPerf benchmark;
- collect full NCU measured PKA features in the first pass;
- feed every generated kernel directly into exact-cycle simulation;
- prove final simulator accuracy.

Those are later stages. The first pass proves scale pressure and compression necessity.

## 5. Local Environment Assumptions

Current machine facts observed on 2026-04-27:

- GPU: two NVIDIA GeForce RTX 5090 devices, each with about 32GB memory.
- Tools available: `nsys`, `ncu`, and `nvcc`.
- Current base Python has no PyTorch.
- Current `trace_gen` environment has CPU-only PyTorch and `transformers`.
- NCU performance counters are currently blocked by `ERR_NVGPUCTRPERM`.

Therefore the first implementation should create or use a CUDA-enabled PyTorch environment and use Nsight Systems first. Nsight Compute measured feature collection is deferred until performance counter access is available.

## 6. Data Flow

The experiment data flow is:

```text
large layer harness
  -> nsys profile
  -> exported kernel timeline
  -> invocation table
  -> compression summary
  -> scale proof report
```

### 6.1 Harness Output

The harness should print a small machine-readable run summary:

- model unit name
- batch size
- sequence length
- hidden size
- intermediate size
- number of heads
- dtype
- warmup count
- profiled iteration count
- forward/backward wall time
- peak GPU memory if available

### 6.2 Timeline Output

The Nsight Systems output should be converted into a kernel invocation table with at least:

- invocation id
- kernel name
- start timestamp
- duration
- grid dimensions
- block dimensions
- stream id if available
- source profiler path

This table is the scale-proof equivalent of the existing small-workload invocation table.

### 6.3 Compression Output

The first compression summary can be name-and-shape based, because the first objective is scale pressure. It should report:

- total kernel invocations
- unique kernel names
- unique kernel name plus launch-shape groups
- top groups by runtime coverage
- representative count at several coverage thresholds
- compression ratio under each grouping policy

Once NCU counter access is available, the same invocation table can be extended into PKA measured feature records.

## 7. Output Artifacts

The first implementation should write artifacts under:

```text
experiments/large_training_layer/
```

Recommended files:

- `run_llama_layer_train.py`
- `run_nsys_layer.sh`
- `parse_nsys_kernels.py`
- `summarize_compression_scale.py`
- `results/llama_layer_b1_s2048_h4096/`

Recommended generated result files:

- `run_summary.json`
- `nsys_report.nsys-rep`
- `nsys_kernel_stats.csv`
- `kernel_invocation_table.json`
- `compression_scale_summary.json`
- `scale_proof_report.md`

## 8. Success Criteria

The experiment is successful if it produces a report containing:

| Evidence | Required Meaning |
|---|---|
| total kernel invocations | single-layer training is already non-trivial |
| unique kernel groups | there is visible heterogeneity and grouping structure |
| top runtime coverage | a small number of groups likely dominate execution time |
| trace/report size | full multi-layer tracing would scale poorly |
| forward/backward wall time | exact-cycle simulation budget can be projected |
| compressed representative count | representative compression reduces backend candidate count |

The first pass does not need to show final simulator speedup. It needs to show that the uncompressed input scale makes a direct backend path unreasonable.

## 9. Risk Controls

### 9.1 Trace Too Large

If the first `seq_len=2048` run is too large, reduce sequence length to `1024` while keeping the same hidden size. The report must record the fallback shape.

### 9.2 Out of Memory

If `bf16/fp16` training with backward does not fit, use gradient checkpointing or reduce sequence length before reducing hidden size. The goal is to preserve large-layer structure.

### 9.3 NCU Counter Permission

Do not block the first experiment on NCU. Use `nsys` for timeline and scale evidence first. Mark PKA measured features as a follow-up acquisition step.

### 9.4 Synthetic Data Criticism

The experiment uses synthetic activations/tokens but real layer computation. The report must state this clearly: the proof target is GPU kernel scale and trace burden for a real training layer shape, not model quality or dataset convergence.

## 10. Follow-Up Path

After the first report exists, the next stages are:

1. Add measured PKA features for representative kernel groups when NCU permissions allow.
2. Compare compression summaries across `seq_len=1024`, `2048`, and possibly `4096`.
3. Add one non-Transformer training layer, such as DLRM-style embedding/MLP or RetinaNet-style vision training, to test whether compression difficulty changes across workload families.
4. Feed only selected representatives into the backend planning path.

## 11. Acceptance Gate

Before implementation starts, this spec should be reviewed for:

- whether the single-layer Llama-style target is the right first workload;
- whether `seq_len=2048, hidden=4096` is the correct first shape;
- whether `nsys`-first evidence is sufficient for the first scale proof;
- whether the output artifacts are enough to connect back to A-line frontend compression.
