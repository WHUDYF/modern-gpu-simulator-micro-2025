# BERT-base Pretraining Full Step: Resource Bound Configuration

## Confirmed Resource Ceiling

| Resource | Limit | Rationale |
|----------|-------|-----------|
| Per-GPU memory | <= 28 GiB | RTX A6000 48 GiB total, conservative headroom after driver + framework |
| Trace + artifact size per unit | <= 500 GiB | Available storage budget per workload unit |
| Single complete iteration | <= 2 hours | Wall-clock time for export + simulate + analyze cycle |

## Batch-Scaling Strategy

1. Start from a small batch size (1-4).
2. Double batch size at each step.
3. Record resource usage at each step.
4. Stop when **any** resource ceiling is reached.
5. Record which limit stopped the run.

## Per-Step Recording Fields

| Field | Unit | Description |
|-------|------|-------------|
| batch_size | int | Number of samples per batch |
| per_gpu_memory_used | GiB | Peak GPU memory during export + simulation |
| trace_size | GiB | Compressed trace file size |
| artifact_size | GiB | Generated artifact size |
| export_time | s | Trace generation/export wall time |
| sim_time | s | Simulator wall time |
| analysis_time | s | Result processing wall time |
| total_iteration_time | s | End-to-end wall time |
| stopped_by | string | Which limit stopped scaling, or "none" |

## Status

**Pending measurement.** Awaiting BERT-base pretraining full step trace generation and simulation runs.
