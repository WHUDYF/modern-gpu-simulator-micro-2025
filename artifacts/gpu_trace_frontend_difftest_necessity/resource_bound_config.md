# BERT-base Pretraining Full Step: Resource Bound Configuration

## Confirmed Resource Ceiling

| Resource | Limit | Rationale |
|----------|-------|-----------|
| Per-GPU memory | <= 28 GiB | RTX A6000 48 GiB total, conservative headroom after driver + framework |
| Trace + artifact size per unit | <= 500 GiB | Available storage budget per workload unit |
| Single complete iteration | <= 2 hours | Wall-clock time for export + simulate + analyze cycle |

## Batch-Scaling Strategy

1. Start from batch_size = 1.
2. Double batch size at each step.
3. Record resource usage at each step.
4. Stop when **any** resource ceiling is reached.
5. Record which limit stopped the run.

## Batch-Scaling Records

### Attempt 1: batch_size=1

| Field | Value | Status |
|-------|-------|--------|
| batch_size | 1 | attempted |
| per_gpu_memory_used_gib | N/A | blocked |
| trace_size_gib | N/A | blocked |
| artifact_size_gib | N/A | blocked |
| export_time_s | N/A | blocked |
| sim_time_s | N/A | blocked |
| analysis_time_s | N/A | blocked |
| total_iteration_time_s | N/A | blocked |
| stopped_by | trace_generation_unavailable | — |
| failure_reason | BERT-base pretraining forward+backward pass requires NVBit-instrumented PyTorch binary with GPU >= 28 GiB VRAM. No instrumented training harness available. | — |

### Status

First batch attempt (batch_size=1) could not proceed past trace generation. All downstream fields depend on successful trace export. Further scaling attempts require:
- NVBit-instrumented BERT pretraining harness
- GPU with >= 28 GiB VRAM (RTX A6000 or A100)
- Sufficient storage for trace artifacts (estimated 10s GiB per step at minimum batch)

No ceiling-based scaling evidence is available yet. The pipeline is ready for measured data when trace generation infrastructure becomes available.
