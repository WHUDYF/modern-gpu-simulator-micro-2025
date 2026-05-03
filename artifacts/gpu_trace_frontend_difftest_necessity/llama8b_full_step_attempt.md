# Llama 3.1 8B Full-Step Validation Attempt

**Attempt**: 1
**Date**: 2026-05-04
**Status**: Abandoned

## Failure Details

| Field | Value |
|-------|-------|
| Failure Stage | Trace generation |
| Failure Reason | Llama 3.1 8B full training-step trace requires NVBit instrumentation of a full forward+backward pass on GPU with >= 48 GiB VRAM. No instrumented binary or pretraining harness available. Trace size estimate (~100 GiB) would approach the 500 GiB storage ceiling. |

## Required Artifacts Preserved

All required evidence artifacts remain intact:

- `workload_evidence_table.json`/`.md`
- `trace_to_sim_formula.json`/`.md`
- `complete_flow_burden_ratio.json`/`.md`
- `difftest_reduction_model.json`/`.md`
- `paper_argument_matrix.json`/`.md`
- `prototype_equivalence_report.json`/`.md`
- `resource_bound_config.json`/`.md`

## Notes

Per AC-10, this is a non-blocking tail task. The required evidence line (BERT-base full step, Llama 3.1 8B decoder layer slice) remains the primary deliverable. A documented failed attempt is acceptable.
