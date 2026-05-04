# BERT-base Pretraining Full Step Resource Bounds

Generated: 2026-05-05

## Confirmed Ceiling

| Resource | Limit |
|----------|------:|
| Per-GPU memory | 28 GiB |
| Trace + artifacts per workload unit | 500 GiB |
| Single complete iteration | 2 hours |

## Measured Batch-1 Complete Iteration

| Field | Value |
|-------|------:|
| Peak GPU memory | 1.0183 GiB |
| Trace size | 18.6277 GiB |
| Export time | 2805.5586 s |
| Simulator wall time | 7.1908 s |
| Analysis time | 0.25 s |
| Total iteration | 2813.0 s |

Evidence: `bert_full_step_attempt.json`, `frontend_timing_breakdown_bert-base-pretraining-full-step.json`, and `redundancy_profile_bert-base-pretraining-full-step.json`.

## Scaling Records

| Batch | Status | Peak Memory (GiB) | Trace Size (GiB) | Stop |
|------:|--------|------------------:|-----------------:|------|
| 1 | complete iteration measured | 1.0183 | 18.6277 | none |
| 2 | direct validation passed, trace projected | 1.0194 | 37.2554 | none |
| 4 | direct validation passed, trace projected | 1.0229 | 74.5108 | none |
| 8 | direct validation passed, trace projected | 1.0323 | 149.0216 | none |
| 16 | direct validation passed, trace projected | 1.0467 | 298.0432 | none |
| 32 | not run | N/A | 596.0864 | projected trace size exceeds 500 GiB |

Batch 32 is the first doubling point whose projected trace size exceeds the plan ceiling. Further scaling requires explicit user approval.
