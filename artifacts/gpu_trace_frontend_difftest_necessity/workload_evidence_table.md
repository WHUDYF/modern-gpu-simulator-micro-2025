# GPU Trace Frontend Necessity Workload Evidence Table

Generated: 2026-05-04

## Go/No-Go

**Verdict**: NO-GO
- Rule: P_trace_to_sim_slice > 15% OR P_trace_to_sim_step > 15%
- Eligible measured claim-bearing rows: 3
- Detail: Fully measured claim-bearing rows exist, but none exceed the 15% threshold.

## Evidence Rows

| Workload ID | Unit | Data Label | Claim-Bearing | P_trace_to_sim (%) | Source Artifact | Provenance |
|-------------|------|------------|---------------|--------------------|-----------------|------------|
| bert-base-encoder-layer-slice | slice | measured | True | 11.49 | complete_flow_burden_ratio.json | All complete-flow components loaded from measured claim-bearing source record |
| bert-base-pretraining-full-step | step | measured | True | 0.00 | complete_flow_burden_ratio.json | All complete-flow components loaded from measured claim-bearing source record |
| llama3.1-8b-decoder-layer-slice | slice | measured | True | 0.09 | complete_flow_burden_ratio.json | All complete-flow components loaded from measured claim-bearing source record |
| llama3.1-8b-full-step | step | modeled | True | 9.47 | complete_flow_burden_ratio.json | Planning row retained for scale context only; not eligible for go/no-go |

## Control Workloads

Control workloads validate tooling only. They do not satisfy claim-bearing go/no-go requirements.
